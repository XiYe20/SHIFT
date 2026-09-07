# =========================
# WAN2.2 Flow-Matching utils
# =========================
import math
import torch
from tqdm import tqdm

class FlowMatchSchedulerWan:
    """
    Copied from DiffSynth-Studio (FlowMatchScheduler, template='Wan'), simplified to only keep WAN.
    Discrete schedule with 1000 train timesteps by default.
    """
    def __init__(self):
        self.num_train_timesteps = 1000
        self.training = False
        self.sigmas = None
        self.timesteps = None
        self.linear_timesteps_weights = None

    @staticmethod
    def set_timesteps_wan(num_inference_steps=1000, denoising_strength=1.0, shift=5):
        """
        DiffSynth-Studio only perform finetuning on the discrete sigmas we will use in the inference process
        Align the train-test behaviour. Thus, the training timesteps depends on num_inference_steps. 
        """
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000

        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_training_weight(self):
        # Copied from DiffSynth; produces a per-timestep weight curve
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        weights = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            weights = weights * (len(self.timesteps) / steps)
            weights = weights + weights[1]
        self.linear_timesteps_weights = weights

    def set_timesteps(self, num_inference_steps=1000, denoising_strength=1.0, training=False, shift=5):
        self.sigmas, self.timesteps = self.set_timesteps_wan(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            shift=shift,
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def _timestep_id(self, timestep):
        # timestep is a scalar tensor (float), match to closest discrete timestep
        if isinstance(timestep, torch.Tensor):
            t_cpu = timestep.detach().to("cpu")
        else:
            t_cpu = torch.tensor(float(timestep))
        return torch.argmin((self.timesteps - t_cpu).abs())

    def add_noise(self, original_samples, noise, timestep):
        # x_t = (1 - sigma) * x + sigma * eps
        tid = self._timestep_id(timestep)
        sigma = self.sigmas[tid].to(device=original_samples.device, dtype=original_samples.dtype)
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample, noise, timestep):
        # Flow matching target
        return noise - sample

    def training_weight(self, timestep):
        if self.linear_timesteps_weights is None:
            raise RuntimeError("training_weight called before set_timesteps(..., training=True)")
        tid = self._timestep_id(timestep.to(self.timesteps.device))
        return self.linear_timesteps_weights[tid].to(device=timestep.device, dtype=timestep.dtype)


def flow_match_sft_loss_wan(
    scheduler: FlowMatchSchedulerWan,
    model_pred: torch.Tensor,
    input_latents: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
):
    """
    Equivalent to DiffSynth FlowMatchSFTLoss core:
    mse(pred, noise - input_latents) * weight(t)
    """
    target = scheduler.training_target(input_latents, noise, timestep)
    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float())
    loss = loss * scheduler.training_weight(timestep)
    return loss


@torch.no_grad()
def wan22_sample_batched(
    pipe,
    image_01_bchw: torch.Tensor,     # (B,3,H,W) in [0,1]
    prompts,                         # list[str] length B
    negative_prompt: str,
    num_frames: int,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    max_sequence_length: int = 512,
    generator: torch.Generator = None,
    show_pgbar: bool = False,
):
    """
    Minimal WAN2.2 I2V batched sampler that bypasses WanImageToVideoPipeline.__call__.
    Fixes the expand_timesteps batch bug by NOT repeating latent_condition.
    Returns:
      video_btchw: (B,T,C,H,W) in [0,1], torch.float32 on pipe device.
    Assumptions:
      - pipe.config.expand_timesteps == True (WAN2.2)
      - pipe.config.boundary_ratio is None and pipe.transformer_2 is None (single-stage denoising)
    """
    device = pipe._execution_device if hasattr(pipe, "_execution_device") else next(pipe.transformer.parameters()).device

    if getattr(pipe.config, "expand_timesteps", False) is not True:
        raise ValueError("wan22_sample_batched expects pipe.config.expand_timesteps=True (WAN2.2).")
    if getattr(pipe.config, "boundary_ratio", None) is not None or getattr(pipe, "transformer_2", None) is not None:
        raise NotImplementedError("This minimal sampler assumes boundary_ratio=None and no transformer_2.")

    # 1) preprocess image like pipeline
    img = pipe.video_processor.preprocess(image_01_bchw, height=height, width=width).to(device, dtype=torch.float32)  # (B,C,H,W)
    B = img.shape[0]

    # 2) encode prompt / negative prompt (batch-aware)
    do_cfg = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompts,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_cfg,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    transformer_dtype = pipe.transformer.dtype
    prompt_embeds = prompt_embeds.to(dtype=transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=transformer_dtype)

    # 3) timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # 4) latents init
    num_latent_frames = (num_frames - 1) // int(pipe.vae_scale_factor_temporal) + 1
    latent_h = height // int(pipe.vae_scale_factor_spatial)
    latent_w = width // int(pipe.vae_scale_factor_spatial)
    z_dim = int(pipe.vae.config.z_dim)

    if generator is None:
        latents = torch.randn((B, z_dim, num_latent_frames, latent_h, latent_w), device=device, dtype=torch.float32)
    else:
        latents = torch.randn((B, z_dim, num_latent_frames, latent_h, latent_w), device=device, dtype=torch.float32, generator=generator)

    # 5) condition for expand_timesteps=True: only first frame
    video_condition = img.unsqueeze(2)  # (B,C,1,H,W)
    enc = pipe.vae.encode(video_condition.to(device=device, dtype=pipe.vae.dtype))
    if hasattr(enc, "latent_dist"):
        latent_condition = enc.latent_dist.mode()
    elif hasattr(enc, "latents"):
        latent_condition = enc.latents
    else:
        raise AttributeError("Could not access latents from VAE encoder output")

    # normalize condition exactly like pipeline (NO repeat(batch_size,...))
    latents_mean = torch.tensor(pipe.vae.config.latents_mean, device=device, dtype=latent_condition.dtype).view(1, z_dim, 1, 1, 1)
    latents_std_inv = (1.0 / torch.tensor(pipe.vae.config.latents_std, device=device, dtype=latent_condition.dtype)).view(1, z_dim, 1, 1, 1)
    condition = (latent_condition - latents_mean) * latents_std_inv  # (B,z,1,h,w)

    # 6) first_frame_mask (broadcast over batch)
    first_frame_mask = torch.ones(1, 1, num_latent_frames, latent_h, latent_w, dtype=torch.float32, device=device)
    first_frame_mask[:, :, 0] = 0

    # 7) denoise
    ps = int(pipe.transformer.config.patch_size[1])  # usually 2

    it = timesteps
    if show_pgbar:
        it = tqdm(timesteps, total=len(timesteps), desc="WAN sampling", dynamic_ncols=True)
    for t in it:
        latent_model_input = (1.0 - first_frame_mask) * condition + first_frame_mask * latents
        latent_model_input = latent_model_input.to(transformer_dtype)

        temp_ts = (first_frame_mask[0, 0][:, ::ps, ::ps] * t).flatten()  # (seq_len,)
        timestep = temp_ts.unsqueeze(0).expand(B, -1).to(device=device, dtype=transformer_dtype)

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_image=None,
            attention_kwargs=None,
            return_dict=False,
        )[0]

        if do_cfg:
            noise_uncond = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                encoder_hidden_states_image=None,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # final merge for expand_timesteps=True
    latents = (1.0 - first_frame_mask) * condition + first_frame_mask * latents

    # decode to [0,1] using same unnorm as pipeline
    latents = latents.to(dtype=pipe.vae.dtype)
    latents_std_inv = (1.0 / torch.tensor(pipe.vae.config.latents_std, device=device, dtype=latents.dtype)).view(1, z_dim, 1, 1, 1)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean, device=device, dtype=latents.dtype).view(1, z_dim, 1, 1, 1)
    latents = latents / latents_std_inv + latents_mean

    video_bcthw = pipe.vae.decode(latents, return_dict=False)[0]
    video_btchw = pipe.video_processor.postprocess_video(video_bcthw, output_type="pt")
    return video_btchw