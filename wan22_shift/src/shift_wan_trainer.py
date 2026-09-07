import sys
from pathlib import Path
# Make reward_model_pretrain importable
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../MotionAlignmentVDM
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'reward_model_pretrain'))

import math
import gc
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl

from peft import LoraConfig, inject_adapter_in_model, get_peft_model_state_dict, set_peft_model_state_dict

from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from wan22_flow_match_utils import FlowMatchSchedulerWan, wan22_sample_batched

from reward_model_pretrain.reward_models import MotionRewardModel
from utils import log_hist, sft_grad_norm_log_from_grads, aw_sft_grad_norm_log_from_grads, layer_grad_norm_mean_ratio
from utils import _downsample_btchw
from utils import attach_peft_mixin, clear_peft_merged_state
from collections import defaultdict

def _to_pil_rgb(img_chw_01: torch.Tensor):
    """
    img_chw_01: (3,H,W) float in [0,1]
    """
    from PIL import Image
    x = (img_chw_01.clamp(0, 1) * 255.0).to(torch.uint8)
    x = x.permute(1, 2, 0).cpu().numpy()  # HWC
    return Image.fromarray(x, mode="RGB")


def _frames_to_torch_video(frames):
    """
    frames: either
      - np.ndarray (T,H,W,C) in [0,1] or [0,255]
      - list of np.ndarray/PIL (each HWC)
    returns: torch.FloatTensor (T,C,H,W) in [0,1]
    """
    if isinstance(frames, np.ndarray):
        arr = frames
    else:
        # list of frames
        arr_list = []
        for f in frames:
            if hasattr(f, "convert"):  # PIL
                f = np.array(f.convert("RGB"))
            else:
                f = np.array(f)
            arr_list.append(f)
        arr = np.stack(arr_list, axis=0)

    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32)

    # if [0,255], normalize
    if arr.max() > 1.5:
        arr = arr / 255.0

    # (T,H,W,C) -> (T,C,H,W)
    t = torch.from_numpy(arr).float().permute(0, 3, 1, 2).contiguous()
    return t


def flow_match_sft_loss_wan_vec(scheduler: FlowMatchSchedulerWan,
                               model_pred: torch.Tensor,
                               input_latents: torch.Tensor,
                               timestep: torch.Tensor,
                               noise: torch.Tensor):
    """
    Per-sample flow-matching loss, returns (B,).
    """
    target = scheduler.training_target(input_latents, noise, timestep)
    mse = (model_pred.float() - target.float()).pow(2)
    # reduce all dims except batch
    loss_vec = mse.flatten(start_dim=1).mean(dim=1)  # (B,)
    loss_vec = loss_vec * scheduler.training_weight(timestep)
    return loss_vec


class ShiftTrainerWAN(pl.LightningModule):
    """
    SHIFT for WAN2.2 (TI2V-5B Diffusers).
    Key difference vs SVD SHIFT: buffer stores LATENTS (not pixels) for feasibility at 704x1280.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sampling_cfg = getattr(self.config, "sampling", None)
        self.automatic_optimization = False

        # dtype
        if self.config.precision == "bf16":
            self.torch_dtype = torch.bfloat16
        elif self.config.precision == "32":
            self.torch_dtype = torch.float32
        else:
            raise ValueError(f"Invalid precision '{self.config.precision}'")

        # FlowMatch scheduler (discrete sigmas aligned with inference steps)
        self.fm_scheduler = FlowMatchSchedulerWan()
        self.fm_scheduler.set_timesteps(
            num_inference_steps=self.sampling_cfg['num_inference_steps'],
            training=True,
            shift=getattr(self.config, "sigma_shift", 5),
        )

        self._initialize_pipeline()
        self._initialize_denoiser_and_lora()
        self._initialize_reward_models()
        self._initialize_buffer()

        self.g_step = 0
        self.d_step = 0

        self._capture_grads = False
    # ---------------------------
    # init
    # ---------------------------
    def _initialize_pipeline(self):
        model_id = self.config.pretrained_wan_path
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

        self.wan_pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=self.torch_dtype,
        )

        self.denoiser = self.wan_pipe.transformer
        self.vae = self.wan_pipe.vae
        self.text_encoder = self.wan_pipe.text_encoder

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def _initialize_denoiser_and_lora(self):
        self.denoiser.requires_grad_(False)

        # =================================================================
        # Load pretrained SFT LoRA and merge into base model (if provided)
        # =================================================================
        if hasattr(self.config, "pretrained_sft_lora_path") and self.config.pretrained_sft_lora_path:
            sft_lora_path = self.config.pretrained_sft_lora_path
            print(f"\n[SFT Merge] Detected SFT LoRA path: {sft_lora_path}")
            print("[SFT Merge] Loading and merging SFT LoRA weights into base model...")
            try:
                checkpoint = torch.load(sft_lora_path, map_location='cpu', weights_only=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint from {sft_lora_path}: {e}")

            self.denoiser = attach_peft_mixin(self.denoiser)

            train_cfg = checkpoint.get("config_dict", None)
            if train_cfg is None or "lora_config" not in train_cfg:
                raise ValueError("Checkpoint is missing 'config_dict.lora_config'; cannot rebuild LoRA adapter.")

            sft_lora_config = dict(train_cfg["lora_config"])  # shallow copy
            sft_lora_config_obj = LoraConfig(**sft_lora_config)

            adapter_name_sft = train_cfg.get("lora_adapter_name")
            if not adapter_name_sft:
                adapter_name_sft = "sft_pretrained"
            self.denoiser.add_adapter(sft_lora_config_obj, adapter_name=adapter_name_sft)

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Extract LoRA weights (remove 'denoiser.' prefix)
                lora_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('denoiser.'):
                        lora_state_dict[key[len('denoiser.'):]] = value

                if lora_state_dict:
                    set_peft_model_state_dict(self.denoiser, lora_state_dict, adapter_name=adapter_name_sft)
                    print(f"Successfully loaded {len(lora_state_dict)} LoRA parameters from checkpoint: {sft_lora_path}")
                    # Fuse SFT LoRA into base weights so the training adapter builds on top
                    self.denoiser.set_adapter(adapter_name_sft)
                    self.denoiser.fuse_lora(adapter_names=[adapter_name_sft])
                    self.denoiser.delete_adapters(adapter_name_sft)
                    clear_peft_merged_state(self.denoiser)
                    print(f"[SFT Merge] Successfully fused adapter '{adapter_name_sft}' into base denoiser.")
                else:
                    raise ValueError(f"No LoRA weights found in checkpoint: {sft_lora_path}")
            else:
                raise ValueError(f"Checkpoint is missing 'state_dict': {sft_lora_path}")

        lora_cfg = LoraConfig(**dict(self.config.lora_config))
        self.denoiser = inject_adapter_in_model(lora_cfg, self.denoiser)

        self.trainable_params = [p for p in self.denoiser.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in self.trainable_params)
        print(f"Total trainable LoRA parameters: {total_params/1e6:.3f} M")

        self._lora_param_names = []
        for name, p in self.denoiser.named_parameters():
            if not p.requires_grad:
                continue
            self._lora_param_names.append(name)

            def _make_hook(n):
                def _hook(grad):
                    if (not getattr(self, "_capture_grads", False)) or grad is None:
                        return grad
                    g = grad.detach().float()
                    prev = self._captured_total_grads.get(n, None)
                    self._captured_total_grads[n] = g if prev is None else (prev + g)
                    return grad
                return _hook

            p.register_hook(_make_hook(name))

    def _initialize_reward_models(self):
        self.motion_reward_model = MotionRewardModel(self.config.ic_reward_model_config, self.config.lc_reward_model_config)

        self.ic_reward_weight = float(getattr(self.config, "ic_reward_model_config", {}).get("reward_weight", 1.0))
        self.lc_reward_weight = float(getattr(self.config, "lc_reward_model_config", {}).get("reward_weight", 1.0))
        self.train_ic_reward = self.config.train_reward_models and self.ic_reward_weight > 0.0
        self.train_lc_reward = self.config.train_reward_models and self.lc_reward_weight > 0.0

        if self.config.train_reward_models:
            if self.train_ic_reward:
                self.motion_reward_model.ic_reward_model.unfreeze_params()
            if self.train_lc_reward:
                self.motion_reward_model.lc_reward_model.unfreeze_params()
        else:
            self.motion_reward_model.freeze_params()

        self.motion_reward_model.eval()

        self.motion_reward_ema = None
        if getattr(self.config, "reward_model_ema", True):
            from torch_ema import ExponentialMovingAverage
            params = []
            if self.train_ic_reward:
                params += self.motion_reward_model.ic_reward_model.get_trainable_params()
            if self.train_lc_reward:
                params += self.motion_reward_model.lc_reward_model.get_trainable_params()
            if len(params) > 0:
                decay = float(getattr(self.config, "reward_model_ema_decay", 0.999))
                self.motion_reward_ema = ExponentialMovingAverage(params, decay=decay)

    def _initialize_buffer(self):
        self.buffer_size = int(self.config.buffer_size)
        self.buffer_device = str(getattr(self.config, "buffer_device", "cpu"))

        global_world_size = int(self.config.num_nodes) * int(self.config.gpus_per_node)
        self.iterations_per_chunk = self.buffer_size // (global_world_size * int(self.config.batch_size))

        self.passes_per_buffer = int(self.config.passes_per_buffer)
        self.buffer_update_frequency = self.iterations_per_chunk * self.passes_per_buffer

        self.buffer_initialized = False
        self.current_pass = 0

        # Latent buffer variables (NOT pixel videos)
        self.buffer_variables = [
            "prompts",                  # list[str] length B
            "prompt_embeds",            # (B, seq, dim)
            "real_latents",             # (B, z, Tz, Hz, Wz)
            "cond_latents",             # (B, z, Tz, Hz, Wz)
            "real_video",               # (B, c, T, H, W)
            "fake_video",               # (P, B, c, T, H, W)
            "fake_latents",             # (P, B, z, Tz, Hz, Wz)
            "fake_ave_rewards",         # (P, B)
            "fake_ic_rewards",          # (P, B)
            "fake_lc_rewards",          # (P, B)
            "fake_advantage",           # (P, B)
            "unweighted_fake_advantage", # (P, B)
            "real_advantage",           # (B,)
            "real_ave_rewards",         # (B,)
            "real_ic_rewards",          # (B,)
            "real_lc_rewards",          # (B,)
        ]

        self.buffer = {k: None for k in self.buffer_variables}

    def _flush_unet_optimizer_if_needed(self, batch_idx: int):
        """
        If buffer ends mid-accumulation, force an optimizer step like svd_shift.
        """
        accum_steps = int(getattr(self.config, "accum_grad_steps", 1))
        accum_steps = max(1, accum_steps)

        if ((batch_idx + 1) % accum_steps) == 0:
            return
        if not any((p.grad is not None) for p in self.trainable_params):
            return

        opt = self.optimizers()
        opt = opt[0] if isinstance(opt, (list, tuple)) else opt

        self.clip_gradients(opt, gradient_clip_val=float(getattr(self.config, "gradient_clip", 1.0)), gradient_clip_algorithm="norm")
        opt.step()
        opt.zero_grad(set_to_none=True)
        self.g_step += 1

        lr_schedulers = self.lr_schedulers()
        lr0 = lr_schedulers[0] if isinstance(lr_schedulers, (list, tuple)) else lr_schedulers
        if lr0 is not None:
            lr0.step()

    # ---------------------------
    # lightning hooks
    # ---------------------------
    def on_fit_start(self):
        self.wan_pipe.to(self.device)
        self.denoiser.to(self.device).train()

        self.motion_reward_model.to(device=self.device, dtype=torch.float32)
        self.motion_reward_model.eval()

        if getattr(self, "motion_reward_ema", None) is not None:
            for sp in self.motion_reward_ema.shadow_params:
                if sp.device != self.device:
                    sp.data = sp.data.to(self.device)

        if not self.buffer_initialized:
            self.recompute_buffer(1)

    def configure_optimizers(self):
        # Param group 0: denoiser LoRA
        param_groups = [dict(
            params=self.trainable_params,
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=float(getattr(self.config, "weight_decay", 0.0)),
        )]

        # Param group 1/2: reward models (optional)
        if self.train_ic_reward:
            param_groups.append(dict(
                params=self.motion_reward_model.ic_reward_model.get_trainable_params(),
                lr=float(getattr(self.config, "ic_reward_model_lr", self.config.lr)),
                betas=(float(getattr(self.config, "ic_reward_model_beta1", self.config.beta1)),
                       float(getattr(self.config, "ic_reward_model_beta2", self.config.beta2))),
                weight_decay=float(getattr(self.config, "ic_reward_model_weight_decay", 0.0)),
            ))
        if self.train_lc_reward:
            param_groups.append(dict(
                params=self.motion_reward_model.lc_reward_model.get_trainable_params(),
                lr=float(getattr(self.config, "lc_reward_model_lr", self.config.lr)),
                betas=(float(getattr(self.config, "lc_reward_model_beta1", self.config.beta1)),
                       float(getattr(self.config, "lc_reward_model_beta2", self.config.beta2))),
                weight_decay=float(getattr(self.config, "lc_reward_model_weight_decay", 0.0)),
            ))

        optimizer = torch.optim.AdamW(param_groups)

        if self.config.scheduler == "linear_warmup":
            def lr_lambda(step):
                w = int(getattr(self.config, "warmup_steps", 0))
                if w <= 0:
                    return 1.0
                if step < w:
                    return float(step) / float(max(1, w))
                return 1.0
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda),
                "interval": "step",
                "frequency": 1,
            }
        elif self.config.scheduler == "stepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.accum_grad_steps, gamma=0.99)
        else:
            raise ValueError(f"Scheduler {self.config.scheduler} not supported")

        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        # save LoRA weights only
        lora_sd = get_peft_model_state_dict(self.denoiser, adapter_name="default")
        checkpoint["state_dict"] = {f"denoiser.{k}": v.cpu() for k, v in lora_sd.items()}

        if getattr(self.config, "train_reward_models", False):
            rm_sd = self.motion_reward_model.state_dict()
            checkpoint["motion_reward_state_dict"] = {f"motion_reward_model.{k}": v.cpu() for k, v in rm_sd.items()}

        if hasattr(self, "config"):
            import yaml
            if hasattr(self.config, "config_path") and self.config.config_path:
                cfg_path = Path(self.config.config_path)
                if cfg_path.exists():
                    checkpoint["config_yaml"] = cfg_path.read_text()
                    checkpoint["config_path"] = str(cfg_path)
            checkpoint["config_dict"] = vars(self.config)

        if getattr(self, "motion_reward_ema", None) is not None:
            try:
                checkpoint["motion_reward_ema_state"] = self.motion_reward_ema.state_dict()
            except Exception as e:
                print(f"Warning: failed to save motion_reward_ema state: {e}")

    # ---------------------------
    # core: shared loss on LATENTS
    # ---------------------------
    def shared_step_latents(self,
                           input_latents: torch.Tensor,
                           cond_latents: torch.Tensor,
                           prompt_embeds: torch.Tensor,
                           stage: str = "train",
                           noise: torch.Tensor = None,
                           timestep: torch.Tensor = None):
        """
        input_latents: (B, z, Tz, Hz, Wz) normalized
        cond_latents:  (B, z, Tz, Hz, Wz) normalized, first frame condition encoded as in pipeline
        prompt_embeds: (B, seq, dim)
        Returns:
          loss_vec: (B,)
          noise, timestep (to reuse in AW part)
        """
        B = input_latents.shape[0]

        # sample timestep/noise (shared)
        if timestep is None:
            max_b = float(getattr(self.config, "max_timestep_boundary", 1.0))
            min_b = float(getattr(self.config, "min_timestep_boundary", 0.0))
            max_id = int(max_b * len(self.fm_scheduler.timesteps))
            min_id = int(min_b * len(self.fm_scheduler.timesteps))
            tid = torch.randint(min_id, max_id, (1,))
            timestep = self.fm_scheduler.timesteps[tid].to(dtype=torch.float32, device=self.device)
        if noise is None:
            noise = torch.randn_like(input_latents)

        noisy_latents = self.fm_scheduler.add_noise(input_latents, noise, timestep)
        transformer_dtype = self.denoiser.dtype

        # expand_timesteps logic (same as WAN2.2 SFT)
        assert getattr(self.wan_pipe.config, "expand_timesteps", True) is True, "WAN2.2 expects expand_timesteps=True"
        num_latent_frames = cond_latents.shape[2]
        latent_h, latent_w = cond_latents.shape[3], cond_latents.shape[4]

        first_frame_mask = torch.ones(1, 1, num_latent_frames, latent_h, latent_w, dtype=torch.float32, device=self.device)
        first_frame_mask[:, :, 0] = 0

        latent_model_input = (1 - first_frame_mask) * cond_latents + first_frame_mask * noisy_latents
        latent_model_input = latent_model_input.to(transformer_dtype)

        ps = int(self.denoiser.config.patch_size[1])
        temp_ts = (first_frame_mask[0, 0][:, ::ps, ::ps] * timestep).flatten()
        timestep_model = temp_ts.unsqueeze(0).expand(B, -1).to(dtype=transformer_dtype)

        prompt_embeds = prompt_embeds.to(dtype=transformer_dtype)
        image_embeds = None  # WAN2.2 TI2V-5B
        with torch.set_grad_enabled(stage == "train"):
            model_pred = self.denoiser(
                hidden_states=latent_model_input,
                timestep=timestep_model,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]

        loss_vec = flow_match_sft_loss_wan_vec(
            scheduler=self.fm_scheduler,
            model_pred=model_pred,
            input_latents=input_latents,
            timestep=timestep,
            noise=noise,
        )
        return loss_vec, noise, timestep

    # ---------------------------
    # reward/advantage
    # ---------------------------
    @torch.no_grad()
    def compute_reward(self, sample_video_01_btc_hw: torch.Tensor):
        """
        sample_video_01_btc_hw: (B,T,C,H,W) in [0,1]
        """
        prev_motion_training = self.motion_reward_model.training
        prev_trackformer_training = self.motion_reward_model.lc_reward_model.traj_discriminator.track_former.training
        self.motion_reward_model.eval()
        try:
            if getattr(self, "motion_reward_ema", None) is not None and getattr(self.config, "reward_model_use_ema_for_diffusion", True):
                with self.motion_reward_ema.average_parameters():
                    ic, lc, ave, _ = self.motion_reward_model(sample_video_01_btc_hw)
                    return ic.reshape(-1), lc.reshape(-1), ave.reshape(-1)
            ic, lc, ave, _ = self.motion_reward_model(sample_video_01_btc_hw)
            return ic.reshape(-1), lc.reshape(-1), ave.reshape(-1)
        finally:
            if prev_motion_training:
                self.motion_reward_model.train()
            if prev_trackformer_training:
                self.motion_reward_model.lc_reward_model.traj_discriminator.track_former.train()

    def compute_advantage(self, rewards: torch.Tensor):
        """
        rewards: (P,B)
        """
        mean = rewards.mean(dim=0, keepdim=True)
        std = rewards.std(dim=0, keepdim=True) + 1e-8
        if getattr(self.config, "advantage_norm_with_std", False):
            adv = (rewards - mean) / std
        else:
            adv = rewards - mean

        adv_clip = float(getattr(self.config, "adv_clip_max", 10.0))
        unweighted = torch.clamp(adv, -adv_clip, adv_clip)

        pos_lambda = float(getattr(self.config, "pos_adv_lambda", 1.0))
        weighted = unweighted * pos_lambda * (unweighted > 0) + unweighted * (unweighted <= 0)
        return weighted, unweighted

    # ---------------------------
    # encoding helpers
    # ---------------------------
    @torch.no_grad()
    def _encode_video_to_latents(self, video_btchw: torch.Tensor) -> torch.Tensor:
        """
        video_btchw: (B,T,C,H,W) in [-1,1]
        returns latents: (B,z,Tz,Hz,Wz) normalized (matches WAN pipeline)
        """
        if video_btchw.dim() != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(video_btchw.shape)}")

        b, t, c, h, w = video_btchw.shape
        t_factor = int(getattr(self.wan_pipe, "vae_scale_factor_temporal", 4))
        if t % t_factor != 1:
            new_t = ((t - 1) // t_factor) * t_factor + 1
            video_btchw = video_btchw[:, :new_t]
            b, t, c, h, w = video_btchw.shape

        video_bcthw = video_btchw.permute(0, 2, 1, 3, 4).contiguous()
        vae_dtype = next(self.vae.parameters()).dtype
        video_bcthw = video_bcthw.to(device=self.device, dtype=vae_dtype)

        enc = self.vae.encode(video_bcthw)

        if hasattr(enc, "latent_dist"):
            latents = enc.latent_dist.mode()
        elif hasattr(enc, "latents"):
            latents = enc.latents
        else:
            raise AttributeError("Could not access latents from VAE encoder output")

        z_dim = int(self.vae.config.z_dim)
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, z_dim, 1, 1, 1)
        latents_std_inv = (1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype)).view(1, z_dim, 1, 1, 1)
        latents = (latents - latents_mean) * latents_std_inv
        return latents.to(dtype=self.torch_dtype)

    @torch.no_grad()
    def _encode_prompt(self, prompts):
        """
        prompts: list[str] length B
        returns prompt_embeds: (B, seq, dim)
        """
        pe, _ = self.wan_pipe.encode_prompt(
            prompt=prompts,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=getattr(self.config, "max_sequence_length", 512),
            device=self.device,
        )
        return pe.to(dtype=self.torch_dtype)

    # ---------------------------
    # buffer collection
    # ---------------------------
    @torch.no_grad()
    def collect_data(self, batch, batch_start: int):
        """
        batch: dict from dataset
          - batch["video"]: (B,T,C,H,W) in [-1,1]
          - batch["prompt"]: list[str] (or str)
        """
        video = batch["video"].to(self.device, non_blocking=True).to(self.torch_dtype)
        prompt = batch["prompt"]
        if isinstance(prompt, str):
            prompts = [prompt] * video.shape[0]
        else:
            prompts = list(prompt)

        B, T, C, H, W = video.shape

        # donwsmaple real video for bufer
        ds_ratio = int(getattr(self.config, "buffer_video_downsample_ratio", 1))
        real_video_buf_01 = ((video.float() + 1.0) * 0.5).clamp(0, 1)     # (B,T,C,H,W)
        real_video_buf_01 = _downsample_btchw(real_video_buf_01, ds_ratio)           # (B,T,C,H//r,W//r)
        real_ic_reward, real_lc_reward, real_ave_reward = self.compute_reward(real_video_buf_01.float())  # (B,)
        if self.config.debug_mode and self.global_rank == 0:
            if not hasattr(self, 'real_vis_idx'):
                self.real_vis_idx = 0
            else:
                self.real_vis_idx += 1
            print(f'real{self.real_vis_idx}', real_ic_reward, real_lc_reward, real_ave_reward)
            if getattr(self.config, "visualize_example", False):
                import imageio.v3 as iio
                temp = real_video_buf_01.float() * 255
                temp = temp.cpu().numpy().astype(np.uint8)
                temp = temp.transpose(0, 1, 3, 4, 2)
                iio.imwrite(f'real_sample{self.real_vis_idx}.mp4', temp[0])
        # seed for stable sampling per buffer batch
        gpu_num = torch.cuda.current_device()
        pl.seed_everything(int(gpu_num + self.config.seed + 10 * self.global_step + 100 * batch_start))

        # real latents / cond latents / prompt embeds
        prompt_embeds = self._encode_prompt(prompts)

        real_latents = self._encode_video_to_latents(video)

        # build first-frame condition video, then encode to condition latents
        first = video[:, 0].to(dtype=torch.float32)  # (B,C,H,W)
        video_condition = torch.cat(
            [first.unsqueeze(2), torch.zeros((B, C, T - 1, H, W), device=self.device, dtype=first.dtype)],
            dim=2,
        )  # (B,C,T,H,W)
        cond_latents = self._encode_video_to_latents(video_condition.permute(0, 2, 1, 3, 4).contiguous())

        total_p = int(self.config.num_sample_videos_per_prompt)
        group_p = int(getattr(self.config, "sample_collect_group_size", total_p))
        group_p = max(1, min(group_p, total_p))

        all_fake_latents = []
        all_fake_videos = []
        all_fake_ic = []
        all_fake_lc = []
        all_fake_ave = []

        # --------------------------
        # sampling config (aligned)
        # --------------------------
        sampling_cfg = self.sampling_cfg
        # size: prefer sampling.target_size, fallback to training target_vid_size
        target_size = sampling_cfg.get("target_size", getattr(self.config, "target_vid_size", None))
        if target_size is None:
            raise ValueError("Missing sampling target size: set sampling.target_size or target_vid_size")
        num_frames, height, width = [int(x) for x in target_size]

        # fixed guidance scale: sampling.guidance_scale
        fixed_guidance = sampling_cfg.get("guidance_scale", None)

        # negative prompt: prefer sampling.negative_prompt, fallback to sample_negative_prompt
        neg_prompt = sampling_cfg.get("negative_prompt", getattr(self.config, "sample_negative_prompt", None))

        # pipeline sampling (cur_p loops)
        for start_p in range(0, total_p, group_p):
            cur_p = min(group_p, total_p - start_p)

            cur_group_latents = []
            cur_group_video = []
            cur_group_ic = []
            cur_group_lc = []
            cur_group_ave = []

            for _ in range(cur_p):
                # disable tqdm unless rank0
                if self.global_rank != 0 and hasattr(self.wan_pipe, "set_progress_bar_config"):
                    self.wan_pipe.set_progress_bar_config(disable=True)

                image_01_bchw = ((first + 1.0) * 0.5).clamp(0, 1).to(dtype=torch.float32)

                fake_video_01 = wan22_sample_batched(
                    pipe=self.wan_pipe,
                    image_01_bchw=image_01_bchw,
                    prompts=prompts,
                    negative_prompt=neg_prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=int(sampling_cfg.get("num_inference_steps")),
                    guidance_scale=fixed_guidance,
                    max_sequence_length=int(getattr(self.config, "max_sequence_length", 512)),
                    show_pgbar=(self.global_rank == 0)
                )
                fake_video_buf_01 = _downsample_btchw(fake_video_01.float(), ds_ratio)  # (B,T,C,H//r,W//r)
                #use downsampled fake video for reward computation
                #since the later reward model training use the downsampled video
                ic_r, lc_r, ave_r = self.compute_reward(fake_video_buf_01.float())
                if self.config.debug_mode and self.global_rank == 0:
                    if not hasattr(self, 'vis_idx'):
                        self.vis_idx = 0
                    else:
                        self.vis_idx += 1
                    print(f'fake{self.vis_idx}', ic_r, lc_r, ave_r) 
                    if getattr(self.config, "visualize_example", False):
                        import imageio.v3 as iio
                        temp = fake_video_buf_01.float() * 255
                        temp = temp.cpu().numpy().astype(np.uint8)
                        temp = temp.transpose(0, 1, 3, 4, 2)
                        iio.imwrite(f'fake_sample{self.vis_idx}.mp4', temp[0])
                # encode fake pixels -> fake latents (for training buffer)
                fake_video_m11 = fake_video_01 * 2.0 - 1.0
                fake_latents = self._encode_video_to_latents(fake_video_m11)

                cur_group_latents.append(fake_latents.unsqueeze(0))       # (1,B,z,Tz,Hz,Wz)
                cur_group_video.append(fake_video_buf_01.unsqueeze(0))
                cur_group_ic.append(ic_r.unsqueeze(0))                    # (1,B)
                cur_group_lc.append(lc_r.unsqueeze(0))
                cur_group_ave.append(ave_r.unsqueeze(0))

                # free
                del fake_video_01, fake_video_m11, fake_latents, fake_video_buf_01

            all_fake_latents.append(torch.cat(cur_group_latents, dim=0))  # (cur_p,B,...)
            all_fake_videos.append(torch.cat(cur_group_video, dim=0))
            all_fake_ic.append(torch.cat(cur_group_ic, dim=0))            # (cur_p,B)
            all_fake_lc.append(torch.cat(cur_group_lc, dim=0))
            all_fake_ave.append(torch.cat(cur_group_ave, dim=0))

        fake_latents = torch.cat(all_fake_latents, dim=0)         # (P,B,z,Tz,Hz,Wz)
        fake_videos_buf_01 = torch.cat(all_fake_videos, dim=0)
        fake_ic_rewards = torch.cat(all_fake_ic, dim=0)           # (P,B)
        fake_lc_rewards = torch.cat(all_fake_lc, dim=0)
        fake_ave_rewards = torch.cat(all_fake_ave, dim=0)
        
        fake_adv, unweighted_adv = self.compute_advantage(fake_ave_rewards)

        # compute real_advantage (hinge) for asymmetric_grpo, like svd_shift
        adv_clip = float(getattr(self.config, "adv_clip_max", 10.0))
        fake_mean_reward = fake_ave_rewards.float().mean(dim=0)           # (B,)
        real_advantage = (real_ave_reward.float() - fake_mean_reward).clamp(min=0.0, max=adv_clip)

        torch.cuda.empty_cache()

        return {
            "prompts": prompts,  # list[str]
            "prompt_embeds": prompt_embeds,
            "real_video": real_video_buf_01,
            "fake_video": fake_videos_buf_01,
            "real_latents": real_latents,
            "cond_latents": cond_latents,
            "fake_latents": fake_latents.to(self.torch_dtype),
            "fake_ave_rewards": fake_ave_rewards,
            "fake_ic_rewards": fake_ic_rewards,
            "fake_lc_rewards": fake_lc_rewards,
            "fake_advantage": fake_adv,
            "unweighted_fake_advantage": unweighted_adv,
            "real_advantage": real_advantage,
            "real_ave_rewards": real_ave_reward,
            "real_ic_rewards": real_ic_reward,
            "real_lc_rewards": real_lc_reward,
        }

    # ---------------------------
    # buffer mechanics (copied structure from svd_shift)
    # ---------------------------
    @torch.no_grad()
    def recompute_buffer(self, chunk_id: int):
        self.denoiser.eval()
        self.motion_reward_model.eval()

        zero_based_chunk = chunk_id - 1
        datamodule = self.trainer.datamodule
        sampler = datamodule.sampler

        chunk_indices = sampler.get_chunk_indices(zero_based_chunk)

        local_rank = self.global_rank
        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else torch.cuda.device_count()

        per_gpu_count = len(chunk_indices) // world_size
        start_idx = local_rank * per_gpu_count
        end_idx = start_idx + per_gpu_count if local_rank < world_size - 1 else len(chunk_indices)
        local_indices = chunk_indices[start_idx:end_idx]

        processing_batch_size = int(self.config.batch_size)

        from torch.utils.data import DataLoader
        from utils import ListSampler  # from wan22_shift/src/utils.py

        if not hasattr(self, "_recompute_sampler"):
            self._recompute_sampler = ListSampler([])
            nw = int(getattr(datamodule, "num_workers", 0))
            loader_kwargs = dict(
                batch_size=processing_batch_size,
                sampler=self._recompute_sampler,
                drop_last=False,
            )
            if nw > 0:
                loader_kwargs.update(dict(
                    num_workers=nw,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=2,
                ))
            else:
                loader_kwargs.update(dict(num_workers=0))
            self._recompute_loader = DataLoader(datamodule.train_dataset, **loader_kwargs)

        self._recompute_sampler.set_indices(local_indices)

        accumulator = {k: [] for k in self.buffer_variables}

        for batch_id, batch in enumerate(self._recompute_loader):
            batch_start = batch_id * processing_batch_size
            collected = self.collect_data(batch, batch_start)

            # move tensors to buffer device if needed; keep prompts as python list
            for k in self.buffer_variables:
                v = collected[k]
                if torch.is_tensor(v) and self.buffer_device == "cpu":
                    collected[k] = v.to("cpu", non_blocking=True).pin_memory()

            for k in self.buffer_variables:
                accumulator[k].append(collected[k])

            if self.global_rank == 0:
                print(f"[chunk {chunk_id}] rank0 processed {batch_id+1} batches")

        for k in self.buffer_variables:
            self.buffer[k] = accumulator[k]

        self.buffer_initialized = True

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        self.denoiser.train()

    def reshuffle_buffer(self):
        if not self.buffer_initialized:
            return
        num_batches = len(self.buffer["real_latents"])
        if num_batches <= 1:
            return

        seed = int(self.config.seed + self.global_step * 100 + self.current_pass * 10000)
        orig_rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        perm = torch.randperm(num_batches).tolist()
        torch.random.set_rng_state(orig_rng_state)

        for k in self.buffer_variables:
            if self.buffer[k] is None:
                continue
            original = self.buffer[k].copy()
            for i, idx in enumerate(perm):
                self.buffer[k][i] = original[idx]

        gc.collect()
        torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def generate_train_batch(self, batch_idx: int):
        bi = batch_idx % self.iterations_per_chunk

        prompts = self.buffer["prompts"][bi]
        prompt_embeds = self.buffer["prompt_embeds"][bi].to(self.device, non_blocking=True)
        real_latents = self.buffer["real_latents"][bi].to(self.device, non_blocking=True)
        cond_latents = self.buffer["cond_latents"][bi].to(self.device, non_blocking=True)
        fake_latents = self.buffer["fake_latents"][bi].to(self.device, non_blocking=True)

        fake_adv = self.buffer["fake_advantage"][bi].to(self.device, non_blocking=True)
        fake_ic = self.buffer["fake_ic_rewards"][bi].to(self.device, non_blocking=True)
        fake_lc = self.buffer["fake_lc_rewards"][bi].to(self.device, non_blocking=True)
        fake_ave = self.buffer["fake_ave_rewards"][bi].to(self.device, non_blocking=True)
        real_adv = self.buffer["real_advantage"][bi].to(self.device, non_blocking=True)
        real_ic = self.buffer["real_ic_rewards"][bi].to(self.device, non_blocking=True)
        real_lc = self.buffer["real_lc_rewards"][bi].to(self.device, non_blocking=True)
        real_ave = self.buffer["real_ave_rewards"][bi].to(self.device, non_blocking=True)

        return prompts, prompt_embeds, real_latents, cond_latents, fake_latents, fake_adv, fake_ic, fake_lc, fake_ave, real_adv, real_ic, real_lc, real_ave

    # ---------------------------
    # training loop (manual opt, structure from svd_shift)
    # ---------------------------
    def _ds_set_boundary(self, is_boundary: bool):
        engine = getattr(self.trainer.strategy, "model", None)
        if hasattr(engine, "set_gradient_accumulation_boundary"):
            engine.set_gradient_accumulation_boundary(is_boundary)

    @torch.no_grad()
    def _unnormalize_latents_for_vae(self, latents_norm: torch.Tensor) -> torch.Tensor:
        """
        Inverse of your _encode_video_to_latents() normalization:
        latents_norm = (latents - mean) * (1/std)
        => latents = latents_norm * std + mean
        """
        z_dim = int(self.vae.config.z_dim)
        mean = torch.tensor(self.vae.config.latents_mean, device=latents_norm.device, dtype=latents_norm.dtype).view(1, z_dim, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=latents_norm.device, dtype=latents_norm.dtype).view(1, z_dim, 1, 1, 1)
        return latents_norm * std + mean

    @torch.no_grad()
    def _decode_latents_to_video01(self, latents_norm_bzthw: torch.Tensor, chunk: int = 1) -> torch.Tensor:
        """
        latents_norm_bzthw: (B,z,T,H,W) normalized
        returns video_01: (B,T,C,H,W) in [0,1]
        """
        latents = self._unnormalize_latents_for_vae(latents_norm_bzthw).to(dtype=torch.float32)

        # decode in small batches to save memory
        outs = []
        B = latents.shape[0]
        chunk = max(1, int(chunk))
        for s in range(0, B, chunk):
            x = latents[s:s+chunk]
            dec = self.vae.decode(x)
            dec = dec.sample if hasattr(dec, "sample") else dec
            # expected (b,c,t,h,w) in [-1,1]
            if dec.dim() != 5:
                raise RuntimeError(f"Unexpected VAE decode shape: {tuple(dec.shape)}")
            dec = dec.clamp(-1, 1)
            dec_01 = ((dec + 1.0) * 0.5).clamp(0, 1)
            outs.append(dec_01)
        vid_bcthw = torch.cat(outs, dim=0)
        vid_btchw = vid_bcthw.permute(0, 2, 1, 3, 4).contiguous()
        return vid_btchw

    def train_motion_reward_model_from_buffer(self):
        """
        Reward-model GAN update (parity with svd_shift), but using latent-buffer:
        - sample real_latents and fake_latents from buffer
        - decode to pixels [0,1]
        - train reward discriminators with BCE logits
        """
        opt = self.optimizers()
        opt = opt[0] if isinstance(opt, (list, tuple)) else opt

        self.motion_reward_model.to(self.device)
        self.motion_reward_model.train()

        # buffer lists
        buf_real = self.buffer["real_video"]   # list of (B, T, C, H, W) in [0, 1]
        buf_fake = self.buffer["fake_video"]   # list of (P, B, T, C, H, W) in [0, 1]
        num_buf_batches = len(buf_real)

        # how many RM steps per buffer (same logic as svd_shift)
        unet_accum = max(1, int(getattr(self.config, "accum_grad_steps", 1)))
        unet_opt_steps_per_buffer = int(math.ceil(self.buffer_update_frequency / float(unet_accum)))

        d_to_g = float(getattr(self.config, "reward_model_d_to_g", 1.0))
        rm_steps = int(round(d_to_g * unet_opt_steps_per_buffer))
        rm_steps = max(1, rm_steps)

        K = max(1, int(getattr(self.config, "reward_model_batch_multiplier", 1)))
        
        for step in range(rm_steps):
            idxs = torch.randint(0, num_buf_batches, (K,), device=self.device).tolist()

            real_vid_list = []
            fake_vid_list = []

            for j, bi in enumerate(idxs):
                real_b = buf_real[bi].to(self.device, non_blocking=True).float()     # (B,z,T,H,W)
                fake_b = buf_fake[bi]                                                # (P,B,z,T,H,W)
                p = (step + j) % fake_b.shape[0]
                fake_sel = fake_b[p].to(self.device, non_blocking=True).float()      # (B,z,T,H,W)
                real_vid_list.append(real_b)
                fake_vid_list.append(fake_sel)

            real_vid = torch.cat(real_vid_list, dim=0)
            fake_vid = torch.cat(fake_vid_list, dim=0)

            opt.zero_grad(set_to_none=True)
            ic_loss = torch.tensor(0.0, device=self.device)
            if getattr(self, "train_ic_reward", False):
                _, ic_logits_real = self.motion_reward_model.ic_reward_model(real_vid)
                ic_loss_real = F.binary_cross_entropy_with_logits(ic_logits_real.float(), torch.ones_like(ic_logits_real).float())
                if getattr(self.config, "use_deepspeed", False):
                    self._ds_set_boundary(False)
                self.manual_backward(self.ic_reward_weight * ic_loss_real)

                _, ic_logits_fake = self.motion_reward_model.ic_reward_model(fake_vid)
                ic_loss_fake = F.binary_cross_entropy_with_logits(ic_logits_fake.float(), torch.zeros_like(ic_logits_fake).float())
                if getattr(self.config, "use_deepspeed", False):
                    self._ds_set_boundary(False)
                self.manual_backward(self.ic_reward_weight * ic_loss_fake)

                ic_loss = ic_loss_real + ic_loss_fake
                if self.global_rank == 0 and self.config.debug_mode:
                    print('step', step, 'ic_loss_real', ic_loss_real.item(), 'ic_loss_fake', ic_loss_fake.item())
                    print('step', step, 'ic_logits_real', ic_logits_real.mean().item(), 'ic_logits_fake', ic_logits_fake.mean().item())

            lc_loss = torch.tensor(0.0, device=self.device)
            if getattr(self, "train_lc_reward", False):
                _, lc_logits_real, queries_point = self.motion_reward_model.lc_reward_model(real_vid, queries=None)
                lc_loss_real = F.binary_cross_entropy_with_logits(lc_logits_real.float(), torch.ones_like(lc_logits_real).float())
                if getattr(self.config, "use_deepspeed", False):
                    self._ds_set_boundary(False)
                self.manual_backward(self.lc_reward_weight * lc_loss_real)

                _, lc_logits_fake, _ = self.motion_reward_model.lc_reward_model(fake_vid, queries=queries_point)
                lc_loss_fake = F.binary_cross_entropy_with_logits(lc_logits_fake.float(), torch.zeros_like(lc_logits_fake).float())
                if getattr(self.config, "use_deepspeed", False):
                    self._ds_set_boundary(True)
                self.manual_backward(self.lc_reward_weight * lc_loss_fake)

                lc_loss = lc_loss_real + lc_loss_fake
                if self.global_rank == 0 and self.config.debug_mode:
                    print('step', step, 'lc_loss_real', lc_loss_real.item(), 'lc_loss_fake', lc_loss_fake.item())
                    print('step', step, 'lc_logits_real', lc_logits_real.mean().item(), 'lc_logits_fake', lc_logits_fake.mean().item())
                
            loss = ic_loss + lc_loss
            self.clip_gradients(opt, gradient_clip_val=float(getattr(self.config, "gradient_clip", 1.0)), gradient_clip_algorithm="norm")
            opt.step()
            self.d_step += 1
            if getattr(self, "motion_reward_ema", None) is not None:
                self.motion_reward_ema.update()

            # log
            lr_groups = [pg["lr"] for pg in opt.param_groups]
            lr_ic = lr_groups[1] if getattr(self, "train_ic_reward", False) else 0.0
            lr_lc = (lr_groups[2] if (getattr(self, "train_ic_reward", False) and getattr(self, "train_lc_reward", False))
                    else (lr_groups[1] if getattr(self, "train_lc_reward", False) else 0.0))

            self.log("train_motion_reward_model_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=real_vid.size(0))
            self.log("train_motion_reward_model_loss_ic", ic_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=real_vid.size(0))
            self.log("train_motion_reward_model_loss_lc", lc_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=real_vid.size(0))
            self.log("train_motion_reward_model_ic_lr", torch.tensor(lr_ic, device=self.device), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=real_vid.size(0))
            self.log("train_motion_reward_model_lc_lr", torch.tensor(lr_lc, device=self.device), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=real_vid.size(0))

        torch.cuda.empty_cache()
        gc.collect()
        
        self.motion_reward_model.eval()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt = opt[0] if isinstance(opt, (list, tuple)) else opt

        accum_steps = int(getattr(self.config, "accum_grad_steps", 1))
        accum_steps = max(1, accum_steps)

        # gradnorm logging toggle (parity with svd_shift)
        is_first_micro = ((batch_idx % accum_steps) == 0)
        every = int(getattr(self.config, "log_grad_norm_every_steps", 0))
        do_grad_norm_log = (every > 0) and is_first_micro and (self.global_step % every == 0)

        if do_grad_norm_log:
            self._capture_grads = True
            self._captured_total_grads = {}

        prompts, prompt_embeds, real_latents, cond_latents, fake_latents, fake_adv, fake_ic, fake_lc, fake_ave, real_advantage, real_ic, real_lc, real_ave = \
            self.generate_train_batch(batch_idx)
        
        asym_grpo = bool(getattr(self.config, "asymmetric_grpo", False))
        min_sft_loss_lambda = float(getattr(self.config, "min_sft_loss_lambda", 1.0))
        match_sigmas = bool(getattr(self.config, "match_sigmas", True))
        aw_sft_loss_lambda = float(getattr(self.config, "aw_sft_loss_lambda", 1.0))

        # real SFT
        loss_vec, noise, timestep = self.shared_step_latents(
            input_latents=real_latents,
            cond_latents=cond_latents,
            prompt_embeds=prompt_embeds,
            stage="train",
            noise=None,
            timestep=None,
        )
        sft_loss = loss_vec.mean()

        if asym_grpo:
            sft_loss_effective = ((real_advantage + min_sft_loss_lambda) * loss_vec).mean()
        else:
            sft_loss_effective = min_sft_loss_lambda * loss_vec.mean()

        if getattr(self.config, "use_deepspeed", False):
            self._ds_set_boundary(False)
        self.manual_backward(sft_loss_effective)

        gn_sft, gn_aw = None, None
        g_sft_snap_cpu = None
        if do_grad_norm_log:
            g_sft_snap = {k: self._captured_total_grads.get(k, None) for k in self._lora_param_names}
            g_sft_snap_cpu, gn_sft = sft_grad_norm_log_from_grads(
                g_sft_snap,
                getattr(self.config, "gradnorm_group_prefix_regex", r"^(.*)\.blocks\.\d+\.")
            )

        # advantage-weighted SFT on fake samples (per p)
        P = int(self.config.num_sample_videos_per_prompt)
        scale = aw_sft_loss_lambda / max(1, P)
        aw_loss_logged = 0.0

        for p in range(P):
            fake_loss_vec, _, _ = self.shared_step_latents(
                input_latents=fake_latents[p],
                cond_latents=cond_latents,
                prompt_embeds=prompt_embeds,
                stage="train",
                noise=noise if match_sigmas else None,
                timestep=timestep if match_sigmas else None,
            )
            batch_aw = (fake_adv[p] * fake_loss_vec).mean()
            aw_loss_logged = aw_loss_logged + batch_aw.detach()

            is_last = (p == P - 1)
            if getattr(self.config, "use_deepspeed", False):
                self._ds_set_boundary(is_last)
            self.manual_backward(scale * batch_aw)

        aw_sft_loss_avg = aw_loss_logged / max(1, P)
        if do_grad_norm_log:
            total_snap = {k: self._captured_total_grads.get(k, None) for k in self._lora_param_names}
            gn_aw = aw_sft_grad_norm_log_from_grads(
                total_snap,
                g_sft_snap_cpu or {},
                getattr(self.config, "gradnorm_group_prefix_regex", r"^(.*)\.blocks\.\d+\.")
            )
            _, norm_ratios_vec = layer_grad_norm_mean_ratio(gn_aw, gn_sft, eps=1e-9)
            self._capture_grads = False

        # step on accumulation boundary
        should_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == self.trainer.num_training_batches)
        if should_step:
            self.clip_gradients(
                opt,
                gradient_clip_val=float(getattr(self.config, "gradient_clip", 1.0)),
                gradient_clip_algorithm="norm",
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            self.g_step += 1

            lr_schedulers = self.lr_schedulers()
            lr0 = lr_schedulers[0] if isinstance(lr_schedulers, (list, tuple)) else lr_schedulers
            if lr0 is not None:
                lr0.step()

        total_loss = sft_loss_effective.detach() + aw_sft_loss_lambda * aw_sft_loss_avg

        # log
        self.log("train_sft_loss", sft_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=real_latents.size(0))
        self.log("train_aw_sft_loss", aw_sft_loss_avg, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=real_latents.size(0))
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=real_latents.size(0))
        self.log("train_fake_ic_rewards", fake_ic.mean().detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=real_latents.size(0))
        self.log("train_fake_lc_rewards", fake_lc.mean().detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=real_latents.size(0))
        self.log("train_fake_ave_rewards", fake_ave.mean().detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=real_latents.size(0))
        self.log("train_lr", torch.tensor(opt.param_groups[0]["lr"], device=self.device), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=real_latents.size(0))
        self.log("train_real_ic_rewards", real_ic.mean().detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=real_latents.size(0))
        self.log("train_real_lc_rewards", real_lc.mean().detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=real_latents.size(0))
        self.log("train_real_ave_rewards", real_ave.mean().detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=real_latents.size(0))

        tb_exp = getattr(getattr(self, "logger", None), "experiment", None)
        if self.global_rank == 0:
            log_hist(tb_exp, {
                "fake_advantage": fake_adv,
                "fake_ic_rewards": fake_ic,
                "fake_lc_rewards": fake_lc,
                "fake_ave_rewards": fake_ave,
                "real_advantage": real_advantage,
                "real_ic_rewards": real_ic,
                "real_lc_rewards": real_lc,
            }, stage="train", global_step=self.global_step)

            if gn_sft is not None and gn_aw is not None:
                log_hist(tb_exp, gn_sft, stage="sft_grad_norm", global_step=self.global_step)
                log_hist(tb_exp, gn_aw, stage="aw_sft_grad_norm", global_step=self.global_step)
                log_hist(tb_exp, {"grad_norm_ratio": norm_ratios_vec}, stage="aw_over_sft", global_step=self.global_step)
                
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # end of buffer (all passes)
        if (batch_idx + 1) % self.buffer_update_frequency == 0:
            # flush any pending accumulation before changing buffer (parity with svd_shift)
            self._flush_unet_optimizer_if_needed(batch_idx)

            # reward model training from current buffer (parity with svd_shift)
            if getattr(self.config, "train_reward_models", False):
                self.train_motion_reward_model_from_buffer()

            sampler = self.trainer.datamodule.sampler
            num_chunks = max(1, sampler.num_chunks)
            chunks_completed = (batch_idx + 1) // self.buffer_update_frequency
            next_chunk_id = (chunks_completed % num_chunks) + 1
            self.recompute_buffer(next_chunk_id)
            self.current_pass = 0

        # end of pass within same buffer
        elif (batch_idx + 1) % self.iterations_per_chunk == 0:
            self.current_pass += 1
            self.reshuffle_buffer()
            if self.global_rank == 0:
                print(f"Buffer reshuffled. Starting pass {self.current_pass + 1}/{self.passes_per_buffer}")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if not hasattr(self, "_val_hist_cache"):
            self._val_hist_cache = defaultdict(list)
        # expensive (samples videos). keep val frequency low.
        collected = self.collect_data(batch, batch_idx)

        prompt_embeds = collected["prompt_embeds"].to(self.device)
        real_latents = collected["real_latents"].to(self.device)
        cond_latents = collected["cond_latents"].to(self.device)
        fake_latents = collected["fake_latents"].to(self.device)
        fake_adv = collected["fake_advantage"].to(self.device)
        fake_ave = collected["fake_ave_rewards"].to(self.device)
        fake_ic = collected["fake_ic_rewards"].to(self.device)
        fake_lc = collected["fake_lc_rewards"].to(self.device)
        real_advantage = collected["real_advantage"].to(self.device)
        real_ic = collected["real_ic_rewards"].to(self.device)
        real_lc = collected["real_lc_rewards"].to(self.device)
        real_ave = collected["real_ave_rewards"].to(self.device)

        self._val_hist_cache["fake_advantage"].append(fake_adv.detach().flatten().cpu())
        self._val_hist_cache["fake_ave_rewards"].append(fake_ave.detach().flatten().cpu())
        self._val_hist_cache["real_advantage"].append(real_advantage.detach().flatten().cpu())
        self._val_hist_cache["real_ave_rewards"].append(real_ave.detach().flatten().cpu())
        self._val_hist_cache["fake_ic_rewards"].append(fake_ic.detach().flatten().cpu())
        self._val_hist_cache["fake_lc_rewards"].append(fake_lc.detach().flatten().cpu())
        self._val_hist_cache["real_ic_rewards"].append(real_ic.detach().flatten().cpu())
        self._val_hist_cache["real_lc_rewards"].append(real_lc.detach().flatten().cpu())

        asym_grpo = bool(getattr(self.config, "asymmetric_grpo", False))
        min_sft_loss_lambda = float(getattr(self.config, "min_sft_loss_lambda", 1.0))
        match_sigmas = bool(getattr(self.config, "match_sigmas", True))
        aw_sft_loss_lambda = float(getattr(self.config, "aw_sft_loss_lambda", 1.0))


        loss_vec, noise, timestep = self.shared_step_latents(real_latents, cond_latents, prompt_embeds, stage="val")
        sft_loss = loss_vec.mean()
        if asym_grpo:
            sft_loss_effective = ((real_advantage + min_sft_loss_lambda) * loss_vec).mean()
        else:
            sft_loss_effective = min_sft_loss_lambda * loss_vec.mean()

        P = fake_latents.shape[0]
        aw = 0.0
        for p in range(P):
            fake_loss_vec, _, _ = self.shared_step_latents(fake_latents[p], cond_latents, 
                                                           prompt_embeds, stage="val", 
                                                           noise=noise if match_sigmas else None, 
                                                           timestep=timestep if match_sigmas else None)
            aw = aw + (fake_adv[p] * fake_loss_vec).mean().detach()
        aw = aw / max(1, P)

        total = sft_loss_effective.detach() + aw_sft_loss_lambda * aw

        self.log("val_sft_loss", sft_loss.detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=real_latents.size(0))
        self.log("val_aw_sft_loss", aw, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=real_latents.size(0))
        self.log("val_total_loss", total, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=real_latents.size(0))
        self.log("val_fake_ave_rewards", fake_ave.mean().detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=real_latents.size(0))
        self.log("val_real_ave_rewards", real_ave.mean().detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=real_latents.size(0))
        self.log("val_real_ic_rewards", real_ic.mean().detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=real_latents.size(0))
        self.log("val_real_lc_rewards", real_lc.mean().detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=real_latents.size(0))

        return total
    
    def on_validation_epoch_end(self):
        if not hasattr(self, "_val_hist_cache"):
            return
        tb_exp = getattr(getattr(self, "logger", None), "experiment", None)
        if self.global_rank == 0:
            for name, chunks in self._val_hist_cache.items():
                if len(chunks) == 0:
                    continue
                vals = torch.cat(chunks, dim=0)
                log_hist(tb_exp, {name: vals}, stage="val", global_step=self.global_step)
        self._val_hist_cache = defaultdict(list)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config=None, **kwargs):
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if config is None:
            import argparse
            config = argparse.Namespace(**ckpt["config_dict"])

        model = cls(config, **kwargs)

        # load LoRA
        sd = ckpt.get("state_dict", {})
        lora_sd = {}
        for k, v in sd.items():
            if k.startswith("denoiser."):
                lora_sd[k[len("denoiser."):]] = v
        if lora_sd:
            set_peft_model_state_dict(model.denoiser, lora_sd, adapter_name="default")

        # load reward models
        if "motion_reward_state_dict" in ckpt:
            rm_sd = {}
            sd = ckpt["motion_reward_state_dict"]
            for k, v in sd.items():
                if k.startswith("motion_reward_model."):
                    rm_sd[k[len("motion_reward_model."):]] = v
            model.motion_reward_model.load_state_dict(rm_sd, strict=True)

        # load EMA for reward model if present
        if "motion_reward_ema_state" in ckpt and getattr(model, "motion_reward_ema", None) is not None:
            try:
                model.motion_reward_ema.load_state_dict(ckpt["motion_reward_ema_state"])
            except Exception as e:
                print(f"Warning: failed to load motion_reward_ema_state: {e}")

        return model
    # for pytorch lightning automatically resume
    def on_load_checkpoint(self, checkpoint):
        # The lora weights are saved in "state_dict" key of the checkpoint, pytorch lightning is able to restore it
        # automatically.
        # restore reward-model weights (saved separately)
        if "motion_reward_state_dict" in checkpoint:
            raw = checkpoint["motion_reward_state_dict"]
            rm_sd = {}
            for k, v in raw.items():
                if k.startswith("motion_reward_model."):
                    rm_sd[k[len("motion_reward_model."):]] = v
            self.motion_reward_model.load_state_dict(rm_sd, strict=False)

        # restore EMA (also saved separately)
        if "motion_reward_ema_state" in checkpoint and getattr(self, "motion_reward_ema", None) is not None:
            self.motion_reward_ema.load_state_dict(checkpoint["motion_reward_ema_state"])