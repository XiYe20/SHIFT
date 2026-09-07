import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import pytorch_lightning as pl
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from peft import inject_adapter_in_model

from diffusers.training_utils import cast_training_params
from einops import rearrange, repeat
import gc
from pathlib import Path
from utils import load_config

from torch.utils.checkpoint import checkpoint
# WAN2.2 TI2V (mirrors wan22_inference.py)
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from wan22_flow_match_utils import FlowMatchSchedulerWan, flow_match_sft_loss_wan

class SFTTrainerWAN(pl.LightningModule):
    def __init__(self, config):
        """Initialize the base SFT trainer"""
        super().__init__()
        self.config = config

        # dtype
        if self.config.precision == "bf16":
            self.torch_dtype = torch.bfloat16
        elif self.config.precision == "32":
            self.torch_dtype = torch.float32
        else:
            raise ValueError(f"Invalid precision '{self.config.precision}'")
        
        # initialize FlowMatch scheduler (Discrete 1000 steps)
        self.fm_scheduler = FlowMatchSchedulerWan()
        self.fm_scheduler.set_timesteps(num_inference_steps=self.config.num_inference_steps, training=True, shift=getattr(self.config, "sigma_shift", 5))

        # Initialize pipeline, model and scheduler
        self._initialize_pipeline()
        self._initialize_denoiser_and_lora()

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
        self.vae.requires_grad_(False)
        self.text_encoder = self.wan_pipe.text_encoder
        self.text_encoder.requires_grad_(False)
        # enable gradient checkpointing
        # TO DO

    def _initialize_denoiser_and_lora(self):
        """Initialize and setup UNet model"""
        # Store initial weights
        self.denoiser.requires_grad_(False)
        # self.lora_name = getattr(self.config, "lora_adapter_name", "default")

        # Initialize denoiser Lora
        lora_config = dict(self.config.lora_config)
        lora_cfg = LoraConfig(**lora_config)
        self.denoiser = inject_adapter_in_model(lora_cfg, self.denoiser)
        # if hasattr(self.denoiser, "peft_config") and "default" in self.denoiser.peft_config and self.lora_name != "default":
        #     self.denoiser.peft_config[self.lora_name] = self.denoiser.peft_config.pop("default")

        # collect trainable params
        self.trainable_params = [p for p in self.denoiser.parameters() if p.requires_grad]
        # count the totale number of parameters in the trainable_params
        total_params = sum(p.numel() for p in self.trainable_params)
        print(f"Total trainable LoRA parameters: {total_params/1e6} M")

    def on_fit_start(self):
        """Setup before training starts"""
        self.wan_pipe.to(self.device)
        self.denoiser.to(self.device).train()

    def configure_optimizers(self):
        """Setup optimizer and learning rate scheduler"""
        # Check if we should use DeepSpeed CPU optimizer
        use_deepspeed = getattr(self.config, 'use_deepspeed', False)
        use_deepspeed_adam = getattr(self.config, 'use_deepspeed_adam', False)
        cpu_offload = getattr(self.config, 'cpu_offload', True)

        if use_deepspeed_adam:
            # Use DeepSpeed's CPU Adam optimizer for better performance with offloading
            if cpu_offload:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                print("Using DeepSpeed CPU Adam optimizer for ZeRO-Offload")
                optimizer = DeepSpeedCPUAdam(
                    self.trainable_params,
                    lr=self.config.lr,
                    betas=(self.config.beta1, self.config.beta2),
                    weight_decay=0.0,
                    eps=1e-8
                )
            else:
                from deepspeed.ops.adam import FusedAdam
                print("DeepSpeed CPU Adam not available, falling back to FusedAdam")
                optimizer = FusedAdam(
                    self.trainable_params, 
                    lr=self.config.lr,
                    betas=(self.config.beta1, self.config.beta2), 
                    weight_decay=0.0
                )
        else:
            if self.config.optimizer == "adamw":
                optimizer = torch.optim.AdamW(
                    self.trainable_params, 
                    lr=self.config.lr,
                    betas=(self.config.beta1, self.config.beta2), 
                    weight_decay=0.0
                )
            else:
                raise ValueError(f"Invalid optimizer: {self.config.optimizer}")

        # Setup scheduler
        if self.config.scheduler == "stepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.accum_grad_steps, gamma=0.99
            )
        elif self.config.scheduler == "linear_warmup": 
            def lr_lambda(current_step):
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))
                return 1.0

            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda),
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise ValueError(f"Scheduler {self.config.scheduler} not supported")

        return [optimizer], [scheduler]
    
    def log_metrics(self, prefix, metrics_dict, batch_size):
        """Log all metrics with consistent naming and parameters"""
        for name, value in metrics_dict.items():
            self.log(
                f"{prefix}_{name}",
                value.detach(),
                on_step=(prefix == "train"), 
                on_epoch=True,
                prog_bar=True, 
                logger=True,
                sync_dist=True,
                rank_zero_only=False,
                batch_size=batch_size,
            )
    
    def on_save_checkpoint(self, checkpoint):
        """Save config directly in the checkpoint file"""
        lora_sd = get_peft_model_state_dict(
            self.denoiser, adapter_name='default'
        )
        # Lightning expects keys to be prefixed with the attribute name (`unet.`)
        checkpoint["state_dict"] = {f"denoiser.{k}": v.cpu() for k, v in lora_sd.items()}

        if hasattr(self, "config"):
            import yaml
            from pathlib import Path
            
            if hasattr(self.config, "config_path"):
                config_path = Path(self.config.config_path)
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        yaml_content = f.read()
                    checkpoint['config_yaml'] = yaml_content
                    checkpoint['config_path'] = str(config_path)
            
            checkpoint['config_dict'] = vars(self.config)
        # TO DO
        # save the optimizer state for resuming training

    def on_train_epoch_end(self):
        """Access epoch-averaged training metrics from Lightning aggregation."""
        metrics = getattr(self.trainer, 'callback_metrics', {})
        avg_cn = metrics.get('train_control_norm', None)
        if avg_cn is not None:
            # Store for downstream use
            self.epoch_avg_train_control_norm = avg_cn.detach()
            # Optional: print or log a separate key if desired
            if getattr(self.config, 'verbose', False) and self.global_rank == 0:
                print(f"Epoch {self.current_epoch} avg train_control_norm: {float(avg_cn):.6f}")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config=None, **kwargs):
        """
        Custom load_from_checkpoint method to handle LoRA weights loading.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            config: Configuration object (if None, will try to load from checkpoint)
            **kwargs: Additional arguments to pass to __init__
        """
        # Load the checkpoint
        if not isinstance(checkpoint_path, (str, Path)):
            raise ValueError(f"checkpoint_path must be a string or Path, got {type(checkpoint_path)}")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        
        # If config is not provided, try to load it from the checkpoint
        if config is None:
            if 'config_dict' in checkpoint:
                # Reconstruct config from saved dict
                from config_utils import Config
                config_dict = checkpoint['config_dict']
                config = Config(**config_dict)
                print(f"Loaded config from checkpoint: {checkpoint_path}")
            else:
                raise ValueError("Config not provided and not found in checkpoint")
        else:
            # do backwar-compat, for checkpoints trained before gan-training feature upgrade
            if not hasattr(config, 'cotracker_reward_gan_training'):
                setattr(config, 'cotracker_reward_gan_training', False)

        # Create the model instance
        try:
            model = cls(config, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create model instance: {e}")
        
        # Load LoRA weights from the checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            
            # Extract LoRA weights (remove 'denoiser.' prefix)
            lora_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('denoiser.'):
                    # Remove the 'denoiser.' prefix to get the LoRA parameter name
                    lora_key = key[5:]  # Remove 'denoiser.' prefix
                    lora_state_dict[lora_key] = value
            
            # Load the LoRA weights into the model
            if lora_state_dict:
                try:
                    set_peft_model_state_dict(model.denoiser, lora_state_dict, adapter_name=model.lora_name)
                    print(f"Successfully loaded {len(lora_state_dict)} LoRA parameters from checkpoint: {checkpoint_path}")
                except Exception as e:
                    print(f"Warning: Failed to load LoRA weights: {e}")
                    print("Continuing with randomly initialized LoRA weights...")
            else:
                print(f"Warning: No LoRA weights found in checkpoint: {checkpoint_path}")
                print("Continuing with randomly initialized LoRA weights...")
        else:
            print(f"Warning: No state_dict found in checkpoint: {checkpoint_path}")
            print("Continuing with randomly initialized LoRA weights...")

        return model

    # Abstract methods to be implemented by subclasses
    def shared_step(self, batch, batch_id, stage='train'):
        """Main training loop
        Expected batch format:
          - video tensor in [-1,1], (B, T, C, H, W)
          - prompt string (or list[str])
        """
        video = batch["video"].to(self.device)
        prompt = batch["prompt"]

        b, t, c, h, w = video.shape
        # Ensure temporal constraint (same logic as in _encode_video_to_latents)
        t_factor = int(getattr(self.wan_pipe, "vae_scale_factor_temporal", 4))
        if t % t_factor != 1:
            new_t = ((t - 1) // t_factor) * t_factor + 1
            video = video[:, :new_t]
            b, t, c, h, w = video.shape

        # encode the video to latents
        with torch.no_grad():
            input_latents = self._encode_video_to_latents(video)  # float32 normalized

        # sample discrete timesteps
        max_b = getattr(self.config, "max_timestep_boundary", 1.0)
        min_b = getattr(self.config, "min_timestep_boundary", 0.0)
        max_id = int(max_b * len(self.fm_scheduler.timesteps))
        min_id = int(min_b * len(self.fm_scheduler.timesteps))
        timestep_id = torch.randint(min_id, max_id, (1,))
        timestep = self.fm_scheduler.timesteps[timestep_id].to(dtype=torch.float32, device=self.device)

        # forward diffusion
        noise = torch.randn_like(input_latents)
        noisy_latents = self.fm_scheduler.add_noise(input_latents, noise, timestep)

        # build first frame conditional information, video_condition = [image, zeros_remaining_frames]
        image = video[:, 0, :, :, :].to(dtype=torch.float32)
        image = image.unsqueeze(2) #(b, c, 1, h, w)
        video_condition = torch.cat([image, image.new_zeros(b, c, t - 1, h, w)], dim=2)  # (B,C,T,H,W)
        with torch.no_grad():
            latent_condition = self._encode_video_to_latents(video_condition.permute(0, 2, 1, 3, 4).contiguous())

        transformer_dtype = self.denoiser.dtype
        # expand_timesteps for wan2.2
        assert getattr(self.wan_pipe.config, "expand_timesteps") == True, "expand_timesteps must be True for wan2.2"
        num_latent_frames = latent_condition.shape[2]
        latent_h, latent_w = latent_condition.shape[3], latent_condition.shape[4]
        first_frame_mask = torch.ones(1, 1, num_latent_frames, latent_h, latent_w, dtype=torch.float32, device=self.device)
        first_frame_mask[:, :, 0] = 0

        condition = latent_condition  # matches pipeline: returns (latents, condition, first_frame_mask)
        latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * noisy_latents
        latent_model_input = latent_model_input.to(transformer_dtype)

        # timestep is per-token sequence, masked at first frame
        ps = int(self.denoiser.config.patch_size[1])  # usually 2
        temp_ts = (first_frame_mask[0, 0][:, ::ps, ::ps] * timestep).flatten()
        timestep_model = temp_ts.unsqueeze(0).expand(latent_model_input.shape[0], -1).to(dtype=transformer_dtype)

        # encode the prompt
        with torch.no_grad():
            prompt_embeds, _ = self.wan_pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                max_sequence_length=getattr(self.config, "max_sequence_length", 512),
                device=self.device,
            )
        prompt_embeds = prompt_embeds.to(dtype=transformer_dtype)

        # 5) image_embeds only for WAN2.1 I2V (per your source code). For WAN2.2 TI2V-5B this is typically None.
        image_embeds = None # for wan2.2 ti2v-5b
        # 6) Denoiser forward
        with torch.set_grad_enabled(stage == "train"):
            model_pred = self.denoiser(
                hidden_states=latent_model_input,
                timestep=timestep_model,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]

        # 7) Flow matching loss (target = noise - x) with scheduler weighting
        loss = flow_match_sft_loss_wan(
            scheduler=self.fm_scheduler,
            model_pred=model_pred,
            input_latents=input_latents,
            timestep=timestep,
            noise=noise,
        )
        return loss
    
    def _encode_video_to_latents(self, video_btchw: torch.Tensor) -> torch.Tensor:
        """
        Encode a real video to WAN VAE latents, matching diffusers pipeline normalization:
        - VAE encode
        - use argmax/mode (deterministic) like sample_mode="argmax"
        - normalize using (latents - mean) * (1/std)
        Input:
        video_btchw: (B, T, C, H, W) in [-1, 1]
        Output:
        latents: (B, z_dim, T_lat, H_lat, W_lat) normalized
        """
        if video_btchw.dim() != 5:
            raise ValueError(f"Expected video tensor with shape (B,T,C,H,W), got {tuple(video_btchw.shape)}")

        b, t, c, h, w = video_btchw.shape

        # Match pipeline temporal constraint: num_frames % vae_scale_factor_temporal == 1
        t_factor = int(getattr(self.wan_pipe, "vae_scale_factor_temporal", 4))
        if t % t_factor != 1:
            new_t = (t // t_factor) * t_factor + 1
            if new_t > t:
                new_t = ((t - 1) // t_factor) * t_factor + 1
            if new_t < 1:
                raise ValueError(f"Cannot adjust T={t} to satisfy T % {t_factor} == 1")
            video_btchw = video_btchw[:, :new_t]
            b, t, c, h, w = video_btchw.shape

        video_bcthw = video_btchw.permute(0, 2, 1, 3, 4).contiguous()
        # Diffusers pipeline encodes in float32
        vae_dtype = next(self.vae.parameters()).dtype
        video_bcthw = video_bcthw.to(device=self.device, dtype=vae_dtype)
        
        enc = self.vae.encode(video_bcthw)

        # retrieve_latents(..., sample_mode="argmax")
        if hasattr(enc, "latent_dist"):
            latents = enc.latent_dist.mode()
        elif hasattr(enc, "latents"):
            latents = enc.latents
        else:
            raise AttributeError("Could not access latents from VAE encoder output")

        # Normalize using VAE config (matches prepare_latents() in diffusers pipeline)
        z_dim = int(self.vae.config.z_dim)
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, z_dim, 1, 1, 1)
        latents_std_inv = (1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype)).view(1, z_dim, 1, 1, 1)
        latents = (latents - latents_mean) * latents_std_inv

        return latents
    
    def training_step(self, batch, batch_id):
        loss= self.shared_step(batch, batch_id, stage='train')
        metric_dict = {
            "loss": loss,
        }
        self.log_metrics("train", metric_dict, batch_size = batch['video'].size(0))

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss= self.shared_step(batch, batch_idx, stage='val')
        metric_dict = {
            "loss": loss,
        }
        self.log_metrics("val", metric_dict, batch_size = batch['video'].size(0))

        return loss

if __name__ == "__main__":
    device = 'cuda:1'
    config = load_config('train_config.yaml')
    trainer = SFTTrainerWAN(config).to(device)
    from utils import VideoDataModule
    data_module = VideoDataModule(config.batch_size,
                                    train_video_path_json=config.train_video_path_json,
                                    val_video_path_json=config.val_video_path_json,
                                    target_vid_size=config.target_vid_size,
                                    vid_data_type=config.vid_data_type,
                                    num_workers=config.num_workers)

    train_loader = data_module.train_dataloader()
    for idx, batch in enumerate(train_loader):
        loss = trainer.training_step(batch, idx)
        print(f"Batch {idx} loss: {loss}")