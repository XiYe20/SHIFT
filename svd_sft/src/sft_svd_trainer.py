import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import pytorch_lightning as pl
from utils import get_model
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.training_utils import cast_training_params
from einops import rearrange, repeat
import gc
from pathlib import Path
from utils import load_config
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing
from torch.utils.checkpoint import checkpoint
from utils import rand_log_normal

class SFTTrainerSVD(pl.LightningModule):
    def __init__(self, config):
        """Initialize the base SFT trainer"""
        super().__init__()
        self.config = config
        
        # Initialize pipeline, model and scheduler
        self._initialize_pipeline()
        self._initialize_unet()
        self.generator = None
        
        # Initialize tracking variables
        self.eval_step_outputs = dict(val=[], test=[])
        self.global_batch_step = 0

    def _initialize_pipeline(self):
        """Setup the Stable Video Diffusion pipeline"""
        self.svd_pipeline = get_model(
            self.config.model_name,
            use_compile=False,
            bfloat_dtype=(self.config.precision == 'bf16'),
        )
        self.feature_extractor = self.svd_pipeline.feature_extractor
        self.image_encoder = self.svd_pipeline.image_encoder
        self.vae = self.svd_pipeline.vae

        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        # Setup appropriate data type
        if self.config.precision == 'bf16':
            self.torch_dtype = torch.bfloat16
        elif self.config.precision == '32':
            self.torch_dtype = torch.float32
        else:
            raise ValueError(f"Invalid precision '{self.config.precision}'. Allowed values are: 'bf16', '32'.")
        # enable gradient checkpointing
        self.svd_pipeline.unet.enable_gradient_checkpointing()
        self.svd_pipeline.vae.enable_gradient_checkpointing()

    def _initialize_unet(self):
        """Initialize and setup UNet model"""
        # Store initial weights
        self.unet = self.svd_pipeline.unet
        self.unet.requires_grad_(False)

        # Initialize UNet Lora
        lora_config = self.config.lora_config
        if self.config.only_lora_on_temporal_module:
            #only inject lora to all the motion_modules layers
            motion_target_modules = []
            for k, _ in self.unet.named_modules():
                if "temporal_transformer_blocks" in k:
                    for m in lora_config['target_modules']:
                        if m in k:
                            motion_target_modules.append(k)
        else:
            # inject lora to all blocks transformer blocks
            motion_target_modules = []
            for k, _ in self.unet.named_modules():
                for m in lora_config['target_modules']:
                    if m in k:
                        motion_target_modules.append(k)
        lora_config['target_modules'] = motion_target_modules
        unet_lora_config = LoraConfig(**lora_config)
        # Add adapter and make sure the trainable params are in float32.
        # Normally, LoRA doesn't benefit from EMA: https://github.com/huggingface/diffusers/issues/9998
        self.unet.add_adapter(unet_lora_config, adapter_name=self.config.lora_adapter_name)

        assert self.config.lora_precision in ["fp32", "bfloat16", "fp16"], f"Invalid LoRA precision: {self.config.lora_precision}"
        if self.config.lora_precision == "fp32":
            # only upcast LoRA parameters into fp32
            cast_training_params(self.unet, dtype=torch.float32)

        lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        self.trainable_params = lora_layers
        # count the totale number of parameters in the lora_layers
        total_params = sum(p.numel() for p in lora_layers)
        print(f"Total trainable LoRA parameters: {total_params/1e6} M")

        # use the _LoraWrapper to wrap the unet_init and the unet
        # base_unet = self.unet_init
        self.lora_name = self.config.lora_adapter_name
        self.svd_pipeline.lora_name = self.lora_name
        # self.unet_init = _LoraWrapper(base_unet, None)
        # self.unet = _LoraWrapper(base_unet, lora_name)

        # Set the pipeline to use our trainable UNet
        self.svd_pipeline.unet = self.unet
        self.svd_pipeline.unet.enable_gradient_checkpointing()

    def on_fit_start(self):
        """Setup before training starts"""
        self.svd_pipeline.to(self.device)
        self.unet.to(self.device).train()

    def switch_to_train(self):
        """Set models to training mode"""
        self.unet.train()
        # self.unet_init.train()

    def switch_to_eval(self):
        """Set models to evaluation mode"""
        self.unet.eval()
        # self.unet_init.eval()

    def gather_across_gpus(self, tensor):
        """Concatenate a tensor across all GPUs"""
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)
    
    def average_across_gpus(self, tensor):
        """Average a tensor across all GPUs"""
        world_size = dist.get_world_size()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
        return tensor

    def configure_optimizers(self):
        """Setup optimizer and learning rate scheduler"""
        # Check if we should use DeepSpeed CPU optimizer
        use_deepspeed = getattr(self.config, 'use_deepspeed', False)
        cpu_offload = getattr(self.config, 'cpu_offload', True)
        
        if use_deepspeed and cpu_offload:
            # Use DeepSpeed's CPU Adam optimizer for better performance with offloading
            try:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                print("Using DeepSpeed CPU Adam optimizer for ZeRO-Offload")
                
                optimizer = DeepSpeedCPUAdam(
                    self.trainable_params,
                    lr=self.config.lr,
                    betas=(self.config.beta1, self.config.beta2),
                    weight_decay=0.0,
                    eps=1e-8
                )
            except ImportError:
                print("DeepSpeed CPU Adam not available, falling back to PyTorch AdamW")
                optimizer = torch.optim.AdamW(
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
        # get lora state
        lora_sd = get_peft_model_state_dict(
            self.unet, adapter_name=self.lora_name
        )

        # Lightning expects keys to be prefixed with the attribute name (`unet.`)
        checkpoint["state_dict"] = {f"unet.{k}": v.cpu() for k, v in lora_sd.items()}


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
            # Also persist dynamic fields explicitly for robust resume
            try:
                checkpoint['current_reward_multiplier'] = float(getattr(self.config, 'reward_multiplier', 10000.0))
            except Exception:
                pass
        
        if hasattr(self, 'cotracker_reward_ema') and self.cotracker_reward_ema is not None:
            try:
                checkpoint['discriminator_ema_state'] = self.cotracker_reward_ema.state_dict()
            except Exception as e:
                print(f"Warning: failed to save discriminator EMA state: {e}")

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
            
            # Extract LoRA weights (remove 'unet.' prefix)
            lora_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('unet.'):
                    # Remove the 'unet.' prefix to get the LoRA parameter name
                    lora_key = key[5:]  # Remove 'unet.' prefix
                    lora_state_dict[lora_key] = value
            
            # Load the LoRA weights into the model
            if lora_state_dict:
                try:
                    set_peft_model_state_dict(model.unet, lora_state_dict, adapter_name=model.lora_name)
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

    def edm_loss(self, sigmas, model_pred, noisy_latents, target):
        # Denoise the latents
        c_out = -sigmas / ((sigmas**2 + 1)**0.5)
        c_skip = 1 / (sigmas**2 + 1)
        denoised_latents = model_pred * c_out + c_skip * noisy_latents
        weighing = (1 + sigmas ** 2) * (sigmas**-2.0)
        # MSE loss
        loss = weighing.float() * F.mse_loss(denoised_latents, target.to(denoised_latents.dtype))
        return loss.mean()

    # Abstract methods to be implemented by subclasses
    def shared_step(self, batch, batch_id, stage='train'):
        """Main training loop"""
        vid = batch
        vid = vid.to(self.device)
        # encode the video to latents
        with torch.no_grad():
            latents, noisy_latents, inp_noisy_latents, encoder_hidden_states, added_time_ids, timesteps, noise, sigmas, cond_sigmas = self.encode_pixels(vid, target_fps=self.config.target_fps)
            torch.cuda.empty_cache()
            gc.collect()
        image_embeddings = encoder_hidden_states

        def unet_forward_fn(model_input, timestep, encoder_states, time_ids):
            unet_dtype = next(self.svd_pipeline.unet.parameters()).dtype
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
                return self.svd_pipeline.unet(
                    model_input.to(unet_dtype),
                    timestep.to(unet_dtype),
                    encoder_hidden_states=encoder_states.to(unet_dtype),
                    added_time_ids=time_ids.to(unet_dtype),
                    return_dict=False,
                )[0]

        with torch.set_grad_enabled(stage == 'train'):
            model_output = checkpoint(
                unet_forward_fn,
                inp_noisy_latents,
                timesteps,
                image_embeddings,
                added_time_ids,
                use_reentrant=False  # Recommended for newer PyTorch versions
            )
            model_output = model_output.to(inp_noisy_latents.dtype)
        model_losses = self.edm_loss(sigmas, model_output, noisy_latents, latents)
        
        return model_losses
        
    def encode_image(self, pixel_values):
        device = self.unet.device
        weight_dtype = next(self.unet.parameters()).dtype

        # pixel: [-1, 1]
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        # We unnormalize it after resizing.
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize the image with for CLIP input
        pixel_values = self.feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(
            device=device, dtype=weight_dtype)
        if next(self.image_encoder.parameters()).device != device:
            self.image_encoder.to(device)

        image_embeddings = self.image_encoder(pixel_values).image_embeds
        return image_embeddings

    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * \
            len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    def tensor_to_vae_latent(self, t):
        video_length = t.shape[1]

        t = rearrange(t, "b f c h w -> (b f) c h w")
        latents = self.vae.encode(t).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
        latents = latents * self.vae.config.scaling_factor

        return latents

    def encode_pixels(self, vid_tensor, noise=None, cond_sigmas=None, sigmas=None, target_fps=7):
        """Encode the input video pixels to latents
        Modified from: https://github.com/pixeli99/SVD_Xtend/blob/main/train_svd_lora.py
        Args:
            vid_tensor: (B, T, C, H, W), with pixel values in range [-1, 1]
        """
        device = self.unet.device
        # first, convert images to latent space.
        pixel_values = vid_tensor.to(
            device, non_blocking=True
        )
        conditional_pixel_values = pixel_values[:, 0:1, :, :, :]
        latents = self.tensor_to_vae_latent(pixel_values)

        # Sample noise that we'll add to the latents
        if noise is None:
            noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        if cond_sigmas is None:
            cond_sigmas = rand_log_normal(shape=[bsz,], loc=-3.0, scale=0.5).to(latents)
        noise_aug_strength = cond_sigmas[0] # TODO: support batch > 1
        cond_sigmas = cond_sigmas[:, None, None, None, None]
        conditional_pixel_values = \
            torch.randn_like(conditional_pixel_values) * cond_sigmas + conditional_pixel_values
        conditional_latents = self.tensor_to_vae_latent(conditional_pixel_values)[:, 0, :, :, :]
        conditional_latents = conditional_latents / self.vae.config.scaling_factor

        # Sample a random timestep for each image
        # P_mean=0.7 P_std=1.6
        if sigmas is None:
            sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents.device)
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            sigmas = sigmas[:, None, None, None, None]
        noisy_latents = latents + noise * sigmas
        timesteps = torch.Tensor(
            [0.25 * sigma.log() for sigma in sigmas]).to(device)

        inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

        # Get the text embedding for conditioning.
        encoder_hidden_states = self.encode_image(
            pixel_values[:, 0, :, :, :].float())

        # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
        # However, I am unable to fully align with the calculation method of the motion score,
        # so I adopted this approach. The same applies to the 'fps' (frames per second).
        added_time_ids = self._get_add_time_ids(
            target_fps, # fixed
            127, # motion_bucket_id = 127, fixed
            noise_aug_strength, # noise_aug_strength == cond_sigmas
            encoder_hidden_states.dtype,
            bsz,
        )
        added_time_ids = added_time_ids.to(latents.device)

        # Conditioning dropout to support classifier-free guidance during inference. For more details
        # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
        if self.config.conditioning_dropout_prob is not None:
            if self.generator is None:
                self.generator = torch.Generator(device=latents.device).manual_seed(self.config.seed)
            random_p = torch.rand(
                bsz, device=latents.device, generator=self.generator)
            # Sample masks for the edit prompts.
            prompt_mask = random_p < 2 * self.config.conditioning_dropout_prob
            prompt_mask = prompt_mask.reshape(bsz, 1, 1)
            # Final text conditioning.
            null_conditioning = torch.zeros_like(encoder_hidden_states)
            encoder_hidden_states = torch.where(
                prompt_mask, null_conditioning.unsqueeze(1), encoder_hidden_states.unsqueeze(1))
            # Sample masks for the original images.
            image_mask_dtype = conditional_latents.dtype
            image_mask = 1 - (
                (random_p >= self.config.conditioning_dropout_prob).to(
                    image_mask_dtype)
                * (random_p < 3 * self.config.conditioning_dropout_prob).to(image_mask_dtype)
            )
            image_mask = image_mask.reshape(bsz, 1, 1, 1)
            # Final image conditioning.
            conditional_latents = image_mask * conditional_latents

        # Concatenate the `conditional_latents` with the `noisy_latents`.
        conditional_latents = conditional_latents.unsqueeze(
            1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
        inp_noisy_latents = torch.cat(
            [inp_noisy_latents, conditional_latents], dim=2)

        return latents, noisy_latents, inp_noisy_latents, encoder_hidden_states, added_time_ids, timesteps, noise, sigmas, cond_sigmas
    
    def training_step(self, batch, batch_id):
        loss= self.shared_step(batch, batch_id, stage='train')
        metric_dict = {
            "loss": loss,
        }
        self.log_metrics("train", metric_dict, batch_size = batch[0].size(0))

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss= self.shared_step(batch, batch_idx, stage='val')
        metric_dict = {
            "loss": loss,
        }
        self.log_metrics("val", metric_dict, batch_size = batch[0].size(0))

        return loss

if __name__ == "__main__":
    device = 'cuda:1'
    config = load_config('train_config.yaml')
    trainer = SFTTrainerSVD(config).to(device)
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