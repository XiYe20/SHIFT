import sys
from pathlib import Path
# Make reward_model_pretrain importable
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../MotionAlignmentVDM
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'reward_model_pretrain'))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

import pytorch_lightning as pl
from utils import get_model
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.training_utils import cast_training_params
from einops import rearrange, repeat
import gc
from pathlib import Path
from utils import load_config, ListSampler
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing
from torch.utils.checkpoint import checkpoint
from utils import rand_log_normal, retrieve_timesteps, _append_dims
from utils import sft_grad_norm_log_from_grads, aw_sft_grad_norm_log_from_grads, layer_grad_norm_mean_ratio
from utils import log_hist
from utils import attach_peft_mixin, clear_peft_merged_state

from EDM_Ancestral_scheduler import EDMAncestralScheduler
from reward_model_pretrain.reward_models import MotionRewardModel
from reward_model_pretrain.vbench_reward_models import VBenchAppearanceRewardModel
from contextlib import nullcontext
from tqdm import tqdm
import math
from collections import defaultdict

class ShiftTrainerSVD(pl.LightningModule):
    """SHIFT trainer for SVD"""
    def __init__(self, config):
        """Initialize the base SHIFT trainer"""
        super().__init__()
        self.config = config
        self.automatic_optimization = False  # manual optimization to save memory
        # Initialize pipeline, model and scheduler
        self._initialize_pipeline()
        self._initialize_unet()
        self._initialize_reward_models()
        self._initialize_buffer()
        self.generator = None
        
        # Initialize tracking variables
        self.eval_step_outputs = dict(val=[], test=[])
        self.global_batch_step = 0

        # set the scheduler to be EDMAncestral
        self.set_edm_ancestral_scheduler()

        self.g_step = 0
        self.d_step = 0

        self._capture_grads = False
        self._vbench = {"model": None}

    def _get_aw_sft_loss_lambda(self) -> float:
        """
        Return the current weight for advantage-weighted SFT loss.

        Backward compatible behavior:
        - If `max_aw_sft_loss_lambda` is not set in config, fall back to the fixed
          `aw_sft_loss_lambda` scalar (current behavior).

        Scheduled behavior (linear ramp from 0 -> max):
        - Set `max_aw_sft_loss_lambda` and either:
          - `aw_sft_loss_rampup_epochs` (preferred; converted to optimizer steps), or
          - `aw_sft_loss_rampup_steps` (directly specified in optimizer steps).
        """
        # Old behavior: fixed scalar
        max_lambda = getattr(self.config, "max_aw_sft_loss_lambda", None)
        if max_lambda is None:
            if self.global_rank == 0:
                print('!!use fixed aw_sft_loss_lambda', self.config.aw_sft_loss_lambda)
            return float(self.config.aw_sft_loss_lambda)

        max_lambda = float(max_lambda)
        rampup_steps = getattr(self.config, "aw_sft_loss_rampup_steps", None)

        if rampup_steps is None:
            rampup_epochs = float(getattr(self.config, "aw_sft_loss_rampup_epochs", 0.0))
            if rampup_epochs <= 0:
                return max_lambda

            # Convert epochs -> optimizer steps
            # Prefer Lightning's view if available (it reflects DDP/limit_train_batches, etc.)
            trainer = getattr(self, "trainer", None)
            iters_per_epoch = None
            if trainer is not None and getattr(trainer, "num_training_batches", None) is not None:
                # During sanity-check validation, Lightning may expose `inf` here.
                try:
                    nb = float(trainer.num_training_batches)
                    if math.isfinite(nb) and nb > 0:
                        iters_per_epoch = int(nb)
                except Exception:
                    iters_per_epoch = None
            if iters_per_epoch is None:
                iters_per_epoch = int(getattr(self.config, "iterations_per_epoch", 0) or 0)

            # print('iters_per_epoch', iters_per_epoch, 'rampup_epochs', rampup_epochs, 'max_lambda', max_lambda)

            accum = int(getattr(self.config, "accum_grad_steps", 1) or 1)
            accum = max(1, accum)
            opt_steps_per_epoch = int(math.ceil(iters_per_epoch / float(accum))) if iters_per_epoch > 0 else 0
            rampup_steps = int(rampup_epochs * opt_steps_per_epoch)

        rampup_steps = int(rampup_steps)
        if rampup_steps <= 0:
            return max_lambda

        # Use optimizer-step counter (accounts for grad accumulation)
        step = int(getattr(self, "g_step", 0) or 0)
        progress = min(1.0, max(0.0, step / float(rampup_steps)))
        # print('rampup_steps', rampup_steps, 'max_lambda', max_lambda, max_lambda * progress, step)
        return max_lambda * progress

    def _initialize_pipeline(self):
        """Setup the Stable Video Diffusion pipeline"""
        self.svd_pipeline = get_model(
            self.config.model_name,
            use_compile=False,
            bfloat_dtype=(self.config.precision == 'bf16'),
            use_for_training=True,
            local_pretrained_model=self.config.pretrained_svd_path
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

            self.unet = attach_peft_mixin(self.unet)

            train_cfg = checkpoint.get("config_dict", None)
            if train_cfg is None or "lora_config" not in train_cfg:
                raise ValueError("Checkpoint is missing 'config_dict.lora_config'; cannot rebuild LoRA adapter.")

            sft_lora_config = dict(train_cfg["lora_config"])  # shallow copy
            sft_unet_lora_config = LoraConfig(**sft_lora_config)

            adapter_name_lora = train_cfg.get("lora_adapter_name")
            if not adapter_name_lora:
                adapter_name_lora = "sft_pretrained"
            self.unet.add_adapter(sft_unet_lora_config, adapter_name=adapter_name_lora)

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Extract LoRA weights (remove 'unet.' prefix)
                lora_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('unet.'):
                        lora_key = key[5:]  # Remove 'unet.' prefix
                        lora_state_dict[lora_key] = value

                if lora_state_dict:
                    set_peft_model_state_dict(self.unet, lora_state_dict, adapter_name=adapter_name_lora)
                    print(f"Successfully loaded {len(lora_state_dict)} LoRA parameters from checkpoint: {sft_lora_path}")
                    # Fuse SFT LoRA into base weights so the training adapter builds on top
                    self.unet.set_adapter(adapter_name_lora)
                    self.unet.fuse_lora(adapter_names=[adapter_name_lora])
                    self.unet.delete_adapters(adapter_name_lora)
                    clear_peft_merged_state(self.unet)
                    print(f"[SFT Merge] Successfully fused adapter '{adapter_name_lora}' into base UNet.")
                else:
                    raise ValueError(f"No LoRA weights found in checkpoint: {sft_lora_path}")
            else:
                raise ValueError(f"Checkpoint is missing 'state_dict': {sft_lora_path}")

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
        self.unet.enable_gradient_checkpointing()

        # Register grad hooks for LoRA/trainable params (works under ZeRO-2 even if p.grad becomes None)
        self._lora_param_names = []
        for name, p in self.unet.named_parameters():
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

    def _initialize_buffer(self):
        self.buffer_size = self.config.buffer_size
        self.buffer_device = self.config.buffer_device

        # Use global world size (num_nodes * gpus_per_node) for multi-node runs
        global_world_size = self.config.num_nodes * self.config.gpus_per_node
        self.iterations_per_chunk = self.buffer_size // (global_world_size * self.config.batch_size)

        self.passes_per_buffer = self.config.passes_per_buffer

        # Calculate how often to update the buffer
        self.buffer_update_frequency = self.iterations_per_chunk * self.config.passes_per_buffer

        self.buffer_initialized = False
        self.current_pass = 0  # Track which pass through the buffer we're on

        # Initialize buffer structures for fake examples generation
        self.buffer_variables = [
            'real_videos',
            'real_advantage',
            'real_ic_rewards',
            'real_lc_rewards',
            'real_vbench_rewards',
            'real_ave_rewards',
            'fake_videos', #(B*config.num_sample_videos_per_prompt, T, C, H, W), value in range [0, 1]
            'fake_ave_rewards',
            'fake_ic_rewards',
            'fake_lc_rewards',
            'fake_vbench_rewards',
            'fake_advantage',
            'unweighted_fake_advantage'
        ]
        
        self.buffer = {}
        for var in self.buffer_variables:
            self.buffer[var] = None

    def _initialize_reward_models(self):
        # set up the reward models from the config
        self.motion_reward_model = MotionRewardModel(self.config.ic_reward_model_config, self.config.lc_reward_model_config)

        # NEW: cache reward weights (compat with config)
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
        # NEW: EMA only for trainable/enabled parts
        self.motion_reward_ema = None
        if getattr(self.config, "reward_model_ema", True):
            params = []
            if self.train_ic_reward:
                params += self.motion_reward_model.ic_reward_model.get_trainable_params()
            if self.train_lc_reward:
                params += self.motion_reward_model.lc_reward_model.get_trainable_params()
            if len(params) > 0:
                decay = float(getattr(self.config, "reward_model_ema_decay", 0.999))
                self.motion_reward_ema = ExponentialMovingAverage(params, decay=decay)
        #setup the vbench appearance reward model
        self.vbench_appearance_reward_weight = float(getattr(self.config, "vbench_appearance_advantage_weight", 0.0))
        self.use_vbench_appearance_reward = self.vbench_appearance_reward_weight > 0.0

        self.vbench_appearance_reward_model = None
            
    def on_fit_start(self):
        """Setup before training starts"""
        self.svd_pipeline.to(self.device)
        self.unet.to(self.device).train()
        self.motion_reward_model.to(device=self.device, dtype=torch.float32)
        self.motion_reward_model.to(self.device)
        self.motion_reward_model.eval()

        if self.use_vbench_appearance_reward and self._vbench["model"] is None:
            m = VBenchAppearanceRewardModel(device=str(self.device))
            m.freeze_params()
            m.eval()
            self._vbench["model"] = m

        # move EMA shadow params to device
        if getattr(self, "motion_reward_ema", None) is not None:
            for sp in self.motion_reward_ema.shadow_params:
                if sp.device != self.device:
                    sp.data = sp.data.to(self.device)

        # Initialize the buffer at the start of training
        if not self.buffer_initialized:
            self.recompute_buffer(1)  # Fill with first chunk

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
        use_deepspeed = getattr(self.config, 'use_deepspeed', False)
        cpu_offload = getattr(self.config, 'cpu_offload', False)
        
        # --- Optimizer G (UNet LoRA) ---
        g_param_groups = [dict(
            params=self.trainable_params,
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
            )]

        # --- Optimizer D (reward models) ---
        d_param_groups = []
        if self.train_ic_reward:
            ic_rm_params = self.motion_reward_model.ic_reward_model.get_trainable_params()
            d_param_groups.append(
                dict(
                    params=ic_rm_params,
                    lr=self.config.ic_reward_model_lr,
                    betas=(self.config.ic_reward_model_beta1, self.config.ic_reward_model_beta2),
                    weight_decay=self.config.ic_reward_model_weight_decay,
                )
            )

        if self.train_lc_reward:
            lc_rm_params = self.motion_reward_model.lc_reward_model.get_trainable_params()
            d_param_groups.append(
                dict(
                    params=lc_rm_params,
                    lr=self.config.lc_reward_model_lr,
                    betas=(self.config.lc_reward_model_beta1, self.config.lc_reward_model_beta2),
                    weight_decay=self.config.lc_reward_model_weight_decay,
                )
            )

        # --- Scheduler (keep yours for G; keep D constant) ---
        if self.config.scheduler == "stepLR":
            step_size = self.config.accum_grad_steps
            gamma = 0.99

            def lr_lambda_g(step):
                return gamma ** (step // max(1, step_size))

        elif self.config.scheduler == "linear_warmup":
            warmup = self.config.warmup_steps

            def lr_lambda_g(step):
                if warmup > 0 and step < warmup:
                    return float(step) / float(max(1, warmup))
                return 1.0
        else:
            raise ValueError(f"Scheduler {self.config.scheduler} not supported")
        
        if use_deepspeed:
            if cpu_offload:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                opt = DeepSpeedCPUAdam(g_param_groups + d_param_groups, eps=1e-8)
            else:
                from deepspeed.ops.adam import FusedAdam
                print("Use DeepSpeed FusedAdam")
                opt = FusedAdam(g_param_groups + d_param_groups)
        else:
            opt = torch.optim.AdamW(g_param_groups + d_param_groups)

        lr_lambdas = [lr_lambda_g] + [lambda step: 1.0] * len(d_param_groups)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambdas)
        return [opt], [sched]

    
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

        # save reward model weights if we are training them
        if getattr(self.config, "train_reward_models", False):
            try:
                checkpoint["motion_reward_model_state_dict"] = {
                    k: v.cpu() for k, v in self.motion_reward_model.state_dict().items()
                }
            except Exception as e:
                print(f"Warning: failed to save motion_reward_model state: {e}")

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
        
        if getattr(self, "motion_reward_ema", None) is not None:
            try:
                checkpoint["motion_reward_ema_state"] = self.motion_reward_ema.state_dict()
            except Exception as e:
                print(f"Warning: failed to save motion_reward_ema state: {e}")

    def _flush_unet_optimizer_if_needed(self, batch_idx: int):
        """
        If we're mid-gradient-accumulation when a buffer ends, force an optimizer step
        so each buffer gets an exact, self-contained number of UNet updates.
        """
        accum_steps = int(getattr(self.config, "accum_grad_steps", 1))
        accum_steps = max(1, accum_steps)

        # If we're exactly on an accumulation boundary, training_step already stepped+zeroed.
        if ((batch_idx + 1) % accum_steps) == 0:
            return

        # If there are no grads, nothing to flush.
        if not any((p.grad is not None) for p in self.trainable_params):
            return

        # UNet optimizer is group 0 (same as training_step)
        opt = self.optimizers()
        opt = opt[0] if isinstance(opt, (list, tuple)) else opt

        # Same clipping/step/scheduler logic as training_step
        self.clip_gradients(
            opt,
            gradient_clip_val=self.config.gradient_clip,
            gradient_clip_algorithm="norm",
        )
        opt.step()
        opt.zero_grad(set_to_none=True)
        self.g_step += 1

        lr_schedulers = self.lr_schedulers()
        lr_scheduler0 = lr_schedulers[0] if isinstance(lr_schedulers, (list, tuple)) else lr_schedulers
        if lr_scheduler0 is not None:
            lr_scheduler0.step()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # If we just finished all passes through a buffer:
        if (batch_idx + 1) % self.buffer_update_frequency == 0:
            # flush any pending UNet grad accumulation BEFORE buffer refresh
            self._flush_unet_optimizer_if_needed(batch_idx)
            # train the vid_reward model right after the generator training
            if getattr(self.config, "train_reward_models", False):
                self.train_motion_reward_model_from_buffer()

            sampler = self.trainer.datamodule.sampler
            # num_chunks = len(data_source) // chunk_size, where chunk_size = buffer_size
            num_chunks = max(1, sampler.num_chunks)
            chunks_completed = (batch_idx + 1) // self.buffer_update_frequency
            # in case that chunks_completed > num_chunks, we wrap around to the first chunk
            next_chunk_id = (chunks_completed % num_chunks) + 1
            self.recompute_buffer(next_chunk_id)

            self.current_pass = 0
        # Or if we finished a pass but not all passes:
        elif (batch_idx + 1) % self.iterations_per_chunk == 0:
            self.current_pass += 1
            self.reshuffle_buffer()
            # Print the current pass number
            if self.global_rank == 0:
                print(f"Buffer reshuffled. Starting pass {self.current_pass + 1}/{self.passes_per_buffer} through the current buffer")


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
                import argparse
                config = argparse.Namespace(**checkpoint['config_dict'])
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
        
        # load motion reward model weights if present
        if "motion_reward_model_state_dict" in checkpoint:
            try:
                model.motion_reward_model.load_state_dict(checkpoint["motion_reward_model_state_dict"], strict=False)
                print(f"Successfully loaded motion_reward_model from checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to load motion_reward_model weights: {e}")

        if "motion_reward_ema_state" in checkpoint and getattr(model, "motion_reward_ema", None) is not None:
            try:
                model.motion_reward_ema.load_state_dict(checkpoint["motion_reward_ema_state"])
                print(f"Successfully loaded motion_reward_ema from checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to load motion_reward_ema state: {e}")

        return model


    @torch.no_grad()
    def compute_reward(self, sample_video, lc_queries_point=None):
        """
        compute the reward for the sample video
        Args:
            sample_video: (B, T, C, H, W), with pixel values in range [0, 255] or [0, 1]
        Returns:
            ic_reward: (B,), the reward for the sample video
            lc_reward: (B,), the reward for the sample video
            ave_reward: (B,), the average reward for the sample video
        """
        # Reward scoring should run in eval mode (especially LC TrackFormer) when
        # collecting rollouts / computing advantages.
        prev_motion_training = self.motion_reward_model.training
        prev_trackformer_training = self.motion_reward_model.lc_reward_model.traj_discriminator.track_former.training
        self.motion_reward_model.eval()
        try:
            if getattr(self, "motion_reward_ema", None) is not None and getattr(self.config, "reward_model_use_ema_for_diffusion", True):
                with self.motion_reward_ema.average_parameters():
                    ic_reward, lc_reward, ave_reward, queries_point = self.motion_reward_model(sample_video, lc_queries_point)
            else:
                ic_reward, lc_reward, ave_reward, queries_point = self.motion_reward_model(sample_video, lc_queries_point)
        finally:
            if prev_motion_training:
                self.motion_reward_model.train()
            if prev_trackformer_training:
                self.motion_reward_model.lc_reward_model.traj_discriminator.track_former.train()
        vbench_reward = torch.zeros_like(ave_reward)
        if self.use_vbench_appearance_reward:
            vbench_reward, _ = self._vbench["model"](sample_video) # returns (B,)

        return ic_reward, lc_reward, ave_reward, queries_point, vbench_reward

    def edm_loss(self, sigmas, model_pred, noisy_latents, target):
        # Denoise the latents
        c_out = -sigmas / ((sigmas**2 + 1)**0.5)
        c_skip = 1 / (sigmas**2 + 1)
        denoised_latents = model_pred * c_out + c_skip * noisy_latents
        weighing = (1 + sigmas ** 2) * (sigmas**-2.0)
        # MSE loss
        loss = weighing.float() * F.mse_loss(denoised_latents, target.to(denoised_latents.dtype), reduction='none')
        loss = loss.flatten(start_dim=1).mean(dim=1)

        return loss #(b, )

    def validation_step_loss(self, real_vid, fake_vid, fake_advantage, real_advantage, stage='val'):
        # compute the sft loss with real_vid
        sft_losses = self.shared_step(real_vid, stage=stage)[0] #(b,)
        asym_grpo = bool(getattr(self.config, "asymmetric_grpo", False))
        min_sft_loss_lambda = self.config.min_sft_loss_lambda
        aw_sft_loss_lambda = self._get_aw_sft_loss_lambda()
        
        if asym_grpo:
            sft_losses = ((real_advantage + min_sft_loss_lambda) * sft_losses).mean()
        else:
            sft_losses = min_sft_loss_lambda * sft_losses.mean()

        # compute the advantage-weighted finetuning loss
        assert fake_vid.shape[0] == self.config.num_sample_videos_per_prompt, "Invalid fake_vid"
        aw_sft_losses = 0
        # print(f'local rank {self.global_rank}, fake_vid shape: {fake_vid.shape}, real_vid shape: {real_vid.shape}')
        for p in range(self.config.num_sample_videos_per_prompt):
            batch_fake_vid = fake_vid[p, ...]
            batch_aw_sft_loss = self.shared_step(batch_fake_vid, stage=stage)[0] #(b, )
            batch_aw_sft_loss = fake_advantage[p, ...] * batch_aw_sft_loss
            aw_sft_losses += batch_aw_sft_loss
        aw_sft_losses = aw_sft_losses / self.config.num_sample_videos_per_prompt
        aw_sft_losses = aw_sft_losses.mean()
        
        total_loss = sft_losses + aw_sft_losses * aw_sft_loss_lambda

        return aw_sft_losses, sft_losses, total_loss
    
    # Abstract methods to be implemented by subclasses
    def shared_step(self, batch, stage='train', noise=None, sigmas=None, cond_sigmas=None, random_p=None):
        """Main training loop"""
        vid = batch
        vid = vid.to(self.device)
        # encode the video to latents
        with torch.no_grad():
            latents, noisy_latents, inp_noisy_latents, encoder_hidden_states, added_time_ids, timesteps, noise, sigmas, cond_sigmas, random_p = self.encode_pixels(vid, target_fps=self.config.target_fps, noise=noise, sigmas=sigmas, cond_sigmas=cond_sigmas, random_p=random_p)

        image_embeddings = encoder_hidden_states

        def unet_forward_fn(model_input, timestep, encoder_states, time_ids):
            unet_dtype = next(self.unet.parameters()).dtype
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
                return self.unet(
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
        
        return model_losses, noise, sigmas, cond_sigmas, random_p
        
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

    def encode_pixels(self, vid_tensor, noise=None, cond_sigmas=None, sigmas=None, target_fps=7, random_p=None):
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
        conditional_pixel_values = \
            torch.randn_like(conditional_pixel_values) * cond_sigmas[:, None, None, None, None] + conditional_pixel_values
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
            target_fps - 1, # fixed
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
            if random_p is None:
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

        return latents, noisy_latents, inp_noisy_latents, encoder_hidden_states, added_time_ids, timesteps, noise, sigmas, cond_sigmas, random_p
    
    def _ds_set_boundary(self, is_boundary: bool):
        # Works only under DeepSpeedStrategy
        engine = getattr(self.trainer.strategy, "model", None)
        if hasattr(engine, "set_gradient_accumulation_boundary"):
            engine.set_gradient_accumulation_boundary(is_boundary)

    def training_step(self, batch, batch_idx):
        # Manual optimization
        opt = self.optimizers()
        opt = opt[0] if isinstance(opt, (list, tuple)) else opt
    
        accum_steps = getattr(self.config, "accum_grad_steps", 1)
        # first micro-batch of the accumulation window, for log gradients
        is_first_micro = ((batch_idx % accum_steps) == 0)
        do_grad_norm_log = (is_first_micro and self.global_step % self.config.log_grad_norm_every_steps == 0)
        if do_grad_norm_log:
            self._capture_grads = True
            self._captured_total_grads = {}

        real_vid, fake_vid, fake_advantage, fake_ic_rewards, fake_lc_rewards, fake_vbench_rewards, fake_ave_rewards, real_advantage, real_ic_rewards, real_lc_rewards, real_vbench_rewards, real_ave_rewards = \
            self.generate_train_batch(batch, batch_idx)

        asym_grpo = bool(getattr(self.config, "asymmetric_grpo", False))
        min_sft_loss_lambda = self.config.min_sft_loss_lambda
        aw_sft_loss_lambda = self._get_aw_sft_loss_lambda()

        # ----- real SFT loss -----
        # also return the noise for later aw_sft
        sft_losses, noise, sigmas, cond_sigmas, random_p = self.shared_step(real_vid, stage='train')  # (B,)
        if asym_grpo:
            sft_loss = ((real_advantage + min_sft_loss_lambda) * sft_losses).mean()
        else:
            sft_loss = min_sft_loss_lambda * sft_losses.mean()

        # backward on SFT part
        if self.config.use_deepspeed:
            self._ds_set_boundary(False)
        self.manual_backward(sft_loss)
        # log SFT grad norm, also save as snapshot for later use
        gn_sft, gn_aw = None, None
        if do_grad_norm_log:
            # snapshot right after SFT backward (captured grads = SFT grads)
            g_sft_snap = {k: self._captured_total_grads.get(k, None) for k in self._lora_param_names}
            g_sft_snap_cpu, gn_sft = sft_grad_norm_log_from_grads(g_sft_snap, self.config.gradnorm_group_prefix_regex)

        # ----- AW SFT loss (per-sample backward to save memory) -----
        num_p = self.config.num_sample_videos_per_prompt
        scale = aw_sft_loss_lambda / num_p
        aw_sft_loss_for_logging = 0.0

        for p in range(num_p):
            batch_fake_vid = fake_vid[p, ...]
            if self.config.match_sigmas:
                batch_aw_sft_loss_vec = self.shared_step(batch_fake_vid, stage='train', noise=noise, sigmas=sigmas, cond_sigmas=cond_sigmas, random_p=random_p)[0]  # (B,)
            else:
                # do not use the same random noise from the sft
                batch_aw_sft_loss_vec = self.shared_step(batch_fake_vid, stage='train')[0]  # (B,)
            # weight by advantage and reduce to scalar
            batch_aw_sft_loss = (fake_advantage[p, ...] * batch_aw_sft_loss_vec).mean()

            # accumulate detached value for logging only
            aw_sft_loss_for_logging = aw_sft_loss_for_logging + batch_aw_sft_loss.detach()

            # backward just this chunk; its graph can be freed afterwards
            is_last = (p == num_p - 1)
            if self.config.use_deepspeed:
                self._ds_set_boundary(is_last)
            self.manual_backward(scale * batch_aw_sft_loss)

        # log the aw_sft loss gradient
        if do_grad_norm_log:
            total_snap = {k: self._captured_total_grads.get(k, None) for k in self._lora_param_names}
            gn_aw = aw_sft_grad_norm_log_from_grads(total_snap, g_sft_snap_cpu, self.config.gradnorm_group_prefix_regex)
            # compute the mean norm ratio (gn_aw/(gn_sft + 1e-9)) of each layer 
            _, norm_ratios_vec = layer_grad_norm_mean_ratio(gn_aw, gn_sft, eps = 1e-9)
            # stop capturing after logging (avoid overhead)
            self._capture_grads = False

        aw_sft_loss_avg = aw_sft_loss_for_logging / num_p
        total_loss_for_logging = (
            sft_loss.detach()
            + aw_sft_loss_lambda * aw_sft_loss_avg
        )

        # ----- gradient accumulation / optimizer & scheduler step -----
        # mimic Trainer(accumulate_grad_batches=accum_steps)
        should_step = ((batch_idx + 1) % accum_steps == 0) or (
            batch_idx + 1 == self.trainer.num_training_batches
        )

        if should_step:
            # explicit gradient clipping (was previously done by Trainer)
            self.clip_gradients(
                opt,
                gradient_clip_val=self.config.gradient_clip,
                gradient_clip_algorithm="norm",
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            self.g_step += 1

            # LR scheduler: your configure_optimizers uses step-level schedulers
            lr_schedulers = self.lr_schedulers()
            lr_scheduler0 = lr_schedulers[0] if isinstance(lr_schedulers, (list, tuple)) else lr_schedulers
            if lr_scheduler0 is not None:
                lr_scheduler0.step()

        # ----- logging (same metrics as before, detached) -----
        if hasattr(self, "logger") and self.logger is not None:
            tb_exp = getattr(self.logger, "experiment", None)
            if self.global_rank == 0:
                log_hist(tb_exp, {
                    "fake_advantage": fake_advantage,
                    "fake_ic_rewards": fake_ic_rewards,
                    "fake_lc_rewards": fake_lc_rewards,
                    "fake_vbench_rewards": fake_vbench_rewards,
                    "fake_ave_rewards": fake_ave_rewards,
                    "real_advantage": real_advantage,
                    "real_ic_rewards": real_ic_rewards,
                    "real_lc_rewards": real_lc_rewards,
                    "real_vbench_rewards": real_vbench_rewards,
                    "real_ave_rewards": real_ave_rewards,
                }, stage="train", global_step=self.global_step)
                if gn_aw is not None and gn_sft is not None:
                    log_hist(tb_exp, gn_aw, stage="aw_sft_grad_norm/", global_step=self.global_step)
                    log_hist(tb_exp, gn_sft, stage="sft_grad_norm/", global_step=self.global_step)
                    log_hist(tb_exp, {"grad-norm-ratio": norm_ratios_vec}, stage="aw-sft-over-sft", global_step=self.global_step)

        metric_dict = {
            "aw_sft_loss": aw_sft_loss_avg,
            "aw_sft_loss_lambda": torch.tensor(aw_sft_loss_lambda, device=self.device),
            "sft_loss": sft_loss.detach(),
            "total_loss": total_loss_for_logging,
            "fake_ic_rewards": fake_ic_rewards.mean().detach(),
            "fake_lc_rewards": fake_lc_rewards.mean().detach(),
            "fake_vbench_rewards": fake_vbench_rewards.mean().detach(),
            "fake_ave_rewards": fake_ave_rewards.mean().detach(),
            "real_ic_rewards": real_ic_rewards.mean().detach(),
            "real_lc_rewards": real_lc_rewards.mean().detach(),
            "real_vbench_rewards": real_vbench_rewards.mean().detach(),
            "real_ave_rewards": real_ave_rewards.mean().detach(),
            "learning_rate": torch.tensor(opt.param_groups[0]["lr"], device=self.device)
        }
        self.log_metrics("train", metric_dict, batch_size=real_vid.size(0))

        # In manual optimization the return value is for logging only
        return total_loss_for_logging
    
    def validation_step(self, batch, batch_idx):
        if not hasattr(self, "_val_hist_cache"):
            self._val_hist_cache = defaultdict(list)
        # Process this batch
        collected_data = self.collect_data(batch, batch_idx)
        real_videos = collected_data['real_videos'].to(self.device, non_blocking=True)
        fake_videos = collected_data['fake_videos'].to(self.device, non_blocking=True)
        fake_advantage = collected_data['fake_advantage'].to(self.device, non_blocking=True)
        real_advantage = collected_data['real_advantage'].to(self.device, non_blocking=True)

        # Keep full tensors for histogram logging; compute means for scalar logging
        fake_ic_rewards_full = collected_data['fake_ic_rewards'].to(self.device, non_blocking=True)
        fake_lc_rewards_full = collected_data['fake_lc_rewards'].to(self.device, non_blocking=True)
        fake_ave_rewards_full = collected_data['fake_ave_rewards'].to(self.device, non_blocking=True)
        fake_vbench_rewards_full = collected_data['fake_vbench_rewards'].to(self.device, non_blocking=True)
        fake_ic_rewards = fake_ic_rewards_full.mean()
        fake_lc_rewards = fake_lc_rewards_full.mean()
        fake_vbench_rewards = fake_vbench_rewards_full.mean()
        fake_ave_rewards = fake_ave_rewards_full.mean()

        real_ic_rewards_full = collected_data['real_ic_rewards'].to(self.device, non_blocking=True)
        real_lc_rewards_full = collected_data['real_lc_rewards'].to(self.device, non_blocking=True)
        real_ave_rewards_full = collected_data['real_ave_rewards'].to(self.device, non_blocking=True)
        real_vbench_rewards_full = collected_data['real_vbench_rewards'].to(self.device, non_blocking=True)
        real_ic_rewards = real_ic_rewards_full.mean()
        real_lc_rewards = real_lc_rewards_full.mean()
        real_vbench_rewards = real_vbench_rewards_full.mean()
        real_ave_rewards = real_ave_rewards_full.mean()

        aw_sft_loss, sft_loss, total_loss = self.validation_step_loss(real_videos, fake_videos, fake_advantage, real_advantage, stage='val')

        self._val_hist_cache["fake_advantage"].append(fake_advantage.detach().flatten().cpu())
        self._val_hist_cache["fake_ic_rewards"].append(fake_ic_rewards_full.detach().flatten().cpu())
        self._val_hist_cache["fake_lc_rewards"].append(fake_lc_rewards_full.detach().flatten().cpu())
        self._val_hist_cache["fake_vbench_rewards"].append(fake_vbench_rewards_full.detach().flatten().cpu())
        self._val_hist_cache["fake_ave_rewards"].append(fake_ave_rewards_full.detach().flatten().cpu())
        self._val_hist_cache["real_advantage"].append(real_advantage.detach().flatten().cpu())
        self._val_hist_cache["real_ic_rewards"].append(real_ic_rewards_full.detach().flatten().cpu())
        self._val_hist_cache["real_lc_rewards"].append(real_lc_rewards_full.detach().flatten().cpu())
        self._val_hist_cache["real_vbench_rewards"].append(real_vbench_rewards_full.detach().flatten().cpu())
        self._val_hist_cache["real_ave_rewards"].append(real_ave_rewards_full.detach().flatten().cpu())

        metric_dict = {
            "aw_sft_loss": aw_sft_loss,
            "sft_loss": sft_loss,
            "total_loss": total_loss,
            "fake_ic_rewards": fake_ic_rewards,
            "fake_lc_rewards": fake_lc_rewards,
            "fake_vbench_rewards": fake_vbench_rewards,
            "fake_ave_rewards": fake_ave_rewards,
            "real_ic_rewards": real_ic_rewards,
            "real_lc_rewards": real_lc_rewards,
            "real_vbench_rewards": real_vbench_rewards,
            "real_ave_rewards": real_ave_rewards,
        }
        self.log_metrics("val", metric_dict, batch_size = real_videos.size(0))

        return total_loss
    
    def on_validation_epoch_end(self):
        # log out the histogram of validation rewards and advantage
        tb_exp = getattr(self.logger, "experiment", None)
        for name, chunks in self._val_hist_cache.items():
            if len(chunks) == 0:
                continue
            vals = torch.cat(chunks, dim=0)
            if self.global_rank == 0:
                log_hist(tb_exp, {name: vals}, stage="val", global_step=self.global_step)
        # clear cache
        self._val_hist_cache = defaultdict(list)

    @torch.no_grad()
    def collect_data(self, batch_vid, batch_start):
        """
        Args:
            batch_vid: (B, T, C, H, W), with pixel values in range [-1, 1]
        """
        batch_vid = batch_vid.to(self.device, non_blocking=True).to(self.torch_dtype)
        gpu_num = torch.cuda.current_device()
        pl.seed_everything(gpu_num + self.config.seed + 10 * self.global_batch_step + 100 * batch_start)

        # reward model expects [0,1]
        real_vid_01 = ((batch_vid.float() + 1.0) * 0.5).clamp(0.0, 1.0)   # (B,T,C,H,W)
        real_ic_reward, real_lc_reward, real_ave_reward, lc_queries_point, real_vbench_reward = self.compute_reward(real_vid_01)          # (B,)

        # first, convert images to latent space.
        pixel_values = batch_vid
        conditional_pixel_values = pixel_values[:, 0:1, :, :, :]

        bsz = pixel_values.shape[0]
        cond_sigmas = torch.Tensor(
            [self.config.sample_noise_aug_strength]
        )[:, None, None, None, None]
        conditional_pixel_values = (
            torch.randn_like(conditional_pixel_values)
            * cond_sigmas.to(conditional_pixel_values.device)
            + conditional_pixel_values
        )
        conditional_latents = self.tensor_to_vae_latent(
            conditional_pixel_values.to(self.torch_dtype)
        )[:, 0, :, :, :]
        conditional_latents = conditional_latents / self.vae.config.scaling_factor

        # Get the text embedding for conditioning.
        encoder_hidden_states = self.encode_image(
            pixel_values[:, 0, :, :, :].float()
        )

        # Fixed added_time_ids (per prompt)
        added_time_ids = self._get_add_time_ids(
            self.config.target_fps - 1,           # fixed
            127,                              # motion_bucket_id = 127, fixed
            self.config.sample_noise_aug_strength,
            encoder_hidden_states.dtype,
            bsz,
        )
        added_time_ids = added_time_ids.to(pixel_values.device)

        # 3. Grouped sampling along p (num_sample_videos_per_prompt) axis
        total_p = self.config.num_sample_videos_per_prompt
        group_p = getattr(self.config, "sample_collect_group_size", total_p)
        group_p = max(1, min(group_p, total_p))

        all_fake_videos = []
        all_fake_ic_rewards = []
        all_fake_lc_rewards = []
        all_fake_ave_rewards = []
        all_fake_vbench_rewards = []

        for start_p in range(0, total_p, group_p):
            cur_p = min(group_p, total_p - start_p)

            # 3.0 Prepare timesteps (shared across all groups)
            timesteps, num_inference_steps = retrieve_timesteps(
                self.svd_pipeline.scheduler,
                self.config.num_inference_steps,
                pixel_values.device,
                None,
                sigmas=None,
            )
            # 3.1 Prepare latents for this group: (bsz * cur_p, F, C, H, W)
            latents = self.svd_pipeline.prepare_latents(
                bsz * cur_p,
                self.config.target_vid_size[0],
                self.unet.config.in_channels,
                self.config.target_vid_size[1],
                self.config.target_vid_size[2],
                encoder_hidden_states.dtype,
                pixel_values.device,
                generator=None,
            )

            # 3.2 Repeat conditioning for this group
            cond_latents_group = repeat(
                conditional_latents,
                "b c h w -> (p b) f c h w",
                p=cur_p,
                f=latents.shape[1],
            )
            if self.config.sample_do_classifier_free_guidance:
                cond_latents_group = torch.cat(
                    [torch.zeros_like(cond_latents_group), cond_latents_group],
                    dim=0,
                )

            encoder_hidden_group = repeat(
                encoder_hidden_states,
                "b c -> (p b) c",
                p=cur_p,
            )
            if self.config.sample_do_classifier_free_guidance:
                encoder_hidden_group = torch.cat(
                    [torch.zeros_like(encoder_hidden_group), encoder_hidden_group],
                    dim=0,
                )
            encoder_hidden_group = encoder_hidden_group.unsqueeze(1)

            added_time_ids_group = repeat(
                added_time_ids,
                "b c -> (p b) c",
                p=cur_p,
            )
            if self.config.sample_do_classifier_free_guidance:
                added_time_ids_group = torch.cat(
                    [added_time_ids_group, added_time_ids_group],
                    dim=0,
                )

            # 3.3 Guidance scale for this group
            guidance_scale = torch.linspace(
                self.config.sample_min_guidance_scale,
                self.config.sample_max_guidance_scale,
                self.config.target_vid_size[0],
            ).unsqueeze(0)
            guidance_scale = guidance_scale.to(self.device, latents.dtype)
            guidance_scale = guidance_scale.repeat(bsz * cur_p, 1)
            self.guidance_scale = _append_dims(guidance_scale, latents.ndim)

            # 3.4 Denoising loop for this group
            num_warmup_steps = (
                len(timesteps)
                - num_inference_steps * self.svd_pipeline.scheduler.order
            )
            self._num_timesteps = len(timesteps)
            progress_bar_cm = self.svd_pipeline.progress_bar(total=num_inference_steps) if self.global_rank == 0 else nullcontext()
            with progress_bar_cm as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if self.config.sample_do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.svd_pipeline.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # Concatenate image conditional_latents over channels dimension
                    latent_model_input = torch.cat(
                        [latent_model_input, cond_latents_group], dim=2
                    )
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_group,
                        added_time_ids=added_time_ids_group,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.config.sample_do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = (
                            noise_pred_uncond
                            + self.guidance_scale
                            * (noise_pred_cond - noise_pred_uncond)
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    step_dict = self.svd_pipeline.scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=latents,
                        return_dict=True,
                    )
                    latents = step_dict["prev_sample"]
                    if (
                        i == len(timesteps) - 1
                        or (
                            (i + 1) > num_warmup_steps
                            and (i + 1) % self.svd_pipeline.scheduler.order == 0
                        )
                    ):
                        if progress_bar is not None:
                            progress_bar.update()

                frames = self.svd_pipeline.decode_latents(
                    latents,
                    self.config.target_vid_size[0],
                    self.config.sample_decode_chunk_size,
                )
                fake_video_group = self.svd_pipeline.video_processor.postprocess_video(
                    video=frames.detach(), output_type="pt"
                )  # ((cur_p * bsz), T, C, H, W), [0,1]

            # 3.5 Rewards for this group
            queries_for_fake = repeat(lc_queries_point, "b n c -> (p b) n c", p=cur_p)
            fake_ic, fake_lc, fake_ave, _, fake_vbench = self.compute_reward(fake_video_group, queries_for_fake)
            fake_ic = rearrange(fake_ic, "(p b) -> p b", p=cur_p)
            fake_lc = rearrange(fake_lc, "(p b) -> p b", p=cur_p)
            fake_ave = rearrange(fake_ave, "(p b) -> p b", p=cur_p)
            fake_vbench = rearrange(fake_vbench, "(p b) -> p b", p=cur_p)
            fake_video_group = rearrange(
                fake_video_group,
                "(p b) f c h w -> p b f c h w",
                p=cur_p,
            )

            all_fake_videos.append(fake_video_group)
            all_fake_ic_rewards.append(fake_ic)
            all_fake_lc_rewards.append(fake_lc)
            all_fake_ave_rewards.append(fake_ave)
            all_fake_vbench_rewards.append(fake_vbench)

        # 4. Concatenate all groups back to full shapes (total_p, B, ...)
        fake_video = torch.cat(all_fake_videos, dim=0)             # (P, B, T, C, H, W)
        fake_ic_rewards = torch.cat(all_fake_ic_rewards, dim=0)    # (P, B)
        fake_lc_rewards = torch.cat(all_fake_lc_rewards, dim=0)    # (P, B)
        fake_ave_rewards = torch.cat(all_fake_ave_rewards, dim=0)  # (P, B)
        fake_vbench_rewards = torch.cat(all_fake_vbench_rewards, dim=0)  # (P, B)

        # visualize the generated iamges in debug_mode
        if self.config.debug_mode and self.config.visualize_examples:
            from diffusers.utils import export_to_video
            fake_frames = rearrange(fake_video, 'p b f c h w -> p b f h w c', p=self.config.num_sample_videos_per_prompt, b=bsz)
            for p_idx in range(fake_frames.shape[0]):
                for batch_idx in range(fake_frames.shape[1]):
                    frames = [i.cpu().numpy() for i in fake_frames[p_idx, batch_idx, ...]]
                    export_to_video(frames, f"./generated_ancestral_{num_inference_steps}_rand{self.config.seed}_vaechunk_{self.config.sample_decode_chunk_size}_rank_{self.global_rank}_batch{batch_idx}_rand{p_idx}.mp4", fps=self.config.target_fps)

            pixel_values = (pixel_values + 1) * 0.5
            pixel_values = pixel_values.permute(0, 1, 3, 4, 2).to(fake_frames.dtype)#(B, T, C, H, W), pixel values in range [0, 1]
            for batch_idx in range(pixel_values.shape[0]):
                frames = [i.cpu().numpy() for i in pixel_values[batch_idx]]
                export_to_video(frames, f"./original_{num_inference_steps}_rand{self.config.seed}_vaechunk_{self.config.sample_decode_chunk_size}_rank_{self.global_rank}_batch{batch_idx}.mp4", fps=self.config.target_fps)

        # 5. Compute group advantage (unchanged)
        fake_ave_advantage, unweighted_fake_ave_advantage = self.compute_advantage(fake_ave_rewards)
        if self.use_vbench_appearance_reward:
            fake_vbench_advantage, unweighted_fake_vbench_advantage = self.compute_advantage(fake_vbench_rewards)

            # Compute per-batch std to match scales
            motion_std = fake_ave_advantage.std(dim=0, keepdim=True)  # (1, B)
            vbench_std = fake_vbench_advantage.std(dim=0, keepdim=True)  # (1, B)
            scale = motion_std / (vbench_std + 1e-6)
            # Merge with matched scale
            vbench_weight = float(getattr(self.config, "vbench_appearance_advantage_weight", 1.0))
            fake_ave_advantage = fake_ave_advantage + vbench_weight * (fake_vbench_advantage * scale)

        asym_grpo = bool(getattr(self.config, "asymmetric_grpo", False))

        fake_mean_reward = fake_ave_rewards.float().mean(dim=0)           # (B,)
        real_advantage = (real_ave_reward.float() - fake_mean_reward).clamp(min=0.0, max=self.config.adv_clip_max)  # hinge, (B, )
        if self.use_vbench_appearance_reward:
            fake_mean_vbench_reward = fake_vbench_rewards.float().mean(dim=0) # (B,)
            real_vbench_advantage = (real_vbench_reward.float() - fake_mean_vbench_reward).clamp(min=0.0, max=self.config.adv_clip_max)  # hinge, (B, )

            motion_std = real_advantage.std(dim=0, keepdim=True)  # (1, B)
            vbench_std = real_vbench_advantage.std(dim=0, keepdim=True)  # (1, B)
            scale = motion_std / (vbench_std + 1e-6)
            # Merge with matched scale
            real_advantage = real_advantage + vbench_weight * (real_vbench_advantage * scale)

        fake_video = fake_video * 2.0 - 1.0  # convert to [-1, 1]
        output = {
            "real_videos": batch_vid.to(self.torch_dtype),
            "real_advantage": real_advantage,
            "real_ic_rewards": real_ic_reward,
            "real_lc_rewards": real_lc_reward,
            "real_vbench_rewards": real_vbench_reward,
            "real_ave_rewards": real_ave_reward,
            "fake_videos": fake_video.to(self.torch_dtype),
            "fake_ave_rewards": fake_ave_rewards,
            "fake_ic_rewards": fake_ic_rewards,
            "fake_lc_rewards": fake_lc_rewards,
            "fake_vbench_rewards": fake_vbench_rewards,
            "fake_advantage": fake_ave_advantage,
            "unweighted_fake_advantage": unweighted_fake_ave_advantage,
        }
        return output

    def train_motion_reward_model_from_buffer(self):
        """
        GAN-like discriminator update using (real, fake) videos in current buffer.
        Trains BOTH IC and LC discriminators (trainable parts only), then freezes back.

        New features:
        - reward_model_d_to_g: float, controls RM optimizer steps per UNet optimizer step (per buffer)
        """
        # get reward optimizer (same optimizer object; reward params live in param_group 1)
        opt = self.optimizers()
        opt = opt[0] if isinstance(opt, (list, tuple)) else opt
    
        self.motion_reward_model.to(self.device)
        # Important: keep *frozen* parts in eval() (disables dropout/BN updates, keeps CoTracker stable).
        # We'll selectively set only the trainable submodules to train() via their unfreeze_params().
        self.motion_reward_model.eval()
        if self.train_ic_reward:
            self.motion_reward_model.ic_reward_model.unfreeze_params()
        if self.train_lc_reward:
            self.motion_reward_model.lc_reward_model.unfreeze_params()

        # Optional: disable dropout inside LC updateformer during reward-model updates
        # (helps stabilize the GAN-like discriminator update).
        def _disable_dropout_(m: nn.Module):
            for mm in m.modules():
                if isinstance(mm, nn.Dropout):
                    mm.train(False)
        try:
            _disable_dropout_(self.motion_reward_model.lc_reward_model.traj_discriminator.cotracker.model.updateformer)
        except Exception as e:
            if self.global_rank == 0 and getattr(self.config, "debug_mode", False):
                print(f"Warning: failed to disable dropout in LC updateformer: {e}")

        # Offload diffusion modules to CPU to free memory
        unet_dev = self.unet.device
        vae_dev = next(self.vae.parameters()).device
        imgenc_dev = next(self.image_encoder.parameters()).device

        # ---buffer lists---
        buf_real = self.buffer["real_videos"]   # list of (B, T, C, H, W) in [-1, 1]
        buf_fake = self.buffer["fake_videos"]   # list of (P, B, T, C, H, W) in [-1, 1]
        num_buf_batches = len(buf_real)

        # how many RM steps per buffer: d_to_g * (UNet optimizer steps per buffer)
        unet_accum = int(getattr(self.config, "accum_grad_steps", 1))
        unet_accum = max(1, unet_accum)
        unet_opt_steps_per_buffer = int(math.ceil(self.buffer_update_frequency / float(unet_accum)))
        assert unet_opt_steps_per_buffer > 0, "unet_opt_steps_per_buffer must be greater than 0"

        d_to_g = float(getattr(self.config, "reward_model_d_to_g", 1.0))
        assert d_to_g > 0.0, "reward_model_d_to_g must be greater than 0.0"
        rm_steps = int(round(d_to_g * unet_opt_steps_per_buffer))
        assert rm_steps > 0, f"rm_steps must be greater than 0, d_to_g={d_to_g}, unet_opt_steps_per_buffer={unet_opt_steps_per_buffer}, d_to_g maybe too small"

        # reward batch multiplier K: each RM step consumes K buffer-batches => effective batch ~ K*B
        K = int(getattr(self.config, "reward_model_batch_multiplier", 1))
        K = max(1, K)

        pbar = tqdm(
            range(rm_steps),
            desc=f"MotionReward Steps {rm_steps} (d_to_g={d_to_g:g})",
            disable=(self.global_rank != 0),
            dynamic_ncols=True,
        )
        for step in pbar:
            # sample K buffer-batches (with replacement)
            idxs = torch.randint(0, num_buf_batches, (K,), device=self.device).tolist()

            real_list = []
            fake_list = []

            for j, bi in enumerate(idxs):
                real_b = buf_real[bi].to(self.device, non_blocking=True)  # (B, T, C, H, W)

                fake_b = buf_fake[bi]  # (P, B, T, C, H, W)
                p = (step + j) % fake_b.shape[0]  # cycle
                fake_sel = fake_b[p].to(self.device, non_blocking=True)  # (B, T, C, H, W)

                real_list.append(real_b)
                fake_list.append(fake_sel)

            real_vid = torch.cat(real_list, dim=0).float()  # (K*B, ...)
            fake_vid = torch.cat(fake_list, dim=0).float()  # (K*B, ...)
            real_vid = ((real_vid + 1.0) * 0.5).clamp(0.0, 1.0)  # reward models expect [0,1]
            fake_vid = ((fake_vid + 1.0) * 0.5).clamp(0.0, 1.0)  # reward models expect [0,1]

            if self.config.debug_mode and self.config.visualize_examples:
                from diffusers.utils import export_to_video
                fake_frames = rearrange(fake_vid, 'b f c h w -> b f h w c')
                for batch_idx in range(fake_frames.shape[0]):
                    frames = [i.cpu().numpy() for i in fake_frames[batch_idx, ...]]
                    export_to_video(frames, f"./fake_vid_for_reward_model{batch_idx}.mp4", fps=self.config.target_fps)
            
                real_frames = rearrange(real_vid, 'b f c h w -> b f h w c')
                for batch_idx in range(real_frames.shape[0]):
                    frames = [i.cpu().numpy() for i in real_frames[batch_idx, ...]]
                    export_to_video(frames, f"./real_vid_for_reward_model{batch_idx}.mp4", fps=self.config.target_fps)

            opt.zero_grad(set_to_none=True)
            ic_loss = torch.tensor(0.0, device=self.device)
            if self.train_ic_reward:
                _, ic_logits_real = self.motion_reward_model.ic_reward_model(real_vid)
                ic_loss_real = F.binary_cross_entropy_with_logits(ic_logits_real.float(), torch.ones_like(ic_logits_real).float())
                if self.config.use_deepspeed:
                    self._ds_set_boundary(False)
                self.manual_backward(self.ic_reward_weight * ic_loss_real)
                _, ic_logits_fake = self.motion_reward_model.ic_reward_model(fake_vid)
                ic_loss_fake = F.binary_cross_entropy_with_logits(ic_logits_fake.float(), torch.zeros_like(ic_logits_fake).float())
                if self.config.use_deepspeed:
                    self._ds_set_boundary(False)
                self.manual_backward(self.ic_reward_weight * ic_loss_fake)
                ic_loss = ic_loss_real + ic_loss_fake
                if self.global_rank == 0 and self.config.debug_mode:
                    print('step', step, 'ic_loss_real', ic_loss_real.item(), 'ic_loss_fake', ic_loss_fake.item())
                    print('step', step, 'ic_logits_real', ic_logits_real.mean().item(), 'ic_logits_fake', ic_logits_fake.mean().item())
                
            lc_loss = torch.tensor(0.0, device=self.device)
            if self.train_lc_reward:
                _, lc_logits_real, queries_point = self.motion_reward_model.lc_reward_model(real_vid, queries=None)
                lc_loss_real = F.binary_cross_entropy_with_logits(lc_logits_real.float(), torch.ones_like(lc_logits_real).float())
                if self.config.use_deepspeed:
                    self._ds_set_boundary(False)
                self.manual_backward(self.lc_reward_weight * lc_loss_real)
                _, lc_logits_fake, _ = self.motion_reward_model.lc_reward_model(fake_vid, queries=queries_point)
                lc_loss_fake = F.binary_cross_entropy_with_logits(lc_logits_fake.float(), torch.zeros_like(lc_logits_fake).float())
                if self.config.use_deepspeed:
                    self._ds_set_boundary(True)
                self.manual_backward(self.lc_reward_weight * lc_loss_fake)
                lc_loss = lc_loss_real + lc_loss_fake
                if self.global_rank == 0 and self.config.debug_mode:
                    print('step', step, 'lc_loss_real', lc_loss_real.item(), 'lc_loss_fake', lc_loss_fake.item())
                    print('step', step, 'lc_logits_real', lc_logits_real.mean().item(), 'lc_logits_fake', lc_logits_fake.mean().item())
                
            loss = ic_loss + lc_loss
            self.clip_gradients(opt, gradient_clip_val=self.config.gradient_clip, gradient_clip_algorithm="norm")
            # torch.nn.utils.clip_grad_norm_(self._rm_params, max_norm=self.config.gradient_clip)
            opt.step()  
            self.d_step += 1
            if getattr(self, "motion_reward_ema", None) is not None:
                self.motion_reward_ema.update()
            if self.global_rank == 0:
                pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", bs=str(int(real_vid.size(0) + fake_vid.size(0))))
            # ---- logging ----
            lr_groups = [pg["lr"] for pg in opt.param_groups]
            lr_ic_rm_opt = lr_groups[1] if self.train_ic_reward else 0.0
            lr_lc_rm_opt = lr_groups[2] if (self.train_ic_reward and self.train_lc_reward) else (lr_groups[1] if self.train_lc_reward else 0.0)

            # Normal training loop: self.log is allowed
            self.log_metrics(
                "train",
                {
                    "motion_reward_model_loss": loss.detach(),
                    "motion_reward_model_loss_ic": ic_loss.detach(),
                    "motion_reward_model_loss_lc": lc_loss.detach(),
                    "motion_reward_model_ic_lr": torch.tensor(lr_ic_rm_opt, device=self.device),
                    "motion_reward_model_lc_lr": torch.tensor(lr_lc_rm_opt, device=self.device),
                    "motion_reward_model_effective_batch": torch.tensor(int(real_vid.size(0) + fake_vid.size(0)), device=self.device),
                    "motion_reward_model_steps_per_buffer": torch.tensor(int(rm_steps), device=self.device).to(torch.float),
                    "unet_opt_steps_per_buffer": torch.tensor(int(unet_opt_steps_per_buffer), device=self.device).to(torch.float),
                    "reward_model_d_to_g": torch.tensor(float(d_to_g), device=self.device),
                },
                batch_size=real_vid.size(0),
            )
            del real_vid, fake_vid, real_b, fake_b, fake_sel, real_list, fake_list
            if self.train_ic_reward:
                del ic_logits_real, ic_logits_fake
            if self.train_lc_reward:
                del lc_logits_real, lc_logits_fake
            del loss, ic_loss, lc_loss
            if (step + 1) % 20 == 0:
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        gc.collect()
        
        # Freeze back + restore diffusion modules
        self.motion_reward_model.eval()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @torch.no_grad()
    def recompute_buffer(self, chunk_id):
        """The whole train dataset is divided into num_chunks = len(data_source) // buffer_size
        For example, len(train_source) = 8, buffer_size = 4, then num_chunks = 2
        epoch_chunks would be some random exmaple indices like [[4,1,5,2], [0,3,6,7]] (two chunks, each chunk has buffer_size examples)
        get_chunk_indices(zero_based_chunk) will fetch a list of examples from epoch_chunks, like [4, 1, 5, 2], as the local_indices for all GPUs to train (the current buffer)
        """
        self.unet.eval() #disable training for unet
        self.motion_reward_model.eval() #disable training for reward_model
        # chunk_id is 1-based in our quick calculation above
        # but our sampler's chunk indexing is 0-based
        zero_based_chunk = chunk_id - 1

        # Access the DataModule
        datamodule = self.trainer.datamodule
        sampler = datamodule.sampler

        # Retrieve all indices for this chunk
        chunk_indices = sampler.get_chunk_indices(zero_based_chunk)
        # Get local rank (GPU ID) and world size (total GPUs)
        local_rank = self.global_rank
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = torch.cuda.device_count()
        # print(f'len(chunk_indices): {len(chunk_indices)}, world_size: {world_size}, local_rank: {local_rank}')
        
        # Distribute indices across GPUs
        per_gpu_count = len(chunk_indices) // world_size
        start_idx = local_rank * per_gpu_count
        end_idx = start_idx + per_gpu_count if local_rank < world_size - 1 else len(chunk_indices)

        # Get this GPU's subset of indices
        local_indices = chunk_indices[start_idx:end_idx]
        
        # Define the batch size for processing
        processing_batch_size = self.config.batch_size
        
        # build (once) a persistent-worker loader that reads videos from datamodule.train_dataset
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
                    prefetch_factor=4,
                ))
            else:
                loader_kwargs.update(dict(num_workers=0))

            self._recompute_loader = DataLoader(datamodule.train_dataset, **loader_kwargs)

        # update indices for this chunk/rank
        self._recompute_sampler.set_indices(local_indices)

        # Initialize accumulators for each GPU's results
        self.accumulator = {}
        for var in self.buffer_variables:
            self.accumulator[var] = []
        # Process in batches to avoid memory issues
        for batch_id, batch_vid in enumerate(self._recompute_loader):
            batch_start = batch_id * processing_batch_size
            collected_data = self.collect_data(batch_vid, batch_start)
        
            # print(f"Rank {local_rank}: Processing batch {batch_id + 1}/{len(local_indices)//processing_batch_size} for chunk {chunk_id}")
            # print(f'Rank {local_rank}, len(local_indices): {len(local_indices)}, processing_batch_size: {processing_batch_size}')
            # move the collected data to the buffer device if buffer device is cpu
            for var in self.buffer_variables:
                data = collected_data[var]
                if isinstance(data, torch.Tensor) and self.buffer_device == 'cpu':
                    collected_data[var] = data.to(self.buffer_device, non_blocking=True).pin_memory()
            
            # Accumulate results
            for var in self.buffer_variables:
                self.accumulator[var].append(collected_data[var])
        
        # Store lists directly in buffer - no concatenation needed
        for var in self.buffer_variables:
            self.buffer[var] = self.accumulator[var]
        
        self.buffer_initialized = True
        # print(f"GPU {local_rank}: Buffer updated for chunk {chunk_id} with {len(chunk_indices)} videos")
        
        # Minimal synchronization to ensure all GPUs have finished
        torch.distributed.barrier()
        self.unet.train()

    def reshuffle_buffer(self):
        """
        Reshuffles the trajectories and corresponding data in the buffer
        to create new training batches while maintaining data alignment.
        """
        if not self.buffer_initialized:
            print("Buffer not initialized yet, nothing to reshuffle")
            return
        
        # Count the number of batches in the buffer
        num_batches = len(self.buffer['real_videos'])
        if num_batches <= 1:
            if self.global_rank == 0:
                print("Only one batch in buffer, no reshuffling needed")
            return
        
        # Create a shuffled permutation of batch indices
        # Use the same seed across all GPUs to ensure consistent shuffling
        seed = self.config.seed + self.global_step * 100 + self.current_pass * 10000
        
        # Set the RNG state for consistent shuffling across GPUs
        orig_rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        
        # Generate permutation indices
        perm_indices = torch.randperm(num_batches)
        
        # Restore original RNG state
        torch.random.set_rng_state(orig_rng_state)
        
        # Perform in-place shuffling for each variable
        for var in self.buffer_variables:
            if self.buffer[var] is not None:
                # Create a reference copy of the original list
                original = self.buffer[var].copy()  # This is a shallow copy of the list, not the tensors
                
                # Shuffle in-place
                for i, idx in enumerate(perm_indices):
                    self.buffer[var][i] = original[idx]
        
        # Force garbage collection to clean up references
        del original
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"GPU {self.global_rank}: Buffer reshuffled with {num_batches} batches (memory-efficient)")
        
        # Minimal synchronization to ensure all GPUs have finished
        torch.distributed.barrier()

    def generate_train_batch(self, batch, batch_idx):
        batch_idx_within_chunk = batch_idx % self.iterations_per_chunk

        real_videos = self.buffer['real_videos'][batch_idx_within_chunk].to(self.device, non_blocking=True)
        real_advantage = self.buffer['real_advantage'][batch_idx_within_chunk].to(self.device, non_blocking=True)
        real_ic_rewards = self.buffer['real_ic_rewards'][batch_idx_within_chunk].to(self.device, non_blocking=True)
        real_lc_rewards = self.buffer['real_lc_rewards'][batch_idx_within_chunk].to(self.device, non_blocking=True)
        real_vbench_rewards = self.buffer['real_vbench_rewards'][batch_idx_within_chunk].to(self.device, non_blocking=True)
        real_ave_rewards = self.buffer['real_ave_rewards'][batch_idx_within_chunk].to(self.device, non_blocking=True)
        fake_videos = self.buffer['fake_videos'][batch_idx_within_chunk].to(self.device, non_blocking=True)
        fake_advantage = self.buffer['fake_advantage'][batch_idx_within_chunk].to(self.device, non_blocking=True)
        fake_ic_rewards = self.buffer['fake_ic_rewards'][batch_idx_within_chunk].to(self.device, non_blocking=True)
        fake_lc_rewards = self.buffer['fake_lc_rewards'][batch_idx_within_chunk].to(self.device, non_blocking=True)
        fake_vbench_rewards = self.buffer['fake_vbench_rewards'][batch_idx_within_chunk].to(self.device, non_blocking=True)
        fake_ave_rewards = self.buffer['fake_ave_rewards'][batch_idx_within_chunk].to(self.device, non_blocking=True)

        return real_videos, fake_videos, fake_advantage, fake_ic_rewards, fake_lc_rewards, fake_vbench_rewards, fake_ave_rewards, real_advantage, real_ic_rewards, real_lc_rewards, real_vbench_rewards, real_ave_rewards

    def compute_advantage(self, rewards):
        """Compute group advantages
        Args:
            rewards: (p, b), where p is the number of sample videos per prompt, b is the batch size
        Return:
            advantages: (p, b)
        """
        mean = rewards.mean(dim=0, keepdim=True)
        std = rewards.std(dim=0, keepdim=True) + 1e-8
        if self.config.advantage_norm_with_std:
            advantages = (rewards - mean) / std
        else:
            advantages = rewards - mean
        # clip the advantages to [-adv_clip_max, adv_clip_max]
        unweighted_advantages = torch.clamp(advantages, -self.config.adv_clip_max, self.config.adv_clip_max)

        # apply the weight tricks from "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning"
        # multiply only the positive advantages by factor pos_adv_lambda, keep the negative advantages unchanged
        # pos_adv_lambda \in [0, 1], when it is 1., it is the normal GRPO
        # disable the pos_adv_lambda when asymmetric_grpo is True
        pos_adv_lambda = float(getattr(self.config, "pos_adv_lambda", 1.0))

        advantages = unweighted_advantages * pos_adv_lambda * (unweighted_advantages > 0) + unweighted_advantages * (unweighted_advantages <= 0)

        return advantages, unweighted_advantages

    def set_edm_ancestral_scheduler(self):
        self.original_scheduler = self.svd_pipeline.scheduler
        config = dict(self.original_scheduler.config)  # Convert FrozenDict to regular dict
        if 'clip_sample' in config:
            del config['clip_sample']
        if 'set_alpha_to_one' in config:
            del config['set_alpha_to_one']
        if 'skip_prk_steps' in config:
            del config['skip_prk_steps']
        
        ancestral_scheduler = EDMAncestralScheduler(**config)
        self.svd_pipeline.register_modules(scheduler = ancestral_scheduler)
        print('Set the SVD scheduler to be EDM_Ancestral Scheduler')

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