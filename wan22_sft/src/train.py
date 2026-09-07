import torch
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from utils import VideoDataModule
from utils import load_config, ConfigViewerCallback, ManualLoRACallback, ManualModelCheckpoint
from sft_wan_trainer import SFTTrainerWAN
from pytorch_lightning.strategies import DDPStrategy
from deepspeed_strategy import create_deepspeed_strategy
import os
import yaml

try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError:
    rank_zero_only = None

# Add environment setup function
def setup_memory_environment():
    """Setup environment variables for memory optimization"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
    os.environ['FLASH_ATTENTION_FORCE_CUT_SIZE'] = '1'
    os.environ["TOKENIZERS_PARALLELISM"]="false"

# NEW: helper to know if we're on global rank 0
def is_rank_zero() -> bool:
    # Prefer Lightning's notion of rank if available
    if rank_zero_only is not None and hasattr(rank_zero_only, "rank"):
        return rank_zero_only.rank == 0
    # Fallback to torch.distributed / env vars
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # Single-process case
    return True

def main():
    setup_memory_environment()

    # Parse command line arguments
    config = load_config()

    # Add DeepSpeed option check (add this to your config)
    use_deepspeed = getattr(config, 'use_deepspeed', False)
    is_main = is_rank_zero()

    if config.resume_from_checkpoint is not None:
        config.resume_from_checkpoint = Path(config.resume_from_checkpoint)
        assert config.resume_from_checkpoint.exists()
    
    if config.use_tf32:
        # Enable TF32 globally
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Initializing logger")
    logger_type = getattr(config, "report_to", "wandb")
    print(f"Using logger: {logger_type}")

    if logger_type == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir=config.save_dir,
            name=config.project_name,  # reuse project folder name for consistency
            version=config.exp_name,           # run subfolder
        )
    else:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            project=config.project_name,
            name=config.exp_name,  # used for the checkpoint folder
            save_dir=config.save_dir,
            dir=config.save_dir,
            offline=False,
            config=config,
        )
    # Define explicit checkpoint directory with the run name
    checkpoint_dir = Path(config.save_dir) / config.project_name / config.exp_name
    # Create separate directories for different checkpoint types
    lora_dir = checkpoint_dir / "lora"
    full_dir = checkpoint_dir / "full"
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        lora_dir.mkdir(exist_ok=True)
        full_dir.mkdir(exist_ok=True)
        print(f"Saving checkpoints to: {checkpoint_dir}")
        # save the config into checkpoint_dir
            # Namespace -> dict
        cfg_dict = vars(config)
        cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg_dict.items()}
        with open(Path(checkpoint_dir).joinpath('config.yaml'), 'w') as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=False)

    # Best checkpoint based on val_loss with epoch in filename
    best_loss_callback = ManualModelCheckpoint(
        dirpath=str(checkpoint_dir),  # Explicitly set the checkpoint directory
        save_last=False,          # Don't save last in this callback
        save_top_k=4,             # Save only the best model
        every_n_epochs=1,
        filename="best_{val_loss:.4f}_{epoch}",  # Include epoch and loss
        monitor="val_loss",     # Monitor val_loss
        mode="min",               # Lower loss is better
        verbose=True,             # Print when a better model is saved
    )

    periodic_full_ckpt_callback = ManualModelCheckpoint(
        dirpath=str(full_dir),
        filename="last",
        save_top_k=-1,
        every_n_epochs=1,  
        save_on_train_epoch_end=True,
        monitor=None,
        verbose=True,
    )

    # LoRA checkpoints for evaluation (every epoch, small)
    lora_callback = ManualLoRACallback(
        dirpath=str(lora_dir),
        every_n_epochs=config.lora_checkpoint_every_n_epochs,
        filename_template="epoch_{epoch:03d}_step_{step}",
    )

    pl.seed_everything(config.seed)
    data_module = VideoDataModule(config.batch_size,
                                          train_video_path_json=config.train_video_path_json,
                                          val_video_path_json=config.val_video_path_json,
                                          videos_root_dir=config.videos_root_dir,
                                          target_vid_size=config.target_vid_size,
                                          vid_data_type=config.vid_data_type,
                                          num_workers=config.num_workers)

    config.iterations_per_epoch = len(data_module.train_dataloader())
    sft_wan_trainer = SFTTrainerWAN(config)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    grad_clip_kwargs = {}
    grad_clip_kwargs.update(dict(
        gradient_clip_val=config.gradient_clip,
        gradient_clip_algorithm="norm",
    ))
    callbacks=[
        best_loss_callback,         # Track best by reward values
        lora_callback,
        lr_monitor, 
        ConfigViewerCallback(),
        periodic_full_ckpt_callback,
        ]
    if use_deepspeed:
        strategy = create_deepspeed_strategy(config)
    else:
        strategy = DDPStrategy(process_group_backend="nccl")
    trainer = pl.Trainer(
        limit_train_batches=config.debug_train_num_batches if config.debug_mode else None, # for debug
        limit_val_batches=config.debug_train_num_batches if config.debug_mode else None,
        logger=logger,
        devices=config.gpus_per_node,
        num_nodes=config.num_nodes,
        accelerator="gpu",
        strategy=strategy,
        callbacks=callbacks,
        deterministic=False,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        val_check_interval=config.val_check_interval,
        accumulate_grad_batches=config.accum_grad_steps,
        max_epochs=config.max_epochs,
        precision=config.precision,
        log_every_n_steps=1,
        **grad_clip_kwargs,
    )
    print(f"val_check_interval: {trainer.val_check_interval}")
    print(f"accumulate_grad_batches: {trainer.accumulate_grad_batches}")
    print(f"limit_val_batches: {trainer.limit_val_batches}")
    print(f"check_val_every_n_epoch: {trainer.check_val_every_n_epoch}")
    trainer.fit(sft_wan_trainer, 
                data_module,
                ckpt_path=config.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()
