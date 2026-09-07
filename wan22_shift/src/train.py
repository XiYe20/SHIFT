import os
import yaml
import torch
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from utils import (
    BufferVideoDataModule,
    load_config,
    ConfigViewerCallback,
    ManualLoRACallback,
    ManualModelCheckpoint,
)
from shift_wan_trainer import ShiftTrainerWAN
from deepspeed_strategy import create_deepspeed_strategy

try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError:
    rank_zero_only = None


def setup_memory_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
    os.environ["FLASH_ATTENTION_FORCE_CUT_SIZE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def is_rank_zero() -> bool:
    if rank_zero_only is not None and hasattr(rank_zero_only, "rank"):
        return rank_zero_only.rank == 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def main():
    setup_memory_environment()
    config = load_config()

    use_deepspeed = getattr(config, "use_deepspeed", False)
    is_main = is_rank_zero()

    if config.resume_from_checkpoint is not None:
        config.resume_from_checkpoint = Path(config.resume_from_checkpoint)
        assert config.resume_from_checkpoint.exists()

    if config.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # logger
    logger_type = getattr(config, "report_to", "wandb")
    if logger_type == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=config.save_dir, name=config.project_name, version=config.exp_name)
    else:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=config.project_name, name=config.exp_name, save_dir=config.save_dir, dir=config.save_dir, offline=False, config=config)

    checkpoint_dir = Path(config.save_dir) / config.project_name / config.exp_name
    lora_dir = checkpoint_dir / "lora"
    full_dir = checkpoint_dir / "full"
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        lora_dir.mkdir(exist_ok=True)
        full_dir.mkdir(exist_ok=True)
        cfg_dict = vars(config)
        cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg_dict.items()}
        with open(checkpoint_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=False)

    # monitor reward on val (max)
    best_callback = ManualModelCheckpoint(
        dirpath=str(checkpoint_dir),
        save_last=False,
        save_top_k=4,
        every_n_epochs=1,
        filename="best_{val_ave_rewards:.4f}_{epoch}",
        monitor="val_fake_ave_rewards",
        mode="max",
        verbose=True,
    )

    periodic_full = ManualModelCheckpoint(
        dirpath=str(full_dir),
        filename="last",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        monitor=None,
        verbose=True,
    )

    lora_callback = ManualLoRACallback(
        dirpath=str(lora_dir),
        every_n_epochs=config.lora_checkpoint_every_n_epochs,
        filename_template="epoch_{epoch:03d}_step_{step}",
    )

    pl.seed_everything(config.seed)

    data_module = BufferVideoDataModule(
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        train_video_path_json=config.train_video_path_json,
        val_video_path_json=config.val_video_path_json,
        videos_root_dir=config.videos_root_dir,
        target_vid_size=tuple(config.target_vid_size),
        vid_data_type=config.vid_data_type,
        num_workers=config.num_workers,
        passes_per_epoch=config.passes_per_buffer,
    )

    config.iterations_per_epoch = len(data_module.train_dataloader())

    model = ShiftTrainerWAN(config)

    callbacks = [
        best_callback,
        lora_callback,
        LearningRateMonitor(logging_interval="step"),
        ConfigViewerCallback(),
        periodic_full,
    ]

    if use_deepspeed:
        strategy = create_deepspeed_strategy(config)
    else:
        strategy = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)

    trainer = pl.Trainer(
        limit_train_batches=config.limit_train_batches if getattr(config, "debug_mode", False) else None,
        limit_val_batches=config.limit_val_batches if getattr(config, "debug_mode", False) else None,
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
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module, ckpt_path=config.resume_from_checkpoint)


if __name__ == "__main__":
    main()