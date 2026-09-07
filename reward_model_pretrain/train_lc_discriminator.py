import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from typing import Dict, Optional, Tuple

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load as safetensor_load
from torch.utils.data import random_split

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import cast_training_params
from diffusers.utils.torch_utils import is_compiled_module

import transformers
import pdb
from cotracker_core.cotracker_discriminator import TrajectoryDiscriminator

from accelerate import DistributedDataParallelKwargs
from utils import DiscriminatorDataset
# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

logger = get_logger(__name__, log_level="INFO")

import os
import torch
from torch.utils.data import Dataset

import sys
import os
sys.path.append(os.path.dirname('./cotracker_core/cotracker_source/__init__.py'))


def main(
    name: str,
    
    output_dir: str,
    lc_loss_config: Dict = None,
    report_to: str = "wandb",
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = -1,
    validation_epochs: int = 1,

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 16,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision: str = 'fp16',

    global_seed: int = 42,
    is_debug: bool = False,
):

    train_config = locals().copy()
    check_min_version("0.10.0.dev0")

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    logging_dir = os.path.join(output_dir, "logs")

    #init accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    
    print('gradient_accumulation_steps', gradient_accumulation_steps)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )
    set_seed(global_seed)
    # NEW: ensure tensorboard (or wandb) trackers are initialized
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=folder_name,  # or any string you like
        )
    
    if accelerator.is_main_process and is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_main_process and (not is_debug) and report_to == "wandb":
        wandb.init(project="MotionAlignmentVDM", name=folder_name, config=config)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(OmegaConf.create(train_config), os.path.join(output_dir, 'train_config.yaml'))

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    

    traj_discriminator = TrajectoryDiscriminator(**lc_loss_config.discriminator_config)
    traj_discriminator.to(accelerator.device, dtype=weight_dtype)
    #print the number of parameters in Millions
    if accelerator.is_main_process:
        print('number of parameters: ', sum(p.numel() for p in traj_discriminator.parameters() if p.requires_grad)/1e6, 'M')


    # Add adapter and make sure the trainable params are in float32.
    if mixed_precision == "fp16":
        if lc_loss_config is not None:
            cast_training_params(traj_discriminator.track_former, dtype=torch.float32)
    
    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)

    if lc_loss_config is not None:
        discriminator_optimizer = torch.optim.AdamW(
            traj_discriminator.track_former.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

    # Get the training dataset
    train_dataset = DiscriminatorDataset(real_video_dir=lc_loss_config.real_video_dir, fake_video_dir=lc_loss_config.fake_video_dir)

    # Split dataset into train and validation (90/10)
    dataset_size = len(train_dataset)
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(global_seed))
    print('len(train_dataset)', len(train_dataset), 'len(val_dataset)', len(val_dataset))
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # Get the training iteration
    if max_train_steps == -1:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / gradient_accumulation_steps)

        assert max_train_epoch != -1
        num_training_steps_for_scheduler = max_train_epoch * num_update_steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = max_train_steps * accelerator.num_processes
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)
    
    if validation_steps == -1:
        assert validation_epochs != -1
        validation_steps = validation_epochs * len(train_dataloader)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=discriminator_optimizer,
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
    )
    # Prepare everything with our `accelerator`.
    train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        train_dataloader, val_dataloader, lr_scheduler
    )
    if lc_loss_config is not None:
        traj_discriminator, discriminator_optimizer = accelerator.prepare(traj_discriminator, discriminator_optimizer)

    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps == -1:
        max_train_steps = max_train_epoch * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num validation examples = {len(val_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_total_loss = 0.
        for step, batch in enumerate(train_dataloader):
            ### >>>> Training >>>> ###
            with accelerator.accumulate(traj_discriminator):
                # pretrain the discriminator
                real_videos = batch["real_video"].to(dtype=weight_dtype)
                generated_videos = batch["fake_video"].to(dtype=weight_dtype)
                # -------------------------
                # Discriminator pretrain
                # -------------------------
                # For the discriminator, we use hinge loss:
                #   - For fake samples: loss_fake = ReLU(1 + D(fake))
                #   - For real samples: loss_real = ReLU(1 - D(real))
                # -------------------------
                # enable the discriminator gradient
                
                # Both loss_D_fake and loss_D_real ideally converges to 0.
                #change to normal BCE loss
                fake_logits_disc, queries_point = traj_discriminator(generated_videos.detach())
                loss_D_fake = F.binary_cross_entropy_with_logits(
                    fake_logits_disc,
                    torch.zeros_like(fake_logits_disc)
                )

                # use the same queries point for real videos
                real_logits_disc, _ = traj_discriminator(real_videos, queries_point=queries_point)
                loss_D_real = F.binary_cross_entropy_with_logits(
                    real_logits_disc,
                    torch.ones_like(real_logits_disc)
                )
                d_loss = loss_D_real + loss_D_fake

                accelerator.backward(d_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(traj_discriminator.parameters(), max_grad_norm)
                    discriminator_optimizer.step()
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                
                loss = d_loss

            ### <<<< Training <<<< ###

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                log_info = {"train_total_loss": train_total_loss}
                if lc_loss_config is not None:
                    log_info["lc_loss_d"] = d_loss.item()
                    log_info["lc_loss_d_real"] = loss_D_real.item()
                    log_info["lc_loss_d_fake"] = loss_D_fake.item()

                accelerator.log(log_info, step=global_step)
                if accelerator.is_main_process and (not is_debug) and report_to == "wandb":
                    wandb.log(log_info, step=global_step)
                
                train_total_loss = 0.

                # Save checkpoint
                if accelerator.is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                    # Save the discriminator checkpoint
                    if lc_loss_config is not None:
                        checkpoint_dict = {
                            'global_step': global_step,
                            'discriminator_state_dict': unwrap_model(traj_discriminator).track_former.state_dict(),
                            'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'latest_losses': {
                                'discriminator_loss': d_loss.item(),
                                'discriminator_real_loss': loss_D_real.item(),
                                'discriminator_fake_loss': loss_D_fake.item(),
                            }
                        }
                    
                        weight_name = f"lc_discriminator-{global_step}.ckpt"
                        save_directory = os.path.join(output_dir, 'lc_discriminator')
                        os.makedirs(save_directory, exist_ok=True)
                        torch.save(checkpoint_dict, os.path.join(save_directory, weight_name))
                        logger.info(f"Saved Discriminator checkpoint to {os.path.join(save_directory, weight_name)} (global_step: {global_step})")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
                
            # Periodically validation (distributed)
            if global_step > 0 and global_step % validation_steps == 0:
                traj_discriminator.eval()

                # local accumulators on each rank
                val_loss_sum = torch.tensor(0.0, device=accelerator.device)
                val_correct = torch.tensor(0, device=accelerator.device, dtype=torch.long)
                val_total = torch.tensor(0, device=accelerator.device, dtype=torch.long)

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        real_videos = val_batch["real_video"].to(dtype=weight_dtype)
                        generated_videos = val_batch["fake_video"].to(dtype=weight_dtype)

                        video = torch.cat([generated_videos, real_videos], dim=0)
                        logits_disc, _ = traj_discriminator(video)

                        n_fake = generated_videos.size(0)
                        fake_logits_disc = logits_disc[:n_fake]
                        real_logits_disc = logits_disc[n_fake:]

                        loss_D_fake = F.binary_cross_entropy_with_logits(
                            fake_logits_disc, torch.zeros_like(fake_logits_disc)
                        )
                        loss_D_real = F.binary_cross_entropy_with_logits(
                            real_logits_disc, torch.ones_like(real_logits_disc)
                        )
                        val_d_loss = loss_D_real + loss_D_fake

                        # weight by number of elements so global mean is correct
                        n = torch.tensor(fake_logits_disc.numel() + real_logits_disc.numel(),
                                        device=accelerator.device, dtype=torch.long)
                        val_loss_sum += val_d_loss.detach() * n

                        fake_preds = (torch.sigmoid(fake_logits_disc) > 0.5).long()
                        real_preds = (torch.sigmoid(real_logits_disc) > 0.5).long()
                        val_correct += (fake_preds == 0).sum() + (real_preds == 1).sum()
                        val_total += n

                # reduce across ranks
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_correct, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_total, op=dist.ReduceOp.SUM)

                val_loss_d = (val_loss_sum / val_total.clamp_min(1)).item()
                val_accuracy = (val_correct.float() / val_total.clamp_min(1).float()).item()

                if accelerator.is_main_process:
                    valid_metrics = {"val_loss_d": val_loss_d, "val_accuracy": val_accuracy}
                    accelerator.log(valid_metrics, step=global_step)
                    if (not is_debug) and report_to == "wandb":
                        wandb.log(valid_metrics, step=global_step)
                    logger.info(f"Validation at step {global_step}: loss_d={val_loss_d:.4f}, accuracy={val_accuracy:.4f}")

                accelerator.wait_for_everyone()
                traj_discriminator.train()
    
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--report_to",  type=str, choices=["wandb", "tensorboard"], default="wandb")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, report_to=args.report_to, **config)