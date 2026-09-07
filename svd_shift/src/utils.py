import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Sampler
import json
import torch
import os
import imageio.v3 as iio
import numpy as np
from torchvision import transforms
import glob
import random
from collections import defaultdict
import cv2
import math
import argparse
import yaml
from diffusers import StableVideoDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
from diffusers.loaders import PeftAdapterMixin

from einops import rearrange
from diffusers.utils import BaseOutput
from typing import Callable, Dict, List, Optional, Union
import PIL
from chunked_sampler import ChunkedSampler

import re

def read_video(vid_path, target_size, vid_data_type):
    """Read video file into a pytorch tensor, with shape (T, C, H, W), pixel range [0, 1]
    Args:
        vid_path: path to the video file
        target_size: the target size of the video, tuple of (T, H, W)
    Returns:
        video_tensor: pytorch tensor of shape (T, C, H, W), pixel range [0, 1]
        init_frame: PIL image of the first frame
    """
    if vid_data_type == 'mp4':
        # Read video file using imageio
        # video = iio.imread(vid_path, index=None)  # Read all frames
        # Convert to numpy array if not already
        # video_np = np.array(video)
        # Read video file using OpenCV instead of imageio
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {vid_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB to match imageio output
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {vid_path}")
        
        video_np = np.array(frames)
        # Handle different video formats
        assert video_np.ndim == 4, "Video must have 4 dimensions (T, H, W, C)"

        # Convert from (T, H, W, C) to (T, C, H, W) format
        video_np = np.transpose(video_np, (0, 3, 1, 2))
        
        # Convert to PyTorch tensor
        video_tensor = torch.from_numpy(video_np).float()
        
        # Normalize to [0, 1]
        video_tensor = video_tensor / 255.0

    elif vid_data_type == 'pt':
        data = torch.load(vid_path)
        video_tensor = data['real_video'] #（T, C, H, W)
        assert video_tensor.min() >= 0 and video_tensor.max() <= 1, "Video tensor must be normalized to [0, 1]"

    # Take only the first target_size[0] frames
    assert video_tensor.shape[0] >= target_size[0], "Video must have at least target_size[0] frames"
    video_tensor = video_tensor[:target_size[0]]

    # Resize and center crop to target height and width
    target_h, target_w = target_size[1], target_size[2]
    
    # if the spatio ratio of target_h/target_w does not match the original spatio ratio of the video, then do center crop to the target size
    if video_tensor.shape[2] / float(video_tensor.shape[3]) != target_h / float(target_w):
        # Create transform pipeline for resizing and cropping
        scale_h, scale_w = float(target_h) / video_tensor.shape[2], float(target_w) / video_tensor.shape[3]
        scale_factor = max(scale_h, scale_w)
        # Calculate intermediate size
        intermediate_h = int(video_tensor.shape[2] * scale_factor)
        intermediate_w = int(video_tensor.shape[3] * scale_factor)
        # Create transform pipeline
        resize_transform = transforms.Compose([
            transforms.Resize((intermediate_h, intermediate_w)),
            transforms.CenterCrop((target_h, target_w))
        ])
    else:
        resize_transform = transforms.Resize((target_h, target_w))
    
    # Apply transforms in batch over all frames (T, C, H, W)
    video_tensor = resize_transform(video_tensor)
    init_frame = video_tensor[0, ...]

    return video_tensor, init_frame

class VideoDataset(Dataset):
    def __init__(self, videos_root_dir='davis_2017/real_videos_seq', video_path_json="refl_videos.json", target_vid_size=(24, 576, 1024), vid_data_type='mp4'):
        """
        Args:
            video_path_json: path to the json file containing the video paths
            target_vid_size: the target size of the video, tuple of (num_frames, height, width)
        """
        with open(video_path_json, "r") as f:
            data = json.load(f)
        # prepend the videos_root_dir
        data = {k: os.path.join(videos_root_dir, v) for k, v in data.items()}
        self.data = data
        self.vid_ids = list(data.keys())
        self.target_vid_size = target_vid_size
        self.vid_data_type = vid_data_type

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        """Return the reference real video for on-policy sampling
        Return:
            video_tensor: pytorch tensor of shape (T, C, H, W), pixel range [0, 1]
            init_frame: PIL image of the first frame
        """
        vid_path = self.data[self.vid_ids[idx]]
        video_tensor, init_frame = read_video(vid_path, self.target_vid_size, self.vid_data_type)
        
        # convert the pixel value from range [0, 1] to [-1, 1]
        video_tensor = video_tensor * 2 - 1.

        return video_tensor

class PaddedDataset(Dataset):
    def __init__(self, base_ds, target_len):
        self.base = base_ds
        self.base_len = len(base_ds)
        self.target_len = target_len
    def __len__(self):
        return self.target_len
    def __getitem__(self, idx):
        return self.base[idx % self.base_len]

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=3, videos_root_dir=None, train_video_path_json="refl_videos.json", val_video_path_json="refl_videos.json", target_vid_size=(24, 576, 1024), vid_data_type='mp4'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.videos_root_dir = videos_root_dir
        self.train_video_path_json = train_video_path_json
        self.val_video_path_json = val_video_path_json
        self.target_vid_size = target_vid_size
        self.vid_data_type = vid_data_type
        self.setup()

    def setup(self, stage=None):
        self.train_dataset = VideoDataset(videos_root_dir=self.videos_root_dir, video_path_json=self.train_video_path_json, target_vid_size=self.target_vid_size, vid_data_type=self.vid_data_type)
        self.val_dataset = VideoDataset(videos_root_dir=self.videos_root_dir, video_path_json=self.val_video_path_json, target_vid_size=self.target_vid_size, vid_data_type=self.vid_data_type)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=3,
        )

class IndexOnlyDataset(Dataset):
    """Lightweight dataset that returns indices; avoids decoding videos in the main train loop."""
    def __init__(self, base_dataset):
        self.base = base_dataset
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        return idx

class BufferVideoDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size, 
        buffer_size, 
        num_workers=3, 
        videos_root_dir=None, 
        train_video_path_json="refl_videos.json", 
        val_video_path_json="refl_videos.json", 
        target_vid_size=(24, 576, 1024), 
        vid_data_type='mp4',
        passes_per_epoch: int = 1
    ):
        super().__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.videos_root_dir = videos_root_dir
        self.train_video_path_json = train_video_path_json
        self.val_video_path_json = val_video_path_json
        self.target_vid_size = target_vid_size
        self.vid_data_type = vid_data_type
        self.sampler = None
        self.passes_per_epoch = passes_per_epoch
        self.setup()

    def setup(self, stage=None):
        """
        Called by Lightning before training (or testing).
        """
        self.train_dataset = VideoDataset(videos_root_dir=self.videos_root_dir, video_path_json=self.train_video_path_json, target_vid_size=self.target_vid_size, vid_data_type=self.vid_data_type)
        if len(self.train_dataset) % self.buffer_size != 0:
            # pad the train_dataset to be the largest multiple of buffer_size by split
            padded_len = math.ceil(len(self.train_dataset) / self.buffer_size) * self.buffer_size
            self.train_dataset = PaddedDataset(self.train_dataset, padded_len)
        self.val_dataset = VideoDataset(videos_root_dir=self.videos_root_dir, video_path_json=self.val_video_path_json, target_vid_size=self.target_vid_size, vid_data_type=self.vid_data_type)

    def train_dataloader(self):
        # index-only dataset for the main training loop
        self.train_index_dataset = IndexOnlyDataset(self.train_dataset)

        self.sampler = ChunkedSampler(self.train_index_dataset, chunk_size=self.buffer_size, shuffle=True, passes_per_epoch=self.passes_per_epoch)
        dataloader = DataLoader(
            self.train_index_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=0, #because here it just generates indices
            drop_last=True  # ensures consistent batch size if you want
        )
        return dataloader
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=3,
        )

class ListSampler(Sampler):
    """Sampler whose index list can be updated between iterations (works with persistent workers)."""
    def __init__(self, indices=None):
        self.indices = list(indices) if indices is not None else []
    def set_indices(self, indices):
        self.indices = list(indices)
    def __iter__(self):
        yield from self.indices
    def __len__(self):
        return len(self.indices)

def attach_peft_mixin(model):
    """
    Make an *already-instantiated* UNet gain the PEFT/LoRA API
    (add_adapter, set_adapter, enable_adapters, …) without touching
    the original source code.

    Returns the same object, now with the extra methods.
    """
    # 1.  Do nothing if it already has the mixin
    if isinstance(model, PeftAdapterMixin):
        return model

    # 2.  Create a new class that merges the current class + mixin
    NewCls = type(
        f"PeftWrapped{model.__class__.__name__}",
        (PeftAdapterMixin, model.__class__),   # MRO: mixin first
        {}
    )

    # 3.  Point the instance to the new class
    model.__class__ = NewCls

    # 4.  Run the mixin’s __init__ (it just sets a few attributes)
    PeftAdapterMixin.__init__(model)

    return model

def clear_peft_merged_state(model):
    """Clear stale merged-adapter tracking left after fuse_lora + delete_adapters."""
    for module in model.modules():
        if hasattr(module, "merged_adapters"):
            module.merged_adapters.clear()

def get_model(
    model_name,
    use_compile=True,
    bfloat_dtype=True,
    use_for_training=True,
    local_pretrained_model=None
):
    """
    Get the SVD model based on the model name.
    """
    torch_dtype = torch.bfloat16 if bfloat_dtype else torch.float32 #torch.float16

    if model_name == "stable-video-diffusion":
        if local_pretrained_model is None:
            pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch_dtype
                    )
        else:
            print("Load from local pretrained model")
            pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "/path/to/models--stabilityai--stable-video-diffusion-img2vid-xt/snapshots/9e43909513c6714f1bc78bcb44d96e733cd242aa", torch_dtype=torch_dtype
                )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # add the peft functions to the svd unet
    pipeline.unet = attach_peft_mixin(pipeline.unet)
    
    if use_compile:
        print("Run torch compile")
        pipeline.unet.to(memory_format=torch.channels_last)
        pipeline.vae.to(memory_format=torch.channels_last)

        pipeline.unet = torch.compile(
            pipeline.unet, mode="reduce-overhead", fullgraph=True
        )
        pipeline.vae.decode = torch.compile(
            pipeline.vae.decode, mode="reduce-overhead", fullgraph=True
        )

    if use_for_training:
        pipeline.vae.requires_grad_(False)
        pipeline.unet.requires_grad_(False)

    return pipeline

import torch
import pytorch_lightning as pl
from pathlib import Path
import wandb

class ConfigViewerCallback(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        if not getattr(trainer, "logger", None):
            return
        config = pl_module.config
        if hasattr(config, "config_path"):
            try:
                import wandb
                artifact = wandb.Artifact("config-artifact", type="config")
                artifact.add_file(config.config_path)
                # Only log artifact if the experiment supports it (W&B)
                if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "log_artifact"):
                    trainer.logger.experiment.log_artifact(artifact)
            except Exception:
                # No wandb installed or not a W&B logger; skip artifact logging
                pass

class ManualLoRACallback(pl.Callback):
    """Custom callback that manually saves LoRA weights only"""
    
    def __init__(self, dirpath, every_n_epochs=1, filename_template="lora_epoch_{epoch:03d}"):
        self.dirpath = Path(dirpath)
        self.every_n_epochs = every_n_epochs
        self.filename_template = filename_template
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        # Only let rank 0 / global zero save
        if not getattr(trainer, "is_global_zero", True):
            return
            
        """Save LoRA checkpoint every N epochs"""
        current_epoch = trainer.current_epoch
        
        # Only save every N epochs
        if current_epoch % self.every_n_epochs != 0:
            return
            
        # Create checkpoint data using the model's method
        checkpoint_data = {}
        pl_module.on_save_checkpoint(checkpoint_data)
        
        # Add training metadata  
        checkpoint_data.update({
            'epoch': current_epoch,
            'global_step': trainer.global_step,
            'pytorch-lightning_version': pl.__version__,
            'state_dict': checkpoint_data.get('state_dict', {}),  # LoRA weights from on_save_checkpoint
        })
        if 'discriminator_ema_state' in checkpoint_data:
            del checkpoint_data['discriminator_ema_state']
        
        # Save LoRA checkpoint as .ckpt file
        filename = self.filename_template.format(
            epoch=current_epoch,
            step=trainer.global_step
        ) + ".ckpt"
        
        filepath = self.dirpath / filename
        
        # Manual save using torch.save (bypasses DeepSpeed)
        torch.save(checkpoint_data, filepath)
        print(f"✅ Saved LoRA-only checkpoint: {filepath} ({filepath.stat().st_size / (1024*1024):.1f} MB)")

from pytorch_lightning.callbacks import ModelCheckpoint
class ManualModelCheckpoint(ModelCheckpoint):
    pass

def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
    
# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

def load_config(config_path=None, override_args=None):
    """
    Load configuration from YAML file with optional command-line overrides.
    
    Args:
        config_path: Path to YAML config file
        override_args: Command line arguments that override YAML settings
    
    Returns:
        config: Namespace containing all configuration values
    """
    parser = argparse.ArgumentParser(description="Adjoint Matching for Stable Diffusion")
    parser.add_argument("--config", type=str, default=config_path, help="Path to config YAML file")
    
    # Allow overriding specific config values from command line
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--precision", type=str, help="Training precision (32, 16, bf16)")
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--save_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--verbose", type=str, help="Enable verbose logging")
    parser.add_argument("--videos_root_dir", type=str, help="Override videos_root_dir")

    
    args = parser.parse_args(override_args)
    
    # If config path provided through command line, use that
    config_path = args.config if args.config else config_path
    
    if not config_path:
        raise ValueError("No config file specified. Use --config or provide config_path.")
    
    # Load YAML file
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert dict to Namespace
    config = argparse.Namespace(**config_dict)
    
    # Override with any command line arguments that were explicitly provided
    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            setattr(config, key, value)
    
    # Ensure paths are Path objects
    if hasattr(config, 'resume_from_checkpoint') and config.resume_from_checkpoint:
        config.resume_from_checkpoint = Path(config.resume_from_checkpoint)
    
    # Store the config file path for reference
    config.config_path = config_path
    
    return config
    

def create_video_json(directory, output_file="video_files.json", recursive=False):
    """
    List all MP4 video files in a directory and save their absolute paths to a JSON file.
    
    Args:
        directory (str): Path to the directory to scan for MP4 files
        output_file (str): Path to the output JSON file (default: "video_files.json")
        recursive (bool): Whether to search subdirectories recursively (default: False)
    
    Returns:
        dict: Dictionary mapping string indices to absolute file paths
    """
    # Check if directory exists
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory: {directory}")
    
    # Get absolute path of the directory
    abs_directory = os.path.abspath(directory)
    
    # Find all MP4 files in the directory
    if recursive:
        mp4_pattern = os.path.join(abs_directory, "**", "*.mp4")
        mp4_files = glob.glob(mp4_pattern, recursive=True)
    else:
        mp4_pattern = os.path.join(abs_directory, "*.mp4")
        mp4_files = glob.glob(mp4_pattern)
    
    # Sort files for consistent ordering
    mp4_files.sort()
    
    # Create dictionary with string indices as keys
    video_dict = {}
    for i, file_path in enumerate(mp4_files):
        video_dict[str(i)] = os.path.abspath(file_path)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(video_dict, f, indent=4)
    
    print(f"Found {len(mp4_files)} MP4 files in {abs_directory}")
    if recursive:
        print("(searched recursively)")
    print(f"Saved to {output_file}")
    
    return video_dict

def split_videos_by_object(input_json_path, validation_ratio=0.2, seed=42, output_dir=None):
    """
    Split videos by object into training and validation sets.
    
    Args:
        input_json_path (str): Path to the input JSON file containing video paths
        validation_ratio (float): Ratio of videos to use for validation (0.0 to 1.0)
        seed (int): Random seed for reproducible splits
        output_dir (str): Directory to save output files (default: same as input)
    
    Returns:
        tuple: (train_dict, val_dict) - dictionaries for training and validation sets
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read the input JSON file
    with open(input_json_path, 'r') as f:
        video_dict = json.load(f)
    
    # Group videos by object name
    object_videos = defaultdict(list)
    
    for key, video_path in video_dict.items():
        # Extract object name from filename
        filename = os.path.basename(video_path)
        # Remove file extension and sequence number to get object name
        # Example: "bear_s00000.mp4" -> "bear"
        object_name = filename.split('_s')[0] if '_s' in filename else filename.split('.')[0]
        object_videos[object_name].append((key, video_path))
    
    print(f"Found {len(object_videos)} different objects:")
    for obj_name, videos in object_videos.items():
        print(f"  {obj_name}: {len(videos)} videos")
    
    # Split each object's videos into train/val
    train_videos = {}
    val_videos = {}
    train_counter = 0
    val_counter = 0
    
    for object_name, videos in object_videos.items():
        # Shuffle videos for this object
        random.shuffle(videos)
        
        # Calculate split point
        num_videos = len(videos)
        num_val = max(1, int(num_videos * validation_ratio))  # At least 1 video for validation
        num_train = num_videos - num_val
        
        print(f"\n{object_name}: {num_train} train, {num_val} validation")
        
        # Split videos
        train_object_videos = videos[:num_train]
        val_object_videos = videos[num_train:]
        
        # Add to train set with new indices
        for original_key, video_path in train_object_videos:
            train_videos[str(train_counter)] = video_path
            train_counter += 1
        
        # Add to validation set with new indices
        for original_key, video_path in val_object_videos:
            val_videos[str(val_counter)] = video_path
            val_counter += 1
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_json_path)
    
    # Save training set
    train_output_path = os.path.join(output_dir, "train_videos.json")
    with open(train_output_path, 'w') as f:
        json.dump(train_videos, f, indent=4)
    
    # Save validation set
    val_output_path = os.path.join(output_dir, "val_videos.json")
    with open(val_output_path, 'w') as f:
        json.dump(val_videos, f, indent=4)
    
    print(f"\nSaved {len(train_videos)} training videos to {train_output_path}")
    print(f"Saved {len(val_videos)} validation videos to {val_output_path}")
    
    return train_videos, val_videos


def group_param_norms_by_transformer_block(
    param_norms: Dict[str, Union[torch.Tensor, float]],
    prefix_regex: Optional[str] = r"^(.*)\.temporal_transformer_blocks\.\d+\.",
):
    """
    Args:
        param_norms: dict[name -> norm] for each parameter (norm can be float or 0-d/1-d tensor).
        prefix_regex: a regex string with ONE capture group for the "layer key".
            Default groups by prefix before '.temporal_transformer_blocks.<idx>.'

            Example regex:
              r"^(.*)\.temporal_transformer_blocks\.\d+\."
            Example name:
              "up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn1.to_v.lora_A...."
            Captured layer:
              "up_blocks.3.attentions.0"

    Returns:
        dict[layer_key -> 1D CPU float tensor of concatenated norms for that layer]
    """
    if prefix_regex is None or prefix_regex == "":
        raise ValueError("prefix_regex must be a non-empty regex string")

    cre = re.compile(prefix_regex)

    groups = defaultdict(list)

    for name, v in param_norms.items():
        m = cre.match(name)
        if m is None:
            continue
        layer = m.group(1)

        if isinstance(v, torch.Tensor):
            t = v.detach().flatten()
        else:
            t = torch.tensor([float(v)])

        groups[layer].append(t.float().cpu())

    out = {}
    for layer, chunks in groups.items():
        out[layer] = torch.cat(chunks, dim=0) if len(chunks) > 0 else torch.empty(0)
    return out

def sft_grad_norm_log_from_grads(grad_snap: dict, prefix_regex: str):
    """
    grad_snap: dict[name -> grad_tensor] (tensor can be on GPU; we detach+cpu inside)
    """
    snap_cpu = {}
    norms = {}
    for name, g in grad_snap.items():
        if g is None:
            continue
        g = g.detach()
        snap_cpu[name] = g.float().cpu()
        norms[name] = torch.sqrt(torch.square(g).sum()).float().cpu()
    norms = group_param_norms_by_transformer_block(norms, prefix_regex)
    return snap_cpu, norms

def aw_sft_grad_norm_log_from_grads(total_grad_snap: dict, base_snapshot_cpu: dict, prefix_regex: str):
    """
    total_grad_snap: dict[name -> total_grad_tensor] (after SFT + AW backprops)
    base_snapshot_cpu: dict[name -> grad_tensor_cpu] (SFT snapshot from earlier)
    Returns grouped norms of (total - base).
    """
    norms = {}
    for name, g in total_grad_snap.items():
        if g is None:
            continue
        g = g.detach().float().cpu()
        base = base_snapshot_cpu.get(name, None)
        if base is None:
            base = torch.zeros_like(g)
        norms[name] = torch.sqrt(torch.square(g - base).sum()).cpu()
    norms = group_param_norms_by_transformer_block(norms, prefix_regex)
    return norms

def layer_grad_norm_mean_ratio(gn_aw: dict, gn_sft: dict, eps: float = 1e-8):
    """
    gn_*: dict[layer_key -> 1D tensor] (can be CPU tensors already)
    Returns:
      ratios_by_layer: dict[layer_key -> 0D tensor]
      ratios_vec: 1D tensor [num_layers] for histogram
    """
    ratios_by_layer = {}
    ratios = []

    common_layers = sorted(set(gn_aw.keys()) & set(gn_sft.keys()))
    for layer in common_layers:
        aw = gn_aw[layer]
        sft = gn_sft[layer]
        if aw is None or sft is None:
            continue
        if aw.numel() == 0 or sft.numel() == 0:
            continue

        aw_m = aw.float().mean()
        sft_m = sft.float().mean()
        r = aw_m / (sft_m + eps)

        ratios_by_layer[layer] = r.detach().cpu()
        ratios.append(r.detach().cpu().view(1))

    ratios_vec = torch.cat(ratios, dim=0) if len(ratios) > 0 else torch.empty(0)
    return ratios_by_layer, ratios_vec

def log_hist(tb_logger, tensors, stage="train", global_step=None):
    """
    tensors: dict[str, Tensor] or list[tuple[str, Tensor]]
    """
    if tb_logger is None or not hasattr(tb_logger, "add_histogram"):
        return
    if global_step is None:
        global_step = getattr(tb_logger, "global_step", 0)

    if isinstance(tensors, dict):
        items = tensors.items()
    else:
        items = tensors  # list of (name, tensor)

    for name, t in items:
        tb_logger.add_histogram(
            f"hist/{stage}_{name}",
            t.detach().flatten().cpu(),
            global_step=global_step,
        )

if __name__ == '__main__':
    pass
    # video_dataset = VideoDataset('/path/to/refl_videos.json')
    # print(len(video_dataset))

    # import pdb; pdb.set_trace()
    # a = [video_dataset[i] for i in [0,1,2]]
    
    # vid_tensor, init_frame = video_dataset.__getitem__(1)
    # import pdb; pdb.set_trace()


    # directory_path = "/path/to/datasets/davis_2017/real_videos_seq"
    # result = create_video_json(directory_path, "/path/to/soc-fine-tuning-sd/configs/refl_davis_videos.json")
    # print(f"Created JSON with {len(result)} videos")

    # input_file = "/path/to/soc-fine-tuning-sd/configs/refl_davis_videos.json"
    # output_dir = "/path/to/soc-fine-tuning-sd/configs"
    # # Split with 20% validation ratio
    # train_dict, val_dict = split_videos_by_object(
    #     input_json_path=input_file,
    #     validation_ratio=0.1,
    #     seed=42,
    #     output_dir=output_dir
    # )
    
    # print(f"\nFinal split:")
    # print(f"Training set: {len(train_dict)} videos")
    # print(f"Validation set: {len(val_dict)} videos")

    # vid_path = '/path/to/datasets/davis_2017/real_videos_seq/bear_s00000.mp4'
    # target_size = (24, 320, 576)
    # vid_data_type = 'mp4'
    # video_tensor, init_frame = read_video(vid_path, target_size, vid_data_type)
    # # save the init_frame tensor as pil image
    # init_frame = transforms.ToPILImage()(init_frame)
    # init_frame.save('init_frame_new.png')

    
