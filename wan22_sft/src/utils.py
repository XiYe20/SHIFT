import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
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
    
    # Apply transforms to each frame
    resized_frames = []
    for i in range(video_tensor.shape[0]):
        frame = video_tensor[i]
        resized_frame = resize_transform(frame)
        resized_frames.append(resized_frame.unsqueeze(0))
    
    # Stack frames back into a video tensor
    video_tensor = torch.cat(resized_frames, dim=0)
    init_frame = video_tensor[0, ...]

    return video_tensor, init_frame

class VideoDataset(Dataset):
    def __init__(
        self,
        videos_root_dir='wisa80k/collision',
        video_prompt_json_file='train_wisa_80k_videos_prompts.json',
        target_vid_size=(24, 576, 1024),
        vid_data_type='mp4'
    ):
        self.videos_root_dir = videos_root_dir
        self.target_vid_size = target_vid_size
        self.vid_data_type = vid_data_type

        self.samples = []
        with open(video_prompt_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            video_name = item["video_name"]
            prompt = item["captions"]
            video_path = os.path.join(videos_root_dir, video_name)
            self.samples.append(
                {"video_path": video_path, "prompt": prompt}
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        vid_path = item["video_path"]
        prompt = item["prompt"]
        video_tensor, _ = read_video(vid_path, self.target_vid_size, self.vid_data_type)

        # convert the pixel value from range [0, 1] to [-1, 1]
        video_tensor = video_tensor * 2 - 1.

        return {"video": video_tensor, "prompt": prompt}
            
class VideoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=3, videos_root_dir=None, train_video_path_json="refl_videos.json", val_video_path_json="refl_videos.json", target_vid_size=(49, 704, 1280), vid_data_type='mp4'):
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
        self.train_dataset = VideoDataset(videos_root_dir=self.videos_root_dir, video_prompt_json_file=self.train_video_path_json, target_vid_size=self.target_vid_size, vid_data_type=self.vid_data_type)
        self.val_dataset = VideoDataset(videos_root_dir=self.videos_root_dir, video_prompt_json_file=self.val_video_path_json, target_vid_size=self.target_vid_size, vid_data_type=self.vid_data_type)

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

import torch
import pytorch_lightning as pl
from pathlib import Path

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
    parser.add_argument("--videos_root_dir", type=str, help="Local path to extracted video directory")
    
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
    

def create_video_json(directory, output_file="video_files.json", recursive=False, prompt_json_file=None, num_examples=5000):
    """
    If prompt_json_file is None:
      - old behavior: save { "0": "/abs/a.mp4", ... }

    If prompt_json_file is provided (list of {"video_name": "<hash>.mp4", "captions": "..."}):
      - new behavior: save [ {"video_name": "<actual_file_in_directory>.mp4", "captions": "..."} , ... ]
        where files like "real_<hash>-seg0.mp4" are matched to "<hash>.mp4" captions.
    """
    # Check if directory exists
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory: {directory}")

    abs_directory = os.path.abspath(directory)

    # Find all MP4 files
    if recursive:
        mp4_pattern = os.path.join(abs_directory, "**", "*.mp4")
        mp4_files = glob.glob(mp4_pattern, recursive=True)
    else:
        mp4_pattern = os.path.join(abs_directory, "*.mp4")
        mp4_files = glob.glob(mp4_pattern)

    mp4_files.sort()

    # ---------- NEW: prompt-json mode ----------
    if prompt_json_file is not None:
        import re

        # Load prompt map: {"<hash>.mp4": "caption", ...}
        with open(prompt_json_file, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        prompt_map = {os.path.basename(it["video_name"]): it.get("captions", "") for it in prompt_data}

        def _infer_original_name(file_basename: str) -> str:
            """
            Map:
              real_<hash>-seg0.mp4 -> <hash>.mp4
              real_<hash>-seg1.mp4 -> <hash>.mp4
            Fallback: strip leading 'real_' and strip '-seg\\d+'.
            """
            m = re.match(r"^(?:real_)?([0-9a-fA-F]{64})(?:-seg\d+)?\.mp4$", file_basename)
            if m:
                return f"{m.group(1).lower()}.mp4"
            # fallback heuristic
            name = file_basename
            if name.startswith("real_"):
                name = name[len("real_"):]
            name = re.sub(r"-seg\d+(?=\.mp4$)", "", name)
            return name

        out_list = []
        missing = 0
        for file_path in mp4_files:
            bn = os.path.basename(file_path)
            origin_name = _infer_original_name(bn)
            caption = prompt_map.get(origin_name)
            if caption is None:
                missing += 1
                # skip videos that have no caption match
                continue
            out_list.append({"video_name": bn, "captions": caption})

        # randomly sample num_examples from video_dict
        if num_examples < len(out_list):
            out_list = random.sample(out_list, num_examples)
        else:
            print(f"num_examples {num_examples} is larger than the number of videos {len(out_list)}, using all videos")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(out_list, f, ensure_ascii=False, indent=4)

        print(f"Found {len(mp4_files)} MP4 files in {abs_directory}")
        print(f"Matched captions for {len(out_list)} files; missing captions for {missing} files")
        print(f"Saved to {output_file}")
        return out_list

    # ---------- OLD: path-json mode ----------
    video_dict = {}
    for i, file_path in enumerate(mp4_files):
        video_dict[str(i)] = os.path.abspath(file_path)

    # randomly sample num_examples from video_dict
    if num_examples < len(video_dict):
        video_dict = dict(random.sample(list(video_dict.items()), num_examples))
    else:
        print(f"num_examples {num_examples} is larger than the number of videos {len(video_dict)}, using all videos")
    with open(output_file, "w") as f:
        json.dump(video_dict, f, indent=4)

    print(f"Found {len(mp4_files)} MP4 files in {abs_directory}")
    if recursive:
        print("(searched recursively)")
    print(f"Saved to {output_file}")
    return video_dict

def split_train_val_json(input_videos_prompt_json, train_output_json, val_output_json, validation_ratio=0.2, seed=42):
    """
    Split a videos_prompt_json (list of {"video_name","captions"}) into train/val json files.
    """
    with open(input_videos_prompt_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of {video_name, captions} items")

    rng = random.Random(seed)
    rng.shuffle(data)

    n = len(data)
    n_val = int(round(n * validation_ratio))
    n_val = max(1, n_val) if n > 0 and validation_ratio > 0 else 0

    val_data = data[:n_val]
    train_data = data[n_val:]

    with open(train_output_json, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(val_output_json, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)

    print(f"Split {n} items -> train={len(train_data)}, val={len(val_data)} (ratio={validation_ratio}, seed={seed})")
    return train_data, val_data


if __name__ == '__main__':
    # video_dataset = VideoDataset('/path/to/refl_videos.json')
    # print(len(video_dataset))

    # import pdb; pdb.set_trace()
    # a = [video_dataset[i] for i in [0,1,2]]
    
    # vid_tensor, init_frame = video_dataset.__getitem__(1)
    # import pdb; pdb.set_trace()
    directory = '/path/to/datasets/wisa80k_correct/processed_704x1280x49_fps24/deformation/real_videos'
    create_video_json(directory, output_file="video_files.json", recursive=False, prompt_json_file='/path/to/datasets/MotionAlignmentDatasets/wisa80k_correct/wisa_80k_captions.json', num_examples=5500)
    
    train_output_json = '/path/to/datasets/wisa80k_correct/processed_704x1280x49_fps24/deformation/wisa80k_deformation_train.json'
    val_output_json = '/path/to/datasets/wisa80k_correct/processed_704x1280x49_fps24/deformation/wisa80k_deformation_val.json'
    split_train_val_json('./video_files.json', train_output_json, val_output_json, validation_ratio=0.0909, seed=42)
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

    
