# SVD SFT (Supervised Fine-Tuning)

LoRA-based supervised fine-tuning for Stable Video Diffusion (SVD-XT).

## Directory Structure

```
svd_sft/
├── train_config.yaml          # Training configuration
├── train_shell.sh             # Launch script
├── train_davis_videos.json    # Training video list
├── val_davis_videos.json      # Validation video list
└── src/
    ├── train.py               # Entry point
    ├── sft_svd_trainer.py     # SFTTrainerSVD (LightningModule)
    ├── utils.py               # Dataset, config loading, callbacks
    ├── deepspeed_strategy.py  # DeepSpeed ZeRO config
    └── memory_optimization_mixin.py
```

## Key Config Paths

Edit `train_config.yaml` before training:

| Field | Description |
|-------|-------------|
| `videos_root_dir` | Root directory containing video files |
| `train_video_path_json` | Path to training video list JSON |
| `val_video_path_json` | Path to validation video list JSON |
| `save_dir` | Checkpoint output directory |
| `project_name` | Project name (subfolder under `save_dir`) |
| `exp_name` | Experiment name (subfolder under `project_name`) |
| `resume_from_checkpoint` | Path to `.ckpt` for resuming (or `null`) |

The pretrained SVD model is loaded from HuggingFace cache automatically (`stabilityai/stable-video-diffusion-img2vid-xt`).

## Dataset Format

**JSON structure** (dict mapping indices to filenames):
```json
{
  "0": "real_bear_s00044.mp4",
  "1": "real_bear_s00023.mp4",
  ...
}
```

**Video directory layout:**
```
videos_root_dir/
├── real_bear_s00044.mp4
├── real_bear_s00023.mp4
├── real_dog_s00010.mp4
└── ...
```

- Keys: string indices (`"0"`, `"1"`, ...)
- Values: video filenames (relative to `videos_root_dir`)
- Videos are loaded as MP4, resized/center-cropped to `target_vid_size: [24, 320, 576]` (T, H, W)
- Pixel values normalized to `[-1, 1]`

## Launch Training

```bash
cd svd_sft

# Simple launch (videos already on local disk):
python src/train.py --config train_config.yaml

# Override videos_root_dir from CLI:
python src/train.py --config train_config.yaml --videos_root_dir /path/to/videos
```

Checkpoints are saved to `{save_dir}/{project_name}/{exp_name}/`:
- `lora/` -- LoRA-only weights (lightweight, for evaluation)
- `full/` -- Full DeepSpeed checkpoints (for resuming training)
