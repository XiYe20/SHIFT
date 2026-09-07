# WAN2.2 SFT (Supervised Fine-Tuning)

LoRA-based supervised fine-tuning for WAN2.2 TI2V-5B (Text/Image-to-Video, Diffusers version).

## Directory Structure

```
wan22_sft/
├── train_config.yaml          # Training configuration
├── train_shell.sh             # Launch script (stages data + trains)
├── wisa80k_debug_train.json   # Debug training set (small subset)
├── wisa80k_debug_val.json     # Debug validation set (small subset)
└── src/
    ├── train.py               # Entry point
    ├── sft_wan_trainer.py     # SFTTrainerWAN (LightningModule)
    ├── utils.py               # Dataset, config loading, callbacks
    ├── wan22_flow_match_utils.py  # Flow-matching scheduler & loss
    ├── deepspeed_strategy.py  # DeepSpeed ZeRO config
    └── memory_optimization_mixin.py
```

## Key Config Paths

Edit `train_config.yaml` before training:

| Field | Description |
|-------|-------------|
| `pretrained_wan_path` | Local path to WAN2.2-TI2V-5B-Diffusers snapshot |
| `videos_root_dir` | Root directory or OSS URI for video data |
| `train_video_path_json` | Training video+caption JSON |
| `val_video_path_json` | Validation video+caption JSON |
| `save_dir` | Checkpoint output directory |
| `resume_from_checkpoint` | Path to `full/last.ckpt` for resuming (or `null`) |

## Dataset Format

**JSON structure** (list of objects with video name and caption):
```json
[
  {
    "video_name": "real_ca763e76...ec14b-seg0.mp4",
    "captions": "The video captures a sequence where a white car..."
  },
  {
    "video_name": "real_306a951e...98f5c0-seg0.mp4",
    "captions": "The video depicts a close-up view of two hands..."
  },
  ...
]
```

**Video directory layout:**
```
videos_root_dir/
├── real_ca763e76...ec14b-seg0.mp4
├── real_306a951e...98f5c0-seg0.mp4
└── ...
```

- Each entry has `video_name` (filename relative to `videos_root_dir`) and `captions` (text prompt)
- Videos resized/center-cropped to `target_vid_size: [49, 480, 832]` (T, H, W)
- Pixel values normalized to `[-1, 1]`; flow-matching loss used for training

## Launch Training

```bash
cd wan22_sft

# Simple launch (videos already on local disk):
python src/train.py --config train_config.yaml

# Override videos_root_dir from CLI:
python src/train.py --config train_config.yaml --videos_root_dir /path/to/videos
```

The provided `train_shell.sh` handles OSS data staging if `videos_root_dir` is a cloud URI.

Checkpoints are saved to `{save_dir}/{project_name}/{exp_name}/`:
- `lora/` -- LoRA-only weights (for evaluation)
- `full/` -- Full DeepSpeed checkpoints (for resuming)
