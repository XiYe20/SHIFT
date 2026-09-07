# WAN2.2 SHIFT

LoRA-based SHIFT fine-tuning for WAN2.2 TI2V-5B, using advantage-weighted SFT with IC and LC reward models. Buffer stores latents (not pixels) for memory efficiency at high resolutions.

## Directory Structure

```
wan22_shift/
├── train_config.yaml          # Training configuration
├── train.sh                   # Launch script (stages data + trains)
└── src/
    ├── train.py               # Entry point
    ├── shift_wan_trainer.py   # ShiftTrainerWAN (LightningModule)
    ├── utils.py               # Dataset, config loading, callbacks
    ├── wan22_flow_match_utils.py  # Flow-matching scheduler & loss
    ├── chunked_sampler.py     # Buffer-based chunked sampler
    ├── deepspeed_strategy.py  # DeepSpeed ZeRO config
    └── memory_optimization_mixin.py
```

## Key Config Paths

Edit `train_config.yaml` before training:

| Field | Description |
|-------|-------------|
| `pretrained_wan_path` | Local path to WAN2.2-TI2V-5B-Diffusers snapshot |
| `pretrained_sft_lora_path` | Path to SFT LoRA checkpoint (loaded & fused before SHIFT training), or `null` |
| `videos_root_dir` | Root directory or OSS URI for video data |
| `train_video_path_json` | Training video+caption JSON |
| `val_video_path_json` | Validation video+caption JSON |
| `save_dir` | Checkpoint output directory |
| `resume_from_checkpoint` | Path to `full/last.ckpt` for resuming (or `null`) |
| `ic_reward_model_config.pretrained_ckpt` | Pretrained IC reward model checkpoint |
| `lc_reward_model_config.pretrained_ckpt` | Pretrained LC reward model checkpoint |
| `ic_reward_model_config.*.raft_ckpt_file` | SEA-RAFT checkpoint for optical flow |
| `lc_reward_model_config.discriminator_config.pretrain_cotracker_ckpt_file` | CoTracker checkpoint |

## Dataset Format

**JSON structure** (list of objects with video name and caption):
```json
[
  {
    "video_name": "real_ca763e76...ec14b-seg0.mp4",
    "captions": "The video captures a sequence where a white car..."
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
- Buffer-based training: dataset split into chunks of `buffer_size`, each reused for `passes_per_buffer` passes
- The dataset JSON files are referenced from remote paths in the config (not included locally)

## Launch Training

```bash
cd wan22_shift
bash train.sh
```

**What `train.sh` does:**
1. Reads `videos_root_dir` (OSS URI) from `train_config.yaml`
2. Downloads and extracts the tar.gz to `/local_workspace/`
3. Runs `python src/train.py --config train_config.yaml --videos_root_dir <local_path>`

**If videos are already local**, run directly:
```bash
python src/train.py --config train_config.yaml --videos_root_dir /path/to/videos
```

Checkpoints are saved to `{save_dir}/{project_name}/{exp_name}/`:
- `lora/` -- LoRA-only weights (for evaluation)
- `full/` -- Full DeepSpeed checkpoints (for resuming)
