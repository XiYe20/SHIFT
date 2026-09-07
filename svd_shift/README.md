# SVD SHIFT

LoRA-based SHIFT fine-tuning for Stable Video Diffusion, using advantage-weighted SFT with IC and LC reward models.

## Directory Structure

```
svd_shift/
├── train_config.yaml          # Training configuration
├── train.sh                   # Launch script (stages data + trains)
├── train_davis_videos.json    # Training video list
├── val_davis_videos.json      # Validation video list
└── src/
    ├── train.py               # Entry point
    ├── shift_svd_trainer.py   # ShiftTrainerSVD (LightningModule)
    ├── utils.py               # Dataset, config loading, callbacks
    ├── EDM_Ancestral_scheduler.py  # Custom EDM scheduler
    ├── chunked_sampler.py     # Buffer-based chunked sampler
    ├── deepspeed_strategy.py  # DeepSpeed ZeRO config
    └── memory_optimization_mixin.py
```

## Key Config Paths

Edit `train_config.yaml` before training:

| Field | Description |
|-------|-------------|
| `pretrained_svd_path` | Local path to SVD-XT model snapshot |
| `pretrained_sft_lora_path` | Path to SFT LoRA checkpoint (loaded & fused before SHIFT training), or `null` |
| `videos_root_dir` | Root directory or OSS URI for video data |
| `train_video_path_json` | Training video list JSON |
| `val_video_path_json` | Validation video list JSON |
| `save_dir` | Checkpoint output directory |
| `resume_from_checkpoint` | Path to `full/last.ckpt` for resuming (or `null`) |
| `ic_reward_model_config.pretrained_ckpt` | Pretrained IC reward model checkpoint |
| `lc_reward_model_config.pretrained_ckpt` | Pretrained LC reward model checkpoint |
| `ic_reward_model_config.*.raft_ckpt_file` | SEA-RAFT checkpoint for optical flow |
| `lc_reward_model_config.discriminator_config.pretrain_cotracker_ckpt_file` | CoTracker checkpoint |

## Dataset Format

**JSON structure** (dict mapping indices to filenames):
```json
{
  "0": "real_flamingo_s00022-seg0.mp4",
  "1": "real_cows_s00065-seg0.mp4",
  ...
}
```

**Video directory layout:**
```
videos_root_dir/
├── real_flamingo_s00022-seg0.mp4
├── real_cows_s00065-seg0.mp4
└── ...
```

- Keys: string indices; Values: video filenames relative to `videos_root_dir`
- Videos resized/center-cropped to `target_vid_size: [24, 320, 576]` (T, H, W)
- Buffer-based training: dataset split into chunks of `buffer_size`, each reused for `passes_per_buffer` passes

## Launch Training

```bash
cd svd_shift
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
