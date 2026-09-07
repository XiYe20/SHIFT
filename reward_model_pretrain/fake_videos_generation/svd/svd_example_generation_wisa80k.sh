#!/bin/bash

# Base directories
BASE_REAL_VIDEO_DIR="/path/to/wisa80k/original_videos"
BASE_OUTPUT_DIR="/path/to/wisa80k/processed_320x576x24_fps24"

# Create base output directory if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR"

# Loop through all subdirectories in the real_video_dir
for subdir_name in "collision" "deformation" "rigid body motion" "elastic motion" "explosion" "gas motion" "liquid motion" "vaporization" "combustion"; do
    subdir="$BASE_REAL_VIDEO_DIR/$subdir_name"
    
    echo "Processing subdirectory: $subdir_name"
    
    # Define input and output paths for this subdirectory
    input_path="$BASE_REAL_VIDEO_DIR/$subdir_name"
    output_path="$BASE_OUTPUT_DIR/$subdir_name"
    
    # Create output subdirectory if it doesn't exist
    mkdir -p "$output_path"
    
    # Run the python script for this subdirectory
    python svd_example_generation.py \
        --gpu_ids 0,1,2,3 \
        --num_process_per_gpu 2 \
        --output_dir "$output_path" \
        --real_video_dir "$input_path" \
        --target_frames 24 --target_height 320 --target_width 576 \
        --target_fps 24 \
        --match_target_fps \
        --noise_aug_strength 0.02 \
        --num_fake_example_per_real_video 2 \
        --num_inference_steps 25 \
        --scheduler_type edm_ancestral \
        --pretrained_svd_path /path/to/models--stabilityai--stable-video-diffusion-img2vid-xt/snapshots/9e43909513c6714f1bc78bcb44d96e733cd242aa
    
    echo "Finished processing subdirectory: $subdir_name"
    echo "---"
done

echo "All subdirectories processed!"