#!/bin/bash
NODE_RANK="${NODE_RANK:-0}"       # 0..WORLD_SIZE-1
NUM_NODES="${WORLD_SIZE:-1}"      # node count
echo "[node ${NODE_RANK}/${NUM_NODES}] starting wan2.2 generation"

pip install -r MotionAlignmentVDM/requirements.txt
# Base directories
BASE_REAL_VIDEO_DIR="/path/to/datasets/MotionAlignmentDatasets/wisa80k_correct/original_videos"
BASE_OUTPUT_DIR="/path/to/datasets/wisa80k/processed_704x1280x49_fps24_20steps"

# =========================================================
# Cross-node barrier (shared filesystem)
# =========================================================
BARRIER_DIR="${BASE_OUTPUT_DIR}/_barrier"          # must be on shared FS visible to all nodes
MASTER_PORT="${MASTER_PORT:-29500}"               # any stable job-level id
RUN_TAG="wan22_job_${MASTER_PORT}"                # stable across nodes for this job

barrier () {
  local stage="$1"
  mkdir -p "${BARRIER_DIR}"
  touch "${BARRIER_DIR}/${RUN_TAG}.${stage}.node${NODE_RANK}"
  while [[ "$(ls -1 "${BARRIER_DIR}/${RUN_TAG}.${stage}.node"* 2>/dev/null | wc -l)" -lt "${NUM_NODES}" ]]; do
    sleep 10
  done
}

# Create base output directory if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR"

# Loop through all subdirectories in the real_video_dir
for subdir_name in "deformation"; do
    subdir="$BASE_REAL_VIDEO_DIR/$subdir_name"
    
    echo "Processing subdirectory: $subdir_name"
    
    # Define input and output paths for this subdirectory
    input_path="$BASE_REAL_VIDEO_DIR/$subdir_name"
    output_path="$BASE_OUTPUT_DIR/$subdir_name"
    
    # Create output subdirectory if it doesn't exist
    mkdir -p "$output_path"
    
    # Run the python script for this subdirectory
    python wan22_example_generation.py \
        --gpu_ids 0,1,2,3 \
        --num_process_per_gpu 1 \
        --output_dir "$output_path" \
        --real_video_dir "$input_path" \
        --target_frames 49 --target_height 704 --target_width 1280 \
        --target_fps 24 \
        --match_target_fps \
        --num_fake_example_per_real_video 5 \
        --num_inference_steps 20 \
        --pretrained_wan_path /path/to/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/b8fff7315c768468a5333511427288870b2e9635 \
        --prompt_json_file /path/to/datasets/MotionAlignmentDatasets/wisa80k_correct/wisa_80k_captions.json \
        # --only_process_real_videos
    
    echo "Finished processing subdirectory: $subdir_name"
    echo "---"
done
barrier "all_sampling_done"
echo "[Node ${NODE_RANK}/${NUM_NODES}] All done."
echo "All subdirectories processed!"