#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# Number of GPUs per node (adjust based on your k8s pod configuration)
NUM_GPUS_PER_NODE=8

# Kubernetes multi-node training script
# K8s automatically provides: NODE_RANK, WORLD_SIZE (or NUM_NODES)
# You need to set: MASTER_ADDR (service name or pod DNS), MASTER_PORT
export TORCH_HOME=/path/to/custom_torch_home
# Get node rank from k8s environment (supports multiple env var names)
NODE_RANK=${NODE_RANK:-${GROUP_RANK:-${RANK:-0}}}

# Get number of nodes from k8s environment
NUM_NODES=${WORLD_SIZE:-${NUM_NODES:-${NNODES:-1}}}

# Master address - should be set to your Kubernetes service name or pod DNS
# Option 1: Use service name (if using headless service)
# MASTER_ADDR=${MASTER_ADDR:-"your-training-service-name"}
# Option 2: Use pod DNS of rank 0 (if using StatefulSet)
# MASTER_ADDR=${MASTER_ADDR:-"your-training-service-0.your-namespace.svc.cluster.local"}
# Option 3: Set via environment variable in your k8s deployment
MASTER_ADDR=${MASTER_ADDR:-"localhost"}  # fallback for local testing

# Master port
MASTER_PORT=${MASTER_PORT:-29514}

echo "Starting training on node ${NODE_RANK}/${NUM_NODES}"
echo "Master address: ${MASTER_ADDR}:${MASTER_PORT}"
echo "GPUs per node: ${NUM_GPUS_PER_NODE}"

# copy datasets to local directory
# 2) choose a local disk path (edit this to your cluster's local SSD mount)
LOCAL_DATASET="/local_workspace"
CLOUD_DATASET="/path/to/TempFolder/correct_processed_320x576x24_fps14.tar.gz"


# =========================================================
# Cross-node barrier (shared filesystem)
# =========================================================
BARRIER_DIR="/path/to/checkpoints/DAVIS_Pretrain_LC_Reward_24x320x576_fps14/_barrier"
RUN_TAG="train_${MASTER_PORT}"   # stable across nodes for this job

barrier () {
  local stage="$1"
  mkdir -p "${BARRIER_DIR}"
  touch "${BARRIER_DIR}/${RUN_TAG}.${stage}.node${NODE_RANK}"
  while [[ "$(ls -1 "${BARRIER_DIR}/${RUN_TAG}.${stage}.node"* 2>/dev/null | wc -l)" -lt "${NUM_NODES}" ]]; do
    sleep 5
  done
}

# 3) stage once per node
mkdir -p "$LOCAL_DATASET"
MARKER="$LOCAL_DATASET/.staged_ok"
if [ ! -f "$MARKER" ]; then
  echo "[stage] copy from $CLOUD_DATASET -> $LOCAL_DATASET"
  cd "$LOCAL_DATASET"
  cp -r "${CLOUD_DATASET}" "${LOCAL_DATASET}/"
  tar -xzf "${LOCAL_DATASET}/correct_processed_320x576x24_fps14.tar.gz" -C "${LOCAL_DATASET}"
  touch "$MARKER"
else
  echo "[stage] already staged at $LOCAL_DATASET"
fi

barrier "dataset_staged"

cd "${SCRIPT_DIR}"
echo "current working dir:"
pwd

accelerate launch \
    --num_processes $(($NUM_NODES * $NUM_GPUS_PER_NODE)) \
    --num_machines $NUM_NODES \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    train_lc_discriminator.py \
    --config ./configs/lc_discriminator_train_config.yaml \
    --report_to tensorboard


# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29514 train_lc_discriminator.py --config ./configs/lc_discriminator_train_config.yaml --report_to tensorboard
# CUDA_VISIBLE_DEVICES=0,3,4,5,6,7,8,9 python /path/to/eval_phyvid.py --master_port 12491

