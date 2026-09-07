#!/usr/bin/env bash
set -euo pipefail
set -x
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

export TORCH_HOME=/path/to/custom_torch_home
export VBENCH_CACHE_DIR=/path/to/vbench_cache_dir

# 1) read videos_root_dir from YAML (cloud tar.gz URI)
VIDEOS_ROOT_DIR=$(
python - <<'PY'
import yaml
cfg = yaml.safe_load(open("train_config.yaml","r"))
print(cfg["videos_root_dir"])
PY
)

# 2) stage to local disk
LOCAL_DATASET="/local_workspace"
TAR_NAME="$(basename "${VIDEOS_ROOT_DIR}")"
EXTRACT_DIR_NAME="${TAR_NAME%.tar.gz}"
EXTRACT_DIR_NAME="${EXTRACT_DIR_NAME%.tgz}"
EXTRACTED_REAL_VIDEOS="${LOCAL_DATASET}/${EXTRACT_DIR_NAME}/real_videos"

# 3) stage once per node
mkdir -p "$LOCAL_DATASET"
MARKER="$LOCAL_DATASET/.staged_ok"
if [ ! -f "$MARKER" ]; then
  echo "[stage] copy from ${VIDEOS_ROOT_DIR} -> ${LOCAL_DATASET}"
  cd "$LOCAL_DATASET"
  cp -r "${VIDEOS_ROOT_DIR}" "${LOCAL_DATASET}/"
  echo "[stage] extract ${TAR_NAME} -> ${LOCAL_DATASET}"
  tar -xzf "${LOCAL_DATASET}/${TAR_NAME}" -C "${LOCAL_DATASET}"
  rm -f "${LOCAL_DATASET}/${TAR_NAME}"
  if [ ! -d "${EXTRACTED_REAL_VIDEOS}" ]; then
    echo "[stage][error] expected extracted real_videos not found: ${EXTRACTED_REAL_VIDEOS}" >&2
    exit 1
  fi
  touch "$MARKER"
else
  echo "[stage] already staged at $LOCAL_DATASET"
fi

cd "${SCRIPT_DIR}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/train.py --config train_config.yaml --videos_root_dir "$EXTRACTED_REAL_VIDEOS"
