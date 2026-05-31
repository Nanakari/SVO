#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_PYTHON="${ENV_PYTHON:-/root/autodl-tmp/conda_envs/SVO/bin/python}"
export PATH="$(dirname "$ENV_PYTHON"):$PATH"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip_cache}"

CONFIG="${CONFIG:-configs/default_autodl.yaml}"
GPU="${GPU:-0}"
THRESHOLDS="${THRESHOLDS:-0.5 1.0 1.5 2.0}"

if [[ "${1:-}" == "--dry-run" ]]; then
  bash scripts/tune_svo_threshold.sh --config "$CONFIG" --thresholds "$THRESHOLDS" --gpu "$GPU" --dry-run
  bash scripts/run_all.sh \
    --config "$CONFIG" \
    --dry-run \
    --datasets coco_chair \
    --methods base,svo,verify_all,random_verify \
    --risk-threshold "${SVO_RISK_THRESHOLD:-1.0}" \
    --gpu "$GPU"
  exit 0
fi

bash scripts/tune_svo_threshold.sh --config "$CONFIG" --thresholds "$THRESHOLDS" --gpu "$GPU"

if [[ -z "${SVO_RISK_THRESHOLD:-}" ]]; then
  cat >&2 <<'MSG'
Set SVO_RISK_THRESHOLD from validation metrics before running the COCO main experiment.
Example:
  SVO_RISK_THRESHOLD=1.0 bash scripts/run_coco_main_autodl.sh --main-only
MSG
  exit 2
fi

bash scripts/run_all.sh \
  --config "$CONFIG" \
  --datasets coco_chair \
  --methods base,svo,verify_all,random_verify \
  --risk-threshold "$SVO_RISK_THRESHOLD" \
  --gpu "$GPU"
