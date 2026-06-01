#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_PYTHON="${ENV_PYTHON:-/root/autodl-tmp/conda_envs/SVO/bin/python}"
export PATH="$(dirname "$ENV_PYTHON"):$PATH"
export PYTHON="${PYTHON:-$ENV_PYTHON}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip_cache}"

CONFIG="${CONFIG:-configs/default_autodl.yaml}"
GPU="${GPU:-0}"
THRESHOLDS="${THRESHOLDS:-0.5 1.0 1.5 2.0}"
MODE="full"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_coco_main_autodl.sh [--dry-run|--tune-only|--main-only]

AutoDL/SeeTaCloud helper for the COCO SVO workflow.

Modes:
  --dry-run     Print threshold-tuning and main-experiment commands only.
  --tune-only   Run validation threshold tuning, then stop.
  --main-only   Run the final COCO main experiment only. Requires SVO_RISK_THRESHOLD.

Environment:
  CONFIG              Global config path. Default: configs/default_autodl.yaml
  GPU                 CUDA_VISIBLE_DEVICES value. Default: 0
  THRESHOLDS          Validation thresholds. Default: "0.5 1.0 1.5 2.0"
  SVO_RISK_THRESHOLD  Frozen validation-selected threshold for --main-only/full main run.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --tune-only)
      MODE="tune-only"
      shift
      ;;
    --main-only)
      MODE="main-only"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run_tuning() {
  local args=(scripts/tune_svo_threshold.sh --config "$CONFIG" --thresholds "$THRESHOLDS" --gpu "$GPU")
  if [[ "$MODE" == "dry-run" ]]; then
    args+=(--dry-run)
  fi
  bash "${args[@]}"
}

run_main() {
  local threshold="${SVO_RISK_THRESHOLD:-}"
  if [[ -z "$threshold" && "$MODE" == "dry-run" ]]; then
    threshold="1.0"
  fi
  if [[ -z "$threshold" ]]; then
    cat >&2 <<'MSG'
Set SVO_RISK_THRESHOLD from validation metrics before running the COCO main experiment.
Example:
  SVO_RISK_THRESHOLD=1.0 bash scripts/run_coco_main_autodl.sh --main-only
MSG
    exit 2
  fi
  local args=(
    scripts/run_all.sh
    --config "$CONFIG"
    --datasets coco_chair
    --methods base,svo,verify_all,random_verify
    --risk-threshold "$threshold"
    --gpu "$GPU"
  )
  if [[ "$MODE" == "dry-run" ]]; then
    args+=(--dry-run)
  fi
  bash "${args[@]}"
}

case "$MODE" in
  dry-run)
    run_tuning
    run_main
    ;;
  tune-only)
    run_tuning
    ;;
  main-only)
    run_main
    ;;
  full)
    run_tuning
    run_main
    ;;
esac
