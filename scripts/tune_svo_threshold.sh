#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="configs/default.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/validation}"
SAMPLE_SIZE="${SAMPLE_SIZE:-5000}"
SEED="${SEED:-42}"
ANNOTATIONS="data/coco/annotations/instances_train2017.json"
SPLIT_FILE=""
IMAGE_ROOT=""
BACKEND=""
LIMIT=""
DRY_RUN=0
OVERWRITE=0
THRESHOLDS=(0.5 1.0 1.5 2.0)

usage() {
  cat <<'EOF'
Usage: bash scripts/tune_svo_threshold.sh [options]

Run the validation-only SVO risk-threshold tuning pipeline on COCO train2017.
The final test split is not used by this script.

Options:
  --dry-run                 Print commands without executing them.
  --config PATH             Global YAML config. Default: configs/default.yaml
  --output-dir DIR          Validation output root. Default: outputs/validation
  --sample-size N           Validation split size. Default: 5000
  --seed N                  Validation split seed. Default: 42
  --annotations PATH        COCO train2017 instances JSON.
  --split-file PATH         Validation split file.
  --image-root DIR          train2017 subset/full image root.
  --thresholds "A B C"      Space- or comma-separated risk thresholds.
  --limit N                 Debug limit for captions and verification records.
  --backend NAME            Object extraction backend for evaluation, e.g. rule.
  --overwrite               Overwrite per-step JSONL outputs where supported.
  --gpu IDS                 Export CUDA_VISIBLE_DEVICES.
  -h, --help                Show this help.

Optional environment variables:
  PYTHON                    Python executable. Default: python
  OUTPUT_DIR                Validation output root.
  SAMPLE_SIZE               Validation split size.
  SEED                      Validation split seed.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --annotations)
      ANNOTATIONS="$2"
      shift 2
      ;;
    --split-file)
      SPLIT_FILE="$2"
      shift 2
      ;;
    --image-root)
      IMAGE_ROOT="$2"
      shift 2
      ;;
    --thresholds)
      IFS=', ' read -r -a THRESHOLDS <<< "$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --gpu)
      export CUDA_VISIBLE_DEVICES="$2"
      shift 2
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

cd "$ROOT_DIR"

if [[ -z "$SPLIT_FILE" ]]; then
  SPLIT_FILE="configs/splits/coco_train2017_val${SAMPLE_SIZE}_seed${SEED}.txt"
fi

if [[ -z "$IMAGE_ROOT" ]]; then
  if [[ -d "data/coco/train2017_val${SAMPLE_SIZE}" ]]; then
    IMAGE_ROOT="data/coco/train2017_val${SAMPLE_SIZE}"
  else
    IMAGE_ROOT="data/coco/train2017"
  fi
fi

if [[ "${#THRESHOLDS[@]}" -eq 0 ]]; then
  echo "At least one threshold is required." >&2
  exit 2
fi

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$OUTPUT_DIR"/{predictions,objects,priors,sweeps}
fi

limit_args=()
if [[ -n "$LIMIT" ]]; then
  limit_args=(--limit "$LIMIT")
fi

sweep_limit_args=()
if [[ -n "$LIMIT" ]]; then
  sweep_limit_args=(--limit "$LIMIT")
fi

backend_args=()
if [[ -n "$BACKEND" ]]; then
  backend_args=(--backend "$BACKEND")
fi

overwrite_args=()
if [[ "$OVERWRITE" -eq 1 ]]; then
  overwrite_args=(--overwrite)
fi

captions="$OUTPUT_DIR/predictions/coco_train2017_val${SAMPLE_SIZE}_base_captions.jsonl"
prior="$OUTPUT_DIR/priors/coco_static_prior.json"
objects="$OUTPUT_DIR/objects/coco_train2017_val${SAMPLE_SIZE}_svo_objects.jsonl"
sweep_output="$OUTPUT_DIR/sweeps/risk_threshold"

run_cmd "$PYTHON" scripts/make_val_split.py \
  --coco-annotations "$ANNOTATIONS" \
  --sample-size "$SAMPLE_SIZE" \
  --seed "$SEED" \
  --output "$SPLIT_FILE"

run_cmd "$PYTHON" scripts/run_caption.py \
  --config "$CONFIG" \
  --dataset configs/datasets/coco_chair.yaml \
  --method configs/methods/base.yaml \
  --output "$captions" \
  --set "dataset.paths.image_root=$IMAGE_ROOT" \
  --set "dataset.paths.annotation_file=$ANNOTATIONS" \
  --set "dataset.paths.split_file=$SPLIT_FILE" \
  "${limit_args[@]}" \
  "${overwrite_args[@]}"

run_cmd "$PYTHON" scripts/build_static_prior.py \
  --config "$CONFIG" \
  --captions "$captions" \
  --coco-annotations "$ANNOTATIONS" \
  --output "$prior" \
  "${backend_args[@]}"

run_cmd "$PYTHON" scripts/extract_objects.py \
  --config "$CONFIG" \
  --method configs/methods/svo.yaml \
  --input "$captions" \
  --output "$objects" \
  --set "risk_scoring.static_prior_path=$prior" \
  "${backend_args[@]}" \
  "${overwrite_args[@]}"

run_cmd "$PYTHON" scripts/sweep_thresholds.py \
  --config "$CONFIG" \
  --dataset coco_chair \
  --method configs/methods/svo.yaml \
  --objects "$objects" \
  --base-predictions "$captions" \
  --coco-annotations "$ANNOTATIONS" \
  --output-dir "$sweep_output" \
  --thresholds "${THRESHOLDS[@]}" \
  "${backend_args[@]}" \
  "${sweep_limit_args[@]}" \
  "${overwrite_args[@]}"

cat <<EOF

Validation threshold tuning outputs:
  Captions: $captions
  Static prior: $prior
  Objects: $objects
  Sweep tables: $sweep_output/tables/

Choose the final --risk-threshold from validation metrics only, then freeze it before running
scripts/run_all.sh on the final COCO/POPE/AMBER evaluations.
EOF
