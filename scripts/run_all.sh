#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="configs/default.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
DATASETS="coco_chair,pope"
METHODS="base,svo,verify_all,random_verify"
RISK_THRESHOLD="${RISK_THRESHOLD:-}"
LIMIT="${LIMIT:-}"
PRIOR_LIMIT="${PRIOR_LIMIT:-5000}"
PRIOR_SPLIT_FILE="${PRIOR_SPLIT_FILE:-}"
PRIOR_IMAGE_ROOT="${PRIOR_IMAGE_ROOT:-}"
STATIC_PRIOR_PATH="${STATIC_PRIOR_PATH:-}"
STATIC_PRIOR_PATH_EXPLICIT=0
BUILD_PRIOR=1
FORCE_PRIOR=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_all.sh [options]

Run or print the SVO reproduction pipeline. Use --dry-run before launching real jobs.

Options:
  --dry-run                 Print commands without executing them.
  --config PATH             Global YAML config. Default: configs/default.yaml
  --output-dir DIR          Output root. Default: outputs
  --datasets LIST           Comma list: coco_chair,pope,amber_object,all. Default: coco_chair,pope
  --methods LIST            Comma list: base,svo,verify_all,random_verify,ablations,components,all.
  --risk-threshold FLOAT    SVO risk threshold. Required for real SVO verification unless config sets it.
  --limit N                 Limit caption/POPE inference for debugging.
  --prior-limit N           COCO train2017 images for static prior captions. Default: 5000
  --prior-split-file PATH   Split file for validation/static-prior captions.
  --prior-image-root DIR    Image root for validation/static-prior captions.
  --static-prior-path PATH  Static prior JSON used during object risk scoring.
  --skip-prior              Do not build the COCO static prior.
  --force-prior             Rebuild the static prior even if the output file already exists.
  --gpu IDS                 Export CUDA_VISIBLE_DEVICES.
  -h, --help                Show this help.

Optional environment variables:
  AMBER_PREDICTIONS         JSONL file for AMBER Object Subset evaluation.
  PYTHON                    Python executable. Default: python
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
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --methods)
      METHODS="$2"
      shift 2
      ;;
    --risk-threshold)
      RISK_THRESHOLD="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --prior-limit)
      PRIOR_LIMIT="$2"
      shift 2
      ;;
    --prior-split-file)
      PRIOR_SPLIT_FILE="$2"
      shift 2
      ;;
    --prior-image-root)
      PRIOR_IMAGE_ROOT="$2"
      shift 2
      ;;
    --static-prior-path)
      STATIC_PRIOR_PATH="$2"
      STATIC_PRIOR_PATH_EXPLICIT=1
      shift 2
      ;;
    --skip-prior)
      BUILD_PRIOR=0
      shift
      ;;
    --force-prior)
      FORCE_PRIOR=1
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

if [[ -z "$STATIC_PRIOR_PATH" ]]; then
  STATIC_PRIOR_PATH="$OUTPUT_DIR/priors/coco_static_prior.json"
fi
static_prior_args=(--set "risk_scoring.static_prior_path=$STATIC_PRIOR_PATH")

contains_item() {
  local list="$1"
  local item="$2"
  [[ "$list" == "all" ]] || [[ ",$list," == *",$item,"* ]]
}

dataset_enabled() {
  contains_item "$DATASETS" "$1"
}

method_enabled() {
  contains_item "$METHODS" "$1" \
    || { [[ "$1" == svo_without_* ]] && contains_item "$METHODS" "ablations"; } \
    || { [[ "$1" == svo_only_* ]] && contains_item "$METHODS" "components"; }
}

requires_risk_threshold() {
  method_enabled "svo" \
    || method_enabled "svo_without_uncertainty" \
    || method_enabled "svo_without_position" \
    || method_enabled "svo_without_prior" \
    || method_enabled "svo_only_uncertainty" \
    || method_enabled "svo_only_position" \
    || method_enabled "svo_only_prior" \
    || method_enabled "random_verify"
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

maybe_mkdir_outputs() {
  if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$OUTPUT_DIR"/{predictions,objects,verifications,revisions,metrics,tables,priors}
  fi
}

limit_args=()
if [[ -n "$LIMIT" ]]; then
  limit_args=(--limit "$LIMIT")
fi

pope_limit_args=()
if [[ -n "$LIMIT" ]]; then
  pope_limit_args=(--limit-per-setting "$LIMIT")
fi

verification_limit_args=()
if [[ -n "$LIMIT" ]]; then
  verification_limit_args=(--limit "$LIMIT")
fi

risk_args=()
if [[ -n "$RISK_THRESHOLD" ]]; then
  risk_args=(--risk-threshold "$RISK_THRESHOLD")
elif [[ "$DRY_RUN" -eq 0 ]] && requires_risk_threshold; then
  configured_threshold="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
threshold = (config.get("risk_scoring") or {}).get("threshold")
print("" if threshold is None else threshold)
PY
)"
  if [[ -z "$configured_threshold" ]]; then
    cat >&2 <<'MSG'
ERROR: --risk-threshold is required for real SVO/random-verify runs unless
risk_scoring.threshold is set in the selected config.

Tune on validation first, then rerun for example:
  bash scripts/run_all.sh --risk-threshold <VAL_THRESHOLD>
MSG
    exit 2
  fi
  echo "Using risk_scoring.threshold=$configured_threshold from $CONFIG." >&2
fi

run_static_prior() {
  if [[ -f "$STATIC_PRIOR_PATH" && "$FORCE_PRIOR" -eq 0 ]]; then
    echo "Static prior already exists, skipping: $STATIC_PRIOR_PATH"
    echo "Use --force-prior to rebuild it."
    return
  fi
  local prior_captions="$OUTPUT_DIR/predictions/coco_train2017_val${PRIOR_LIMIT}_base_captions.jsonl"
  local prior_split="${PRIOR_SPLIT_FILE:-configs/splits/coco_train2017_val${PRIOR_LIMIT}_seed42.txt}"
  local prior_image_root="$PRIOR_IMAGE_ROOT"
  if [[ -z "$prior_image_root" ]]; then
    if [[ -d "data/coco/train2017_val${PRIOR_LIMIT}" ]]; then
      prior_image_root="data/coco/train2017_val${PRIOR_LIMIT}"
    else
      prior_image_root="data/coco/train2017"
    fi
  fi
  run_cmd "$PYTHON" scripts/make_val_split.py \
    --coco-annotations data/coco/annotations/instances_train2017.json \
    --sample-size "$PRIOR_LIMIT" \
    --seed 42 \
    --output "$prior_split"
  run_cmd "$PYTHON" scripts/run_caption.py \
    --config "$CONFIG" \
    --dataset configs/datasets/coco_chair.yaml \
    --method configs/methods/base.yaml \
    --output "$prior_captions" \
    --limit "$PRIOR_LIMIT" \
    --set "dataset.paths.image_root=$prior_image_root" \
    --set dataset.paths.annotation_file=data/coco/annotations/instances_train2017.json \
    --set "dataset.paths.split_file=$prior_split"
  run_cmd "$PYTHON" scripts/build_static_prior.py \
    --config "$CONFIG" \
    --captions "$prior_captions" \
    --coco-annotations data/coco/annotations/instances_train2017.json \
    --output "$STATIC_PRIOR_PATH"
}

generate_coco_base_captions() {
  local captions="$OUTPUT_DIR/predictions/coco_chair_base_captions.jsonl"
  run_cmd "$PYTHON" scripts/run_caption.py \
    --config "$CONFIG" \
    --dataset configs/datasets/coco_chair.yaml \
    --method configs/methods/base.yaml \
    --output "$captions" \
    "${limit_args[@]}"
}

evaluate_coco_base() {
  local captions="$OUTPUT_DIR/predictions/coco_chair_base_captions.jsonl"
  run_cmd "$PYTHON" scripts/evaluate.py \
    --config "$CONFIG" \
    --dataset coco_chair \
    --task chair \
    --predictions "$captions" \
    --output "$OUTPUT_DIR/metrics/coco_chair_base_chair.json" \
    --method base
}

extract_coco_objects() {
  local method="$1"
  local captions="$OUTPUT_DIR/predictions/coco_chair_base_captions.jsonl"
  local objects="$OUTPUT_DIR/objects/coco_chair_${method}_objects.jsonl"
  run_cmd "$PYTHON" scripts/extract_objects.py \
    --config "$CONFIG" \
    --method "configs/methods/${method}.yaml" \
    --input "$captions" \
    --output "$objects" \
    "${static_prior_args[@]}"
}

verify_and_revise_coco() {
  local method="$1"
  local objects="$2"
  local reference="${3:-}"
  local verifications="$OUTPUT_DIR/verifications/coco_chair_${method}.jsonl"
  local revisions="$OUTPUT_DIR/revisions/coco_chair_${method}_revisions.jsonl"
  local verify_cmd=("$PYTHON" scripts/verify_objects.py --config "$CONFIG" --method "configs/methods/${method}.yaml" --input "$objects" --output "$verifications" "${verification_limit_args[@]}")
  if [[ "${#risk_args[@]}" -gt 0 && "$method" != "verify_all" ]]; then
    verify_cmd+=("${risk_args[@]}")
  fi
  if [[ -n "$reference" ]]; then
    verify_cmd+=(--reference "$reference")
  fi
  run_cmd "${verify_cmd[@]}"
  run_cmd "$PYTHON" scripts/revise_captions.py \
    --config "$CONFIG" \
    --input "$verifications" \
    --output "$revisions"
  run_cmd "$PYTHON" scripts/evaluate.py \
    --config "$CONFIG" \
    --dataset coco_chair \
    --task chair \
    --predictions "$revisions" \
    --text-field revised_caption \
    --output "$OUTPUT_DIR/metrics/coco_chair_${method}_chair.json" \
    --method "$method"
  run_cmd "$PYTHON" scripts/evaluate.py \
    --config "$CONFIG" \
    --dataset coco_chair \
    --task efficiency \
    --objects "$objects" \
    --verifications "$verifications" \
    --base-predictions "$OUTPUT_DIR/predictions/coco_chair_base_captions.jsonl" \
    --output "$OUTPUT_DIR/metrics/coco_chair_${method}_efficiency.json" \
    --method "$method"
  run_cmd "$PYTHON" scripts/evaluate.py \
    --config "$CONFIG" \
    --dataset coco_chair \
    --task false_correction \
    --predictions "$revisions" \
    --output "$OUTPUT_DIR/metrics/coco_chair_${method}_false_correction.json" \
    --method "$method"
}

verify_coco_reference() {
  local objects="$1"
  local reference="$OUTPUT_DIR/verifications/coco_chair_svo.jsonl"
  local verify_cmd=("$PYTHON" scripts/verify_objects.py --config "$CONFIG" --method configs/methods/svo.yaml --input "$objects" --output "$reference" "${verification_limit_args[@]}")
  if [[ "${#risk_args[@]}" -gt 0 ]]; then
    verify_cmd+=("${risk_args[@]}")
  fi
  run_cmd "${verify_cmd[@]}"
}

run_coco_methods() {
  if [[ "$BUILD_PRIOR" -eq 1 ]]; then
    run_static_prior
  fi
  if method_enabled "base" || method_enabled "svo" || method_enabled "verify_all" || method_enabled "random_verify" || method_enabled "svo_without_uncertainty" || method_enabled "svo_without_position" || method_enabled "svo_without_prior" || method_enabled "svo_only_uncertainty" || method_enabled "svo_only_position" || method_enabled "svo_only_prior"; then
    generate_coco_base_captions
  fi
  if method_enabled "base"; then
    evaluate_coco_base
  fi

  local svo_objects="$OUTPUT_DIR/objects/coco_chair_svo_objects.jsonl"
  if method_enabled "svo" || method_enabled "verify_all" || method_enabled "random_verify"; then
    extract_coco_objects "svo"
  fi
  if method_enabled "svo"; then
    verify_and_revise_coco "svo" "$svo_objects"
  fi
  if method_enabled "verify_all"; then
    verify_and_revise_coco "verify_all" "$svo_objects"
  fi
  if method_enabled "random_verify"; then
    if ! method_enabled "svo"; then
      verify_coco_reference "$svo_objects"
    fi
    verify_and_revise_coco "random_verify" "$svo_objects" "$OUTPUT_DIR/verifications/coco_chair_svo.jsonl"
  fi

  for ablation in svo_without_uncertainty svo_without_position svo_without_prior; do
    if method_enabled "$ablation"; then
      extract_coco_objects "$ablation"
      verify_and_revise_coco "$ablation" "$OUTPUT_DIR/objects/coco_chair_${ablation}_objects.jsonl"
    fi
  done
  for component in svo_only_uncertainty svo_only_position svo_only_prior; do
    if method_enabled "$component"; then
      extract_coco_objects "$component"
      verify_and_revise_coco "$component" "$OUTPUT_DIR/objects/coco_chair_${component}_objects.jsonl"
    fi
  done
}

run_pope_pipeline() {
  local predictions="$OUTPUT_DIR/predictions/pope_base_pope.jsonl"
  if method_enabled "base" || method_enabled "svo"; then
    run_cmd "$PYTHON" scripts/run_pope.py \
      --config "$CONFIG" \
      --dataset configs/datasets/pope.yaml \
      --method configs/methods/base.yaml \
      --output "$predictions" \
      "${pope_limit_args[@]}"
  fi

  if method_enabled "base"; then
    run_cmd "$PYTHON" scripts/evaluate.py \
      --config "$CONFIG" \
      --dataset pope \
      --task pope \
      --predictions "$predictions" \
      --output "$OUTPUT_DIR/metrics/pope_base_pope.json" \
      --method base
  fi

  if method_enabled "svo"; then
    local objects="$OUTPUT_DIR/objects/pope_svo_objects.jsonl"
    local verifications="$OUTPUT_DIR/verifications/pope_svo.jsonl"
    local revisions="$OUTPUT_DIR/revisions/pope_svo_revised.jsonl"
    run_cmd "$PYTHON" scripts/extract_objects.py \
      --config "$CONFIG" \
      --method configs/methods/svo.yaml \
      --input "$predictions" \
      --text-field question \
      --target-field target_object \
      --output "$objects" \
      "${static_prior_args[@]}"
    run_cmd "$PYTHON" scripts/verify_objects.py \
      --config "$CONFIG" \
      --method configs/methods/svo.yaml \
      --input "$objects" \
      --output "$verifications" \
      "${risk_args[@]}" \
      "${verification_limit_args[@]}"
    run_cmd "$PYTHON" scripts/revise_pope.py \
      --config "$CONFIG" \
      --predictions "$predictions" \
      --verifications "$verifications" \
      --output "$revisions"
    run_cmd "$PYTHON" scripts/evaluate.py \
      --config "$CONFIG" \
      --dataset pope \
      --task pope \
      --predictions "$revisions" \
      --output "$OUTPUT_DIR/metrics/pope_svo_pope.json" \
      --method svo
  fi
}

run_amber_eval() {
  local predictions="${AMBER_PREDICTIONS:-}"
  if [[ -z "$predictions" ]]; then
    echo "Skipping AMBER: set AMBER_PREDICTIONS=/path/to/amber_predictions.jsonl to evaluate." >&2
    return
  fi
  run_cmd "$PYTHON" scripts/evaluate.py \
    --config "$CONFIG" \
    --dataset amber_object \
    --task amber \
    --predictions "$predictions" \
    --output "$OUTPUT_DIR/metrics/amber_object_svo_amber.json"
}

maybe_mkdir_outputs

if dataset_enabled "coco_chair"; then
  run_coco_methods
fi
if dataset_enabled "pope"; then
  run_pope_pipeline
fi
if dataset_enabled "amber_object"; then
  run_amber_eval
fi

run_cmd "$PYTHON" scripts/export_results.py \
  --config "$CONFIG" \
  --metrics-dir "$OUTPUT_DIR/metrics" \
  --out "$OUTPUT_DIR/tables" \
  --missing-value NA
