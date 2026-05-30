#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
CONFIRM=0
DOWNLOAD_COCO_REQUIRED=0
DOWNLOAD_COCO_ANNOTATIONS=0
DOWNLOAD_TRAIN2017_FULL=0
PREPARE_TRAIN2017_SUBSET=0
TRAIN2017_SUBSET_SIZE=5000
TRAIN2017_SPLIT_FILE=""
TRAIN2017_SOURCE_ROOT=""
DOWNLOAD_MISSING_SUBSET=0

usage() {
  cat <<'EOF'
Usage: bash scripts/prepare_data.sh [options]

Create the expected data layout and optionally download public COCO files.
POPE and AMBER require manual placement according to their licenses/sources.

Options:
  --data-root DIR              Data root. Default: ./data
  --download-coco-required     Download COCO val2014 images and full annotations.
  --download-coco-annotations  Download only COCO annotation ZIP files.
  --download-train2017-full    Also download full COCO train2017 images.
  --prepare-train2017-subset   Prepare train2017 validation subset images from split ids.
  --subset-size N              Subset size when creating split. Default: 5000
  --split-file PATH            Split file path. Default: configs/splits/coco_train2017_val<N>_seed42.txt
  --source-image-root DIR      Full train2017 directory to copy from. Default: data/coco/train2017
  --download-missing-subset    Download subset images missing from source-image-root.
  --confirm                    Required for any download.
  -h, --help                   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --download-coco-required)
      DOWNLOAD_COCO_REQUIRED=1
      shift
      ;;
    --download-coco-annotations)
      DOWNLOAD_COCO_ANNOTATIONS=1
      shift
      ;;
    --download-train2017-full)
      DOWNLOAD_TRAIN2017_FULL=1
      shift
      ;;
    --prepare-train2017-subset)
      PREPARE_TRAIN2017_SUBSET=1
      shift
      ;;
    --subset-size)
      TRAIN2017_SUBSET_SIZE="$2"
      shift 2
      ;;
    --split-file)
      TRAIN2017_SPLIT_FILE="$2"
      shift 2
      ;;
    --source-image-root)
      TRAIN2017_SOURCE_ROOT="$2"
      shift 2
      ;;
    --download-missing-subset)
      DOWNLOAD_MISSING_SUBSET=1
      shift
      ;;
    --confirm)
      CONFIRM=1
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

mkdir -p \
  "$DATA_ROOT/coco/annotations" \
  "$DATA_ROOT/coco/train2017" \
  "$DATA_ROOT/coco/val2014" \
  "$DATA_ROOT/pope" \
  "$DATA_ROOT/amber/images"

download_and_unzip() {
  local url="$1"
  local zip_path="$2"
  local dest="$3"
  if [[ "$CONFIRM" -ne 1 ]]; then
    echo "Refusing to download $url without --confirm." >&2
    exit 2
  fi
  mkdir -p "$(dirname "$zip_path")" "$dest"
  if [[ ! -f "$zip_path" ]]; then
    if command -v curl >/dev/null 2>&1; then
      curl -L --fail "$url" -o "$zip_path"
    elif command -v wget >/dev/null 2>&1; then
      wget "$url" -O "$zip_path"
    else
      echo "Neither curl nor wget is available." >&2
      exit 1
    fi
  fi
  unzip -n "$zip_path" -d "$dest"
}

if [[ "$DOWNLOAD_COCO_REQUIRED" -eq 1 ]]; then
  download_and_unzip "http://images.cocodataset.org/zips/val2014.zip" "$DATA_ROOT/coco/val2014.zip" "$DATA_ROOT/coco"
  DOWNLOAD_COCO_ANNOTATIONS=1
fi

if [[ "$DOWNLOAD_TRAIN2017_FULL" -eq 1 ]]; then
  download_and_unzip "http://images.cocodataset.org/zips/train2017.zip" "$DATA_ROOT/coco/train2017.zip" "$DATA_ROOT/coco"
fi

if [[ "$DOWNLOAD_COCO_ANNOTATIONS" -eq 1 ]]; then
  download_and_unzip "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" "$DATA_ROOT/coco/annotations_trainval2014.zip" "$DATA_ROOT/coco"
  download_and_unzip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$DATA_ROOT/coco/annotations_trainval2017.zip" "$DATA_ROOT/coco"
fi

if [[ "$PREPARE_TRAIN2017_SUBSET" -eq 1 ]]; then
  split_file="${TRAIN2017_SPLIT_FILE:-configs/splits/coco_train2017_val${TRAIN2017_SUBSET_SIZE}_seed42.txt}"
  source_root="${TRAIN2017_SOURCE_ROOT:-$DATA_ROOT/coco/train2017}"
  subset_args=(
    python "$ROOT_DIR/scripts/prepare_coco_subset.py"
    --coco-annotations "$DATA_ROOT/coco/annotations/instances_train2017.json"
    --split-file "$split_file"
    --sample-size "$TRAIN2017_SUBSET_SIZE"
    --output-image-root "$DATA_ROOT/coco/train2017_val${TRAIN2017_SUBSET_SIZE}"
    --source-image-root "$source_root"
  )
  if [[ "$DOWNLOAD_MISSING_SUBSET" -eq 1 ]]; then
    subset_args+=(--download-missing)
  fi
  if [[ "$CONFIRM" -eq 1 ]]; then
    subset_args+=(--confirm)
  fi
  "${subset_args[@]}"
fi

cat <<EOF
Data directories are ready under:
  $DATA_ROOT

Expected remaining manual files:
  $DATA_ROOT/pope/random.jsonl
  $DATA_ROOT/pope/popular.jsonl
  $DATA_ROOT/pope/adversarial.jsonl
  $DATA_ROOT/amber/object_subset.jsonl
  $DATA_ROOT/amber/images/

Optional train2017 validation subset:
  $DATA_ROOT/coco/train2017_val${TRAIN2017_SUBSET_SIZE}/
  configs/splits/coco_train2017_val${TRAIN2017_SUBSET_SIZE}_seed42.txt

Update configs/datasets/*.yaml if your paths differ.
EOF
