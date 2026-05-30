#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
CONFIRM=0
DOWNLOAD_COCO_REQUIRED=0

usage() {
  cat <<'EOF'
Usage: bash scripts/prepare_data.sh [options]

Create the expected data layout and optionally download public COCO files.
POPE and AMBER require manual placement according to their licenses/sources.

Options:
  --data-root DIR              Data root. Default: ./data
  --download-coco-required     Download COCO val2014, train2017, and annotations.
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
  download_and_unzip "http://images.cocodataset.org/zips/train2017.zip" "$DATA_ROOT/coco/train2017.zip" "$DATA_ROOT/coco"
  download_and_unzip "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" "$DATA_ROOT/coco/annotations_trainval2014.zip" "$DATA_ROOT/coco"
  download_and_unzip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$DATA_ROOT/coco/annotations_trainval2017.zip" "$DATA_ROOT/coco"
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

Update configs/datasets/*.yaml if your paths differ.
EOF
