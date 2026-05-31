#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models}"
LLAVA_ID="${LLAVA_MODEL_ID:-llava-hf/llava-1.5-7b-hf}"
LLAVA_DIR="${LLAVA_LOCAL_DIR:-$MODEL_ROOT/llava-1.5-7b-hf}"
GROUNDINGDINO_REPO_DIR="${GROUNDINGDINO_REPO_DIR:-$MODEL_ROOT/GroundingDINO}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-$MODEL_ROOT/groundingdino_swint_ogc.pth}"
GROUNDINGDINO_CHECKPOINT_URL="${GROUNDINGDINO_CHECKPOINT_URL:-https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth}"
GROUNDINGDINO_CHECKPOINT_HF_REPO="${GROUNDINGDINO_CHECKPOINT_HF_REPO:-ShilongLiu/GroundingDINO}"
GROUNDINGDINO_CHECKPOINT_HF_FILE="${GROUNDINGDINO_CHECKPOINT_HF_FILE:-groundingdino_swint_ogc.pth}"
GROUNDINGDINO_CHECKPOINT_SOURCE="${GROUNDINGDINO_CHECKPOINT_SOURCE:-url}"
CONFIRM=0
SKIP_LLAVA=0
SKIP_GROUNDINGDINO=0
INSTALL_GROUNDINGDINO=0

usage() {
  cat <<'EOF'
Usage: bash scripts/download_models.sh --confirm [options]

Download model assets needed for real experiments. This script never runs unless --confirm is set.

Options:
  --confirm                  Confirm that large downloads are allowed.
  --model-root DIR           Destination model root. Default: ./models
  --llava-id ID              Hugging Face model id. Default: llava-hf/llava-1.5-7b-hf
  --groundingdino-source SRC  Checkpoint source: url, hf, or auto. Default: url
  --skip-llava               Do not download LLaVA.
  --skip-groundingdino       Do not clone/download GroundingDINO assets.
  --install-groundingdino    Run pip install -e on the cloned GroundingDINO repo.
  -h, --help                 Show this help.

Environment variables:
  HF_ENDPOINT                Optional Hugging Face endpoint/mirror used by hf download.
  HF_TOKEN                   Optional Hugging Face token used by hf/huggingface-cli.
  GROUNDINGDINO_CHECKPOINT_SOURCE  url, hf, or auto.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm)
      CONFIRM=1
      shift
      ;;
    --model-root)
      MODEL_ROOT="$2"
      LLAVA_DIR="$MODEL_ROOT/llava-1.5-7b-hf"
      GROUNDINGDINO_REPO_DIR="$MODEL_ROOT/GroundingDINO"
      GROUNDINGDINO_CHECKPOINT="$MODEL_ROOT/groundingdino_swint_ogc.pth"
      shift 2
      ;;
    --llava-id)
      LLAVA_ID="$2"
      shift 2
      ;;
    --groundingdino-source)
      GROUNDINGDINO_CHECKPOINT_SOURCE="$2"
      shift 2
      ;;
    --skip-llava)
      SKIP_LLAVA=1
      shift
      ;;
    --skip-groundingdino)
      SKIP_GROUNDINGDINO=1
      shift
      ;;
    --install-groundingdino)
      INSTALL_GROUNDINGDINO=1
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

if [[ "$CONFIRM" -ne 1 ]]; then
  echo "Refusing to download model assets without --confirm." >&2
  usage >&2
  exit 2
fi

mkdir -p "$MODEL_ROOT"

download_file() {
  local url="$1"
  local output="$2"
  if [[ -f "$output" ]]; then
    echo "Exists: $output"
    return
  fi
  mkdir -p "$(dirname "$output")"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --continue-at - "$url" -o "$output"
  elif command -v wget >/dev/null 2>&1; then
    wget -c "$url" -O "$output"
  else
    echo "Neither curl nor wget is available to download $url" >&2
    exit 1
  fi
}

download_hf_model() {
  local repo_id="$1"
  local local_dir="$2"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo_id" --local-dir "$local_dir"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo_id" --local-dir "$local_dir"
  else
    echo "No Hugging Face downloader found. Install huggingface_hub first:" >&2
    echo "  python -m pip install -U huggingface_hub" >&2
    exit 1
  fi
}

download_hf_file() {
  local repo_id="$1"
  local filename="$2"
  local output="$3"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo_id" "$filename" --local-dir "$tmp_dir"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo_id" "$filename" --local-dir "$tmp_dir"
  else
    rm -rf "$tmp_dir"
    return 127
  fi
  mkdir -p "$(dirname "$output")"
  mv "$tmp_dir/$filename" "$output"
  rm -rf "$tmp_dir"
}

if [[ "$SKIP_LLAVA" -eq 0 ]]; then
  mkdir -p "$LLAVA_DIR"
  echo "Downloading LLaVA model: $LLAVA_ID"
  download_hf_model "$LLAVA_ID" "$LLAVA_DIR"
fi

if [[ "$SKIP_GROUNDINGDINO" -eq 0 ]]; then
  if [[ -d "$GROUNDINGDINO_REPO_DIR/.git" ]]; then
    echo "GroundingDINO repo exists: $GROUNDINGDINO_REPO_DIR"
  else
    git clone https://github.com/IDEA-Research/GroundingDINO.git "$GROUNDINGDINO_REPO_DIR"
  fi
  case "$GROUNDINGDINO_CHECKPOINT_SOURCE" in
    url)
      download_file "$GROUNDINGDINO_CHECKPOINT_URL" "$GROUNDINGDINO_CHECKPOINT"
      ;;
    hf)
      download_hf_file "$GROUNDINGDINO_CHECKPOINT_HF_REPO" "$GROUNDINGDINO_CHECKPOINT_HF_FILE" "$GROUNDINGDINO_CHECKPOINT"
      ;;
    auto)
      if ! download_hf_file "$GROUNDINGDINO_CHECKPOINT_HF_REPO" "$GROUNDINGDINO_CHECKPOINT_HF_FILE" "$GROUNDINGDINO_CHECKPOINT"; then
        echo "HF checkpoint download failed; falling back to URL." >&2
        download_file "$GROUNDINGDINO_CHECKPOINT_URL" "$GROUNDINGDINO_CHECKPOINT"
      fi
      ;;
    *)
      echo "Unsupported --groundingdino-source: $GROUNDINGDINO_CHECKPOINT_SOURCE" >&2
      exit 2
      ;;
  esac
  if [[ "$INSTALL_GROUNDINGDINO" -eq 1 ]]; then
    python -m pip install -e "$GROUNDINGDINO_REPO_DIR" --no-build-isolation
  fi
fi

cat <<EOF

Model assets prepared.

Use these config overrides if you keep the default layout:
  --set generation.model_name_or_path=$LLAVA_DIR
  --set verification.groundingdino.config_path=$GROUNDINGDINO_REPO_DIR/groundingdino/config/GroundingDINO_SwinT_OGC.py
  --set verification.groundingdino.checkpoint_path=$GROUNDINGDINO_CHECKPOINT
EOF
