#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models}"
LLAVA_ID="${LLAVA_MODEL_ID:-llava-hf/llava-1.5-7b-hf}"
LLAVA_DIR="${LLAVA_LOCAL_DIR:-$MODEL_ROOT/llava-1.5-7b-hf}"
GROUNDINGDINO_REPO_DIR="${GROUNDINGDINO_REPO_DIR:-$MODEL_ROOT/GroundingDINO}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-$MODEL_ROOT/groundingdino_swint_ogc.pth}"
GROUNDINGDINO_CHECKPOINT_URL="${GROUNDINGDINO_CHECKPOINT_URL:-https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth}"
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
  --skip-llava               Do not download LLaVA.
  --skip-groundingdino       Do not clone/download GroundingDINO assets.
  --install-groundingdino    Run pip install -e on the cloned GroundingDINO repo.
  -h, --help                 Show this help.

Environment variables:
  HF_TOKEN                   Optional Hugging Face token used by huggingface-cli.
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
    curl -L --fail "$url" -o "$output"
  elif command -v wget >/dev/null 2>&1; then
    wget "$url" -O "$output"
  else
    echo "Neither curl nor wget is available to download $url" >&2
    exit 1
  fi
}

if [[ "$SKIP_LLAVA" -eq 0 ]]; then
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found. Install huggingface_hub first:" >&2
    echo "  python -m pip install huggingface_hub" >&2
    exit 1
  fi
  mkdir -p "$LLAVA_DIR"
  echo "Downloading LLaVA model: $LLAVA_ID"
  huggingface-cli download "$LLAVA_ID" --local-dir "$LLAVA_DIR"
fi

if [[ "$SKIP_GROUNDINGDINO" -eq 0 ]]; then
  if [[ -d "$GROUNDINGDINO_REPO_DIR/.git" ]]; then
    echo "GroundingDINO repo exists: $GROUNDINGDINO_REPO_DIR"
  else
    git clone https://github.com/IDEA-Research/GroundingDINO.git "$GROUNDINGDINO_REPO_DIR"
  fi
  download_file "$GROUNDINGDINO_CHECKPOINT_URL" "$GROUNDINGDINO_CHECKPOINT"
  if [[ "$INSTALL_GROUNDINGDINO" -eq 1 ]]; then
    python -m pip install -e "$GROUNDINGDINO_REPO_DIR"
  fi
fi

cat <<EOF

Model assets prepared.

Use these config overrides if you keep the default layout:
  --set generation.model_name_or_path=$LLAVA_DIR
  --set verification.groundingdino.config_path=$GROUNDINGDINO_REPO_DIR/groundingdino/config/GroundingDINO_SwinT_OGC.py
  --set verification.groundingdino.checkpoint_path=$GROUNDINGDINO_CHECKPOINT
EOF
