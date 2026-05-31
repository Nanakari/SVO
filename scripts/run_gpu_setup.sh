#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_PYTHON="${ENV_PYTHON:-/root/autodl-tmp/conda_envs/SVO/bin/python}"
export PATH="$(dirname "$ENV_PYTHON"):$PATH"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip_cache}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available; wait until the GPU is attached." >&2
  exit 1
fi

nvidia-smi
"$ENV_PYTHON" -m pip install -U -r requirements-models-cu12.txt
"$ENV_PYTHON" -m pip install -U ninja wheel
"$ENV_PYTHON" -m pip install -e models/GroundingDINO --no-build-isolation

"$ENV_PYTHON" - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available after installing the model stack.")
PY

"$ENV_PYTHON" - <<'PY'
from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration
from groundingdino.util.inference import load_model, load_image, predict

config = AutoConfig.from_pretrained("models/llava-1.5-7b-hf", local_files_only=True)
processor = AutoProcessor.from_pretrained("models/llava-1.5-7b-hf", local_files_only=True)
print("llava_config_model_type", config.model_type)
print("llava_processor", processor.__class__.__name__)
print("llava_class", LlavaForConditionalGeneration.__name__)
print("groundingdino_inference_imports_ok", bool(load_model and load_image and predict))
PY

"$ENV_PYTHON" scripts/check_assets.py --sha256
echo "GPU model stack is installed and assets are visible."
