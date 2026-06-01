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
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc is not available on PATH; GroundingDINO CUDA extension builds may fail." >&2
fi
"$ENV_PYTHON" -m pip install -U -r requirements-models-cu12.txt
"$ENV_PYTHON" -m pip install -U ninja wheel
"$ENV_PYTHON" -m pip uninstall -y groundingdino || true
rm -rf models/GroundingDINO/build
find models/GroundingDINO/groundingdino -name '*.so' -type f -delete
"$ENV_PYTHON" -m pip install -e models/GroundingDINO --no-build-isolation --force-reinstall --no-deps

"$ENV_PYTHON" - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available after installing the model stack.")
PY

"$ENV_PYTHON" - <<'PY'
from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoTokenizer, BertModel

config = AutoConfig.from_pretrained("models/llava-1.5-7b-hf", local_files_only=True)
processor = AutoProcessor.from_pretrained("models/llava-1.5-7b-hf", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
bert = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
print("llava_config_model_type", config.model_type)
print("llava_processor", processor.__class__.__name__)
print("llava_class", LlavaForConditionalGeneration.__name__)
print("groundingdino_text_tokenizer", tokenizer.__class__.__name__)
print("groundingdino_text_encoder", bert.__class__.__name__)
PY

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$ENV_PYTHON" - <<'PY'
from groundingdino.util.inference import load_model, load_image, predict

model = load_model(
    "models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "models/groundingdino_swint_ogc.pth",
    device="cuda",
)
print("groundingdino_inference_imports_ok", bool(load_model and load_image and predict))
print("groundingdino_model", model.__class__.__name__)
PY

"$ENV_PYTHON" scripts/check_assets.py \
  --strict \
  --sha256 \
  --assets llava,groundingdino,groundingdino_text_encoder \
  --datasets coco_chair
echo "GPU model stack is installed and assets are visible."
