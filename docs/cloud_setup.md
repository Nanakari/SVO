# Cloud Setup

Target platform: Ubuntu 22.04/24.04, NVIDIA CUDA 12.x, conda or mamba, Python 3.10.

## 1. Clone and Create the Environment

```bash
git clone https://github.com/Nanakari/SVO.git
cd SVO
bash scripts/setup_cloud.sh --env-name SVO --python 3.10
conda activate SVO
```

For real model inference on CUDA 12:

```bash
bash scripts/setup_cloud.sh --env-name SVO --with-models
```

The model stack is pinned for CUDA 12.1 in `requirements-models-cu12.txt`.
`requirements-lock-cu121.txt` records the stricter reference stack used on the cloud host.

| Host component | Use | Notes |
| --- | --- | --- |
| CUDA toolkit 12.1 from `/usr/local/cuda/bin/nvcc` | cu121 wheels | Verified setup |
| `torch==2.4.1+cu121` | with `torchvision==0.19.1+cu121` | Main supported pair |
| Driver reports CUDA runtime 13.0 | still use cu121 wheels | Follow `nvcc`, not the runtime label |

Do not automatically upgrade to cu13 wheels. GroundingDINO compiles against the local CUDA toolkit,
so rebuild its extension after changing torch or CUDA.

spaCy extraction is installed by default because `configs/default.yaml` uses `backend: spacy`.
For dependency-light smoke/test-only setup, skip it:

```bash
bash scripts/setup_cloud.sh --env-name SVO --without-nlp
```

## 2. Check the Environment

```bash
bash scripts/setup_cloud.sh --check-only
python scripts/check_assets.py \
  --assets llava,groundingdino,groundingdino_text_encoder \
  --datasets coco_chair
bash scripts/smoke_test.sh
```

The smoke test uses toy files under `examples/smoke/` and does not load LLaVA, torch, or
GroundingDINO.

## 3. Real Experiment Readiness

Before running real experiments, make sure:

- COCO, POPE, and AMBER paths match `configs/datasets/*.yaml`.
- LLaVA and GroundingDINO paths are either configured in `configs/default.yaml` or passed via
  `--set` overrides.
- `bert-base-uncased` is present in the Hugging Face cache for GroundingDINO's BERT text encoder.
- The GroundingDINO commit printed by `scripts/download_models.sh` is saved in the experiment log.
- `risk_scoring.threshold` is tuned on validation data or passed with `--risk-threshold`.
- `outputs/` is empty or intentionally being resumed.

On AutoDL/SeeTaCloud-style instances, the prepared helper scripts are:

```bash
bash scripts/run_gpu_setup.sh
bash scripts/run_coco_main_autodl.sh --dry-run
```

`run_gpu_setup.sh` installs the pinned CUDA 12.1 model stack, rebuilds GroundingDINO with
`--no-build-isolation --no-deps`, verifies local LLaVA processor loading, and checks the prepared
assets.
