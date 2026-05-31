# Cloud Setup

Target platform: Ubuntu 22.04/24.04, NVIDIA CUDA 12.x, conda or mamba, Python 3.10.

## 1. Clone and Create the Environment

```bash
git clone <your-repo-url> paper_reproduce
cd paper_reproduce
bash scripts/setup_cloud.sh --env-name SVO --python 3.10
conda activate SVO
```

For real model inference on CUDA 12:

```bash
bash scripts/setup_cloud.sh --env-name SVO --with-models
```

The model stack is pinned for CUDA 12.1 in `requirements-models-cu12.txt`. This avoids a
PyTorch/CUDA mismatch when GroundingDINO compiles its CUDA extension with `/usr/local/cuda`.
If your host only provides a different CUDA toolkit, update the PyTorch pins and rebuild
GroundingDINO in the same environment.

spaCy extraction is installed by default because `configs/default.yaml` uses `backend: spacy`.
For dependency-light smoke/test-only setup, skip it:

```bash
bash scripts/setup_cloud.sh --env-name SVO --without-nlp
```

## 2. Check the Environment

```bash
bash scripts/setup_cloud.sh --check-only
python scripts/check_assets.py
bash scripts/smoke_test.sh
```

The smoke test uses toy files under `examples/smoke/` and does not load LLaVA, torch, or
GroundingDINO.

## 3. Real Experiment Readiness

Before running real experiments, make sure:

- COCO, POPE, and AMBER paths match `configs/datasets/*.yaml`.
- LLaVA and GroundingDINO paths are either configured in `configs/default.yaml` or passed via
  `--set` overrides.
- `risk_scoring.threshold` is tuned on validation data or passed with `--risk-threshold`.
- `outputs/` is empty or intentionally being resumed.

On AutoDL/SeeTaCloud-style instances, the prepared helper scripts are:

```bash
bash scripts/run_gpu_setup.sh
bash scripts/run_coco_main_autodl.sh --dry-run
```

`run_gpu_setup.sh` installs the pinned CUDA 12.1 model stack, installs GroundingDINO with
`--no-build-isolation`, verifies local LLaVA processor loading, and checks the prepared assets.
