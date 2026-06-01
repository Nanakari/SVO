# SVO

SVO is a reproducible experiment scaffold for Selective Verification of Objects: a
training-free post-processing pipeline for reducing object hallucinations in LVLM captions and
object-existence answers.

The public project name is `SVO`. The Python package is still named `paper_reproduce` internally to
avoid a broad import migration; that package name is not the display name of the project.

## Quick Start

Target cloud environment: Ubuntu 22.04/24.04, Python 3.10, CUDA toolkit 12.1 for real GPU runs.

```bash
git clone https://github.com/Nanakari/SVO.git
cd SVO
bash scripts/setup_cloud.sh --env-name SVO --python 3.10
conda activate SVO
bash scripts/smoke_test.sh
bash scripts/run_all.sh --dry-run
```

The smoke test uses toy fixtures and does not load LLaVA, torch, or GroundingDINO.

`requirements.txt` is for development and tests; it intentionally does not install the CUDA model
stack. For real LLaVA/GroundingDINO runs, install `requirements-models-cu12.txt` on the GPU host.

## Model And Data Assets

Prepare only the assets needed for the COCO/CHAIR workflow:

```bash
python -m pip install -U huggingface_hub
bash scripts/download_models.sh --confirm
bash scripts/prepare_data.sh --download-coco-required --confirm
bash scripts/prepare_data.sh \
  --prepare-train2017-subset \
  --subset-size 5000 \
  --download-missing-subset \
  --confirm
```

Expected model assets:

- `models/llava-1.5-7b-hf/`
- `models/GroundingDINO/`
- `models/groundingdino_swint_ogc.pth`
- Hugging Face cache entries for `bert-base-uncased`, used by GroundingDINO's text encoder

Expected COCO assets:

- `data/coco/val2014/`
- `data/coco/annotations/instances_val2014.json`
- `data/coco/annotations/captions_val2014.json`
- `data/coco/annotations/instances_train2017.json`
- `data/coco/annotations/captions_train2017.json`
- `data/coco/train2017_val5000/`
- `configs/splits/coco_train2017_val5000_seed42.txt`

COCO-only asset check:

```bash
python scripts/check_assets.py \
  --strict \
  --assets llava,groundingdino,groundingdino_text_encoder \
  --datasets coco_chair
```

Do not use a full strict check unless POPE and AMBER files are also prepared.

## CUDA Reproduction

Main installation entry:

```bash
python -m pip install -r requirements-models-cu12.txt
python -m pip install -U ninja wheel
python -m pip install -e models/GroundingDINO --no-build-isolation --no-deps
```

Verified cloud combination:

| CUDA toolkit used by `nvcc` | PyTorch wheels | Notes |
| --- | --- | --- |
| 12.1 | `torch==2.4.1+cu121`, `torchvision==0.19.1+cu121` | Main supported setup |

NVIDIA drivers may report a newer CUDA runtime in `nvidia-smi`; GroundingDINO compilation should
still follow `/usr/local/cuda/bin/nvcc`. Do not automatically upgrade to cu13 wheels.

`requirements-lock-cu121.txt` is a reference lock for the tested CUDA 12.1 cloud stack. Keep
`requirements-models-cu12.txt` as the default install entry unless you intentionally need the
stricter reference file.

## Run Experiments

Tune the SVO risk threshold on the validation-only COCO train2017 subset:

```bash
bash scripts/tune_svo_threshold.sh \
  --config configs/default_autodl.yaml \
  --thresholds "0.5 1.0 1.5 2.0" \
  --gpu 0
```

If captions, priors, and objects already exist, rerun only the threshold sweep:

```bash
bash scripts/tune_svo_threshold.sh \
  --config configs/default_autodl.yaml \
  --sweep-only \
  --thresholds "0.5 1.0 1.5 2.0" \
  --gpu 0
```

Run the main COCO/POPE workflow after selecting a validation threshold:

```bash
bash scripts/run_all.sh \
  --config configs/default_autodl.yaml \
  --datasets coco_chair,pope \
  --methods base,svo,verify_all,random_verify \
  --risk-threshold <VAL_THRESHOLD> \
  --gpu 0
```

AutoDL/SeeTaCloud helper:

```bash
bash scripts/run_gpu_setup.sh
SVO_RISK_THRESHOLD=<VAL_THRESHOLD> bash scripts/run_coco_main_autodl.sh --main-only
```

`run_all.sh` fails fast for real SVO/random-verify runs if neither `--risk-threshold` nor
`risk_scoring.threshold` is set. Existing static priors are reused automatically; pass
`--force-prior` only when you intentionally want to rebuild them.

Caption and POPE generation check the first 100 pending image paths before loading LLaVA. Use
`--check-images all` for full preflight or `--check-images none` to disable the check.

## Evaluation

`scripts/evaluate.py --task chair` defaults to an official-compatible CHAIR backend:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --dataset coco_chair \
  --task chair \
  --chair-backend official \
  --predictions outputs/revisions/coco_chair_svo_revisions.jsonl \
  --text-field revised_caption
```

The official-compatible scorer follows the original CHAIR metric behavior for COCO object
synonyms, tokenization, special cases, and image-level ground truth from COCO instances plus
reference captions when available. The previous project-local extractor scorer remains available
only as an explicit fallback.

The compatibility target is the CHAIR scorer from
[LisaAnne/Hallucination](https://github.com/LisaAnne/Hallucination). Because that upstream
repository has no explicit license, this project keeps a small compatible implementation and wrapper
instead of vendoring the external repository.

To use the internal fallback explicitly:

```bash
python scripts/evaluate.py ... --task chair --chair-backend internal
```

Main reported CHAIR results should use the official-compatible `--chair-backend official`.

To check local parity against a cloned LisaAnne/Hallucination repository:

```bash
python scripts/check_chair_parity.py --official-chair-repo /path/to/Hallucination
```

POPE evaluation uses the public POPE script-style answer normalizer by default:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --dataset pope \
  --task pope \
  --pope-normalizer official \
  --predictions outputs/revisions/pope_svo_revised.jsonl
```

## Repository Layout

```text
configs/                 Experiment, dataset, method, and vocabulary configs
docs/                    Reproduction notes and result schema docs
examples/smoke/          No-model smoke fixtures
scripts/                 CLI entry points and cloud helpers
src/paper_reproduce/     Internal Python package
tests/                   Unit tests
```

Ignored local directories include `data/`, `models/`, and `outputs/`. Do not commit model weights,
datasets, or generated experiment outputs.

## Verification Commands

```bash
python -m compileall -q src scripts tests
python -m pytest
git diff --check
```

On Linux or Git Bash:

```bash
bash -n scripts/*.sh
bash scripts/smoke_test.sh
```

Additional docs:

- `docs/cloud_setup.md`
- `docs/reproduction.md`
- `docs/model_zoo.md`
- `docs/troubleshooting.md`
- `docs/result_schemas.md`
