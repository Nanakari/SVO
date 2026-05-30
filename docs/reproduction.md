# Reproduction Workflow

This page describes the intended cloud workflow. It generates outputs only from commands you run.

## 1. Sanity Checks

```bash
conda activate SVO
bash scripts/setup_cloud.sh --check-only
python scripts/check_assets.py
bash scripts/smoke_test.sh
bash scripts/run_all.sh --dry-run
```

## 2. Prepare Real Assets

```bash
bash scripts/prepare_data.sh --download-coco-required --confirm
bash scripts/download_models.sh --confirm --install-groundingdino
python scripts/check_assets.py --strict
```

Update `configs/default.yaml` or pass `--set` overrides for local model paths.

## 3. Tune SVO Threshold on Validation Data

Generate validation captions and static priors, then choose the SVO threshold from validation metrics
only. The repository does not include a hard-coded paper threshold.

```bash
bash scripts/run_all.sh --dry-run --datasets coco_chair --methods base,svo --risk-threshold <VAL_THRESHOLD>
```

Replace `<VAL_THRESHOLD>` only after validating it on the COCO train2017 validation split.

## 4. Run Main Experiments

```bash
bash scripts/run_all.sh \
  --datasets coco_chair,pope \
  --methods base,svo,verify_all,random_verify \
  --risk-threshold <VAL_THRESHOLD> \
  --gpu 0
```

Risk-score ablations:

```bash
bash scripts/run_all.sh \
  --datasets coco_chair \
  --methods ablations \
  --risk-threshold <VAL_THRESHOLD> \
  --gpu 0
```

AMBER Object Subset evaluation needs an existing prediction JSONL:

```bash
AMBER_PREDICTIONS=outputs/predictions/amber_object_svo.jsonl \
  bash scripts/run_all.sh --datasets amber_object --methods svo
```

## 5. Export Tables

Tables are generated from metrics JSON files only:

```bash
python scripts/export_results.py --metrics-dir outputs/metrics --out outputs/tables --missing-value NA
```

Missing cells mean the corresponding real metric file was not found.
