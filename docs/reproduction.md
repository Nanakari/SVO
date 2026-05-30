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
bash scripts/prepare_data.sh --prepare-train2017-subset --subset-size 5000 --confirm
bash scripts/download_models.sh --confirm --install-groundingdino
python scripts/check_assets.py --strict
```

Update `configs/default.yaml` or pass `--set` overrides for local model paths.

## 3. Tune SVO Threshold on Validation Data

Generate validation captions and static priors on the fixed 5000-image train2017 split, then choose
the SVO threshold from validation metrics only. The repository does not include a hard-coded paper
threshold.

```bash
bash scripts/tune_svo_threshold.sh \
  --thresholds "0.5 1.0 1.5 2.0" \
  --gpu 0
```

This writes validation-only outputs under `outputs/validation/`, including
`outputs/validation/sweeps/risk_threshold/tables/threshold_sweep.md`. Replace
`<VAL_THRESHOLD>` only after selecting it from this train2017 validation split.

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

Single-component risk analysis:

```bash
bash scripts/run_all.sh \
  --datasets coco_chair \
  --methods components \
  --risk-threshold <VAL_THRESHOLD> \
  --gpu 0
```

Threshold sensitivity analysis:

```bash
bash scripts/tune_svo_threshold.sh \
  --thresholds "0.5 1.0 1.5 2.0" \
  --gpu 0
```

Detector sensitivity analysis:

```bash
python scripts/sweep_detector_thresholds.py \
  --objects outputs/validation/objects/coco_train2017_val5000_svo_objects.jsonl \
  --base-predictions outputs/validation/predictions/coco_train2017_val5000_base_captions.jsonl \
  --risk-threshold <VAL_THRESHOLD> \
  --box-thresholds 0.25 0.35 0.45 \
  --text-thresholds 0.20 0.25 0.30 \
  --coco-annotations data/coco/annotations/instances_train2017.json \
  --output-dir outputs/validation/sweeps/detector_thresholds
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
