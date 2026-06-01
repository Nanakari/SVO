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
bash scripts/prepare_data.sh \
  --prepare-train2017-subset \
  --subset-size 5000 \
  --download-missing-subset \
  --confirm
bash scripts/download_models.sh --confirm --install-groundingdino
python scripts/check_assets.py \
  --strict \
  --assets llava,groundingdino,groundingdino_text_encoder \
  --datasets coco_chair
```

This is a COCO-only check. A full strict check will also require POPE and AMBER files. Update
`configs/default.yaml` or pass `--set` overrides for local model paths.

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

To reuse completed captions, priors, and extracted objects after an interrupted run:

```bash
bash scripts/tune_svo_threshold.sh \
  --sweep-only \
  --thresholds "0.5 1.0 1.5 2.0" \
  --gpu 0
```

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
  --coco-caption-annotations data/coco/annotations/captions_train2017.json \
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

## 6. CHAIR Backend

`scripts/evaluate.py --task chair` uses `--chair-backend official` by default. This
official-compatible scorer follows the COCO synonym, tokenization, and special-case behavior of
the original CHAIR metric and uses COCO instance annotations plus reference captions when the
caption annotation file is present.

Compatibility is checked against the behavior of the
[LisaAnne/Hallucination](https://github.com/LisaAnne/Hallucination) CHAIR scorer. The upstream
repository does not provide an explicit license, so this project keeps a lightweight compatible
implementation and wrapper instead of vendoring the external codebase.

Use the project-local extractor scorer only for fallback or sanity checks:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --dataset coco_chair \
  --task chair \
  --chair-backend internal \
  --predictions outputs/revisions/coco_chair_svo_revisions.jsonl
```

Main results should report the official-compatible backend.

For a local alignment check against a cloned LisaAnne/Hallucination checkout:

```bash
python scripts/check_chair_parity.py \
  --official-chair-repo /path/to/Hallucination
```

The optional `--check-synonym-coverage` flag also checks every alias in the official
`data/synonyms.txt` resource. That stricter check is useful when extending the local synonym map.

## 7. POPE Answer Normalization

POPE evaluation defaults to `--pope-normalizer official`, matching the public POPE evaluation
script: keep only the first sentence, remove commas, classify answers containing `no` or `not` as
`no`, and classify the remaining answers as `yes`.

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --dataset pope \
  --task pope \
  --pope-normalizer official \
  --predictions outputs/revisions/pope_svo_revised.jsonl
```

Use `--pope-normalizer strict` only for exact yes/no fixture tests or ablations.
