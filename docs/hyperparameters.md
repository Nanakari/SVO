# Hyperparameters

This project separates the main reproduction protocol from optional sensitivity analysis.
Main-result tables should use a fixed protocol. Extra sweeps are useful diagnostics, but they should
be reported separately and selected only from validation data.

## Default Protocol

| Parameter Group | Default | Role | Main Protocol |
| --- | --- | --- | --- |
| LLaVA decoding | `temperature=0.0`, `top_p=1.0`, `do_sample=false`, `max_new_tokens=128` | Makes Base generation deterministic | Fixed |
| SVO risk weights | `uncertainty=1.0`, `position=1.0`, `prior=1.0` | Combines the three risk signals | Fixed for main runs |
| SVO risk threshold | `null` in config; pass `--risk-threshold` | Selects which objects are verified | Tune on validation only |
| GroundingDINO thresholds | `box_threshold=0.35`, `text_threshold=0.25` | Controls detector strictness | Fixed unless running detector sensitivity |
| Evidence threshold | defaults to box threshold | Decides whether a verified object has visual evidence | Fixed with detector setting |
| Static prior smoothing | `min_prior_count=5`, `fallback_prior=dataset_mean` | Stabilizes low-frequency object priors | Fixed |
| Caption revision rules | conservative local edits | Prevents broad false corrections | Fixed |

The default protocol intentionally avoids searching over many knobs. This makes the comparison
between Base, Verify-All, Random-Verify, and SVO easier to audit.

## What Should Be Tuned

The primary tunable value is `risk_scoring.threshold` / `--risk-threshold`. It controls the central
SVO tradeoff:

- lower threshold: more objects verified, higher cost, usually fewer missed hallucinations;
- higher threshold: fewer objects verified, lower cost, usually fewer corrections.

Choose this value on the COCO train2017 random 5000-image validation split, then freeze it before
running the final test metrics.

## Optional Sensitivity Analysis

Risk threshold sweep:

```bash
bash scripts/tune_svo_threshold.sh \
  --thresholds "0.5 1.0 1.5 2.0" \
  --gpu 0
```

The wrapper writes validation-only artifacts under `outputs/validation/`, builds the static prior
from train2017 validation captions, and evaluates every threshold against
`instances_train2017.json` with the fixed split file.

GroundingDINO threshold sweep:

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

The validation wrapper writes under `outputs/validation/...`. The lower-level sweep scripts write
under the `--output-dir` you pass. These outputs are generated from real metric JSON files; missing
cells indicate that a metric was not produced.

## Risk Component Analysis

The project includes both removal and singleton component analyses:

```text
configs/methods/svo_without_uncertainty.yaml
configs/methods/svo_without_position.yaml
configs/methods/svo_without_prior.yaml
configs/methods/svo_only_uncertainty.yaml
configs/methods/svo_only_position.yaml
configs/methods/svo_only_prior.yaml
```

Run singleton component analysis with:

```bash
bash scripts/run_all.sh \
  --datasets coco_chair \
  --methods components \
  --risk-threshold <VAL_THRESHOLD>
```

## Reporting Rules

- Do not tune any parameter on the final test set.
- If GroundingDINO thresholds are changed, all methods that use GroundingDINO must share the same
  thresholds.
- If risk weights are tuned, report the search space and treat the result as an extra experiment,
  not as the default reproduction protocol.
- Do not hand-fill table values. Use script-generated metrics and table exports only.
