# Experiment Matrix

## Core Methods

| Method | Caption or Answer Source | Verification Policy | Revision | Status |
| --- | --- | --- | --- | --- |
| Base | LLaVA-1.5-7B | none | none | implemented |
| Verify-All | Base output | all extracted objects | conservative | implemented |
| Random-Verify | Base output | matched random count | conservative | implemented |
| Ours/SVO | Base output | risk threshold | conservative | implemented |

## Risk Ablations

| Method Config | Disabled Term |
| --- | --- |
| `configs/methods/svo_without_uncertainty.yaml` | uncertainty |
| `configs/methods/svo_without_position.yaml` | position |
| `configs/methods/svo_without_prior.yaml` | static prior |

## Risk Component Analysis

| Method Config | Enabled Term |
| --- | --- |
| `configs/methods/svo_only_uncertainty.yaml` | uncertainty |
| `configs/methods/svo_only_position.yaml` | position |
| `configs/methods/svo_only_prior.yaml` | static prior |

## Sensitivity Analysis

| Script | Variable | Output |
| --- | --- | --- |
| `scripts/tune_svo_threshold.sh` | Validation-only SVO risk threshold | `outputs/validation/sweeps/risk_threshold/tables/threshold_sweep.*` |
| `scripts/sweep_thresholds.py` | SVO risk threshold | `outputs/sweeps/risk_threshold/tables/threshold_sweep.*` |
| `scripts/sweep_detector_thresholds.py` | GroundingDINO box/text thresholds | `outputs/sweeps/detector_thresholds/tables/detector_sensitivity.*` |

## Datasets and Metrics

| Dataset | Script Task | Metrics |
| --- | --- | --- |
| COCO/CHAIR | `evaluate.py --task chair --chair-backend official` | CHAIRs, CHAIRi, average length, correct object coverage |
| POPE | `evaluate.py --task pope --pope-normalizer official` | accuracy, precision, recall, F1, yes ratio |
| AMBER Object Subset | `evaluate.py --task amber` | object-existence yes/no metrics |
| Efficiency | `evaluate.py --task efficiency` | verification rate, external queries/image, relative latency |
| False Correction | `evaluate.py --task false_correction` | hallucinated removal, correct retention, false correction rate |

## Reserved Interfaces

VCD and OPERA are listed in configs and export table templates but are not run by default. Their
cells remain empty until real metric JSON files are produced by a connected implementation.
