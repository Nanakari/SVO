# scripts

Command-line entry points:

- `run_caption.py` for Base caption generation. Existing complete outputs are detected before
  LLaVA is loaded.
- `run_pope.py` for POPE inference. Existing complete outputs are detected before LLaVA is loaded.
  Answers default to the public POPE `official` normalizer.
- `extract_objects.py` for object phrase extraction and SVO risk scoring.
- `build_static_prior.py` for COCO validation-set static hallucination priors.
- `verify_objects.py` for GroundingDINO visual verification.
- `revise_captions.py` for conservative caption revision.
- `revise_pope.py` for POPE Yes-to-No revision.
- `evaluate.py` for CHAIR, POPE, AMBER, efficiency, and false-correction metrics. CHAIR defaults
  to the official-compatible `--chair-backend official`; `--chair-backend internal` is an explicit
  fallback. POPE defaults to `--pope-normalizer official`.
- `check_chair_parity.py` for local CHAIR official-compatible parity checks against a cloned
  LisaAnne/Hallucination repository via `--official-chair-repo`.
- `export_results.py` for table templates and metric-based exports.
- `setup_cloud.sh` for creating/checking the Ubuntu CUDA 12 conda environment.
- `prepare_data.sh` for creating the data layout and optionally downloading public COCO files.
- `prepare_coco_subset.py` for copying/downloading train2017 split images without requiring full train2017.
- `download_models.sh` for confirmed LLaVA/GroundingDINO asset preparation, including the
  `bert-base-uncased` GroundingDINO text encoder cache by default.
- `run_all.sh` for dry-run or real end-to-end experiment orchestration.
- `tune_svo_threshold.sh` for validation-only SVO risk-threshold tuning on train2017_val5000.
  Use `--sweep-only` to reuse completed captions, priors, and objects.
- `run_coco_main_autodl.sh` for cloud COCO orchestration. It supports `--dry-run`,
  `--tune-only`, and `--main-only`; `--main-only` requires `SVO_RISK_THRESHOLD`.
- `smoke_test.sh` for the no-model toy fixture pipeline.
- `run_tests.sh` for compile, pytest, and smoke checks.
- `make_val_split.py` for deterministic COCO train2017 validation splits.
- `check_assets.py` for local dataset/model path validation. Use `--assets` and `--datasets` to
  check only the prepared scope.
- `sweep_thresholds.py` for SVO risk-threshold sensitivity tables.
- `sweep_detector_thresholds.py` for GroundingDINO box/text-threshold sensitivity tables.

The SVO pipeline is intentionally decomposed into explicit steps (`extract_objects.py`,
`verify_objects.py`, `revise_captions.py`, and `evaluate.py`) so each intermediate file can be
audited before tables are exported.

All scripts must support YAML config files plus command-line overrides.

Shell scripts target Ubuntu/cloud runs. On Windows, run the Python entry points directly or execute
the shell scripts from Git Bash/WSL.
