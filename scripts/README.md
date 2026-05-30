# scripts

Command-line entry points are added phase by phase:

- `run_caption.py` for Base caption generation. Implemented in stage 2.
- `run_pope.py` for POPE inference. Implemented in stage 2.
- `extract_objects.py` for object phrase extraction and SVO risk scoring. Implemented in stage 3.
- `build_static_prior.py` for COCO validation-set static hallucination priors. Implemented in stage 3.
- `verify_objects.py` for GroundingDINO visual verification. Implemented in stage 4.
- `revise_captions.py` for conservative caption revision. Implemented in stage 5.
- `revise_pope.py` for POPE Yes-to-No revision. Implemented in stage 5.
- `evaluate.py` for CHAIR, POPE, AMBER, efficiency, and false-correction metrics. Implemented in stage 6.
- `export_results.py` for table templates and metric-based exports. Implemented in stage 7.
- `setup_cloud.sh` for creating/checking the Ubuntu CUDA 12 conda environment.
- `prepare_data.sh` for creating the data layout and optionally downloading public COCO files.
- `download_models.sh` for confirmed LLaVA/GroundingDINO asset preparation.
- `run_all.sh` for dry-run or real end-to-end experiment orchestration.
- `smoke_test.sh` for the no-model toy fixture pipeline.
- `run_tests.sh` for compile, pytest, and smoke checks.
- `make_val_split.py` for deterministic COCO train2017 validation splits.
- `check_assets.py` for local dataset/model path validation.
- `sweep_thresholds.py` for SVO risk-threshold sensitivity tables.
- `sweep_detector_thresholds.py` for GroundingDINO box/text-threshold sensitivity tables.

The SVO pipeline is intentionally decomposed into explicit steps (`extract_objects.py`,
`verify_objects.py`, `revise_captions.py`, and `evaluate.py`) so each intermediate file can be
audited before tables are exported.

All scripts must support YAML config files plus command-line overrides.

Shell scripts target Ubuntu/cloud runs. On Windows, run the Python entry points directly or execute
the shell scripts from Git Bash/WSL.
