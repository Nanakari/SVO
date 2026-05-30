# Contributing

This repository is organized as a reproducible experiment scaffold. Please keep changes auditable:

- Do not commit real datasets, checkpoints, generated outputs, cache directories, or private tokens.
- Do not hand-fill paper tables. Add metrics JSON files only when they are produced by scripts.
- Keep new model and dataset integrations behind small adapters or CLI flags.
- Add a smoke or unit test when changing extraction, scoring, verification, revision, evaluation, or export behavior.
- Document any upstream dependency version that affects reproducibility.

Before opening a pull request, run:

```bash
bash scripts/run_tests.sh
bash scripts/run_all.sh --dry-run
python scripts/check_assets.py
```
