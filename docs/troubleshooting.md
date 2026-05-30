# Troubleshooting

## `risk_scoring.threshold` is null

SVO verification needs either a tuned config value or `--risk-threshold`:

```bash
bash scripts/run_all.sh --risk-threshold <VAL_THRESHOLD>
```

Do not tune this value on test metrics.

## spaCy model is missing

Use the rule extractor for smoke tests:

```bash
python scripts/extract_objects.py ... --backend rule
```

Or install the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## GroundingDINO import fails

Install the official package in the active environment and check paths:

```bash
bash scripts/download_models.sh --confirm --skip-llava --install-groundingdino
python scripts/check_assets.py
```

## Tables are empty or show `NA`

`scripts/export_results.py` only reads metric JSON files from `outputs/metrics`. Run the relevant
`scripts/evaluate.py` command first and confirm the `method`, `dataset`, and `task` fields match the
table template.

## Generated outputs should not be committed

`outputs/`, `data/`, and `models/` are ignored by default except for their README files. Keep real
assets and result files outside version control unless you intentionally publish a separate artifact.
