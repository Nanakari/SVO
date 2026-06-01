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

If torch or CUDA changed, remove the old build artifacts before reinstalling:

```bash
python -m pip uninstall -y groundingdino || true
rm -rf models/GroundingDINO/build
find models/GroundingDINO/groundingdino -name '*.so' -type f -delete
python -m pip install -e models/GroundingDINO --no-build-isolation --no-deps
```

## GroundingDINO BERT text encoder is missing

The detector uses `bert-base-uncased` through Hugging Face/transformers. Pre-cache it before GPU
runs:

```bash
bash scripts/download_models.sh --confirm --skip-llava
python scripts/check_assets.py \
  --assets groundingdino,groundingdino_text_encoder \
  --datasets coco_chair
```

Respect `HF_HOME`, `HF_ENDPOINT`, and `HF_TOKEN` when using a mirror or private cache.

## CHAIR results differ from older internal scorer runs

`scripts/evaluate.py --task chair` now defaults to `--chair-backend official`, which means the
official-compatible backend. Use this backend for main results. The internal extractor-based scorer
is still available for sanity checks:

```bash
python scripts/evaluate.py ... --task chair --chair-backend internal
```

Do not compare official-compatible-backend numbers directly with previous internal-backend runs
without recording the backend in the metric metadata.

Run the local parity harness when changing CHAIR synonyms or tokenization:

```bash
python scripts/check_chair_parity.py --official-chair-repo /path/to/Hallucination
```

## POPE answers look different from strict yes/no parsing

POPE uses the public evaluation script-style normalizer by default. It keeps the first sentence,
removes commas, maps answers containing `no` or `not` to `no`, and maps the rest to `yes`:

```bash
python scripts/evaluate.py ... --task pope --pope-normalizer official
```

Use `--pope-normalizer strict` only when you want exact yes/no aliases and `unknown` handling.

## Tables are empty or show `NA`

`scripts/export_results.py` only reads metric JSON files from `outputs/metrics`. Run the relevant
`scripts/evaluate.py` command first and confirm the `method`, `dataset`, and `task` fields match the
table template.

Sweep scripts write their own tables under `outputs/sweeps/<sweep_name>/tables`. If those tables
show `NA`, inspect the matching `outputs/sweeps/<sweep_name>/metrics` directory and confirm the
per-threshold evaluation commands completed.

## Generated outputs should not be committed

`outputs/`, `data/`, and `models/` are ignored by default except for their README files. Keep real
assets and result files outside version control unless you intentionally publish a separate artifact.
