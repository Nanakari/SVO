# Data Preparation

This project does not redistribute datasets. Put files under `data/` or symlink them there.

## Expected Layout

```text
data/
  coco/
    train2017/
    val2014/
    annotations/
      instances_train2017.json
      instances_val2014.json
  pope/
    random.jsonl
    popular.jsonl
    adversarial.jsonl
  amber/
    images/
    object_subset.jsonl
```

## COCO

Create directories only:

```bash
bash scripts/prepare_data.sh
```

Download the public COCO files required by the default configs:

```bash
bash scripts/prepare_data.sh --download-coco-required --confirm
```

If you store COCO elsewhere, update `configs/datasets/coco_chair.yaml` and
`configs/datasets/pope.yaml`.

## POPE

Place the three POPE JSONL files at:

```text
data/pope/random.jsonl
data/pope/popular.jsonl
data/pope/adversarial.jsonl
```

The loader expects each record to contain enough information for image path, question, and label.
Use `scripts/run_pope.py --help` and `src/paper_reproduce/datasets/pope.py` to inspect accepted
field aliases.

## AMBER Object Subset

Place object-existence-only AMBER samples at:

```text
data/amber/object_subset.jsonl
data/amber/images/
```

The current evaluator scores only yes/no object existence. Attribute, relation, and OCR subsets are
intentionally out of scope for this SVO reproduction.

## Validation Split

Generate a deterministic COCO train2017 validation split for threshold tuning:

```bash
python scripts/make_val_split.py \
  --coco-annotations data/coco/annotations/instances_train2017.json \
  --sample-size 500 \
  --seed 42 \
  --output configs/splits/coco_train2017_val500_seed42.txt
```
