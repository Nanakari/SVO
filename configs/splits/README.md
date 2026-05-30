# Splits

Validation splits are generated from local dataset annotations and are not committed as paper
results. Use `scripts/make_val_split.py` to create deterministic image-id lists, for example:

```bash
python scripts/make_val_split.py \
  --coco-annotations data/coco/annotations/instances_train2017.json \
  --sample-size 5000 \
  --seed 42 \
  --output configs/splits/coco_train2017_val5000_seed42.txt
```

The generated file contains only image ids. It is used for threshold tuning; test sets should not
be used for threshold selection.
