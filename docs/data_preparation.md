# Data Preparation

This project does not redistribute datasets. Put files under `data/` or symlink them there.

## Expected Layout

```text
data/
  coco/
    train2017/
    train2017_val5000/
    val2014/
    annotations/
      instances_train2017.json
      instances_val2014.json
      captions_train2017.json
      captions_val2014.json
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

Download the public COCO files required by the default configs. This downloads val2014 images and
full annotation files, but not full train2017 images:

```bash
bash scripts/prepare_data.sh --download-coco-required --confirm
```

Prepare the default 5000-image train2017 validation subset:

```bash
bash scripts/prepare_data.sh \
  --prepare-train2017-subset \
  --subset-size 5000 \
  --download-missing-subset \
  --confirm
```

If a full `data/coco/train2017/` directory already exists, the subset script copies from it. If it
does not exist, add `--download-missing-subset --confirm` to download only the split images.

The split and subset scripts stream the COCO JSON arrays instead of loading the full
`instances_train2017.json` into memory. This keeps preparation usable on small cgroup memory limits
while still producing the same deterministic split.

`captions_val2014.json` and `captions_train2017.json` are used by the official-compatible CHAIR
backend to mirror the original image-level ground-truth construction. If they are absent, CHAIR can
fall back to instance annotations only, but main results should keep the caption annotations.

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

Generated POPE answers and POPE evaluation default to the public POPE script-style `official`
answer normalizer. This keeps free-form responses comparable with common POPE reporting.

POPE object verification prefers a structured `target_object` field. If the field is absent, the
loader parses common question templates such as `Is there a dining table in the image?`; if parsing
fails, object extraction falls back to the configured NLP/rule extractor.

`run_caption.py` and `run_pope.py` check the first 100 pending image paths before loading LLaVA.
Use `--check-images all` when validating a newly prepared dataset.

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
  --sample-size 5000 \
  --seed 42 \
  --output configs/splits/coco_train2017_val5000_seed42.txt
```

The COCO loader supports both subset and full image roots:

```bash
# subset directory
--set dataset.paths.image_root=data/coco/train2017_val5000

# full train2017 directory, still filtered by split_file
--set dataset.paths.image_root=data/coco/train2017
--set dataset.paths.split_file=configs/splits/coco_train2017_val5000_seed42.txt
```
