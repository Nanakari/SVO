# data

Place dataset files or symlinks here. This directory is intentionally ignored except for this README.

Expected layout:

```text
data/
  coco/
    train2017/
    train2017_val2000/
    val2014/
    val2017/
    annotations/
  pope/
    random.jsonl
    popular.jsonl
    adversarial.jsonl
  amber/
    images/
    object_subset.jsonl
```

Update `configs/datasets/*.yaml` if your local paths differ.

Use `bash scripts/prepare_data.sh` from the project root to create this layout. Add
`--download-coco-required --confirm` when you want the script to download public COCO val2014 images
and full annotation files. Add `--prepare-train2017-subset --subset-size 2000 --confirm` to prepare
the default train2017 validation subset. POPE and AMBER files must be placed manually according to
their source licenses.
