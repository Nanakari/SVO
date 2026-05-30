# data

Place dataset files or symlinks here. This directory is intentionally ignored except for this README.

Expected layout:

```text
data/
  coco/
    train2017/
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
`--download-coco-required --confirm` only when you want the script to download the public COCO files.
POPE and AMBER files must be placed manually according to their source licenses.
