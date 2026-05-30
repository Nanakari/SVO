from __future__ import annotations

import json
from pathlib import Path

from paper_reproduce.datasets.coco import load_coco_caption_samples


def test_coco_loader_filters_and_orders_split_file(tmp_path: Path) -> None:
    annotation = tmp_path / "instances.json"
    split = tmp_path / "split.txt"
    images = tmp_path / "images"
    annotation.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "one.jpg", "height": 10, "width": 20},
                    {"id": 2, "file_name": "two.jpg", "height": 20, "width": 30},
                    {"id": 3, "file_name": "three.jpg", "height": 30, "width": 40},
                ]
            }
        ),
        encoding="utf-8",
    )
    split.write_text("3\n1\n", encoding="utf-8")

    samples = load_coco_caption_samples(
        {
            "name": "coco_chair",
            "paths": {
                "image_root": str(images),
                "annotation_file": str(annotation),
                "split_file": str(split),
            },
        },
        tmp_path,
    )

    assert [sample.image_id for sample in samples] == ["3", "1"]
    assert [sample.image_path.name for sample in samples] == ["three.jpg", "one.jpg"]
    assert samples[0].metadata["split_file"] == str(split)


def test_coco_loader_without_split_keeps_full_annotation_order(tmp_path: Path) -> None:
    annotation = tmp_path / "instances.json"
    images = tmp_path / "images"
    annotation.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "one.jpg"},
                    {"id": 2, "file_name": "two.jpg"},
                ]
            }
        ),
        encoding="utf-8",
    )

    samples = load_coco_caption_samples(
        {
            "name": "coco_chair",
            "paths": {
                "image_root": str(images),
                "annotation_file": str(annotation),
            },
        },
        tmp_path,
    )

    assert [sample.image_id for sample in samples] == ["1", "2"]
