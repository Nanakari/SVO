from __future__ import annotations

import json
from pathlib import Path

from paper_reproduce.datasets.coco import load_coco_caption_samples
from paper_reproduce.datasets.coco_stream import sample_coco_image_ids
from paper_reproduce.evaluation.coco_gt import load_coco_gt_objects
from paper_reproduce.extraction import ObjectVocabulary


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


def test_streaming_coco_helpers_sample_and_load_gt(tmp_path: Path) -> None:
    annotation = tmp_path / "instances.json"
    annotation.write_text(
        json.dumps(
            {
                "info": {"description": "tiny fixture"},
                "images": [
                    {"id": 1, "file_name": "one.jpg"},
                    {"id": 2, "file_name": "two.jpg"},
                    {"id": 3, "file_name": "three.jpg"},
                ],
                "annotations": [
                    {"image_id": 1, "category_id": 10},
                    {"image_id": 2, "category_id": 11},
                    {"image_id": 2, "category_id": 10},
                ],
                "categories": [
                    {"id": 10, "name": "person"},
                    {"id": 11, "name": "dog"},
                ],
            }
        ),
        encoding="utf-8",
    )
    vocabulary = ObjectVocabulary(
        categories=frozenset({"person", "dog"}),
        aliases={"person": "person", "dog": "dog"},
        filters=frozenset(),
    )

    sampled = sample_coco_image_ids(annotation, sample_size=2, seed=42)
    gt = load_coco_gt_objects(annotation, vocabulary)

    assert len(sampled) == 2
    assert gt == {"1": {"person"}, "2": {"person", "dog"}}
