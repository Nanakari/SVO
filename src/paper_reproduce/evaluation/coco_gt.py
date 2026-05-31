"""COCO ground-truth object loading for CHAIR-style evaluation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from paper_reproduce.datasets.coco_stream import iter_coco_annotations, iter_coco_categories
from paper_reproduce.extraction import ObjectVocabulary


def load_coco_gt_objects(
    annotation_path: str | Path, vocabulary: ObjectVocabulary
) -> dict[str, set[str]]:
    """Load image_id -> normalized COCO object categories."""

    categories = {}
    for category in iter_coco_categories(annotation_path):
        normalized = vocabulary.normalize(category["name"]) or str(category["name"]).lower()
        categories[int(category["id"])] = normalized

    gt_by_image: dict[str, set[str]] = defaultdict(set)
    for annotation in iter_coco_annotations(annotation_path):
        image_id = str(annotation.get("image_id", ""))
        category_id = annotation.get("category_id")
        if image_id and category_id in categories:
            gt_by_image[image_id].add(categories[category_id])
    return dict(gt_by_image)
