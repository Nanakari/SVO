"""COCO ground-truth object loading for CHAIR-style evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from paper_reproduce.extraction import ObjectVocabulary


def load_coco_gt_objects(
    annotation_path: str | Path, vocabulary: ObjectVocabulary
) -> dict[str, set[str]]:
    """Load image_id -> normalized COCO object categories."""

    with Path(annotation_path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    categories = {}
    for category in data.get("categories", []):
        normalized = vocabulary.normalize(category["name"]) or str(category["name"]).lower()
        categories[int(category["id"])] = normalized

    gt_by_image: dict[str, set[str]] = defaultdict(set)
    for annotation in data.get("annotations", []):
        image_id = str(annotation.get("image_id", ""))
        category_id = annotation.get("category_id")
        if image_id and category_id in categories:
            gt_by_image[image_id].add(categories[category_id])
    return dict(gt_by_image)
