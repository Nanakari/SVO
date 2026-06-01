"""COCO ground-truth object loading for CHAIR-style evaluation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from paper_reproduce.datasets.coco_stream import (
    iter_coco_annotations,
    iter_coco_array,
    iter_coco_categories,
)
from paper_reproduce.evaluation.chair import OfficialChairMapper
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


def load_coco_category_names(annotation_path: str | Path) -> list[str]:
    """Load COCO category names in annotation order."""

    return [str(category["name"]) for category in iter_coco_categories(annotation_path)]


def load_coco_gt_objects_official(
    annotation_path: str | Path,
    mapper: OfficialChairMapper,
    *,
    caption_annotation_path: str | Path | None = None,
) -> dict[str, set[str]]:
    """Load image_id -> official-compatible CHAIR GT objects.

    The original CHAIR evaluator builds image-level ground truth from COCO instance
    categories and, when available, objects mentioned in COCO reference captions.
    """

    categories = {}
    for category in iter_coco_categories(annotation_path):
        canonical = mapper.canonicalize_phrase(str(category["name"]))
        if canonical is not None:
            categories[int(category["id"])] = canonical

    gt_by_image: dict[str, set[str]] = defaultdict(set)
    for annotation in iter_coco_annotations(annotation_path):
        image_id = str(annotation.get("image_id", ""))
        category_id = annotation.get("category_id")
        if image_id and category_id in categories:
            gt_by_image[image_id].add(categories[category_id])

    if caption_annotation_path is not None:
        caption_path = Path(caption_annotation_path)
        if not caption_path.exists():
            raise FileNotFoundError(f"COCO caption annotation file not found: {caption_path}")
        for annotation in iter_coco_array(caption_path, "annotations"):
            image_id = str(annotation.get("image_id", ""))
            if not image_id:
                continue
            _, object_names, _, _ = mapper.caption_to_objects(str(annotation.get("caption", "")))
            gt_by_image[image_id].update(object_names)

    return dict(gt_by_image)
