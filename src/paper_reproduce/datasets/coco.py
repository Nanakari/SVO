"""COCO/CHAIR dataset reader for Base caption generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from paper_reproduce.datasets.coco_stream import (
    iter_coco_images,
    load_coco_image_index,
    read_image_id_split,
)
from paper_reproduce.datasets.types import CaptionSample
from paper_reproduce.utils.config import resolve_path


def load_coco_caption_samples(
    dataset_config: Mapping[str, Any],
    project_root: str | Path,
    *,
    limit: int | None = None,
) -> list[CaptionSample]:
    """Load COCO image samples from an instances annotation file.

    The reader only prepares image records for caption generation. CHAIR-specific object
    annotations are consumed by later evaluation stages.
    """

    paths = dataset_config.get("paths", {})
    image_root = resolve_path(paths.get("image_root"), project_root)
    annotation_file = resolve_path(paths.get("annotation_file"), project_root)
    split_file = resolve_path(paths.get("split_file"), project_root)
    if image_root is None:
        raise ValueError("COCO dataset config is missing paths.image_root")
    if annotation_file is None:
        raise ValueError("COCO dataset config is missing paths.annotation_file")
    if not annotation_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")
    if split_file is not None and not split_file.exists():
        raise FileNotFoundError(f"COCO split file not found: {split_file}")

    if split_file is not None:
        split_ids = read_image_id_split(split_file)
        image_by_id = load_coco_image_index(annotation_file, split_ids)
        missing = [image_id for image_id in split_ids if image_id not in image_by_id]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(
                f"Split file contains {len(missing)} image ids not found in annotation file: {preview}"
            )
        images = (image_by_id[image_id] for image_id in split_ids)
    else:
        images = iter_coco_images(annotation_file)

    samples: list[CaptionSample] = []
    for image in images:
        image_id = str(image["id"])
        file_name = image.get("file_name")
        if not file_name:
            raise ValueError(f"COCO image entry is missing file_name for image id {image_id}")
        samples.append(
            CaptionSample(
                sample_id=image_id,
                image_id=image_id,
                image_path=image_root / file_name,
                dataset=str(dataset_config.get("name", "coco_chair")),
                metadata={
                    "file_name": file_name,
                    "height": image.get("height"),
                    "width": image.get("width"),
                    "annotation_file": str(annotation_file),
                    "split_file": str(split_file) if split_file else None,
                },
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples
