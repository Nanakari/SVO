"""COCO/CHAIR dataset reader for Base caption generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

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

    with annotation_file.open("r", encoding="utf-8-sig") as handle:
        annotation_data = json.load(handle)
    images = annotation_data.get("images")
    if not isinstance(images, list):
        raise ValueError(f"COCO annotation file has no images list: {annotation_file}")

    if split_file is not None:
        images = _filter_images_by_split(images, split_file)

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


def _filter_images_by_split(images: list[Mapping[str, Any]], split_file: Path) -> list[Mapping[str, Any]]:
    split_ids = _read_split_image_ids(split_file)
    image_by_id = {str(image.get("id")): image for image in images if "id" in image}
    missing = [image_id for image_id in split_ids if image_id not in image_by_id]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Split file contains {len(missing)} image ids not found in annotation file: {preview}"
        )
    return [image_by_id[image_id] for image_id in split_ids]


def _read_split_image_ids(split_file: Path) -> list[str]:
    if split_file.suffix.lower() == ".json":
        with split_file.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if isinstance(data, Mapping):
            values = data.get("image_ids", [])
        else:
            values = data
        return [str(value) for value in values]

    image_ids: list[str] = []
    with split_file.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{"):
                record = json.loads(stripped)
                if "image_id" not in record:
                    raise ValueError(f"JSONL split line {line_number} is missing image_id")
                image_ids.append(str(record["image_id"]))
            else:
                image_ids.append(stripped.split()[0])
    return image_ids
