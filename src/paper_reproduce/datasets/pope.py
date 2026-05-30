"""POPE dataset reader for object-existence yes/no inference."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from paper_reproduce.datasets.types import PopeSample
from paper_reproduce.utils.config import resolve_path
from paper_reproduce.utils.io import read_jsonl


def load_pope_samples(
    dataset_config: Mapping[str, Any],
    project_root: str | Path,
    *,
    settings: list[str] | None = None,
    limit_per_setting: int | None = None,
) -> list[PopeSample]:
    """Load POPE random/popular/adversarial JSONL files.

    The loader accepts the common POPE JSONL variants used in public reproductions, where
    image and label keys may be named slightly differently.
    """

    paths = dataset_config.get("paths", {})
    image_root = resolve_path(paths.get("image_root"), project_root)
    annotation_files = paths.get("annotation_files", {})
    if image_root is None:
        raise ValueError("POPE dataset config is missing paths.image_root")
    if not isinstance(annotation_files, Mapping):
        raise ValueError("POPE dataset config must define paths.annotation_files")

    requested_settings = settings or list(annotation_files.keys())
    samples: list[PopeSample] = []
    for setting in requested_settings:
        annotation_path = resolve_path(annotation_files.get(setting), project_root)
        if annotation_path is None:
            raise ValueError(f"POPE annotation file for setting `{setting}` is not configured")
        if not annotation_path.exists():
            raise FileNotFoundError(f"POPE annotation file not found: {annotation_path}")

        count = 0
        for index, record in enumerate(read_jsonl(annotation_path)):
            image_ref = _first_present(record, ["image", "image_path", "file_name", "filename"])
            image_id = str(record.get("image_id") or _image_id_from_ref(image_ref))
            question = _first_present(record, ["question", "text", "query"])
            if question is None:
                raise ValueError(f"POPE record is missing question/text/query: {annotation_path}:{index + 1}")
            label = _normalise_label(_first_present(record, ["answer", "label", "gt_answer", "truth"]))
            local_id = record.get("sample_id") or record.get("question_id") or record.get("id") or index
            sample_id = f"{setting}:{local_id}"
            image_path = _resolve_image_path(image_root, image_ref, image_id)
            samples.append(
                PopeSample(
                    sample_id=sample_id,
                    image_id=image_id,
                    image_path=image_path,
                    question=str(question),
                    label=label,
                    setting=setting,
                    raw=record,
                )
            )
            count += 1
            if limit_per_setting is not None and count >= limit_per_setting:
                break
    return samples


def _first_present(record: Mapping[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return default


def _normalise_label(value: Any) -> str | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"yes", "y", "true", "1"}:
        return "yes"
    if lowered in {"no", "n", "false", "0"}:
        return "no"
    return lowered


def _image_id_from_ref(image_ref: Any) -> str:
    if image_ref is None:
        return ""
    stem = Path(str(image_ref)).stem
    match = re.search(r"(\d+)$", stem)
    return match.group(1) if match else stem


def _resolve_image_path(image_root: Path, image_ref: Any, image_id: str) -> Path:
    if image_ref:
        image_path = Path(str(image_ref))
        if image_path.is_absolute():
            return image_path
        return image_root / image_path
    if image_id:
        if str(image_id).isdigit():
            return image_root / f"COCO_val2014_{int(image_id):012d}.jpg"
        return image_root / str(image_id)
    raise ValueError("POPE record must provide an image reference or image_id")
