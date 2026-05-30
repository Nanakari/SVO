"""AMBER Object Subset reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from paper_reproduce.datasets.types import AmberObjectSample
from paper_reproduce.utils.config import resolve_path
from paper_reproduce.utils.io import read_jsonl


def load_amber_object_samples(
    dataset_config: Mapping[str, Any],
    project_root: str | Path,
    *,
    limit: int | None = None,
) -> list[AmberObjectSample]:
    """Load AMBER object-existence samples from a flexible JSONL schema."""

    paths = dataset_config.get("paths", {})
    image_root = resolve_path(paths.get("image_root"), project_root)
    annotation_file = resolve_path(paths.get("annotation_file"), project_root)
    if image_root is None:
        raise ValueError("AMBER dataset config is missing paths.image_root")
    if annotation_file is None or not annotation_file.exists():
        raise FileNotFoundError(f"AMBER Object Subset annotation file not found: {annotation_file}")

    filters = dataset_config.get("filter", {})
    samples: list[AmberObjectSample] = []
    for index, record in enumerate(read_jsonl(annotation_file)):
        if not _is_object_existence_record(record, filters):
            continue
        image_ref = _first_present(record, ["image", "image_path", "file_name", "filename"])
        image_id = str(record.get("image_id") or _image_id_from_ref(image_ref) or index)
        sample_id = str(record.get("sample_id") or record.get("id") or f"amber:{index}")
        samples.append(
            AmberObjectSample(
                sample_id=sample_id,
                image_id=image_id,
                image_path=_resolve_image_path(image_root, image_ref, image_id),
                question=_first_present(record, ["question", "query", "text", "prompt"]),
                label=_normalise_label(_first_present(record, ["answer", "label", "gt_answer", "truth"])),
                target_object=_first_present(record, ["object", "target_object", "category", "class"]),
                raw=record,
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _is_object_existence_record(record: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    record_type = " ".join(
        str(record.get(key, "")).lower()
        for key in ["type", "task", "subtask", "category", "question_type"]
    )
    if filters.get("exclude_attributes", True) and "attribute" in record_type:
        return False
    if filters.get("exclude_relations", True) and "relation" in record_type:
        return False
    if filters.get("exclude_ocr", True) and "ocr" in record_type:
        return False
    if not filters.get("include_object_existence_only", True):
        return True
    if not record_type.strip():
        return True
    return "object" in record_type or "exist" in record_type


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
    return Path(str(image_ref)).stem


def _resolve_image_path(image_root: Path, image_ref: Any, image_id: str) -> Path:
    if image_ref:
        image_path = Path(str(image_ref))
        if image_path.is_absolute():
            return image_path
        return image_root / image_path
    return image_root / str(image_id)
