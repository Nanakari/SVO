"""Memory-conscious helpers for COCO instances JSON files."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def iter_coco_array(annotation_path: str | Path, key: str) -> Iterator[dict[str, Any]]:
    """Yield objects from a top-level COCO JSON array without loading the whole file."""

    path = Path(annotation_path)
    decoder = json.JSONDecoder()
    key_token = json.dumps(key)
    chunk_size = 1024 * 1024

    with path.open("r", encoding="utf-8-sig") as handle:
        buffer = ""
        pos = 0

        while True:
            found = buffer.find(key_token, pos)
            if found >= 0:
                pos = found + len(key_token)
                break
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"COCO annotation file has no `{key}` array: {path}")
            buffer = buffer[-len(key_token) :] + chunk
            pos = 0

        buffer, pos = _seek_array_start(handle, buffer, pos, chunk_size, path, key)
        buffer = buffer[pos + 1 :]
        pos = 0

        while True:
            buffer, pos = _ensure_data(handle, buffer, pos, chunk_size)
            pos = _skip_ws_and_commas(buffer, pos)
            if pos >= len(buffer):
                continue
            if buffer[pos] == "]":
                return

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        raise
                    buffer += chunk

            if not isinstance(value, dict):
                raise ValueError(f"Expected object in COCO `{key}` array: {path}")
            yield value
            buffer = buffer[end:]
            pos = 0


def iter_coco_images(annotation_path: str | Path) -> Iterator[dict[str, Any]]:
    yield from iter_coco_array(annotation_path, "images")


def iter_coco_annotations(annotation_path: str | Path) -> Iterator[dict[str, Any]]:
    yield from iter_coco_array(annotation_path, "annotations")


def iter_coco_categories(annotation_path: str | Path) -> Iterator[dict[str, Any]]:
    yield from iter_coco_array(annotation_path, "categories")


def load_coco_image_index(
    annotation_path: str | Path, image_ids: Iterable[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Return image metadata keyed by image id, optionally filtered to a split."""

    wanted = {str(image_id) for image_id in image_ids} if image_ids is not None else None
    images: dict[str, dict[str, Any]] = {}
    for image in iter_coco_images(annotation_path):
        if "id" not in image or not image.get("file_name"):
            continue
        image_id = str(image["id"])
        if wanted is None or image_id in wanted:
            images[image_id] = image
            if wanted is not None and len(images) == len(wanted):
                break
    return images


def sample_coco_image_ids(annotation_path: str | Path, sample_size: int, seed: int) -> list[str]:
    """Sample deterministic COCO image ids while keeping memory proportional to image count."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    image_ids = sorted(
        {str(image["id"]) for image in iter_coco_images(annotation_path) if "id" in image},
        key=_image_id_sort_key,
    )
    if sample_size > len(image_ids):
        raise ValueError(f"Requested {sample_size} images, but annotation contains {len(image_ids)}")
    rng = random.Random(seed)
    sampled = list(image_ids)
    rng.shuffle(sampled)
    return sorted(sampled[:sample_size], key=_image_id_sort_key)


def read_image_id_split(split_path: str | Path) -> list[str]:
    path = Path(split_path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        values = data.get("image_ids", []) if isinstance(data, Mapping) else data
        return [str(value) for value in values]

    image_ids: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
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


def _seek_array_start(
    handle: Any, buffer: str, pos: int, chunk_size: int, path: Path, key: str
) -> tuple[str, int]:
    while True:
        while pos < len(buffer):
            char = buffer[pos]
            if char.isspace() or char == ":":
                pos += 1
                continue
            if char == "[":
                return buffer, pos
            raise ValueError(f"COCO `{key}` key is not followed by an array in {path}")
        chunk = handle.read(chunk_size)
        if not chunk:
            raise ValueError(f"COCO `{key}` array is truncated in {path}")
        buffer += chunk


def _ensure_data(handle: Any, buffer: str, pos: int, chunk_size: int) -> tuple[str, int]:
    while pos >= len(buffer):
        chunk = handle.read(chunk_size)
        if not chunk:
            raise ValueError("Unexpected end of COCO JSON array")
        buffer += chunk
    return buffer, pos


def _skip_ws_and_commas(buffer: str, pos: int) -> int:
    while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","):
        pos += 1
    return pos


def _image_id_sort_key(value: str) -> int | str:
    return int(value) if value.isdigit() else value
