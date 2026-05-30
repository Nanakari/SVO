"""Small JSON/JSONL and filesystem helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def ensure_parent(path: str | Path) -> Path:
    """Create a file's parent directory and return the path."""

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {line_number} must be an object: {path}")
            yield item


def append_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Append JSON-serializable records to a JSONL file and return the count."""

    output_path = ensure_parent(path)
    count = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write a JSON object with stable formatting."""

    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def existing_sample_ids(path: str | Path) -> set[str]:
    """Collect sample ids already present in an output JSONL file."""

    path_obj = Path(path)
    if not path_obj.exists():
        return set()
    ids: set[str] = set()
    for record in read_jsonl(path_obj):
        sample_id = record.get("sample_id")
        if sample_id is not None:
            ids.add(str(sample_id))
    return ids
