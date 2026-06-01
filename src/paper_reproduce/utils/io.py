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
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_number} in {path}: {exc}") from exc
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


def repair_jsonl_tail(path: str | Path) -> bool:
    """Truncate one incomplete final JSONL line left by an interrupted append.

    Only the last non-empty line may be repaired. Any malformed line before the tail still
    raises, since silently dropping middle records would make resuming unsafe.
    """

    path_obj = Path(path)
    if not path_obj.exists():
        return False
    data = path_obj.read_bytes()
    if not data:
        return False

    lines = data.splitlines(keepends=True)
    last_nonempty_index: int | None = None
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip():
            last_nonempty_index = index
            break
    if last_nonempty_index is None:
        return False

    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line.decode("utf-8-sig" if index == 0 else "utf-8").strip())
        except json.JSONDecodeError as exc:
            if index != last_nonempty_index:
                raise ValueError(f"Invalid JSONL line {index + 1} in {path_obj}: {exc}") from exc
            path_obj.write_bytes(b"".join(lines[:index]))
            return True
        if not isinstance(item, dict):
            raise ValueError(f"JSONL line {index + 1} must be an object: {path_obj}")
    return False


def existing_sample_ids(path: str | Path, *, repair_tail: bool = True) -> set[str]:
    """Collect sample ids already present in an output JSONL file."""

    path_obj = Path(path)
    if not path_obj.exists():
        return set()
    if repair_tail:
        repair_jsonl_tail(path_obj)
    ids: set[str] = set()
    for record in read_jsonl(path_obj):
        sample_id = record.get("sample_id")
        if sample_id is not None:
            ids.add(str(sample_id))
    return ids
