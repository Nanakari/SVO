"""Common evaluation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from paper_reproduce.utils.io import read_jsonl, write_json


def safe_divide(numerator: float, denominator: float) -> float | None:
    """Return a ratio or None when the denominator is zero."""

    if denominator == 0:
        return None
    return numerator / denominator


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load all JSONL records into memory for metric computation."""

    return list(read_jsonl(path))


def infer_method(records: list[Mapping[str, Any]], default: str = "unknown") -> str:
    """Infer a method name from result records."""

    for record in records:
        method = record.get("method")
        if method:
            return str(method)
    return default


def write_metrics(
    *,
    output_path: str | Path,
    dataset: str,
    task: str,
    method: str,
    metrics: Mapping[str, Any],
    counts: Mapping[str, Any],
    source_files: Iterable[str | Path],
    notes: Iterable[str] | None = None,
) -> None:
    """Write a machine-generated metric JSON file."""

    write_json(
        output_path,
        {
            "dataset": dataset,
            "task": task,
            "method": method,
            "metrics": dict(metrics),
            "counts": dict(counts),
            "source_files": [str(path) for path in source_files],
            "notes": list(notes or []),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def token_length(text: str) -> int:
    """A simple whitespace token count for Average Length."""

    return len(text.strip().split()) if text.strip() else 0
