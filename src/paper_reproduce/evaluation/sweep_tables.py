"""Dynamic table exports for threshold and detector sensitivity sweeps."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

from paper_reproduce.evaluation.export_tables import MetricFile, load_metric_files
from paper_reproduce.utils.io import ensure_parent, write_json


SWEEP_METRIC_COLUMNS: tuple[dict[str, Any], ...] = (
    {"label": "CHAIRs", "task": "chair", "metric_path": ("chairs",)},
    {"label": "CHAIRi", "task": "chair", "metric_path": ("chairi",)},
    {"label": "Avg. Length", "task": "chair", "metric_path": ("average_length",)},
    {
        "label": "Correct Object Coverage",
        "task": "chair",
        "metric_path": ("correct_object_coverage",),
    },
    {"label": "Verification Rate", "task": "efficiency", "metric_path": ("verification_rate",)},
    {
        "label": "External Queries / Image",
        "task": "efficiency",
        "metric_path": ("external_queries_per_image",),
    },
    {"label": "Latency x", "task": "efficiency", "metric_path": ("relative_latency",)},
    {
        "label": "Hallucinated Removal",
        "task": "false_correction",
        "metric_path": ("hallucinated_object_removal",),
    },
    {
        "label": "Correct Retention",
        "task": "false_correction",
        "metric_path": ("correct_object_retention",),
    },
    {"label": "False Correction Rate", "task": "false_correction", "metric_path": ("false_correction_rate",)},
)


def format_float_label(value: float) -> str:
    """Return a path-safe label for a float value."""

    compact = f"{float(value):.6g}"
    return compact.replace("-", "m").replace(".", "p")


def format_float_display(value: float, precision: int = 4) -> str:
    """Return a compact human-readable float display."""

    return f"{float(value):.{precision}f}".rstrip("0").rstrip(".")


def build_sweep_table(
    *,
    name: str,
    title: str,
    rows: Iterable[Mapping[str, Any]],
    metrics_dir: str | Path,
    parameter_columns: Iterable[str],
    dataset: str = "coco_chair",
    missing_value: str = "NA",
    precision: int = 4,
) -> dict[str, Any]:
    """Build a dynamic sweep table from machine-generated metric JSON files."""

    metric_files = load_metric_files(metrics_dir)
    parameter_columns = list(parameter_columns)
    columns = ["Setting", *parameter_columns, *(column["label"] for column in SWEEP_METRIC_COLUMNS)]
    table_rows: list[dict[str, Any]] = []
    for row in rows:
        method = str(row["method"])
        parameters = dict(row.get("parameters", {}))
        display: dict[str, str] = {"Setting": str(row.get("label", method))}
        cells: list[dict[str, Any]] = []

        for column in parameter_columns:
            value = parameters.get(column)
            display[column] = _format_value(value, missing_value=missing_value, precision=precision)

        for column in SWEEP_METRIC_COLUMNS:
            value, source = _lookup_metric(
                metric_files,
                method=method,
                task=str(column["task"]),
                dataset=dataset,
                metric_path=tuple(column["metric_path"]),
            )
            cell = {
                "column": column["label"],
                "value": value,
                "display": _format_value(value, missing_value=missing_value, precision=precision),
                "missing": value is None,
                "source_file": str(source) if source else None,
                "metric_path": ".".join(column["metric_path"]),
                "task": column["task"],
                "dataset": dataset,
            }
            cells.append(cell)
            display[column["label"]] = cell["display"]

        table_rows.append(
            {
                "label": display["Setting"],
                "method": method,
                "parameters": parameters,
                "cells": cells,
                "display": display,
            }
        )

    return {
        "name": name,
        "title": title,
        "dataset": dataset,
        "metrics_dir": str(metrics_dir),
        "columns": columns,
        "rows": table_rows,
        "missing_value": missing_value,
    }


def export_sweep_table(
    table: Mapping[str, Any],
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("md", "csv", "json"),
) -> list[Path]:
    """Export a dynamic sweep table."""

    written: list[Path] = []
    out_dir = Path(output_dir)
    for fmt in formats:
        fmt = fmt.lower()
        if fmt == "json":
            path = ensure_parent(out_dir / f"{table['name']}.json")
            write_json(path, table)
        elif fmt == "csv":
            path = ensure_parent(out_dir / f"{table['name']}.csv")
            _write_csv(path, table)
        elif fmt in {"md", "markdown"}:
            path = ensure_parent(out_dir / f"{table['name']}.md")
            _write_markdown(path, table)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")
        written.append(path)
    return written


def _lookup_metric(
    metric_files: list[MetricFile],
    *,
    method: str,
    task: str,
    dataset: str,
    metric_path: tuple[str, ...],
) -> tuple[Any, Path | None]:
    for metric_file in metric_files:
        if metric_file.method != method:
            continue
        if metric_file.task != task or metric_file.dataset != dataset:
            continue
        value = _nested_get(metric_file.metrics, metric_path)
        if value is not None:
            return value, metric_file.path
    return None, None


def _nested_get(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _format_value(value: Any, *, missing_value: str, precision: int) -> str:
    if value is None:
        return missing_value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_float_display(value, precision=precision)
    return str(value)


def _write_csv(path: Path, table: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table["columns"]))
        writer.writeheader()
        for row in table["rows"]:
            writer.writerow(row["display"])


def _write_markdown(path: Path, table: Mapping[str, Any]) -> None:
    columns = list(table["columns"])
    lines = [f"# {table['title']}", "", _markdown_row(columns), _markdown_row(["---"] * len(columns))]
    for row in table["rows"]:
        lines.append(_markdown_row([row["display"].get(column, "") for column in columns]))
    lines.append("")
    lines.append("Generated from machine-readable metric JSON files. Missing cells indicate unavailable metrics.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_row(values: list[str]) -> str:
    escaped = [str(value).replace("|", "\\|") for value in values]
    return "| " + " | ".join(escaped) + " |"
