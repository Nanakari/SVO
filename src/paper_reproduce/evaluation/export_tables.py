"""Export paper-ready table templates from real metric JSON files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from paper_reproduce.utils.io import ensure_parent, write_json


@dataclass(frozen=True)
class MetricFile:
    """One machine-generated metrics file."""

    path: Path
    dataset: str
    task: str
    method: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RowSpec:
    """Output table row specification."""

    label: str
    methods: tuple[str, ...]


@dataclass(frozen=True)
class ColumnSpec:
    """Output table column specification."""

    label: str
    task: str
    metric_path: tuple[str, ...]
    dataset: str | None = None


@dataclass(frozen=True)
class TableSpec:
    """Output table specification."""

    name: str
    title: str
    rows: tuple[RowSpec, ...]
    columns: tuple[ColumnSpec, ...]
    first_column: str = "Method"


METHOD_ROWS = {
    "main": (
        RowSpec("Base", ("base",)),
        RowSpec("VCD", ("vcd",)),
        RowSpec("OPERA", ("opera",)),
        RowSpec("Verify-All", ("verify_all",)),
        RowSpec("Random-Verify", ("random_verify",)),
        RowSpec("Ours", ("svo", "ours", "ours_full")),
    ),
    "ablation": (
        RowSpec("Ours-Full", ("svo", "ours", "ours_full")),
        RowSpec("w/o Uncertainty", ("svo_without_uncertainty", "without_uncertainty")),
        RowSpec("w/o Position", ("svo_without_position", "without_position")),
        RowSpec("w/o Prior", ("svo_without_prior", "without_prior")),
        RowSpec("Uncertainty Only", ("svo_only_uncertainty", "only_uncertainty")),
        RowSpec("Position Only", ("svo_only_position", "only_position")),
        RowSpec("Prior Only", ("svo_only_prior", "only_prior")),
        RowSpec("Random-Verify", ("random_verify",)),
        RowSpec("Verify-All", ("verify_all",)),
    ),
    "correction": (
        RowSpec("Verify-All", ("verify_all",)),
        RowSpec("Random-Verify", ("random_verify",)),
        RowSpec("Ours", ("svo", "ours", "ours_full")),
    ),
}


TABLE_SPECS: dict[str, TableSpec] = {
    "chair_main": TableSpec(
        name="chair_main",
        title="CHAIR Main Results",
        rows=METHOD_ROWS["main"],
        columns=(
            ColumnSpec("CHAIRs ↓", "chair", ("chairs",), "coco_chair"),
            ColumnSpec("CHAIRi ↓", "chair", ("chairi",), "coco_chair"),
            ColumnSpec("Avg. Length", "chair", ("average_length",), "coco_chair"),
            ColumnSpec("Correct Object Coverage ↑", "chair", ("correct_object_coverage",), "coco_chair"),
        ),
    ),
    "pope_detailed": TableSpec(
        name="pope_detailed",
        title="POPE Detailed Results",
        rows=METHOD_ROWS["main"],
        columns=(
            ColumnSpec("Random Accuracy", "pope", ("random", "accuracy"), "pope"),
            ColumnSpec("Random Precision", "pope", ("random", "precision"), "pope"),
            ColumnSpec("Random Recall", "pope", ("random", "recall"), "pope"),
            ColumnSpec("Random F1", "pope", ("random", "f1"), "pope"),
            ColumnSpec("Random Yes Ratio", "pope", ("random", "yes_ratio"), "pope"),
            ColumnSpec("Popular Accuracy", "pope", ("popular", "accuracy"), "pope"),
            ColumnSpec("Popular Precision", "pope", ("popular", "precision"), "pope"),
            ColumnSpec("Popular Recall", "pope", ("popular", "recall"), "pope"),
            ColumnSpec("Popular F1", "pope", ("popular", "f1"), "pope"),
            ColumnSpec("Popular Yes Ratio", "pope", ("popular", "yes_ratio"), "pope"),
            ColumnSpec("Adversarial Accuracy", "pope", ("adversarial", "accuracy"), "pope"),
            ColumnSpec("Adversarial Precision", "pope", ("adversarial", "precision"), "pope"),
            ColumnSpec("Adversarial Recall", "pope", ("adversarial", "recall"), "pope"),
            ColumnSpec("Adversarial F1", "pope", ("adversarial", "f1"), "pope"),
            ColumnSpec("Adversarial Yes Ratio", "pope", ("adversarial", "yes_ratio"), "pope"),
        ),
    ),
    "pope_summary": TableSpec(
        name="pope_summary",
        title="POPE Summary Results",
        rows=METHOD_ROWS["main"],
        columns=(
            ColumnSpec("Random F1", "pope", ("random", "f1"), "pope"),
            ColumnSpec("Popular F1", "pope", ("popular", "f1"), "pope"),
            ColumnSpec("Adversarial F1", "pope", ("adversarial", "f1"), "pope"),
            ColumnSpec("Avg F1", "pope", ("overall", "f1"), "pope"),
        ),
    ),
    "ablation": TableSpec(
        name="ablation",
        title="Risk Scoring Ablation",
        rows=METHOD_ROWS["ablation"],
        columns=(
            ColumnSpec("CHAIRs ↓", "chair", ("chairs",), "coco_chair"),
            ColumnSpec("CHAIRi ↓", "chair", ("chairi",), "coco_chair"),
            ColumnSpec("Verification Rate ↓", "efficiency", ("verification_rate",), "coco_chair"),
            ColumnSpec("Latency × ↓", "efficiency", ("relative_latency",), "coco_chair"),
        ),
    ),
    "efficiency": TableSpec(
        name="efficiency",
        title="Efficiency Analysis",
        rows=METHOD_ROWS["main"],
        columns=(
            ColumnSpec("CHAIRs ↓", "chair", ("chairs",), "coco_chair"),
            ColumnSpec("CHAIRi ↓", "chair", ("chairi",), "coco_chair"),
            ColumnSpec("Verification Rate ↓", "efficiency", ("verification_rate",), "coco_chair"),
            ColumnSpec("External Queries / Image ↓", "efficiency", ("external_queries_per_image",), "coco_chair"),
            ColumnSpec("Latency × ↓", "efficiency", ("relative_latency",), "coco_chair"),
        ),
    ),
    "false_correction": TableSpec(
        name="false_correction",
        title="False-Correction Analysis",
        rows=METHOD_ROWS["correction"],
        columns=(
            ColumnSpec(
                "Hallucinated Removal ↑",
                "false_correction",
                ("hallucinated_object_removal",),
                "coco_chair",
            ),
            ColumnSpec(
                "Correct Retention ↑",
                "false_correction",
                ("correct_object_retention",),
                "coco_chair",
            ),
            ColumnSpec("FCR ↓", "false_correction", ("false_correction_rate",), "coco_chair"),
        ),
    ),
}


def load_metric_files(metrics_dir: str | Path) -> list[MetricFile]:
    """Load metric JSON files recursively from a directory."""

    root = Path(metrics_dir)
    if not root.exists():
        return []
    metric_files: list[MetricFile] = []
    for path in sorted(root.rglob("*.json")):
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        metrics = data.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        metric_files.append(
            MetricFile(
                path=path,
                dataset=str(data.get("dataset", "")),
                task=str(data.get("task", "")),
                method=str(data.get("method", "")),
                metrics=metrics,
            )
        )
    return metric_files


def build_table(
    spec: TableSpec,
    metric_files: list[MetricFile],
    *,
    missing_value: str = "",
    precision: int = 4,
) -> dict[str, Any]:
    """Build one table payload from real metric files."""

    rows: list[dict[str, Any]] = []
    for row_spec in spec.rows:
        row_cells: list[dict[str, Any]] = []
        row_display: dict[str, str] = {spec.first_column: row_spec.label}
        for column in spec.columns:
            value, source = _lookup_metric(metric_files, row_spec, column)
            cell = {
                "column": column.label,
                "value": value,
                "display": _format_value(value, missing_value=missing_value, precision=precision),
                "missing": value is None,
                "source_file": str(source) if source else None,
                "metric_path": ".".join(column.metric_path),
                "task": column.task,
                "dataset": column.dataset,
            }
            row_cells.append(cell)
            row_display[column.label] = cell["display"]
        rows.append(
            {
                "label": row_spec.label,
                "method_candidates": list(row_spec.methods),
                "cells": row_cells,
                "display": row_display,
            }
        )

    return {
        "name": spec.name,
        "title": spec.title,
        "columns": [spec.first_column] + [column.label for column in spec.columns],
        "rows": rows,
        "missing_value": missing_value,
    }


def export_table(
    table: Mapping[str, Any],
    output_dir: str | Path,
    *,
    formats: Iterable[str],
) -> list[Path]:
    """Export a table in requested formats."""

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


def selected_specs(names: Iterable[str] | None) -> list[TableSpec]:
    """Resolve requested table names."""

    if not names:
        return list(TABLE_SPECS.values())
    specs: list[TableSpec] = []
    for name in names:
        if name not in TABLE_SPECS:
            raise ValueError(f"Unknown table `{name}`. Available: {', '.join(TABLE_SPECS)}")
        specs.append(TABLE_SPECS[name])
    return specs


def _lookup_metric(
    metric_files: list[MetricFile], row: RowSpec, column: ColumnSpec
) -> tuple[Any, Path | None]:
    for method in row.methods:
        for metric_file in metric_files:
            if metric_file.method != method:
                continue
            if column.dataset is not None and metric_file.dataset != column.dataset:
                continue
            if metric_file.task != column.task:
                continue
            value = _nested_get(metric_file.metrics, column.metric_path)
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
        return f"{value:.{precision}f}".rstrip("0").rstrip(".")
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
