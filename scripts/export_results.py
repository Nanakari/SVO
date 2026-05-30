"""Export experiment tables from real metric JSON files only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_reproduce.evaluation.export_tables import (
    build_table,
    export_table,
    load_metric_files,
    selected_specs,
)
from paper_reproduce.utils.config import apply_overrides, load_yaml, resolve_path
from paper_reproduce.utils.io import ensure_parent, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export table templates and filled cells from machine-generated metric JSON files."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to global YAML config. Defaults to configs/default.yaml.",
    )
    parser.add_argument("--metrics-dir", help="Directory containing metric JSON files.")
    parser.add_argument("--out", help="Output table directory.")
    parser.add_argument(
        "--tables",
        nargs="+",
        help="Table names to export. Defaults to all supported tables.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["md", "csv", "json"],
        help="Output formats: md, csv, json.",
    )
    parser.add_argument(
        "--missing-value",
        default="",
        help="Display string for missing cells. Empty string by default.",
    )
    parser.add_argument("--precision", type=int, default=4, help="Decimal places for floats.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config value with dotted key syntax.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config, args.overrides)
    metrics_dir = _metrics_dir(args.metrics_dir, config)
    output_dir = _output_dir(args.out, config)
    metric_files = load_metric_files(metrics_dir)

    specs = selected_specs(args.tables)
    manifest: dict[str, Any] = {
        "metrics_dir": str(metrics_dir),
        "output_dir": str(output_dir),
        "metric_files_loaded": [str(item.path) for item in metric_files],
        "tables": [],
        "missing_value": args.missing_value,
        "formats": args.formats,
    }

    written_paths: list[Path] = []
    for spec in specs:
        table = build_table(
            spec,
            metric_files,
            missing_value=args.missing_value,
            precision=args.precision,
        )
        written = export_table(table, output_dir, formats=args.formats)
        written_paths.extend(written)
        manifest["tables"].append(
            {
                "name": table["name"],
                "title": table["title"],
                "rows": len(table["rows"]),
                "columns": len(table["columns"]),
                "missing_cells": _count_missing_cells(table),
                "files": [str(path) for path in written],
            }
        )

    manifest_path = ensure_parent(output_dir / "manifest.json")
    write_json(manifest_path, manifest)

    print(f"Metric files loaded: {len(metric_files)}")
    print(f"Tables exported: {len(specs)}")
    print(f"Files written: {len(written_paths) + 1}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {manifest_path}")


def _load_config(path: str, overrides: list[str]) -> dict[str, Any]:
    config_path = resolve_path(path, PROJECT_ROOT)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return apply_overrides(load_yaml(config_path), overrides)


def _metrics_dir(explicit: str | None, config: dict[str, Any]) -> Path:
    if explicit:
        return resolve_path(explicit, PROJECT_ROOT)
    output_dir = config.get("project", {}).get("output_dir", "outputs")
    return resolve_path(Path(output_dir) / "metrics", PROJECT_ROOT)


def _output_dir(explicit: str | None, config: dict[str, Any]) -> Path:
    if explicit:
        return resolve_path(explicit, PROJECT_ROOT)
    output_dir = config.get("project", {}).get("output_dir", "outputs")
    return resolve_path(Path(output_dir) / "tables", PROJECT_ROOT)


def _count_missing_cells(table: dict[str, Any]) -> int:
    return sum(1 for row in table["rows"] for cell in row["cells"] if cell["missing"])


if __name__ == "__main__":
    main()
