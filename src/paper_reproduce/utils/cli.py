"""Shared command-line helpers for experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from paper_reproduce.utils.config import apply_overrides, deep_merge, load_yaml


def project_root_from_script(script_file: str | Path) -> Path:
    """Return the repository root for scripts stored under `scripts/`."""

    return Path(script_file).resolve().parents[1]


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    """Add config arguments shared by stage-2 scripts."""

    parser.add_argument("--config", required=True, help="Path to the global YAML config.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset YAML config.")
    parser.add_argument("--method", required=True, help="Path to the method YAML config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config value with dotted key syntax, e.g. generation.max_new_tokens=64.",
    )


def load_cli_config(args: argparse.Namespace) -> dict:
    """Load global, dataset, and method config files plus CLI overrides."""

    base_config = load_yaml(args.config)
    dataset_config = load_yaml(args.dataset)
    method_config = load_yaml(args.method)
    config = deep_merge(base_config, {"dataset": dataset_config, "method": method_config})
    return apply_overrides(config, args.overrides)


def positive_int(value: str) -> int:
    """argparse type for positive integers."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def optional_limit(items: Sequence, limit: int | None) -> Sequence:
    """Return an optionally limited sequence."""

    if limit is None:
        return items
    return items[:limit]
