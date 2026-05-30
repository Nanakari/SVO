"""Configuration loading and command-line override helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    PyYAML is a project dependency, but imports stay lazy so `--help` and static checks work
    before the environment is fully installed.
    """

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "PyYAML is required to load config files. Install project dependencies with "
            "`pip install -e .[nlp,eval]` inside the target environment."
        ) from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {config_path}")
    return data


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings without mutating either input."""

    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config_files(paths: Iterable[str | Path | None]) -> dict[str, Any]:
    """Load and merge config files in order, skipping `None` values."""

    config: dict[str, Any] = {}
    for path in paths:
        if path is None:
            continue
        config = deep_merge(config, load_yaml(path))
    return config


def parse_override(value: str) -> Any:
    """Parse a command-line override value using JSON when possible."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply dotted-key overrides like `generation.max_new_tokens=64`."""

    updated = copy.deepcopy(dict(config))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must use key=value syntax: {override}")
        key, raw_value = override.split("=", 1)
        set_by_dotted_key(updated, key, parse_override(raw_value))
    return updated


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested dictionary value using dotted-key syntax."""

    if not dotted_key:
        raise ValueError("Override key cannot be empty")

    cursor: dict[str, Any] = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if next_value is None:
            next_value = {}
            cursor[part] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set nested key below non-mapping value: {part}")
        cursor = next_value
    cursor[parts[-1]] = value


def get_by_dotted_key(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a nested dictionary value using dotted-key syntax."""

    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def resolve_path(path: str | Path | None, root: str | Path) -> Path | None:
    """Resolve a possibly relative path against the project root."""

    if path is None:
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.exists():
        return path_obj.resolve()
    return Path(root) / path_obj
