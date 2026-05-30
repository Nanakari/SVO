"""Check local dataset and model asset paths before launching experiments."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_reproduce.utils.config import load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate configured model and dataset asset paths.")
    parser.add_argument("--config", default="configs/assets.yaml", help="Asset YAML config.")
    parser.add_argument("--sha256", action="store_true", help="Compute sha256 for files that exist.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any path is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_ROOT)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Asset config not found: {config_path}")
    config = load_yaml(config_path)

    checks = list(_asset_checks(config.get("assets", {}))) + list(
        _dataset_checks(config.get("datasets", {}))
    )
    missing = 0
    for label, path in checks:
        resolved = resolve_path(path, PROJECT_ROOT)
        exists = bool(resolved and resolved.exists())
        missing += int(not exists)
        digest = ""
        if args.sha256 and exists and resolved.is_file():
            digest = f" sha256={_sha256(resolved)}"
        print(f"{'OK' if exists else 'MISSING'} {label}: {resolved}{digest}")

    print(f"Checked paths: {len(checks)}")
    print(f"Missing paths: {missing}")
    if args.strict and missing:
        raise SystemExit(1)


def _asset_checks(assets: Mapping[str, Any]) -> list[tuple[str, str]]:
    checks: list[tuple[str, str]] = []
    for name, payload in assets.items():
        if not isinstance(payload, Mapping):
            continue
        for key in ("local_dir", "repo_dir", "config_path", "checkpoint_path"):
            value = payload.get(key)
            if value:
                checks.append((f"asset.{name}.{key}", str(value)))
    return checks


def _dataset_checks(datasets: Mapping[str, Any]) -> list[tuple[str, str]]:
    checks: list[tuple[str, str]] = []
    for name, payload in datasets.items():
        if not isinstance(payload, Mapping):
            continue
        for key, value in payload.items():
            if value:
                checks.append((f"dataset.{name}.{key}", str(value)))
    return checks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
