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
    parser.add_argument(
        "--assets",
        help="Comma-separated asset names to check, e.g. llava,groundingdino.",
    )
    parser.add_argument(
        "--datasets",
        help="Comma-separated dataset names to check, e.g. coco_chair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_ROOT)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Asset config not found: {config_path}")
    config = load_yaml(config_path)

    asset_filter = _name_filter(args.assets)
    dataset_filter = _name_filter(args.datasets)
    checks = list(_asset_checks(config.get("assets", {}), asset_filter)) + list(
        _dataset_checks(config.get("datasets", {}), dataset_filter)
    )
    missing = 0
    for check in checks:
        exists, resolved, digest = _run_check(check, args.sha256)
        missing += int(not exists)
        print(f"{'OK' if exists else 'MISSING'} {check['label']}: {resolved}{digest}")

    print(f"Checked paths: {len(checks)}")
    print(f"Missing paths: {missing}")
    if args.strict and missing:
        raise SystemExit(1)


def _asset_checks(
    assets: Mapping[str, Any], allowed: set[str] | None
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for name, payload in assets.items():
        if not isinstance(payload, Mapping):
            continue
        if allowed is not None and name not in allowed:
            continue
        for key in ("local_dir", "repo_dir", "config_path", "checkpoint_path"):
            value = payload.get(key)
            if value:
                checks.append({"kind": "path", "label": f"asset.{name}.{key}", "value": str(value)})
        model_id = payload.get("model_id")
        hf_files = payload.get("hf_files", [])
        if model_id and isinstance(hf_files, list):
            for filename in hf_files:
                checks.append(
                    {
                        "kind": "hf_cache",
                        "label": f"asset.{name}.hf_cache.{filename}",
                        "model_id": str(model_id),
                        "filename": str(filename),
                    }
                )
    return checks


def _dataset_checks(
    datasets: Mapping[str, Any], allowed: set[str] | None
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for name, payload in datasets.items():
        if not isinstance(payload, Mapping):
            continue
        if allowed is not None and name not in allowed:
            continue
        for key, value in payload.items():
            if value:
                checks.append({"kind": "path", "label": f"dataset.{name}.{key}", "value": str(value)})
    return checks


def _run_check(check: Mapping[str, str], use_sha256: bool) -> tuple[bool, str, str]:
    if check["kind"] == "hf_cache":
        return _hf_cache_check(check["model_id"], check["filename"])
    resolved = resolve_path(check["value"], PROJECT_ROOT)
    exists = bool(resolved and resolved.exists())
    digest = ""
    if use_sha256 and exists and resolved.is_file():
        digest = f" sha256={_sha256(resolved)}"
    return exists, str(resolved), digest


def _hf_cache_check(model_id: str, filename: str) -> tuple[bool, str, str]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False, f"{model_id}:{filename} (huggingface_hub not installed)", ""
    path = try_to_load_from_cache(model_id, filename)
    if path is None or not isinstance(path, str):
        return False, f"{model_id}:{filename} (not cached)", ""
    return Path(path).exists(), path, ""


def _name_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
