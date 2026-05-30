"""Apply conservative SVO caption revisions from visual verification outputs."""

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

from paper_reproduce.revision import revise_caption
from paper_reproduce.utils.config import apply_overrides, load_yaml, resolve_path
from paper_reproduce.utils.io import append_jsonl, ensure_parent, existing_sample_ids, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply conservative text revisions to verified captions."
    )
    parser.add_argument("--config", required=True, help="Path to global YAML config.")
    parser.add_argument("--input", required=True, help="Verification JSONL from verify_objects.py.")
    parser.add_argument("--output", help="Output revision JSONL path.")
    parser.add_argument(
        "--allow-rule",
        action="append",
        dest="allow_rules",
        help="Restrict allowed revision rules. Can be repeated.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output instead of resuming from existing sample_ids.",
    )
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
    config = apply_overrides(load_yaml(args.config), args.overrides)
    input_path = resolve_path(args.input, PROJECT_ROOT)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Verification JSONL not found: {input_path}")
    output_path = _default_output_path(args.output, input_path)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    skip_ids = existing_sample_ids(output_path)

    allowed_rules = args.allow_rules or list(config.get("revision", {}).get("allow_rules", []))
    read_count = 0
    written = 0
    changed = 0
    skipped_actions = 0
    for read_count, record in enumerate(read_jsonl(input_path), start=1):
        sample_id = str(record.get("sample_id") or read_count)
        if sample_id in skip_ids:
            continue

        caption = str(record.get("caption") or "")
        result = revise_caption(
            caption,
            list(record.get("verified_objects", [])),
            allowed_rules=allowed_rules,
        )
        actions = [action.to_dict() for action in result.actions]
        changed += int(result.revised_caption != result.original_caption)
        skipped_actions += sum(1 for action in actions if action.get("action") == "skip")

        output_record = {
            "sample_id": sample_id,
            "image_id": str(record.get("image_id", "")),
            "image_path": record.get("image_path"),
            "dataset": record.get("dataset"),
            "method": record.get("method"),
            "object_method": record.get("object_method"),
            "caption_method": record.get("caption_method"),
            "original_caption": result.original_caption,
            "revised_caption": result.revised_caption,
            "actions": actions,
            "revision": {
                "strategy": config.get("revision", {}).get("strategy", "conservative"),
                "allow_rules": allowed_rules,
            },
            "source_file": str(input_path),
        }
        append_jsonl(output_path, [output_record])
        written += 1

    print(f"Read records: {read_count}")
    print(f"Skipped existing: {len(skip_ids)}")
    print(f"Wrote records: {written}")
    print(f"Changed captions: {changed}")
    print(f"Skipped revision actions: {skipped_actions}")
    print(f"Output: {output_path}")


def _default_output_path(explicit_output: str | None, input_path: Path) -> Path:
    if explicit_output:
        return ensure_parent(resolve_path(explicit_output, PROJECT_ROOT))
    return ensure_parent(PROJECT_ROOT / "outputs" / "revisions" / f"{input_path.stem}_revisions.jsonl")


if __name__ == "__main__":
    main()
