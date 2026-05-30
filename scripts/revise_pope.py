"""Apply the POPE Yes-to-No SVO revision protocol."""

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

from paper_reproduce.revision import revise_pope_answer
from paper_reproduce.utils.config import apply_overrides, load_yaml, resolve_path
from paper_reproduce.utils.io import append_jsonl, ensure_parent, existing_sample_ids, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Revise POPE answers using visual verification.")
    parser.add_argument("--config", required=True, help="Path to global YAML config.")
    parser.add_argument("--predictions", required=True, help="POPE prediction JSONL from run_pope.py.")
    parser.add_argument("--verifications", required=True, help="POPE verification JSONL.")
    parser.add_argument("--output", help="Output revised POPE JSONL path.")
    parser.add_argument(
        "--allow-no-to-yes",
        action="store_true",
        help="Enable No-to-Yes correction. Off by default for the SVO protocol.",
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
    predictions_path = resolve_path(args.predictions, PROJECT_ROOT)
    verifications_path = resolve_path(args.verifications, PROJECT_ROOT)
    if predictions_path is None or not predictions_path.exists():
        raise FileNotFoundError(f"POPE prediction JSONL not found: {predictions_path}")
    if verifications_path is None or not verifications_path.exists():
        raise FileNotFoundError(f"POPE verification JSONL not found: {verifications_path}")
    output_path = _default_output_path(args.output, predictions_path)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    skip_ids = existing_sample_ids(output_path)

    verifications = _load_by_sample_id(verifications_path)
    no_to_yes = bool(args.allow_no_to_yes or config.get("revision", {}).get("no_to_yes_for_pope", False))

    read_count = 0
    written = 0
    changed = 0
    missing_verifications = 0
    for read_count, prediction in enumerate(read_jsonl(predictions_path), start=1):
        sample_id = str(prediction.get("sample_id") or read_count)
        if sample_id in skip_ids:
            continue

        verification = verifications.get(sample_id)
        missing_verifications += int(verification is None)
        revised = revise_pope_answer(prediction, verification, no_to_yes=no_to_yes)
        changed += int(revised["original_answer"] != revised["revised_answer"])

        output_record: dict[str, Any] = dict(prediction)
        output_record.update(
            {
                "sample_id": sample_id,
                "original_answer": revised["original_answer"],
                "revised_answer": revised["revised_answer"],
                "answer": revised["revised_answer"],
                "action": revised["action"],
                "verified_objects": revised["verified_objects"],
                "revision": {
                    "strategy": "pope_yes_to_no",
                    "no_to_yes_enabled": no_to_yes,
                },
                "source_prediction_file": str(predictions_path),
                "source_verification_file": str(verifications_path),
            }
        )
        append_jsonl(output_path, [output_record])
        written += 1

    print(f"Read predictions: {read_count}")
    print(f"Skipped existing: {len(skip_ids)}")
    print(f"Wrote records: {written}")
    print(f"Changed answers: {changed}")
    print(f"Missing verifications: {missing_verifications}")
    print(f"Output: {output_path}")


def _load_by_sample_id(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(read_jsonl(path), start=1):
        records[str(record.get("sample_id") or index)] = record
    return records


def _default_output_path(explicit_output: str | None, input_path: Path) -> Path:
    if explicit_output:
        return ensure_parent(resolve_path(explicit_output, PROJECT_ROOT))
    return ensure_parent(PROJECT_ROOT / "outputs" / "revisions" / f"{input_path.stem}_pope_revised.jsonl")


if __name__ == "__main__":
    main()
