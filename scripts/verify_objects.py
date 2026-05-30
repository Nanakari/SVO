"""Verify extracted objects with GroundingDINO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_reproduce.utils.config import apply_overrides, deep_merge, load_yaml, resolve_path
from paper_reproduce.utils.io import append_jsonl, ensure_parent, existing_sample_ids, read_jsonl
from paper_reproduce.verification import (
    build_verifier,
    count_reference_verifications,
    select_objects_for_verification,
    selection_config_from_method,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select extracted objects and verify them with GroundingDINO."
    )
    parser.add_argument("--config", required=True, help="Path to global YAML config.")
    parser.add_argument("--method", required=True, help="Method YAML config, e.g. svo or verify_all.")
    parser.add_argument("--input", required=True, help="Object JSONL from extract_objects.py.")
    parser.add_argument("--output", help="Output verification JSONL path.")
    parser.add_argument(
        "--reference",
        help="Reference verification/object JSONL for matched_random per-image counts.",
    )
    parser.add_argument("--risk-threshold", type=float, help="Override risk_scoring.threshold.")
    parser.add_argument(
        "--evidence-threshold",
        type=float,
        help="Override verification.evidence_threshold.",
    )
    parser.add_argument(
        "--policy",
        choices=["all", "risk_threshold", "matched_random"],
        help="Override method.pipeline.verify_policy.",
    )
    parser.add_argument("--limit", type=int, help="Limit the number of input records.")
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
    config = _load_config(args)
    if args.policy:
        config.setdefault("method", {}).setdefault("pipeline", {})["verify_policy"] = args.policy
    if args.evidence_threshold is not None:
        config.setdefault("verification", {})["evidence_threshold"] = args.evidence_threshold
    _fill_default_evidence_threshold(config)

    input_path = resolve_path(args.input, PROJECT_ROOT)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Object JSONL not found: {input_path}")
    output_path = _default_output_path(args.output, input_path, config)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    skip_ids = existing_sample_ids(output_path)

    selection = selection_config_from_method(config, risk_threshold_override=args.risk_threshold)
    reference_counts = _load_reference_counts(args.reference) if args.reference else None

    verifier = None
    read_count = 0
    written = 0
    total_selected = 0
    total_verified = 0
    for read_count, record in enumerate(read_jsonl(input_path), start=1):
        if args.limit is not None and read_count > args.limit:
            break
        sample_id = str(record.get("sample_id") or read_count)
        if sample_id in skip_ids:
            continue

        selected = select_objects_for_verification(
            list(record.get("objects", [])),
            selection,
            sample_id=sample_id,
            reference_counts=reference_counts,
        )
        total_selected += len(selected)

        verified_objects: list[dict[str, Any]] = []
        selected_latency = 0.0
        if selected:
            if verifier is None:
                verifier = build_verifier(config, PROJECT_ROOT)
            for object_record in selected:
                object_name = str(object_record["normalized"])
                result = verifier.verify(str(record.get("image_path")), object_name)
                result_record = result.to_dict()
                result_record["source_object"] = _source_object_payload(object_record)
                verified_objects.append(result_record)
                selected_latency += result.latency_sec

        output_record = {
            "sample_id": sample_id,
            "image_id": str(record.get("image_id", "")),
            "image_path": record.get("image_path"),
            "dataset": record.get("dataset"),
            "method": config.get("method", {}).get("name", record.get("method")),
            "object_method": record.get("method"),
            "caption_method": record.get("caption_method"),
            "caption": record.get("caption"),
            "verify_policy": selection.policy,
            "risk_threshold": selection.risk_threshold,
            "evidence_threshold": config.get("verification", {}).get("evidence_threshold"),
            "selected_objects": [_source_object_payload(item) for item in selected],
            "verified_objects": verified_objects,
            "external_queries": len(selected),
            "latency_sec": selected_latency,
            "source_file": str(input_path),
        }
        append_jsonl(output_path, [output_record])
        written += 1
        total_verified += len(verified_objects)

    print(f"Read records: {read_count}")
    print(f"Skipped existing: {len(skip_ids)}")
    print(f"Wrote records: {written}")
    print(f"Selected objects: {total_selected}")
    print(f"Verified objects: {total_verified}")
    print(f"Output: {output_path}")


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    config = deep_merge(config, {"method": load_yaml(args.method)})
    return apply_overrides(config, args.overrides)


def _fill_default_evidence_threshold(config: dict[str, Any]) -> None:
    verification = config.setdefault("verification", {})
    if verification.get("evidence_threshold") is not None:
        return
    groundingdino = verification.get("groundingdino", {})
    verification["evidence_threshold"] = float(groundingdino.get("box_threshold", 0.35))


def _load_reference_counts(path: str) -> dict[str, int]:
    reference_path = resolve_path(path, PROJECT_ROOT)
    if reference_path is None or not reference_path.exists():
        raise FileNotFoundError(f"Reference JSONL not found: {reference_path}")
    counts: dict[str, int] = {}
    for index, record in enumerate(read_jsonl(reference_path), start=1):
        sample_id = str(record.get("sample_id") or index)
        counts[sample_id] = count_reference_verifications(record)
    return counts


def _default_output_path(
    explicit_output: str | None, input_path: Path, config: Mapping[str, Any]
) -> Path:
    if explicit_output:
        return ensure_parent(resolve_path(explicit_output, PROJECT_ROOT))
    method_name = config.get("method", {}).get("name", "verification")
    return ensure_parent(PROJECT_ROOT / "outputs" / "verifications" / f"{input_path.stem}_{method_name}.jsonl")


def _source_object_payload(object_record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": object_record.get("text"),
        "normalized": object_record.get("normalized"),
        "span": object_record.get("span"),
        "object_index": object_record.get("object_index"),
        "risk": object_record.get("risk"),
        "source": object_record.get("source"),
    }


if __name__ == "__main__":
    main()
