"""Extract object mentions from generated captions and compute SVO risk scores."""

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

from paper_reproduce.extraction import ExtractedObject, ObjectVocabulary, build_extractor
from paper_reproduce.scoring import RiskScorer
from paper_reproduce.utils.config import apply_overrides, deep_merge, load_yaml, resolve_path
from paper_reproduce.utils.io import append_jsonl, ensure_parent, existing_sample_ids, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract visible object phrases from caption JSONL and attach risk scores."
    )
    parser.add_argument("--config", required=True, help="Path to global YAML config.")
    parser.add_argument("--method", help="Optional method YAML config for risk-term ablations.")
    parser.add_argument("--input", required=True, help="Input caption JSONL from run_caption.py.")
    parser.add_argument("--output", help="Output object JSONL path.")
    parser.add_argument("--text-field", default="caption", help="Input field containing text to parse.")
    parser.add_argument(
        "--target-field",
        help="Optional structured object field to prefer before NLP extraction, e.g. target_object.",
    )
    parser.add_argument("--backend", help="Override object_extraction.backend, e.g. spacy or rule.")
    parser.add_argument(
        "--no-risk",
        action="store_true",
        help="Only extract objects; do not attach uncertainty/position/prior risk scores.",
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
    config = _load_config(args)
    if args.backend:
        config.setdefault("object_extraction", {})["backend"] = args.backend

    input_path = resolve_path(args.input, PROJECT_ROOT)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input caption JSONL not found: {input_path}")

    output_path = _default_output_path(args.output, input_path)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    skip_ids = existing_sample_ids(output_path)

    vocabulary = ObjectVocabulary.from_config(config, PROJECT_ROOT) if args.target_field else None
    extractor = None
    scorer = None if args.no_risk else RiskScorer.from_config(config, PROJECT_ROOT)
    processing_method = config.get("method", {}).get("name")

    read_count = 0
    written = 0
    extracted_mentions = 0
    for read_count, record in enumerate(read_jsonl(input_path), start=1):
        sample_id = str(record.get("sample_id") or read_count)
        if sample_id in skip_ids:
            continue

        caption = record.get(args.text_field)
        if caption is None:
            raise ValueError(f"Input record {sample_id} is missing text field `{args.text_field}`")
        caption_text = str(caption)
        mentions = _target_mentions(
            record=record,
            target_field=args.target_field,
            text=caption_text,
            vocabulary=vocabulary,
        )
        if mentions is None:
            if extractor is None:
                extractor = build_extractor(config, PROJECT_ROOT)
            mentions = extractor.extract(caption_text)
        if scorer is None:
            objects = [mention.to_dict() for mention in mentions]
        else:
            objects = scorer.score_objects(
                caption=caption_text,
                objects=mentions,
                token_scores=record.get("token_scores"),
            )

        output_record = {
            "sample_id": sample_id,
            "image_id": str(record.get("image_id", "")),
            "image_path": record.get("image_path"),
            "dataset": record.get("dataset"),
            "method": processing_method or record.get("method"),
            "caption_method": record.get("method"),
            "caption": caption_text,
            "objects": objects,
            "source_file": str(input_path),
            "extraction": {
                "backend": config.get("object_extraction", {}).get("backend"),
                "vocabulary_path": config.get("object_extraction", {}).get("vocabulary_path"),
                "risk_scored": scorer is not None,
                "target_field": args.target_field,
            },
        }
        append_jsonl(output_path, [output_record])
        written += 1
        extracted_mentions += len(objects)

    print(f"Read records: {read_count}")
    print(f"Skipped existing: {len(skip_ids)}")
    print(f"Wrote records: {written}")
    print(f"Extracted object mentions: {extracted_mentions}")
    print(f"Output: {output_path}")


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    if args.method:
        config = deep_merge(config, {"method": load_yaml(args.method)})
    return apply_overrides(config, args.overrides)


def _default_output_path(explicit_output: str | None, input_path: Path) -> Path:
    if explicit_output:
        return ensure_parent(resolve_path(explicit_output, PROJECT_ROOT))
    return ensure_parent(PROJECT_ROOT / "outputs" / "objects" / f"{input_path.stem}_objects.jsonl")


def _target_mentions(
    *,
    record: dict[str, Any],
    target_field: str | None,
    text: str,
    vocabulary: ObjectVocabulary | None,
) -> list[ExtractedObject] | None:
    if not target_field or vocabulary is None:
        return None
    raw_value = record.get(target_field)
    if raw_value is None:
        return None
    target_text = str(raw_value).strip()
    if not target_text:
        return None
    canonical = vocabulary.normalize(target_text)
    if canonical is None:
        return None

    lowered_text = text.lower()
    lowered_target = target_text.lower()
    start = lowered_text.find(lowered_target)
    if start >= 0:
        span = (start, start + len(target_text))
        mention_text = text[span[0] : span[1]]
    else:
        span = (0, 0)
        mention_text = target_text

    return [
        ExtractedObject(
            text=mention_text,
            normalized=canonical,
            span=span,
            object_index=1,
            source="structured_target",
            metadata={"target_field": target_field},
        )
    ]


if __name__ == "__main__":
    main()
