"""Build static object hallucination priors from COCO validation captions."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_reproduce.evaluation import load_coco_gt_objects
from paper_reproduce.extraction import ObjectVocabulary, build_extractor
from paper_reproduce.scoring import compute_static_prior
from paper_reproduce.utils.config import apply_overrides, load_yaml, resolve_path
from paper_reproduce.utils.io import ensure_parent, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute static hallucination priors B(o) from generated captions and COCO GT."
    )
    parser.add_argument("--config", required=True, help="Path to global YAML config.")
    parser.add_argument("--captions", required=True, help="Caption JSONL generated on COCO train2017 validation split.")
    parser.add_argument("--coco-annotations", required=True, help="COCO instances annotation JSON.")
    parser.add_argument("--output", help="Output JSON path. Defaults to risk_scoring.static_prior_path.")
    parser.add_argument("--text-field", default="caption", help="Caption field in the input JSONL.")
    parser.add_argument("--backend", help="Override object_extraction.backend, e.g. spacy or rule.")
    parser.add_argument("--min-count", type=int, help="Low-frequency threshold m.")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Small constant for prior ratios.")
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
    if args.backend:
        config.setdefault("object_extraction", {})["backend"] = args.backend

    captions_path = resolve_path(args.captions, PROJECT_ROOT)
    annotations_path = resolve_path(args.coco_annotations, PROJECT_ROOT)
    if captions_path is None or not captions_path.exists():
        raise FileNotFoundError(f"Caption JSONL not found: {captions_path}")
    if annotations_path is None or not annotations_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {annotations_path}")

    min_count = int(args.min_count or config.get("risk_scoring", {}).get("min_prior_count", 5))
    output_path = _resolve_output_path(args.output, config)

    vocabulary = ObjectVocabulary.from_config(config, PROJECT_ROOT)
    extractor = build_extractor(config, PROJECT_ROOT)
    gt_by_image = load_coco_gt_objects(annotations_path, vocabulary)

    generated_counts: Counter[str] = Counter()
    hallucinated_counts: Counter[str] = Counter()
    missing_gt_images: list[str] = []
    caption_count = 0

    for caption_count, record in enumerate(read_jsonl(captions_path), start=1):
        image_id = str(record.get("image_id", ""))
        gt_objects = gt_by_image.get(image_id)
        if gt_objects is None:
            missing_gt_images.append(image_id)
            continue

        caption = record.get(args.text_field)
        if caption is None:
            raise ValueError(f"Caption record {record.get('sample_id', caption_count)} is missing `{args.text_field}`")
        for mention in extractor.extract(str(caption)):
            generated_counts[mention.normalized] += 1
            if mention.normalized not in gt_objects:
                hallucinated_counts[mention.normalized] += 1

    prior_data = compute_static_prior(
        generated_counts=generated_counts,
        hallucinated_counts=hallucinated_counts,
        min_count=min_count,
        epsilon=args.epsilon,
    )
    prior_data.update(
        {
            "source_captions": str(captions_path),
            "coco_annotation_file": str(annotations_path),
            "caption_records_read": caption_count,
            "missing_gt_image_count": len(missing_gt_images),
            "missing_gt_image_ids_preview": missing_gt_images[:20],
        }
    )
    write_json(output_path, prior_data)

    print(f"Caption records read: {caption_count}")
    print(f"Images missing GT: {len(missing_gt_images)}")
    print(f"Generated object mentions: {prior_data['generated_total']}")
    print(f"Hallucinated object mentions: {prior_data['hallucinated_total']}")
    print(f"Mean prior: {prior_data['mean_prior']}")
    print(f"Output: {output_path}")

def _resolve_output_path(explicit_output: str | None, config: dict[str, Any]) -> Path:
    if explicit_output:
        return ensure_parent(resolve_path(explicit_output, PROJECT_ROOT))
    configured = config.get("risk_scoring", {}).get("static_prior_path")
    if configured:
        return ensure_parent(resolve_path(configured, PROJECT_ROOT))
    return ensure_parent(PROJECT_ROOT / "outputs" / "priors" / "coco_static_prior.json")


if __name__ == "__main__":
    main()
