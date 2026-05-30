"""Create a deterministic COCO image-id split for threshold tuning."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_reproduce.utils.config import resolve_path
from paper_reproduce.utils.io import ensure_parent, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample deterministic image ids from a COCO annotation JSON."
    )
    parser.add_argument("--coco-annotations", required=True, help="COCO instances annotation JSON.")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of image ids to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default="configs/splits/coco_train2017_val500_seed42.txt",
        help="Output split file.",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "json", "jsonl"],
        default="txt",
        help="Output format. txt writes one image_id per line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_path = resolve_path(args.coco_annotations, PROJECT_ROOT)
    if annotation_path is None or not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    image_ids = sorted({str(image["id"]) for image in data.get("images", []) if "id" in image})
    if args.sample_size > len(image_ids):
        raise ValueError(
            f"Requested {args.sample_size} images, but annotation file contains {len(image_ids)}"
        )

    rng = random.Random(args.seed)
    sampled = list(image_ids)
    rng.shuffle(sampled)
    sampled = sorted(sampled[: args.sample_size], key=lambda value: int(value) if value.isdigit() else value)

    output_path = ensure_parent(resolve_path(args.output, PROJECT_ROOT))
    if args.format == "txt":
        output_path.write_text("\n".join(sampled) + "\n", encoding="utf-8")
    elif args.format == "jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for image_id in sampled:
                handle.write(json.dumps({"image_id": image_id}, ensure_ascii=False) + "\n")
    else:
        write_json(
            output_path,
            {
                "source_annotation_file": str(annotation_path),
                "sample_size": args.sample_size,
                "seed": args.seed,
                "image_ids": sampled,
            },
        )

    print(f"Images available: {len(image_ids)}")
    print(f"Images sampled: {len(sampled)}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
