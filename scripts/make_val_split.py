"""Create a deterministic COCO image-id split for threshold tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_reproduce.utils.config import resolve_path
from paper_reproduce.utils.io import ensure_parent, write_json
from paper_reproduce.datasets.coco_stream import sample_coco_image_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample deterministic image ids from a COCO annotation JSON."
    )
    parser.add_argument("--coco-annotations", required=True, help="COCO instances annotation JSON.")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of image ids to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default="configs/splits/coco_train2017_val5000_seed42.txt",
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

    sampled = sample_coco_image_ids(annotation_path, args.sample_size, args.seed)

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

    print(f"Images sampled: {len(sampled)}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
