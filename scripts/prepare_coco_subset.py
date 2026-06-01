"""Prepare a COCO image subset from split image ids.

The script keeps the full COCO annotation file unchanged and prepares only the image files needed
for validation. A full train2017 directory and a subset directory are both compatible with the COCO
loader as long as the same split file is supplied.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_reproduce.datasets.coco_stream import (
    load_coco_image_index,
    read_image_id_split,
    sample_coco_image_ids,
)
from paper_reproduce.utils.config import resolve_path
from paper_reproduce.utils.io import ensure_parent, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy or download the COCO images referenced by a split file."
    )
    parser.add_argument(
        "--coco-annotations",
        default="data/coco/annotations/instances_train2017.json",
        help="Full COCO instances annotation JSON.",
    )
    parser.add_argument(
        "--split-file",
        default="configs/splits/coco_train2017_val2000_seed42.txt",
        help="Split file with one image_id per line. Created if missing.",
    )
    parser.add_argument("--sample-size", type=int, default=2000, help="Image count if split is created.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed if split is created.")
    parser.add_argument(
        "--output-image-root",
        default="data/coco/train2017_val2000",
        help="Destination image subset directory.",
    )
    parser.add_argument(
        "--source-image-root",
        default="data/coco/train2017",
        help="Optional full COCO image directory to copy from.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download images that are missing from --source-image-root.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required when --download-missing is used.",
    )
    parser.add_argument(
        "--base-url",
        default="http://images.cocodataset.org/train2017",
        help="Base URL for downloading missing train2017 images.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to materialize images found in --source-image-root.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying/downloading.")
    parser.add_argument("--manifest", help="Output manifest JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.download_missing and not args.confirm:
        raise ValueError("--download-missing requires --confirm")

    annotation_path = _existing_path(args.coco_annotations, "COCO annotation")
    split_path = resolve_path(args.split_file, PROJECT_ROOT)
    output_root = resolve_path(args.output_image_root, PROJECT_ROOT)
    source_root = resolve_path(args.source_image_root, PROJECT_ROOT)
    manifest_path = resolve_path(
        args.manifest or Path(args.output_image_root) / "subset_manifest.json",
        PROJECT_ROOT,
    )

    image_ids = _load_or_create_split(
        split_path=split_path,
        annotation_path=annotation_path,
        sample_size=args.sample_size,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    images = load_coco_image_index(annotation_path, image_ids)
    missing_annotation_ids = [image_id for image_id in image_ids if image_id not in images]
    if missing_annotation_ids:
        preview = ", ".join(missing_annotation_ids[:10])
        raise ValueError(
            f"Split contains {len(missing_annotation_ids)} ids not found in annotations: {preview}"
        )

    stats = {
        "annotation_file": str(annotation_path),
        "split_file": str(split_path),
        "output_image_root": str(output_root),
        "source_image_root": str(source_root),
        "sample_size": len(image_ids),
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "download_missing": args.download_missing,
        "existing": 0,
        "copied": 0,
        "downloaded": 0,
        "missing": 0,
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        image = images[image_id]
        file_name = str(image["file_name"])
        destination = output_root / file_name
        if destination.exists():
            stats["existing"] += 1
            continue

        source = source_root / file_name if source_root is not None else None
        if source is not None and source.exists():
            _materialize(source, destination, args.copy_mode, dry_run=args.dry_run)
            stats["copied"] += 1
            continue

        if args.download_missing:
            _download_file(f"{args.base_url.rstrip('/')}/{file_name}", destination, dry_run=args.dry_run)
            stats["downloaded"] += 1
            continue

        stats["missing"] += 1

    if not args.dry_run:
        write_json(manifest_path, stats)

    for key in ["existing", "copied", "downloaded", "missing"]:
        print(f"{key}: {stats[key]}")
    print(f"Split images: {len(image_ids)}")
    print(f"Output image root: {output_root}")
    print(f"Manifest: {manifest_path}")
    if stats["missing"]:
        raise SystemExit(
            "Some images are missing. Provide --source-image-root with a full COCO directory "
            "or use --download-missing --confirm."
        )


def _existing_path(value: str, label: str) -> Path:
    path = resolve_path(value, PROJECT_ROOT)
    if path is None or not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _load_or_create_split(
    *,
    split_path: Path,
    annotation_path: Path,
    sample_size: int,
    seed: int,
    dry_run: bool,
) -> list[str]:
    if split_path.exists():
        image_ids = read_image_id_split(split_path)
    else:
        image_ids = sample_coco_image_ids(annotation_path, sample_size, seed)
        if not dry_run:
            ensure_parent(split_path).write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    return image_ids


def _materialize(source: Path, destination: Path, mode: str, *, dry_run: bool) -> None:
    print(f"{mode}: {source} -> {destination}")
    if dry_run:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        destination.hardlink_to(source)
    elif mode == "symlink":
        destination.symlink_to(source)
    else:  # pragma: no cover - argparse prevents this path
        raise ValueError(f"Unsupported copy mode: {mode}")


def _download_file(url: str, destination: Path, *, dry_run: bool) -> None:
    print(f"download: {url} -> {destination}")
    if dry_run:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


if __name__ == "__main__":
    main()
