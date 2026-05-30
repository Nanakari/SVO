"""Sweep GroundingDINO box/text thresholds for SVO sensitivity analysis."""

from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_reproduce.evaluation.sweep_tables import (
    build_sweep_table,
    export_sweep_table,
    format_float_display,
    format_float_label,
)
from paper_reproduce.utils.config import load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SVO while sweeping GroundingDINO box/text thresholds."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Global YAML config.")
    parser.add_argument("--dataset", default="coco_chair", help="Dataset name or YAML path.")
    parser.add_argument("--method", default="configs/methods/svo.yaml", help="SVO method YAML.")
    parser.add_argument("--objects", required=True, help="Object JSONL from extract_objects.py.")
    parser.add_argument("--base-predictions", help="Base caption JSONL for relative latency.")
    parser.add_argument("--coco-annotations", help="Override COCO instances annotation path.")
    parser.add_argument("--output-dir", default="outputs/sweeps/detector_thresholds", help="Sweep output directory.")
    parser.add_argument("--risk-threshold", type=float, help="SVO risk threshold for object selection.")
    parser.add_argument("--box-thresholds", nargs="+", type=float, required=True, help="GroundingDINO box thresholds.")
    parser.add_argument("--text-thresholds", nargs="+", type=float, required=True, help="GroundingDINO text thresholds.")
    parser.add_argument(
        "--evidence-threshold",
        type=float,
        help="Visual evidence threshold. Defaults to each box threshold via verify_objects.py.",
    )
    parser.add_argument("--text-field", default="revised_caption", help="Caption field for CHAIR evaluation.")
    parser.add_argument("--backend", help="Override extraction backend during evaluation.")
    parser.add_argument("--limit", type=int, help="Limit verification records.")
    parser.add_argument("--missing-value", default="NA", help="Missing value display in exported tables.")
    parser.add_argument("--precision", type=int, default=4, help="Decimal places in exported tables.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite per-threshold JSONL outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    metrics_dir = output_dir / "metrics"
    table_dir = output_dir / "tables"
    dataset_name = _dataset_name(args.dataset)
    rows = []

    pairs = list(itertools.product(args.box_thresholds, args.text_thresholds))
    for box_threshold, text_threshold in pairs:
        box_label = format_float_label(box_threshold)
        text_label = format_float_label(text_threshold)
        method_name = f"svo_box_{box_label}_text_{text_label}"
        verification_path = output_dir / "verifications" / f"{method_name}.jsonl"
        revision_path = output_dir / "revisions" / f"{method_name}_revisions.jsonl"

        verify_cmd = [
            sys.executable,
            "scripts/verify_objects.py",
            "--config",
            args.config,
            "--method",
            args.method,
            "--input",
            args.objects,
            "--output",
            str(verification_path),
            "--set",
            f"verification.groundingdino.box_threshold={box_threshold}",
            "--set",
            f"verification.groundingdino.text_threshold={text_threshold}",
        ]
        if args.risk_threshold is not None:
            verify_cmd.extend(["--risk-threshold", str(args.risk_threshold)])
        if args.evidence_threshold is not None:
            verify_cmd.extend(["--evidence-threshold", str(args.evidence_threshold)])
        if args.limit is not None:
            verify_cmd.extend(["--limit", str(args.limit)])
        if args.overwrite:
            verify_cmd.append("--overwrite")

        revise_cmd = [
            sys.executable,
            "scripts/revise_captions.py",
            "--config",
            args.config,
            "--input",
            str(verification_path),
            "--output",
            str(revision_path),
        ]
        if args.overwrite:
            revise_cmd.append("--overwrite")

        chair_cmd = [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            args.config,
            "--dataset",
            args.dataset,
            "--task",
            "chair",
            "--predictions",
            str(revision_path),
            "--text-field",
            args.text_field,
            "--method",
            method_name,
            "--output",
            str(metrics_dir / f"{dataset_name}_{method_name}_chair.json"),
        ]
        _add_optional_eval_args(chair_cmd, args)

        efficiency_cmd = [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            args.config,
            "--dataset",
            args.dataset,
            "--task",
            "efficiency",
            "--objects",
            args.objects,
            "--verifications",
            str(verification_path),
            "--method",
            method_name,
            "--output",
            str(metrics_dir / f"{dataset_name}_{method_name}_efficiency.json"),
        ]
        if args.base_predictions:
            efficiency_cmd.extend(["--base-predictions", args.base_predictions])

        false_cmd = [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            args.config,
            "--dataset",
            args.dataset,
            "--task",
            "false_correction",
            "--predictions",
            str(revision_path),
            "--method",
            method_name,
            "--output",
            str(metrics_dir / f"{dataset_name}_{method_name}_false_correction.json"),
        ]
        _add_optional_eval_args(false_cmd, args)

        for command in (verify_cmd, revise_cmd, chair_cmd, efficiency_cmd, false_cmd):
            _run(command, dry_run=args.dry_run)

        rows.append(
            {
                "label": (
                    f"box={format_float_display(box_threshold)}, "
                    f"text={format_float_display(text_threshold)}"
                ),
                "method": method_name,
                "parameters": {
                    "risk_threshold": args.risk_threshold,
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                    "evidence_threshold": (
                        args.evidence_threshold
                        if args.evidence_threshold is not None
                        else box_threshold
                    ),
                },
            }
        )

    if not args.dry_run:
        table = build_sweep_table(
            name="detector_sensitivity",
            title="GroundingDINO Threshold Sensitivity",
            rows=rows,
            metrics_dir=metrics_dir,
            parameter_columns=[
                "risk_threshold",
                "box_threshold",
                "text_threshold",
                "evidence_threshold",
            ],
            dataset=dataset_name,
            missing_value=args.missing_value,
            precision=args.precision,
        )
        written = export_sweep_table(table, table_dir)
        print(f"Detector threshold pairs: {len(pairs)}")
        print(f"Metrics directory: {metrics_dir}")
        print(f"Tables: {', '.join(str(path) for path in written)}")


def _add_optional_eval_args(command: list[str], args: argparse.Namespace) -> None:
    if args.coco_annotations:
        command.extend(["--coco-annotations", args.coco_annotations])
    if args.backend:
        command.extend(["--backend", args.backend])


def _dataset_name(value: str) -> str:
    path = resolve_path(value, PROJECT_ROOT)
    if path is not None and path.exists():
        return str(load_yaml(path).get("name", path.stem))
    return value


def _run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
