"""Evaluate generated outputs without fabricating experiment results."""

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

from paper_reproduce.evaluation import (
    OfficialChairMapper,
    amber_object_score,
    evaluate_chair_records_internal,
    evaluate_chair_records_official,
    evaluate_efficiency_records,
    evaluate_false_correction_records,
    evaluate_yes_no_records,
    load_coco_category_names,
    load_coco_gt_objects,
    load_coco_gt_objects_official,
)
from paper_reproduce.evaluation.common import infer_method, load_records, write_metrics
from paper_reproduce.extraction import ObjectVocabulary, build_extractor
from paper_reproduce.utils.config import apply_overrides, deep_merge, load_yaml, resolve_path
from paper_reproduce.utils.io import ensure_parent
from paper_reproduce.utils.answers import YES_NO_NORMALIZERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CHAIR, POPE, AMBER, efficiency, and false-correction metrics."
    )
    parser.add_argument("--config", required=True, help="Path to global YAML config.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset YAML path or dataset name under configs/datasets, e.g. coco_chair.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["chair", "pope", "amber", "efficiency", "false_correction"],
        help="Evaluation task to run.",
    )
    parser.add_argument("--predictions", help="Prediction/revision JSONL for the task.")
    parser.add_argument("--objects", help="Object JSONL for efficiency denominator.")
    parser.add_argument("--verifications", help="Verification JSONL for efficiency metrics.")
    parser.add_argument("--base-predictions", help="Base caption JSONL for relative latency.")
    parser.add_argument("--coco-annotations", help="Override COCO instances annotation path.")
    parser.add_argument("--coco-caption-annotations", help="Override COCO caption annotation path.")
    parser.add_argument("--output", help="Output metrics JSON path.")
    parser.add_argument("--method", help="Override method name in metric output.")
    parser.add_argument("--text-field", help="Text field for CHAIR evaluation.")
    parser.add_argument(
        "--chair-backend",
        choices=["official", "internal"],
        default="official",
        help="CHAIR scorer backend. Default: official-compatible.",
    )
    parser.add_argument("--answer-field", default="answer", help="Answer field for yes/no metrics.")
    parser.add_argument("--label-field", default="label", help="Label field for yes/no metrics.")
    parser.add_argument(
        "--pope-normalizer",
        choices=YES_NO_NORMALIZERS,
        default="official",
        help="POPE answer normalizer. Default follows the public POPE evaluation script.",
    )
    parser.add_argument("--group-field", help="Optional grouping field for yes/no metrics.")
    parser.add_argument(
        "--backend",
        help="Override object_extraction.backend for internal CHAIR and false-correction evaluation.",
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
    config, dataset_config, dataset_config_path = _load_configs(args)
    if args.backend:
        config.setdefault("object_extraction", {})["backend"] = args.backend

    dataset_name = str(dataset_config.get("name", Path(dataset_config_path).stem))
    source_files: list[Path] = [Path(args.config), Path(dataset_config_path)]

    if args.task == "chair":
        output_path, method, metrics, counts, task_sources = _run_chair(args, config, dataset_config)
    elif args.task == "false_correction":
        output_path, method, metrics, counts, task_sources = _run_false_correction(
            args, config, dataset_config
        )
    elif args.task == "pope":
        output_path, method, metrics, counts, task_sources = _run_pope(args, dataset_config)
    elif args.task == "amber":
        output_path, method, metrics, counts, task_sources = _run_amber(args, dataset_config)
    elif args.task == "efficiency":
        output_path, method, metrics, counts, task_sources = _run_efficiency(args, dataset_config)
    else:  # pragma: no cover - argparse prevents this path
        raise ValueError(f"Unsupported task: {args.task}")

    source_files.extend(task_sources)
    write_metrics(
        output_path=output_path,
        dataset=dataset_name,
        task=args.task,
        method=method,
        metrics=metrics,
        counts=counts,
        source_files=source_files,
        notes=_notes_for_task(args.task),
    )
    print(f"Task: {args.task}")
    print(f"Dataset: {dataset_name}")
    print(f"Method: {method}")
    print(f"Output: {output_path}")


def _run_chair(
    args: argparse.Namespace, config: dict[str, Any], dataset_config: dict[str, Any]
) -> tuple[Path, str, dict[str, Any], dict[str, Any], list[Path]]:
    predictions_path = _required_path(args.predictions, "predictions")
    records = load_records(predictions_path)
    annotation_path = _coco_annotation_path(args, dataset_config)
    text_field = args.text_field or _default_caption_text_field(records)
    sources = [predictions_path, annotation_path]
    if args.chair_backend == "official":
        category_names = load_coco_category_names(annotation_path)
        mapper = OfficialChairMapper(category_names)
        caption_annotation_path = _coco_caption_annotation_path(args, dataset_config)
        if caption_annotation_path is not None:
            sources.append(caption_annotation_path)
        gt_by_image = load_coco_gt_objects_official(
            annotation_path,
            mapper,
            caption_annotation_path=caption_annotation_path,
        )
        metrics, counts = evaluate_chair_records_official(
            records,
            gt_by_image=gt_by_image,
            mapper=mapper,
            text_field=text_field,
        )
    else:
        extractor = build_extractor(config, PROJECT_ROOT)
        vocabulary = ObjectVocabulary.from_config(config, PROJECT_ROOT)
        gt_by_image = load_coco_gt_objects(annotation_path, vocabulary)
        metrics, counts = evaluate_chair_records_internal(
            records,
            gt_by_image=gt_by_image,
            extractor=extractor,
            text_field=text_field,
        )
    method = args.method or infer_method(records)
    output_path = _default_output_path(args.output, dataset_config, method, "chair")
    return output_path, method, metrics, counts, sources


def _run_false_correction(
    args: argparse.Namespace, config: dict[str, Any], dataset_config: dict[str, Any]
) -> tuple[Path, str, dict[str, Any], dict[str, Any], list[Path]]:
    predictions_path = _required_path(args.predictions, "predictions")
    records = load_records(predictions_path)
    extractor = build_extractor(config, PROJECT_ROOT)
    vocabulary = ObjectVocabulary.from_config(config, PROJECT_ROOT)
    annotation_path = _coco_annotation_path(args, dataset_config)
    gt_by_image = load_coco_gt_objects(annotation_path, vocabulary)
    metrics, counts = evaluate_false_correction_records(
        records, gt_by_image=gt_by_image, extractor=extractor
    )
    method = args.method or infer_method(records)
    output_path = _default_output_path(args.output, dataset_config, method, "false_correction")
    return output_path, method, metrics, counts, [predictions_path, annotation_path]


def _run_pope(
    args: argparse.Namespace, dataset_config: dict[str, Any]
) -> tuple[Path, str, dict[str, Any], dict[str, Any], list[Path]]:
    predictions_path = _required_path(args.predictions, "predictions")
    records = load_records(predictions_path)
    group_field = args.group_field if args.group_field is not None else "setting"
    metrics, counts = evaluate_yes_no_records(
        records,
        group_field=group_field,
        answer_field=args.answer_field,
        label_field=args.label_field,
        answer_normalizer=args.pope_normalizer,
        label_normalizer="strict",
    )
    method = args.method or infer_method(records)
    output_path = _default_output_path(args.output, dataset_config, method, "pope")
    return output_path, method, metrics, counts, [predictions_path]


def _run_amber(
    args: argparse.Namespace, dataset_config: dict[str, Any]
) -> tuple[Path, str, dict[str, Any], dict[str, Any], list[Path]]:
    predictions_path = _required_path(args.predictions, "predictions")
    records = load_records(predictions_path)
    metrics, counts = evaluate_yes_no_records(
        records,
        group_field=args.group_field,
        answer_field=args.answer_field,
        label_field=args.label_field,
    )
    if args.group_field:
        overall = metrics.get("overall", {})
        metrics["overall"]["amber_object_score"] = amber_object_score(overall)
    else:
        metrics["amber_object_score"] = amber_object_score(metrics)
    method = args.method or infer_method(records)
    output_path = _default_output_path(args.output, dataset_config, method, "amber")
    return output_path, method, metrics, counts, [predictions_path]


def _run_efficiency(
    args: argparse.Namespace, dataset_config: dict[str, Any]
) -> tuple[Path, str, dict[str, Any], dict[str, Any], list[Path]]:
    objects_path = _required_path(args.objects, "objects")
    verifications_path = _required_path(args.verifications, "verifications")
    object_records = load_records(objects_path)
    verification_records = load_records(verifications_path)
    base_path = _required_path(args.base_predictions, "base-predictions") if args.base_predictions else None
    base_records = load_records(base_path) if base_path else None
    metrics, counts = evaluate_efficiency_records(
        object_records=object_records,
        verification_records=verification_records,
        base_records=base_records,
    )
    method = args.method or infer_method(verification_records)
    output_path = _default_output_path(args.output, dataset_config, method, "efficiency")
    sources = [objects_path, verifications_path]
    if base_path:
        sources.append(base_path)
    return output_path, method, metrics, counts, sources


def _load_configs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path]:
    base_config = load_yaml(args.config)
    dataset_path = _dataset_config_path(args.dataset)
    dataset_config = load_yaml(dataset_path)
    config = apply_overrides(deep_merge(base_config, {"dataset": dataset_config}), args.overrides)
    return config, dataset_config, dataset_path


def _dataset_config_path(value: str) -> Path:
    direct = resolve_path(value, PROJECT_ROOT)
    if direct is not None and direct.exists():
        return direct
    named = PROJECT_ROOT / "configs" / "datasets" / f"{value}.yaml"
    if named.exists():
        return named
    raise FileNotFoundError(f"Dataset config not found by path or name: {value}")


def _required_path(value: str | None, name: str) -> Path:
    if value is None:
        raise ValueError(f"--{name} is required for this evaluation task")
    path = resolve_path(value, PROJECT_ROOT)
    if path is None or not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    return path


def _coco_annotation_path(args: argparse.Namespace, dataset_config: dict[str, Any]) -> Path:
    configured = args.coco_annotations or dataset_config.get("paths", {}).get("annotation_file")
    path = resolve_path(configured, PROJECT_ROOT)
    if path is None or not path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {path}")
    return path


def _coco_caption_annotation_path(
    args: argparse.Namespace, dataset_config: dict[str, Any]
) -> Path | None:
    configured = args.coco_caption_annotations or dataset_config.get("paths", {}).get(
        "caption_annotation_file"
    )
    path = resolve_path(configured, PROJECT_ROOT)
    if path is None:
        return None
    if path.exists():
        return path
    if args.coco_caption_annotations:
        raise FileNotFoundError(f"COCO caption annotation file not found: {path}")
    print(
        f"Warning: COCO caption annotation file not found; CHAIR GT will use instances only: {path}",
        file=sys.stderr,
    )
    return None


def _default_caption_text_field(records: list[dict[str, Any]]) -> str:
    if records and "revised_caption" in records[0]:
        return "revised_caption"
    return "caption"


def _default_output_path(
    explicit_output: str | None, dataset_config: dict[str, Any], method: str, task: str
) -> Path:
    if explicit_output:
        return ensure_parent(resolve_path(explicit_output, PROJECT_ROOT))
    dataset_name = str(dataset_config.get("name", "dataset"))
    return ensure_parent(PROJECT_ROOT / "outputs" / "metrics" / f"{dataset_name}_{method}_{task}.json")


def _notes_for_task(task: str) -> list[str]:
    if task == "chair":
        return [
            "CHAIR defaults to the official-compatible backend; use --chair-backend internal "
            "only for fallback or sanity checks."
        ]
    if task == "amber":
        return [
            "AMBER Object Subset uses the generic yes/no object-existence scorer; "
            "amber_object_score defaults to F1 until an official AMBER scorer is connected."
        ]
    if task == "false_correction":
        return [
            "False-correction metrics are computed from original/revised captions, COCO GT objects, "
            "and conservative revision actions."
        ]
    return []


if __name__ == "__main__":
    main()
