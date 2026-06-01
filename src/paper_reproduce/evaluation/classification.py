"""Yes/no object-existence classification metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from paper_reproduce.evaluation.common import safe_divide
from paper_reproduce.utils.answers import normalize_yes_no_answer


def evaluate_yes_no_records(
    records: list[Mapping[str, Any]],
    *,
    group_field: str | None = None,
    answer_field: str = "answer",
    label_field: str = "label",
    answer_normalizer: str = "strict",
    label_normalizer: str = "strict",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute accuracy, precision, recall, F1, and yes ratio."""

    if group_field is None:
        metrics, counts = _evaluate_group(
            records,
            answer_field=answer_field,
            label_field=label_field,
            answer_normalizer=answer_normalizer,
            label_normalizer=label_normalizer,
        )
        return metrics, counts

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(group_field, "unknown"))].append(record)

    metrics_by_group: dict[str, Any] = {}
    counts_by_group: dict[str, Any] = {}
    for group_name, group_records in sorted(grouped.items()):
        group_metrics, group_counts = _evaluate_group(
            group_records,
            answer_field=answer_field,
            label_field=label_field,
            answer_normalizer=answer_normalizer,
            label_normalizer=label_normalizer,
        )
        metrics_by_group[group_name] = group_metrics
        counts_by_group[group_name] = group_counts

    overall_metrics, overall_counts = _evaluate_group(
        records,
        answer_field=answer_field,
        label_field=label_field,
        answer_normalizer=answer_normalizer,
        label_normalizer=label_normalizer,
    )
    metrics_by_group["overall"] = overall_metrics
    counts_by_group["overall"] = overall_counts
    return metrics_by_group, counts_by_group


def amber_object_score(metrics: Mapping[str, Any]) -> float | None:
    """AMBER Object Subset score interface.

    The project reports the object-existence F1 as the default object score unless an
    official AMBER-specific scorer is connected later.
    """

    value = metrics.get("f1")
    return float(value) if value is not None else None


def _evaluate_group(
    records: list[Mapping[str, Any]],
    *,
    answer_field: str,
    label_field: str,
    answer_normalizer: str,
    label_normalizer: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tp = fp = tn = fn = unknown = 0
    for record in records:
        pred = normalize_yes_no_answer(
            record.get(answer_field) or record.get("revised_answer"),
            mode=answer_normalizer,
        )
        label = normalize_yes_no_answer(
            record.get(label_field) or record.get("gt_answer") or record.get("answer_label"),
            mode=label_normalizer,
        )
        if pred not in {"yes", "no"} or label not in {"yes", "no"}:
            unknown += 1
            continue
        if pred == "yes" and label == "yes":
            tp += 1
        elif pred == "yes" and label == "no":
            fp += 1
        elif pred == "no" and label == "no":
            tn += 1
        elif pred == "no" and label == "yes":
            fn += 1

    evaluated = tp + fp + tn + fn
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    metrics = {
        "accuracy": safe_divide(tp + tn, evaluated),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": safe_divide(tp + fp, evaluated),
    }
    counts = {
        "records": len(records),
        "evaluated": evaluated,
        "unknown": unknown,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    return metrics, counts
