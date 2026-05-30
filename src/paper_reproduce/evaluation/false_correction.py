"""False-correction and retention metrics for conservative revisions."""

from __future__ import annotations

from typing import Any, Mapping

from paper_reproduce.evaluation.common import safe_divide
from paper_reproduce.extraction import ObjectExtractor

_CORRECTION_ACTIONS = {"delete", "weaken", "yes_to_no"}


def evaluate_false_correction_records(
    records: list[Mapping[str, Any]],
    *,
    gt_by_image: Mapping[str, set[str]],
    extractor: ObjectExtractor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute hallucinated removal, correct retention, and false correction rate."""

    evaluated_images = 0
    missing_gt = 0
    hallucinated_original = 0
    hallucinated_removed = 0
    correct_original = 0
    correct_retained = 0
    correction_actions = 0
    false_corrections = 0

    for record in records:
        image_id = str(record.get("image_id", ""))
        gt_objects = gt_by_image.get(image_id)
        if gt_objects is None:
            missing_gt += 1
            continue

        original_caption = str(record.get("original_caption") or "")
        revised_caption = str(record.get("revised_caption") or original_caption)
        original_mentions = extractor.extract(original_caption)
        revised_names = {mention.normalized for mention in extractor.extract(revised_caption)}

        action_objects = {
            str(action.get("object"))
            for action in record.get("actions", [])
            if action.get("action") in _CORRECTION_ACTIONS and action.get("object")
        }
        correction_actions += len(action_objects)
        false_corrections += sum(1 for name in action_objects if name in gt_objects)

        for mention in original_mentions:
            name = mention.normalized
            if name in gt_objects:
                correct_original += 1
                if name in revised_names and name not in action_objects:
                    correct_retained += 1
            else:
                hallucinated_original += 1
                if name not in revised_names or name in action_objects:
                    hallucinated_removed += 1

        evaluated_images += 1

    metrics = {
        "hallucinated_object_removal": safe_divide(hallucinated_removed, hallucinated_original),
        "correct_object_retention": safe_divide(correct_retained, correct_original),
        "false_correction_rate": safe_divide(false_corrections, correction_actions),
    }
    counts = {
        "records": len(records),
        "evaluated_images": evaluated_images,
        "missing_gt": missing_gt,
        "hallucinated_original": hallucinated_original,
        "hallucinated_removed": hallucinated_removed,
        "correct_original": correct_original,
        "correct_retained": correct_retained,
        "correction_actions": correction_actions,
        "false_corrections": false_corrections,
    }
    return metrics, counts
