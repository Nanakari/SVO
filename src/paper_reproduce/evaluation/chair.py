"""COCO/CHAIR caption hallucination metrics."""

from __future__ import annotations

from typing import Any, Mapping

from paper_reproduce.evaluation.common import safe_divide, token_length
from paper_reproduce.extraction import ObjectExtractor


def evaluate_chair_records(
    records: list[Mapping[str, Any]],
    *,
    gt_by_image: Mapping[str, set[str]],
    extractor: ObjectExtractor,
    text_field: str = "caption",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute CHAIRs, CHAIRi, Average Length, and Correct Object Coverage."""

    evaluated_images = 0
    missing_gt = 0
    hallucinated_sentences = 0
    total_mentions = 0
    hallucinated_mentions = 0
    total_length = 0
    coverage_sum = 0.0
    coverage_denominator = 0

    for record in records:
        image_id = str(record.get("image_id", ""))
        gt_objects = gt_by_image.get(image_id)
        if gt_objects is None:
            missing_gt += 1
            continue
        text = str(record.get(text_field) or record.get("revised_caption") or record.get("caption") or "")
        mentions = extractor.extract(text)
        mention_names = [mention.normalized for mention in mentions]
        hallucinated = [name for name in mention_names if name not in gt_objects]

        evaluated_images += 1
        total_length += token_length(text)
        total_mentions += len(mention_names)
        hallucinated_mentions += len(hallucinated)
        hallucinated_sentences += int(bool(hallucinated))
        if gt_objects:
            coverage_sum += len(set(mention_names).intersection(gt_objects)) / len(gt_objects)
            coverage_denominator += 1

    metrics = {
        "chairs": safe_divide(hallucinated_sentences, evaluated_images),
        "chairi": safe_divide(hallucinated_mentions, total_mentions),
        "average_length": safe_divide(total_length, evaluated_images),
        "correct_object_coverage": safe_divide(coverage_sum, coverage_denominator),
    }
    counts = {
        "records": len(records),
        "evaluated_images": evaluated_images,
        "missing_gt": missing_gt,
        "hallucinated_sentences": hallucinated_sentences,
        "object_mentions": total_mentions,
        "hallucinated_object_mentions": hallucinated_mentions,
        "coverage_images": coverage_denominator,
    }
    return metrics, counts
