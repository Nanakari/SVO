"""Efficiency metrics for selective verification."""

from __future__ import annotations

from typing import Any, Mapping

from paper_reproduce.evaluation.common import safe_divide


def evaluate_efficiency_records(
    *,
    object_records: list[Mapping[str, Any]],
    verification_records: list[Mapping[str, Any]],
    base_records: list[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute verification rate, external queries/image, and relative latency."""

    total_objects = sum(len(record.get("objects", [])) for record in object_records)
    verification_images = len(verification_records)
    external_queries = sum(int(record.get("external_queries", 0) or 0) for record in verification_records)
    verification_latency = sum(float(record.get("latency_sec", 0.0) or 0.0) for record in verification_records)

    base_latency = None
    relative_latency = None
    if base_records is not None:
        base_latency_total = sum(float(record.get("latency_sec", 0.0) or 0.0) for record in base_records)
        base_images = len(base_records)
        base_latency = safe_divide(base_latency_total, base_images)
        method_latency = safe_divide(base_latency_total + verification_latency, base_images)
        if base_latency is not None and base_latency > 0 and method_latency is not None:
            relative_latency = method_latency / base_latency

    metrics = {
        "verification_rate": safe_divide(external_queries, total_objects),
        "external_queries_per_image": safe_divide(external_queries, verification_images),
        "relative_latency": relative_latency,
        "base_latency_sec_per_image": base_latency,
        "verification_latency_sec_per_image": safe_divide(verification_latency, verification_images),
    }
    counts = {
        "object_records": len(object_records),
        "verification_records": len(verification_records),
        "base_records": len(base_records or []),
        "object_mentions": total_objects,
        "external_queries": external_queries,
        "verification_latency_sec": verification_latency,
    }
    return metrics, counts
