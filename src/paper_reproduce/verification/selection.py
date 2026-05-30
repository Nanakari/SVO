"""Object selection policies for visual verification."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for selecting objects to verify."""

    policy: str
    risk_threshold: float | None
    deduplicate: bool = True
    random_seed: int = 42


def selection_config_from_method(
    config: Mapping[str, Any], *, risk_threshold_override: float | None = None
) -> SelectionConfig:
    method = config.get("method", {})
    pipeline = method.get("pipeline", {})
    policy = str(pipeline.get("verify_policy", "risk_threshold")).lower()
    threshold = risk_threshold_override
    if threshold is None:
        configured_threshold = config.get("risk_scoring", {}).get("threshold")
        threshold = float(configured_threshold) if configured_threshold is not None else None
    random_config = method.get("random", {})
    project_config = config.get("project", {})
    verification_config = config.get("verification", {})
    return SelectionConfig(
        policy=policy,
        risk_threshold=threshold,
        deduplicate=bool(verification_config.get("deduplicate_objects", True)),
        random_seed=int(random_config.get("seed", project_config.get("seed", 42))),
    )


def select_objects_for_verification(
    objects: list[Mapping[str, Any]],
    selection: SelectionConfig,
    *,
    sample_id: str,
    reference_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Select object records according to a configured verification policy."""

    candidates = [_copy_object(item) for item in objects if item.get("normalized")]
    if selection.deduplicate:
        candidates = _deduplicate_by_normalized(candidates)

    if selection.policy == "all":
        return candidates

    if selection.policy == "risk_threshold":
        if selection.risk_threshold is None:
            raise ValueError(
                "risk_scoring.threshold or --risk-threshold is required for risk_threshold policy"
            )
        return [
            item
            for item in candidates
            if float(item.get("risk", {}).get("total", float("-inf"))) > selection.risk_threshold
        ]

    if selection.policy == "matched_random":
        if reference_counts is None:
            raise ValueError("--reference is required for matched_random verification")
        if sample_id not in reference_counts:
            raise ValueError(f"Reference count missing for sample_id `{sample_id}`")
        count = min(reference_counts[sample_id], len(candidates))
        rng = random.Random(f"{selection.random_seed}:{sample_id}")
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        selected = shuffled[:count]
        selected.sort(key=lambda item: int(item.get("object_index", 0)))
        return selected

    raise ValueError(f"Unsupported verification policy: {selection.policy}")


def count_reference_verifications(record: Mapping[str, Any]) -> int:
    """Count verified objects in a reference JSONL record."""

    if isinstance(record.get("verified_objects"), list):
        return len(record["verified_objects"])
    if isinstance(record.get("objects"), list):
        return len(record["objects"])
    raise ValueError("Reference record must contain `verified_objects` or `objects`")


def _copy_object(item: Mapping[str, Any]) -> dict[str, Any]:
    return dict(item)


def _deduplicate_by_normalized(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_name: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in objects:
        name = str(item["normalized"])
        current = best_by_name.get(name)
        if current is None:
            best_by_name[name] = item
            order.append(name)
            continue
        current_risk = float(current.get("risk", {}).get("total", float("-inf")))
        item_risk = float(item.get("risk", {}).get("total", float("-inf")))
        if item_risk > current_risk:
            best_by_name[name] = item
    return [best_by_name[name] for name in order]
