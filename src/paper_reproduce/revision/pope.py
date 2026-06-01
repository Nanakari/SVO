"""POPE Yes-to-No revision protocol."""

from __future__ import annotations

from typing import Any, Mapping

from paper_reproduce.utils.answers import normalize_yes_no_answer


def revise_pope_answer(
    prediction: Mapping[str, Any],
    verification: Mapping[str, Any] | None,
    *,
    no_to_yes: bool = False,
    answer_normalizer: str = "official",
) -> dict[str, Any]:
    """Revise POPE answers according to the SVO protocol.

    Only positive answers are changed when the verified target object lacks visual evidence.
    No-to-Yes revision stays disabled unless explicitly requested, matching the experiment plan.
    """

    raw_answer = str(prediction.get("raw_response") or prediction.get("answer") or "").strip()
    revised_answer = normalize_yes_no_answer(raw_answer, mode=answer_normalizer)
    action = {
        "action": "keep",
        "rule": "pope_yes_to_no",
        "reason": "not_positive_answer",
        "object": None,
    }

    verified_objects = list((verification or {}).get("verified_objects", []))
    has_verification = bool(verified_objects)
    has_visual_evidence = any(bool(item.get("has_visual_evidence", False)) for item in verified_objects)
    target_object = _target_object(verified_objects)

    if revised_answer == "yes" and has_verification and not has_visual_evidence:
        revised_answer = "no"
        action = {
            "action": "yes_to_no",
            "rule": "pope_yes_to_no",
            "reason": "no_visual_evidence",
            "object": target_object,
        }
    elif revised_answer == "yes" and not has_verification:
        action = {
            "action": "keep",
            "rule": "pope_yes_to_no",
            "reason": "no_verification_available",
            "object": target_object,
        }
    elif revised_answer == "no" and no_to_yes and has_visual_evidence:
        revised_answer = "yes"
        action = {
            "action": "no_to_yes",
            "rule": "pope_no_to_yes",
            "reason": "visual_evidence_present",
            "object": target_object,
        }

    return {
        "original_answer": raw_answer or "unknown",
        "revised_answer": revised_answer,
        "action": action,
        "verified_objects": verified_objects,
    }


def _target_object(verified_objects: list[Mapping[str, Any]]) -> str | None:
    if not verified_objects:
        return None
    first = verified_objects[0]
    return str(first.get("normalized") or first.get("query") or "") or None
