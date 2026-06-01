from __future__ import annotations

from paper_reproduce.evaluation.classification import evaluate_yes_no_records
from paper_reproduce.revision import revise_pope_answer
from paper_reproduce.utils.answers import normalize_yes_no_answer


def test_pope_official_normalizer_matches_public_script_behavior() -> None:
    assert normalize_yes_no_answer("No, there is not.", mode="official") == "no"
    assert normalize_yes_no_answer("There is not a cat.", mode="official") == "no"
    assert normalize_yes_no_answer("Yes. No in the second sentence.", mode="official") == "yes"
    assert normalize_yes_no_answer("uncertain", mode="official") == "yes"


def test_pope_metrics_use_official_normalizer_for_predictions() -> None:
    records = [
        {"answer": "There is not a bottle.", "label": "no", "setting": "random"},
        {"answer": "The answer is yes.", "label": "yes", "setting": "random"},
    ]

    metrics, counts = evaluate_yes_no_records(
        records,
        group_field="setting",
        answer_normalizer="official",
        label_normalizer="strict",
    )

    assert metrics["random"]["accuracy"] == 1.0
    assert metrics["random"]["yes_ratio"] == 0.5
    assert counts["random"]["evaluated"] == 2


def test_pope_revision_normalizes_raw_response_before_yes_to_no() -> None:
    prediction = {"raw_response": "The answer is yes."}
    verification = {"verified_objects": [{"has_visual_evidence": False, "query": "bottle"}]}

    revised = revise_pope_answer(prediction, verification)

    assert revised["original_answer"] == "The answer is yes."
    assert revised["revised_answer"] == "no"
    assert revised["action"]["action"] == "yes_to_no"
