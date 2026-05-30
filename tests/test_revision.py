from __future__ import annotations

from paper_reproduce.revision import revise_caption, revise_pope_answer


def test_conservative_caption_revision_deletes_failed_coordination_object() -> None:
    caption = "A person with a laptop and a bottle."
    start = caption.index("bottle")
    result = revise_caption(
        caption,
        [
            {
                "has_visual_evidence": False,
                "score": 0.01,
                "source_object": {
                    "normalized": "bottle",
                    "text": "bottle",
                    "span": [start, start + len("bottle")],
                },
            }
        ],
    )

    assert result.revised_caption == "A person with a laptop."
    assert result.actions[0].action == "delete"
    assert result.actions[0].rule == "coordination"


def test_pope_revision_is_yes_to_no_only_by_default() -> None:
    prediction = {"answer": "yes"}
    verification = {
        "verified_objects": [
            {
                "has_visual_evidence": False,
                "source_object": {"normalized": "bottle"},
            }
        ]
    }

    revised = revise_pope_answer(prediction, verification)

    assert revised["revised_answer"] == "no"
    assert revised["action"]["action"] == "yes_to_no"
