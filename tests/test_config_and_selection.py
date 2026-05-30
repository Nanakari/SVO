from __future__ import annotations

from paper_reproduce.utils.config import apply_overrides
from paper_reproduce.verification.selection import (
    SelectionConfig,
    select_objects_for_verification,
)


def test_apply_overrides_parses_json_values() -> None:
    config = {"generation": {"max_new_tokens": 128}, "runtime": {"device": "cuda"}}

    updated = apply_overrides(
        config,
        ["generation.max_new_tokens=64", "runtime.device=\"cpu\"", "flags.enabled=true"],
    )

    assert updated["generation"]["max_new_tokens"] == 64
    assert updated["runtime"]["device"] == "cpu"
    assert updated["flags"]["enabled"] is True


def test_risk_threshold_selection_deduplicates_by_highest_risk() -> None:
    objects = [
        {"normalized": "person", "object_index": 1, "risk": {"total": 0.2}},
        {"normalized": "bottle", "object_index": 2, "risk": {"total": 1.8}},
        {"normalized": "bottle", "object_index": 3, "risk": {"total": 0.4}},
    ]

    selected = select_objects_for_verification(
        objects,
        SelectionConfig(policy="risk_threshold", risk_threshold=1.0),
        sample_id="sample-1",
    )

    assert [item["normalized"] for item in selected] == ["bottle"]
    assert selected[0]["object_index"] == 2
