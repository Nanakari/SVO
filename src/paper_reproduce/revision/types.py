"""Types for conservative revision outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RevisionAction:
    """Single caption or answer revision decision."""

    object: str | None
    action: str
    rule: str
    reason: str
    span: tuple[int, int] | None = None
    replacement: str | None = None
    score: float | None = None
    has_visual_evidence: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "object": self.object,
            "action": self.action,
            "rule": self.rule,
            "reason": self.reason,
            "replacement": self.replacement,
            "score": self.score,
            "has_visual_evidence": self.has_visual_evidence,
        }
        if self.span is not None:
            payload["span"] = [self.span[0], self.span[1]]
        return payload


@dataclass(frozen=True)
class CaptionRevisionResult:
    """Result of conservative caption revision."""

    original_caption: str
    revised_caption: str
    actions: list[RevisionAction]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_caption": self.original_caption,
            "revised_caption": self.revised_caption,
            "actions": [action.to_dict() for action in self.actions],
        }
