"""Types for visual object verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DetectionBox:
    """Single detector box for an object query."""

    box: list[float]
    score: float
    phrase: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": self.box,
            "score": self.score,
            "phrase": self.phrase,
        }


@dataclass(frozen=True)
class VerificationResult:
    """Visual evidence result for one object query."""

    normalized: str
    query: str
    score: float
    has_visual_evidence: bool
    detector: str
    boxes: list[DetectionBox]
    latency_sec: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized": self.normalized,
            "query": self.query,
            "score": self.score,
            "has_visual_evidence": self.has_visual_evidence,
            "detector": self.detector,
            "boxes": [box.to_dict() for box in self.boxes],
            "latency_sec": self.latency_sec,
            "metadata": self.metadata,
        }
