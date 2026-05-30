"""Types for object phrase extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExtractedObject:
    """A visible object mention extracted from generated text."""

    text: str
    normalized: str
    span: tuple[int, int]
    object_index: int
    source: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "normalized": self.normalized,
            "span": [self.span[0], self.span[1]],
            "object_index": self.object_index,
            "source": self.source,
            "metadata": self.metadata,
        }
