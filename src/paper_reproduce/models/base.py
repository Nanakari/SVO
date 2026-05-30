"""Model adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class GenerationResult:
    """Text generated for a single image and prompt."""

    text: str
    latency_sec: float
    token_scores: list[dict[str, Any]] | None = None


class ImageTextGenerator(Protocol):
    """Protocol for replaceable LVLM generators."""

    def generate(self, image_path: str | Path, prompt: str) -> GenerationResult:
        """Generate text conditioned on an image and prompt."""
