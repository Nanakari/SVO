"""Shared dataset sample types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaptionSample:
    """Single image sample for open-ended caption generation."""

    sample_id: str
    image_id: str
    image_path: Path
    dataset: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PopeSample:
    """Single POPE object-existence question."""

    sample_id: str
    image_id: str
    image_path: Path
    question: str
    label: str | None
    setting: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class AmberObjectSample:
    """Single AMBER object-existence subset sample."""

    sample_id: str
    image_id: str
    image_path: Path
    question: str | None
    label: str | None
    target_object: str | None
    raw: dict[str, Any]
