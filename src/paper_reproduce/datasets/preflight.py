"""Dataset preflight checks used before expensive model loading."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence


class HasImagePath(Protocol):
    image_path: Path


def check_sample_image_paths(samples: Sequence[HasImagePath], mode: str) -> None:
    """Check sample image files before loading large models."""

    normalized = mode.lower()
    if normalized == "none":
        return
    if normalized == "first100":
        candidates = samples[:100]
    elif normalized == "all":
        candidates = samples
    else:
        raise ValueError(f"Unsupported image preflight mode: {mode}")

    missing = [str(sample.image_path) for sample in candidates if not Path(sample.image_path).exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} image files among {len(candidates)} checked samples. "
            f"Examples: {preview}"
        )
