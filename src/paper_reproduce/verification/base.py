"""Visual verifier interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from paper_reproduce.verification.types import VerificationResult


class VisualVerifier(Protocol):
    """Protocol for replaceable open-vocabulary visual verifiers."""

    detector_name: str

    def verify(self, image_path: str | Path, object_name: str) -> VerificationResult:
        """Verify whether an object has visual evidence in an image."""
