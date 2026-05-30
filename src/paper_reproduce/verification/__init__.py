"""Open-vocabulary visual verification modules."""

from paper_reproduce.verification.base import VisualVerifier
from paper_reproduce.verification.grounding_dino import GroundingDinoVerifier, build_verifier
from paper_reproduce.verification.selection import (
    SelectionConfig,
    count_reference_verifications,
    select_objects_for_verification,
    selection_config_from_method,
)
from paper_reproduce.verification.types import DetectionBox, VerificationResult

__all__ = [
    "DetectionBox",
    "GroundingDinoVerifier",
    "SelectionConfig",
    "VerificationResult",
    "VisualVerifier",
    "build_verifier",
    "count_reference_verifications",
    "select_objects_for_verification",
    "selection_config_from_method",
]
