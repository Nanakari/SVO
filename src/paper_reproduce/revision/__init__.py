"""Conservative text revision modules."""

from paper_reproduce.revision.caption import failed_objects_from_verification, revise_caption
from paper_reproduce.revision.pope import revise_pope_answer
from paper_reproduce.revision.types import CaptionRevisionResult, RevisionAction

__all__ = [
    "CaptionRevisionResult",
    "RevisionAction",
    "failed_objects_from_verification",
    "revise_caption",
    "revise_pope_answer",
]
