"""Answer normalization helpers for yes/no benchmarks."""

from __future__ import annotations

import re
from typing import Any


YES_NO_NORMALIZERS = ("official", "strict", "first-match")


def normalize_yes_no_answer(value: Any, *, mode: str = "strict") -> str:
    """Normalize free-form yes/no answers.

    `official` follows the public POPE evaluation script: keep only the first sentence, remove
    commas, return `no` when the sentence contains `no` or `not`, and return `yes` otherwise.
    `strict` accepts exact yes/no aliases only. `first-match` preserves the previous project
    behavior of taking the first yes/no token in a free-form answer.
    """

    normalized_mode = mode.replace("_", "-").lower()
    text = str(value or "").strip()
    if normalized_mode == "official":
        return _normalize_pope_official(text)
    if normalized_mode == "strict":
        return _normalize_strict(text)
    if normalized_mode == "first-match":
        return _normalize_first_match(text)
    raise ValueError(f"Unsupported yes/no normalizer: {mode}")


def _normalize_pope_official(text: str) -> str:
    first_sentence = text.split(".", 1)[0] if "." in text else text
    words = first_sentence.replace(",", "").split()
    lowered_words = {word.lower() for word in words}
    if "no" in words or "No" in words or "no" in lowered_words or "not" in lowered_words:
        return "no"
    return "yes"


def _normalize_strict(text: str) -> str:
    lowered = text.strip().lower()
    if lowered in {"yes", "y", "true", "1"}:
        return "yes"
    if lowered in {"no", "n", "false", "0"}:
        return "no"
    return lowered


def _normalize_first_match(text: str) -> str:
    match = re.search(r"\b(yes|no)\b", text.strip(), flags=re.IGNORECASE)
    if not match:
        return "unknown"
    return match.group(1).lower()
