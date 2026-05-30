"""Conservative caption revision rules for SVO."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from paper_reproduce.revision.types import CaptionRevisionResult, RevisionAction

_ARTICLE_BEFORE_RE = re.compile(r"(?:^|(?<=\s))(?:a|an|the)\s+$", re.IGNORECASE)
_PREPOSITION_BEFORE_RE = re.compile(r"\s+(?:with|including|containing)\s*$", re.IGNORECASE)
_COORD_BEFORE_RE = re.compile(r"(?:,\s*(?:and|or)\s+|\s+(?:and|or)\s+)$", re.IGNORECASE)
_COORD_AFTER_RE = re.compile(r"^(?:\s*,\s*|\s+(?:and|or)\s+)", re.IGNORECASE)
_THERE_BE_RE = re.compile(r"\bthere\s+(is|are)\b", re.IGNORECASE)


@dataclass(frozen=True)
class _Edit:
    start: int
    end: int
    replacement: str
    action: RevisionAction


def revise_caption(
    caption: str,
    verified_objects: list[Mapping[str, Any]],
    *,
    allowed_rules: list[str] | None = None,
) -> CaptionRevisionResult:
    """Apply conservative local edits for objects without visual evidence.

    Unsupported or ambiguous syntax is explicitly recorded as `skip` instead of being
    force-deleted.
    """

    allowed = set(allowed_rules or ["coordination", "simple_existence", "simple_with_preposition"])
    failed = [item for item in verified_objects if not bool(item.get("has_visual_evidence", False))]
    failed.sort(key=lambda item: _object_span(item)[0] if _object_span(item) else -1, reverse=True)

    revised = caption
    actions: list[RevisionAction] = []
    occupied_edits: list[tuple[int, int]] = []

    for item in failed:
        span = _object_span(item)
        object_name = _object_name(item)
        if span is None or not _valid_span(caption, span):
            actions.append(
                _action(item, object_name, "skip", "invalid_span", "missing_or_invalid_span", None, None)
            )
            continue
        if _overlaps_any(span, occupied_edits):
            actions.append(_action(item, object_name, "skip", "overlap", "overlapping_prior_edit", span, None))
            continue

        edit = _choose_edit(caption, item, span, object_name, allowed)
        if edit is None:
            actions.append(
                _action(
                    item,
                    object_name,
                    "skip",
                    "unsupported_structure",
                    "no_conservative_rule_matched",
                    span,
                    None,
                )
            )
            continue

        revised = revised[: edit.start] + edit.replacement + revised[edit.end :]
        occupied_edits.append((edit.start, edit.end))
        actions.append(edit.action)

    actions.sort(key=lambda action: action.span[0] if action.span else 10**9)
    return CaptionRevisionResult(
        original_caption=caption,
        revised_caption=_cleanup_spacing(revised),
        actions=actions,
    )


def failed_objects_from_verification(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return verification objects that lack visual evidence."""

    return [
        item
        for item in record.get("verified_objects", [])
        if not bool(item.get("has_visual_evidence", False))
    ]


def _choose_edit(
    caption: str,
    item: Mapping[str, Any],
    span: tuple[int, int],
    object_name: str,
    allowed: set[str],
) -> _Edit | None:
    if "simple_existence" in allowed:
        edit = _simple_existence_edit(caption, item, span, object_name)
        if edit is not None:
            return edit
    if "simple_with_preposition" in allowed:
        edit = _simple_preposition_edit(caption, item, span, object_name)
        if edit is not None:
            return edit
    if "coordination" in allowed:
        edit = _coordination_edit(caption, item, span, object_name)
        if edit is not None:
            return edit
    return None


def _simple_existence_edit(
    caption: str, item: Mapping[str, Any], span: tuple[int, int], object_name: str
) -> _Edit | None:
    sentence_start, sentence_end = _sentence_bounds(caption, span)
    sentence_prefix = caption[sentence_start : span[0]]
    match = _THERE_BE_RE.search(sentence_prefix)
    if match is None:
        return None
    # Keep the rule narrow: if another comma or relative marker appears before the object,
    # this is likely not a simple existence clause.
    tail = sentence_prefix[match.end() :]
    if any(marker in tail.lower() for marker in [",", " that ", " which ", " who "]):
        return None
    start = _expand_article_left(caption, span[0])
    replacement = "objects" if match.group(1).lower() == "are" else "an object"
    return _Edit(
        start=start,
        end=span[1],
        replacement=replacement,
        action=_action(item, object_name, "weaken", "simple_existence", "no_visual_evidence", span, replacement),
    )


def _simple_preposition_edit(
    caption: str, item: Mapping[str, Any], span: tuple[int, int], object_name: str
) -> _Edit | None:
    item_start = _expand_article_left(caption, span[0])
    before_item = caption[:item_start]
    match = _PREPOSITION_BEFORE_RE.search(before_item)
    if match is None:
        return None
    # Only remove a simple prepositional supplement ending at punctuation/comma/end.
    after = caption[span[1] :]
    if after.strip() and after.lstrip()[0] not in {".", ",", ";", "!", "?"}:
        return None
    return _Edit(
        start=match.start(),
        end=span[1],
        replacement="",
        action=_action(item, object_name, "delete", "simple_with_preposition", "no_visual_evidence", span, ""),
    )


def _coordination_edit(
    caption: str, item: Mapping[str, Any], span: tuple[int, int], object_name: str
) -> _Edit | None:
    item_start = _expand_article_left(caption, span[0])
    before_item = caption[:item_start]
    before_match = _COORD_BEFORE_RE.search(before_item)
    if before_match is not None:
        return _Edit(
            start=before_match.start(),
            end=span[1],
            replacement="",
            action=_action(item, object_name, "delete", "coordination", "no_visual_evidence", span, ""),
        )

    after_item = caption[span[1] :]
    after_match = _COORD_AFTER_RE.match(after_item)
    if after_match is not None:
        return _Edit(
            start=item_start,
            end=span[1] + after_match.end(),
            replacement="",
            action=_action(item, object_name, "delete", "coordination", "no_visual_evidence", span, ""),
        )
    return None


def _object_span(item: Mapping[str, Any]) -> tuple[int, int] | None:
    source = item.get("source_object") or item
    span = source.get("span")
    if not isinstance(span, list) or len(span) != 2:
        return None
    try:
        return int(span[0]), int(span[1])
    except (TypeError, ValueError):
        return None


def _object_name(item: Mapping[str, Any]) -> str:
    source = item.get("source_object") or item
    return str(source.get("normalized") or item.get("normalized") or source.get("text") or "")


def _valid_span(text: str, span: tuple[int, int]) -> bool:
    start, end = span
    return 0 <= start < end <= len(text)


def _sentence_bounds(text: str, span: tuple[int, int]) -> tuple[int, int]:
    start = max(text.rfind(".", 0, span[0]), text.rfind("!", 0, span[0]), text.rfind("?", 0, span[0]))
    end_candidates = [index for index in [text.find(".", span[1]), text.find("!", span[1]), text.find("?", span[1])] if index != -1]
    sentence_start = start + 1 if start != -1 else 0
    sentence_end = min(end_candidates) + 1 if end_candidates else len(text)
    return sentence_start, sentence_end


def _expand_article_left(text: str, start: int) -> int:
    prefix = text[:start]
    match = _ARTICLE_BEFORE_RE.search(prefix)
    if match is None:
        return start
    return match.start() if prefix[match.start() :].startswith(("a", "A", "t", "T")) else match.start() + 1


def _overlaps_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in spans)


def _action(
    item: Mapping[str, Any],
    object_name: str | None,
    action: str,
    rule: str,
    reason: str,
    span: tuple[int, int] | None,
    replacement: str | None,
) -> RevisionAction:
    return RevisionAction(
        object=object_name,
        action=action,
        rule=rule,
        reason=reason,
        span=span,
        replacement=replacement,
        score=_optional_float(item.get("score")),
        has_visual_evidence=bool(item.get("has_visual_evidence", False)),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cleanup_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r",\s*([.;!?])", r"\1", text)
    return text.strip()
