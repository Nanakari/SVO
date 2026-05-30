"""Object phrase extractors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

from paper_reproduce.extraction.types import ExtractedObject
from paper_reproduce.extraction.vocabulary import (
    ObjectVocabulary,
    normalise_for_matching,
    tokenize_with_spans,
)


class ObjectExtractor(Protocol):
    """Protocol for replaceable object phrase extractors."""

    def extract(self, text: str) -> list[ExtractedObject]:
        """Extract visible object mentions from generated text."""


class RuleBasedObjectExtractor:
    """Dependency-free vocabulary matcher used for smoke tests and fallback runs."""

    def __init__(self, vocabulary: ObjectVocabulary) -> None:
        self.vocabulary = vocabulary

    def extract(self, text: str) -> list[ExtractedObject]:
        tokens = tokenize_with_spans(text)
        used_token_positions: set[int] = set()
        mentions: list[tuple[int, int, str, str, str]] = []

        for alias, canonical in self.vocabulary.alias_items_by_length():
            alias_words = alias.split()
            width = len(alias_words)
            if width == 0:
                continue
            for start in range(0, len(tokens) - width + 1):
                end = start + width
                if any(index in used_token_positions for index in range(start, end)):
                    continue
                token_words = [normalise_for_matching(token[0]) for token in tokens[start:end]]
                if token_words != alias_words:
                    continue
                used_token_positions.update(range(start, end))
                char_start = tokens[start][1]
                char_end = tokens[end - 1][2]
                mentions.append((char_start, char_end, text[char_start:char_end], canonical, alias))

        mentions.sort(key=lambda item: item[0])
        return [
            ExtractedObject(
                text=mention_text,
                normalized=canonical,
                span=(char_start, char_end),
                object_index=index,
                source="rule_vocabulary",
                metadata={"matched_alias": alias},
            )
            for index, (char_start, char_end, mention_text, canonical, alias) in enumerate(
                mentions, start=1
            )
        ]


class SpacyObjectExtractor:
    """spaCy noun-chunk extractor with COCO/common-object normalization."""

    def __init__(self, vocabulary: ObjectVocabulary, model_name: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - depends on optional nlp deps
            raise RuntimeError(
                "spaCy is required for object_extraction.backend=spacy. Install dependencies "
                "with `pip install -e .[nlp]` and download the configured spaCy model."
            ) from exc

        try:
            self.nlp = spacy.load(model_name)
        except OSError as exc:  # pragma: no cover - depends on local model files
            raise RuntimeError(
                f"spaCy model `{model_name}` is not installed. Run "
                f"`python -m spacy download {model_name}` in the SVO environment."
            ) from exc
        self.vocabulary = vocabulary
        self.rule_fallback = RuleBasedObjectExtractor(vocabulary)

    def extract(self, text: str) -> list[ExtractedObject]:
        doc = self.nlp(text)
        mentions: list[ExtractedObject] = []
        occupied: list[tuple[int, int]] = []

        for chunk in doc.noun_chunks:
            canonical = self.vocabulary.normalize(chunk.text)
            if canonical is None:
                continue
            mentions.append(
                ExtractedObject(
                    text=chunk.text,
                    normalized=canonical,
                    span=(chunk.start_char, chunk.end_char),
                    object_index=0,
                    source="spacy_noun_chunk",
                    metadata={"root": chunk.root.text, "root_pos": chunk.root.pos_},
                )
            )
            occupied.append((chunk.start_char, chunk.end_char))

        # Add vocabulary-only matches for multi-word COCO categories that noun chunks may miss.
        for mention in self.rule_fallback.extract(text):
            if _overlaps_any(mention.span, occupied):
                continue
            mentions.append(mention)
            occupied.append(mention.span)

        mentions.sort(key=lambda item: item.span[0])
        return [
            ExtractedObject(
                text=mention.text,
                normalized=mention.normalized,
                span=mention.span,
                object_index=index,
                source=mention.source,
                metadata=mention.metadata,
            )
            for index, mention in enumerate(mentions, start=1)
        ]


def build_extractor(config: Mapping[str, Any], project_root: str | Path) -> ObjectExtractor:
    """Build the configured object extractor."""

    extraction_config = config.get("object_extraction", {})
    vocabulary = ObjectVocabulary.from_config(config, project_root)
    backend = str(extraction_config.get("backend", "spacy")).lower()
    if backend == "spacy":
        return SpacyObjectExtractor(vocabulary, str(extraction_config.get("spacy_model", "en_core_web_sm")))
    if backend in {"rule", "rules", "vocabulary"}:
        return RuleBasedObjectExtractor(vocabulary)
    raise ValueError(f"Unsupported object_extraction.backend: {backend}")


def _overlaps_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in spans)
