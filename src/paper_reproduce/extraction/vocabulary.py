"""Object vocabulary loading and phrase normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from paper_reproduce.utils.config import load_yaml, resolve_path

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_ARTICLES = {"a", "an", "the"}


@dataclass(frozen=True)
class ObjectVocabulary:
    """COCO/common-object vocabulary with synonym and filter support."""

    categories: frozenset[str]
    aliases: dict[str, str]
    filters: frozenset[str]

    @classmethod
    def from_config(cls, config: Mapping[str, Any], project_root: str | Path) -> "ObjectVocabulary":
        extraction_config = config.get("object_extraction", config)
        vocab_path = resolve_path(extraction_config.get("vocabulary_path"), project_root)
        if vocab_path is None:
            raise ValueError("object_extraction.vocabulary_path is required")
        return cls.from_yaml(vocab_path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ObjectVocabulary":
        data = load_yaml(path)
        categories = {_clean_phrase(item) for item in data.get("categories", [])}
        synonyms = data.get("synonyms", {}) or {}
        aliases = {category: category for category in categories}
        for category in list(categories):
            aliases[_singularise_last_word(category)] = category
        category_by_singular = {_singularise_last_word(category): category for category in categories}
        for alias, target in synonyms.items():
            normalised_alias = _clean_phrase(alias, singularize=True)
            normalised_target = _clean_phrase(target)
            canonical_target = (
                normalised_target
                if normalised_target in categories
                else category_by_singular.get(_singularise_last_word(normalised_target))
            )
            if canonical_target in categories:
                aliases[normalised_alias] = canonical_target
        filters = {_clean_phrase(item) for item in data.get("filters", [])}
        filters.update(_singularise_last_word(item) for item in list(filters))
        return cls(frozenset(categories), aliases, frozenset(filters))

    def normalize(self, phrase: str) -> str | None:
        """Map a phrase to a canonical object category, if possible."""

        cleaned = _clean_phrase(phrase)
        if not cleaned or cleaned in self.filters:
            return None
        if cleaned in self.aliases:
            return self.aliases[cleaned]

        words = cleaned.split()
        # Try longest contiguous sub-phrases first, so "traffic light" wins over "light".
        for width in range(len(words), 0, -1):
            for start in range(0, len(words) - width + 1):
                candidate = " ".join(words[start : start + width])
                if candidate in self.filters:
                    continue
                if candidate in self.aliases:
                    return self.aliases[candidate]

        singular = _singularise_last_word(cleaned)
        if singular in self.aliases:
            return self.aliases[singular]
        return None

    def alias_items_by_length(self) -> list[tuple[str, str]]:
        """Return aliases sorted from longest to shortest for greedy text matching."""

        return sorted(self.aliases.items(), key=lambda item: len(item[0].split()), reverse=True)


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Tokenize lowercase words with character spans."""

    return [(match.group(0).lower(), match.start(), match.end()) for match in _WORD_RE.finditer(text)]


def normalise_for_matching(text: str) -> str:
    """Public normalizer used by scoring alignment."""

    return _clean_phrase(text, singularize=True)


def _clean_phrase(phrase: Any, *, singularize: bool = False) -> str:
    words = [word for word in _WORD_RE.findall(str(phrase).lower()) if word not in _ARTICLES]
    if not words:
        return ""
    normalised = " ".join(words)
    if singularize:
        normalised = _singularise_last_word(normalised)
    return normalised


def _singularise_last_word(phrase: str) -> str:
    words = phrase.split()
    if not words:
        return phrase
    last = words[-1]
    if len(last) > 3 and last.endswith("ies"):
        words[-1] = last[:-3] + "y"
    elif len(last) > 3 and last.endswith(("ses", "xes", "zes", "ches", "shes")):
        words[-1] = last[:-2]
    elif len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        words[-1] = last[:-1]
    return " ".join(words)
