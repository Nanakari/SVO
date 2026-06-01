"""High-risk object scoring for SVO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from paper_reproduce.extraction.types import ExtractedObject
from paper_reproduce.extraction.vocabulary import normalise_for_matching
from paper_reproduce.scoring.static_prior import StaticObjectPrior


@dataclass(frozen=True)
class RiskWeights:
    uncertainty: float = 1.0
    position: float = 1.0
    prior: float = 1.0


class RiskScorer:
    """Compute SVO risk terms: uncertainty, position risk, and static prior."""

    def __init__(
        self,
        *,
        weights: RiskWeights,
        static_prior: StaticObjectPrior,
        enabled_terms: Mapping[str, bool] | None = None,
    ) -> None:
        self.weights = weights
        self.static_prior = static_prior
        self.enabled_terms = {
            "uncertainty": True,
            "position": True,
            "prior": True,
            **dict(enabled_terms or {}),
        }

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], project_root: str | Path | None = None
    ) -> "RiskScorer":
        risk_config = config.get("risk_scoring", {})
        weights_config = risk_config.get("weights", {})
        method_config = config.get("method", {})
        enabled_terms = method_config.get("risk_terms") or config.get("risk_terms") or {}
        weights = RiskWeights(
            uncertainty=float(weights_config.get("uncertainty", 1.0)),
            position=float(weights_config.get("position", 1.0)),
            prior=float(weights_config.get("prior", 1.0)),
        )
        prior_required = bool(enabled_terms.get("prior", True)) and weights.prior > 0.0
        static_prior = StaticObjectPrior.from_config(
            config, project_root, required=prior_required
        )
        return cls(
            weights=weights,
            static_prior=static_prior,
            enabled_terms=enabled_terms,
        )

    def score_objects(
        self,
        *,
        caption: str,
        objects: list[ExtractedObject],
        token_scores: list[Mapping[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Attach risk dictionaries to extracted objects."""

        scored: list[dict[str, Any]] = []
        total_objects = len(objects)
        for mention in objects:
            uncertainty = estimate_object_uncertainty(mention, token_scores)
            position = mention.object_index / total_objects if total_objects else 0.0
            prior = self.static_prior.get(mention.normalized)

            used_uncertainty = uncertainty if uncertainty is not None else 0.0
            missing_terms = []
            if uncertainty is None:
                missing_terms.append("uncertainty")

            risk_total = 0.0
            if self.enabled_terms.get("uncertainty", True):
                risk_total += self.weights.uncertainty * used_uncertainty
            if self.enabled_terms.get("position", True):
                risk_total += self.weights.position * position
            if self.enabled_terms.get("prior", True):
                risk_total += self.weights.prior * prior

            record = mention.to_dict()
            record["risk"] = {
                "uncertainty": uncertainty,
                "position": position,
                "prior": prior,
                "total": risk_total,
                "weights": {
                    "uncertainty": self.weights.uncertainty,
                    "position": self.weights.position,
                    "prior": self.weights.prior,
                },
                "enabled_terms": dict(self.enabled_terms),
                "missing_terms": missing_terms,
            }
            scored.append(record)
        return scored


def estimate_object_uncertainty(
    mention: ExtractedObject, token_scores: list[Mapping[str, Any]] | None
) -> float | None:
    """Estimate U(o) as average negative log probability for matching object tokens."""

    if not token_scores:
        return None

    object_words = set(normalise_for_matching(mention.text).split()) | set(mention.normalized.split())
    logprobs: list[float] = []
    for token in token_scores:
        token_text = normalise_for_matching(str(token.get("token", "")))
        if not token_text:
            continue
        token_words = set(token_text.split())
        if not object_words.intersection(token_words):
            continue
        logprob = token.get("logprob")
        if logprob is None:
            prob = token.get("prob")
            if prob is None:
                continue
            prob_value = max(float(prob), 1e-12)
            logprob = math.log(prob_value)
        logprobs.append(float(logprob))

    if not logprobs:
        return None
    return sum(-value for value in logprobs) / len(logprobs)
