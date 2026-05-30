"""Static object hallucination prior utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from paper_reproduce.utils.config import resolve_path


@dataclass(frozen=True)
class StaticObjectPrior:
    """Lookup object hallucination priors with low-frequency fallback."""

    values: dict[str, float]
    counts: dict[str, int]
    fallback_prior: float
    min_count: int

    @classmethod
    def empty(cls, fallback_prior: float = 0.0, min_count: int = 5) -> "StaticObjectPrior":
        return cls(values={}, counts={}, fallback_prior=fallback_prior, min_count=min_count)

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], project_root: str | Path | None = None
    ) -> "StaticObjectPrior":
        risk_config = config.get("risk_scoring", {})
        min_count = int(risk_config.get("min_prior_count", 5))
        fallback_setting = risk_config.get("fallback_prior", "dataset_mean")
        fallback_prior = float(fallback_setting) if isinstance(fallback_setting, (int, float)) else 0.0
        prior_path = risk_config.get("static_prior_path")
        if prior_path is None:
            return cls.empty(fallback_prior=fallback_prior, min_count=min_count)
        resolved = resolve_path(prior_path, project_root or Path.cwd())
        if resolved is None or not resolved.exists():
            return cls.empty(fallback_prior=fallback_prior, min_count=min_count)
        return cls.from_json(resolved, min_count=min_count, fallback_prior=fallback_prior)

    @classmethod
    def from_json(
        cls, path: str | Path, *, min_count: int = 5, fallback_prior: float = 0.0
    ) -> "StaticObjectPrior":
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)

        if isinstance(data, dict) and "priors" in data:
            priors = data.get("priors", {})
            mean_prior = float(data.get("mean_prior", fallback_prior))
        else:
            priors = data
            mean_prior = fallback_prior

        values: dict[str, float] = {}
        counts: dict[str, int] = {}
        for name, item in priors.items():
            if isinstance(item, Mapping):
                values[str(name)] = float(item.get("prior", item.get("value", 0.0)))
                counts[str(name)] = int(item.get("generated", item.get("count", min_count)))
            else:
                values[str(name)] = float(item)
                counts[str(name)] = min_count
        return cls(values=values, counts=counts, fallback_prior=mean_prior, min_count=min_count)

    def get(self, object_name: str) -> float:
        count = self.counts.get(object_name, 0)
        if count < self.min_count:
            return self.fallback_prior
        return self.values.get(object_name, self.fallback_prior)


def compute_static_prior(
    *,
    generated_counts: Mapping[str, int],
    hallucinated_counts: Mapping[str, int],
    min_count: int,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Compute static hallucination priors from generated and hallucinated counts."""

    priors: dict[str, dict[str, float | int]] = {}
    generated_total = 0
    hallucinated_total = 0
    for object_name in sorted(generated_counts):
        generated = int(generated_counts[object_name])
        hallucinated = int(hallucinated_counts.get(object_name, 0))
        generated_total += generated
        hallucinated_total += hallucinated
        prior = (hallucinated + epsilon) / (generated + epsilon)
        priors[object_name] = {
            "generated": generated,
            "hallucinated": hallucinated,
            "prior": prior,
            "low_frequency": generated < min_count,
        }

    mean_prior = (hallucinated_total + epsilon) / (generated_total + epsilon) if generated_total else 0.0
    return {
        "min_count": min_count,
        "epsilon": epsilon,
        "mean_prior": mean_prior,
        "generated_total": generated_total,
        "hallucinated_total": hallucinated_total,
        "priors": priors,
    }
