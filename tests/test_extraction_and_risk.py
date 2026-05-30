from __future__ import annotations

from pathlib import Path

from paper_reproduce.extraction import build_extractor
from paper_reproduce.scoring import RiskScorer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return {
        "object_extraction": {
            "backend": "rule",
            "vocabulary_path": "configs/vocab/coco_objects.yaml",
        },
        "risk_scoring": {
            "static_prior_path": None,
            "fallback_prior": 0.0,
            "weights": {"uncertainty": 1.0, "position": 1.0, "prior": 1.0},
        },
    }


def test_rule_extractor_finds_smoke_objects() -> None:
    extractor = build_extractor(_config(), PROJECT_ROOT)

    mentions = extractor.extract("A person with a laptop and a bottle.")

    assert [mention.normalized for mention in mentions] == ["person", "laptop", "bottle"]
    assert [mention.object_index for mention in mentions] == [1, 2, 3]


def test_risk_scorer_attaches_terms() -> None:
    extractor = build_extractor(_config(), PROJECT_ROOT)
    mentions = extractor.extract("A person with a laptop and a bottle.")
    scorer = RiskScorer.from_config(_config(), PROJECT_ROOT)

    scored = scorer.score_objects(
        caption="A person with a laptop and a bottle.",
        objects=mentions,
        token_scores=[
            {"token": "person", "logprob": -0.1},
            {"token": "laptop", "logprob": -0.2},
            {"token": "bottle", "logprob": -1.4},
        ],
    )

    bottle = scored[-1]
    assert bottle["normalized"] == "bottle"
    assert bottle["risk"]["uncertainty"] == 1.4
    assert bottle["risk"]["position"] == 1.0
    assert bottle["risk"]["total"] == 2.4
