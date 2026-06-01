"""Check SVO CHAIR behavior against a local LisaAnne/Hallucination checkout.

The upstream CHAIR scorer is Python 2 code with optional NLTK/pattern dependencies. To keep this
repository dependency-light and avoid vendoring unlicensed upstream code, this harness reads the
official `data/synonyms.txt` resource from a local checkout and compares SVO's
official-compatible wrapper on fixed parity cases. Use `--check-synonym-coverage` for a stricter
resource coverage audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_reproduce.evaluation.chair import (  # noqa: E402
    OfficialChairMapper,
    evaluate_chair_records_official,
)


DEFAULT_CASES = [
    {
        "caption": "A woman rides a bike near a traffic signal.",
        "expected": ["person", "bicycle", "traffic light"],
    },
    {
        "caption": "A baby bird sits on a toilet seat with a bow tie.",
        "expected": ["bird", "toilet", "tie"],
    },
    {
        "caption": "A cell phone and sports ball are on a sofa.",
        "expected": ["cell phone", "sports ball", "couch"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate CHAIR official-compatible behavior against a local official repo."
    )
    parser.add_argument(
        "--official-chair-repo",
        required=True,
        help="Path to a local LisaAnne/Hallucination checkout.",
    )
    parser.add_argument(
        "--cases",
        help="Optional JSON file with caption parity cases. Defaults to built-in cases.",
    )
    parser.add_argument(
        "--check-synonym-coverage",
        action="store_true",
        help="Also require every alias in official data/synonyms.txt to map to the same canonical object.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.official_chair_repo).expanduser().resolve()
    synonyms = _load_official_synonyms(repo)
    official_mapper = _OfficialResourceMapper(synonyms)
    project_mapper = OfficialChairMapper(official_mapper.categories)
    cases = _load_cases(args.cases)

    failures: list[str] = []
    for index, case in enumerate(cases, start=1):
        caption = str(case["caption"])
        expected = [str(item) for item in case.get("expected", [])]
        official_objects = official_mapper.caption_to_objects(caption)[1]
        project_objects = project_mapper.caption_to_objects(caption)[1]
        if project_objects != official_objects:
            failures.append(
                f"case {index} object mismatch: expected official {official_objects}, got {project_objects}"
            )
        if expected and project_objects != expected:
            failures.append(f"case {index} fixture mismatch: expected {expected}, got {project_objects}")

    official_metrics, official_counts = _fixture_metrics(official_mapper)
    project_metrics, project_counts = _fixture_metrics(project_mapper)
    for key in ("chairs", "chairi"):
        if project_metrics[key] != official_metrics[key]:
            failures.append(
                f"metric {key} mismatch: expected official {official_metrics[key]}, got {project_metrics[key]}"
            )
    if project_counts["hallucinated_object_mentions"] != official_counts["hallucinated_object_mentions"]:
        failures.append(
            "hallucinated mention count mismatch: expected official "
            f"{official_counts['hallucinated_object_mentions']}, "
            f"got {project_counts['hallucinated_object_mentions']}"
        )

    if args.check_synonym_coverage:
        failures.extend(_synonym_coverage_failures(project_mapper, synonyms))

    if failures:
        print("CHAIR parity check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(f"CHAIR parity cases: {len(cases)}")
    print("Object extraction parity: OK")
    print("Metric parity: OK")
    if args.check_synonym_coverage:
        print("Official synonym coverage: OK")


def _load_cases(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return list(DEFAULT_CASES)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--cases JSON must contain a list")
    return [dict(item) for item in data]


def _load_official_synonyms(repo: Path) -> dict[str, list[str]]:
    synonyms_path = repo / "data" / "synonyms.txt"
    if not synonyms_path.exists():
        raise FileNotFoundError(
            f"Official CHAIR synonyms file not found: {synonyms_path}. "
            "Clone https://github.com/LisaAnne/Hallucination and pass --official-chair-repo."
        )
    synonyms: dict[str, list[str]] = {}
    for line in synonyms_path.read_text(encoding="utf-8").splitlines():
        values = [value.strip() for value in line.split(",") if value.strip()]
        if values:
            synonyms[values[0]] = values
    return synonyms


def _fixture_metrics(mapper: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    records = [
        {"image_id": 1, "caption": "A woman with a bike."},
        {"image_id": 2, "caption": "A dog and a phone."},
    ]
    gt_by_image = {
        "1": {"person", "bicycle"},
        "2": {"dog"},
    }
    return evaluate_chair_records_official(records, gt_by_image=gt_by_image, mapper=mapper)


def _synonym_coverage_failures(
    project_mapper: OfficialChairMapper, synonyms: dict[str, list[str]]
) -> list[str]:
    failures: list[str] = []
    for canonical, aliases in synonyms.items():
        for alias in aliases:
            mapped = project_mapper.canonicalize_phrase(alias)
            if mapped != canonical:
                failures.append(f"official synonym `{alias}` should map to `{canonical}`, got `{mapped}`")
    return failures


class _OfficialResourceMapper:
    """Minimal mapper using official CHAIR synonyms resource and SVO tokenization."""

    def __init__(self, synonyms: dict[str, list[str]]) -> None:
        self.categories = list(synonyms)
        self.delegate = OfficialChairMapper(self.categories)
        self.delegate.inverse_synonym_dict.clear()
        for canonical, aliases in synonyms.items():
            self.delegate.inverse_synonym_dict[canonical] = canonical
            for alias in aliases:
                normalized = self.delegate.canonicalize_phrase(alias) or _normalize_like_project(alias)
                self.delegate.inverse_synonym_dict[normalized] = canonical

    def caption_to_objects(self, caption: str) -> tuple[list[str], list[str], list[int], list[str]]:
        return self.delegate.caption_to_objects(caption)


def _normalize_like_project(value: str) -> str:
    return " ".join(value.lower().replace(",", " ").split())


if __name__ == "__main__":
    main()
