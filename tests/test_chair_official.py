from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from paper_reproduce.evaluation.chair import (
    OfficialChairMapper,
    evaluate_chair_records_internal,
    evaluate_chair_records_official,
)
from paper_reproduce.evaluation.coco_gt import (
    load_coco_category_names,
    load_coco_gt_objects_official,
)
from paper_reproduce.extraction.types import ExtractedObject


class DummyExtractor:
    def extract(self, text: str) -> list[ExtractedObject]:
        names = [token.removesuffix(".") for token in text.lower().split()]
        return [
            ExtractedObject(
                text=name,
                normalized=name,
                span=(0, 0),
                object_index=index,
                source="dummy",
                metadata={},
            )
            for index, name in enumerate(names, start=1)
            if name in {"person", "chair", "dog", "cat"}
        ]


def test_official_chair_metrics_match_fixture_counts() -> None:
    mapper = OfficialChairMapper(["person", "chair", "dog", "cat", "sports ball"])
    records = [
        {"image_id": 1, "caption": "A man sits on a chair."},
        {"image_id": 2, "caption": "A dog holds a sports ball and a cat."},
    ]
    gt_by_image = {
        "1": {"person", "chair"},
        "2": {"dog", "sports ball"},
    }

    metrics, counts = evaluate_chair_records_official(
        records,
        gt_by_image=gt_by_image,
        mapper=mapper,
    )

    assert metrics["chairs"] == 0.5
    assert metrics["chairi"] == 0.2
    assert counts["object_mentions"] == 5
    assert counts["hallucinated_object_mentions"] == 1
    assert counts["chair_backend"] == "official"


def test_official_mapper_handles_chair_special_cases() -> None:
    mapper = OfficialChairMapper(["person", "bird", "tie", "toilet", "sports ball", "cell phone"])

    assert mapper.caption_to_objects("A baby bird with a bow tie.")[1] == ["bird", "tie"]
    assert mapper.caption_to_objects("A toilet seat near two cell phones.")[1] == [
        "toilet",
        "cell phone",
    ]
    assert mapper.caption_to_objects("A player throws sports balls.")[1] == [
        "person",
        "sports ball",
    ]


def test_official_coco_gt_adds_reference_caption_objects(tmp_path: Path) -> None:
    instances = tmp_path / "instances.json"
    captions = tmp_path / "captions.json"
    instances.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "one.jpg"}],
                "annotations": [{"image_id": 1, "category_id": 10}],
                "categories": [
                    {"id": 10, "name": "person"},
                    {"id": 11, "name": "cat"},
                    {"id": 12, "name": "couch"},
                ],
            }
        ),
        encoding="utf-8",
    )
    captions.write_text(
        json.dumps({"annotations": [{"image_id": 1, "caption": "A cat on a sofa."}]}),
        encoding="utf-8",
    )

    mapper = OfficialChairMapper(load_coco_category_names(instances))
    gt_by_image = load_coco_gt_objects_official(
        instances,
        mapper,
        caption_annotation_path=captions,
    )

    assert gt_by_image == {"1": {"person", "cat", "couch"}}


def test_internal_chair_scorer_remains_explicit_fallback() -> None:
    records = [{"image_id": 1, "caption": "person cat"}]
    metrics, counts = evaluate_chair_records_internal(
        records,
        gt_by_image={"1": {"person"}},
        extractor=DummyExtractor(),
    )

    assert metrics["chairs"] == 1.0
    assert metrics["chairi"] == 0.5
    assert counts["chair_backend"] == "internal"


def test_evaluate_cli_defaults_to_official_chair_backend(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = tmp_path / "config.yaml"
    dataset = tmp_path / "dataset.yaml"
    instances = tmp_path / "instances.json"
    captions = tmp_path / "captions.json"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "metrics.json"

    config.write_text("{}\n", encoding="utf-8")
    dataset.write_text(
        "\n".join(
            [
                "name: coco_chair",
                "paths:",
                f"  annotation_file: {instances.as_posix()}",
                f"  caption_annotation_file: {captions.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    instances.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "one.jpg"}],
                "annotations": [{"image_id": 1, "category_id": 1}],
                "categories": [{"id": 1, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )
    captions.write_text(json.dumps({"annotations": []}), encoding="utf-8")
    predictions.write_text(
        json.dumps({"image_id": 1, "caption": "A person with a bottle."}) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            str(config),
            "--dataset",
            str(dataset),
            "--task",
            "chair",
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=True,
    )

    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["counts"]["chair_backend"] == "official"
    assert metrics["metrics"]["chairs"] == 1.0
