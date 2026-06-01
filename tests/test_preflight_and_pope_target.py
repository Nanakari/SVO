from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from paper_reproduce.datasets.pope import load_pope_samples
from paper_reproduce.datasets.preflight import check_sample_image_paths
from paper_reproduce.datasets.types import CaptionSample


def test_preflight_image_check_fails_before_model_loading(tmp_path: Path) -> None:
    samples = [
        CaptionSample(
            sample_id="1",
            image_id="1",
            image_path=tmp_path / "missing.jpg",
            dataset="coco_chair",
            metadata={},
        )
    ]

    with pytest.raises(FileNotFoundError, match="Missing 1 image files"):
        check_sample_image_paths(samples, "first100")


def test_pope_loader_parses_target_object_from_question(tmp_path: Path) -> None:
    annotation = tmp_path / "random.jsonl"
    annotation.write_text(
        json.dumps(
            {
                "image": "COCO_val2014_000000000001.jpg",
                "question": "Is there a dining table in the image?",
                "label": "yes",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "paths": {
            "image_root": str(tmp_path),
            "annotation_files": {"random": str(annotation)},
        }
    }

    samples = load_pope_samples(config, tmp_path)

    assert samples[0].target_object == "dining table"


def test_extract_objects_prefers_structured_target_field(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = tmp_path / "config.yaml"
    input_path = tmp_path / "pope.jsonl"
    output_path = tmp_path / "objects.jsonl"
    config.write_text(
        "\n".join(
            [
                "object_extraction:",
                "  backend: spacy",
                "  spacy_model: missing_model_for_structured_target",
                "  vocabulary_path: configs/vocab/coco_objects.yaml",
            ]
        ),
        encoding="utf-8",
    )
    input_path.write_text(
        json.dumps(
            {
                "sample_id": "random:1",
                "image_id": "1",
                "image_path": "image.jpg",
                "dataset": "pope",
                "method": "base",
                "question": "Is there a dining table in the image?",
                "target_object": "dining table",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/extract_objects.py",
            "--config",
            str(config),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--text-field",
            "question",
            "--target-field",
            "target_object",
            "--no-risk",
        ],
        cwd=project_root,
        check=True,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["objects"] == [
        {
            "text": "dining table",
            "normalized": "dining table",
            "span": [11, 23],
            "object_index": 1,
            "source": "structured_target",
            "metadata": {"target_field": "target_object"},
        }
    ]
