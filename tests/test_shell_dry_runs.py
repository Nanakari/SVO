from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def bash_or_skip() -> str:
    bash = shutil.which("bash")
    if bash is not None:
        return bash
    for candidate in (
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
        Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
    ):
        if candidate.exists():
            return str(candidate)
    pytest.skip("bash is not available")


def run_bash(*args: str) -> str:
    result = subprocess.run(
        [bash_or_skip(), *args],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def test_tune_threshold_dry_run_defaults_to_val2000() -> None:
    output = run_bash("scripts/tune_svo_threshold.sh", "--dry-run", "--thresholds", "0.5")

    assert "--sample-size 2000" in output
    assert "configs/splits/coco_train2017_val2000_seed42.txt" in output
    assert "coco_train2017_val2000_base_captions.jsonl" in output


def test_run_all_dry_run_applies_main_coco_split_by_default() -> None:
    output = run_bash(
        "scripts/run_all.sh",
        "--dry-run",
        "--skip-prior",
        "--datasets",
        "coco_chair",
        "--methods",
        "base,svo",
        "--risk-threshold",
        "1.0",
    )

    assert "--sample-size 5000" in output
    assert "configs/splits/coco_val2014_main5000_seed42.txt" in output
    assert "--set dataset.paths.split_file=configs/splits/coco_val2014_main5000_seed42.txt" in output


def test_run_all_full_dataset_does_not_apply_main_coco_split() -> None:
    output = run_bash(
        "scripts/run_all.sh",
        "--dry-run",
        "--skip-prior",
        "--datasets",
        "coco_chair",
        "--methods",
        "base,svo",
        "--risk-threshold",
        "1.0",
        "--full-dataset",
    )

    assert "Using full COCO/CHAIR dataset" in output
    assert "coco_val2014_main5000_seed42.txt" not in output
