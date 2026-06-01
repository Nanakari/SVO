from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_chair_parity_harness_accepts_local_official_repo_fixture(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "Hallucination"
    data = repo / "data"
    data.mkdir(parents=True)
    (data / "synonyms.txt").write_text(
        "\n".join(
            [
                "person, woman",
                "bicycle, bike",
                "traffic light, traffic signal",
                "bird",
                "toilet",
                "tie, bow tie",
                "cell phone, phone",
                "sports ball, ball",
                "couch, sofa",
                "dog",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_chair_parity.py",
            "--official-chair-repo",
            str(repo),
        ],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Object extraction parity: OK" in result.stdout
    assert "Metric parity: OK" in result.stdout
