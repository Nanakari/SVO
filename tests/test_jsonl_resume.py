from __future__ import annotations

from pathlib import Path

import pytest

from paper_reproduce.utils.io import existing_sample_ids, read_jsonl, repair_jsonl_tail


def test_repair_jsonl_tail_truncates_only_final_partial_line(tmp_path: Path) -> None:
    path = tmp_path / "outputs.jsonl"
    path.write_text('{"sample_id": "a"}\n{"sample_id": "b"', encoding="utf-8")

    assert repair_jsonl_tail(path) is True
    assert list(read_jsonl(path)) == [{"sample_id": "a"}]


def test_existing_sample_ids_repairs_interrupted_append(tmp_path: Path) -> None:
    path = tmp_path / "outputs.jsonl"
    path.write_text('{"sample_id": "a"}\n{"sample_id": "b"', encoding="utf-8")

    assert existing_sample_ids(path) == {"a"}


def test_repair_jsonl_tail_rejects_middle_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "outputs.jsonl"
    path.write_text('{"sample_id": "a"}\nnot-json\n{"sample_id": "b"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        repair_jsonl_tail(path)
