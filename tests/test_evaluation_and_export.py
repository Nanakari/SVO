from __future__ import annotations

import json
from pathlib import Path

from paper_reproduce.evaluation.efficiency import evaluate_efficiency_records
from paper_reproduce.evaluation.export_tables import (
    build_table,
    export_table,
    load_metric_files,
    selected_specs,
)


def test_efficiency_metrics_from_real_counts() -> None:
    metrics, counts = evaluate_efficiency_records(
        object_records=[{"objects": [{"normalized": "person"}, {"normalized": "bottle"}]}],
        verification_records=[{"external_queries": 1, "latency_sec": 0.5}],
        base_records=[{"latency_sec": 1.0}],
    )

    assert metrics["verification_rate"] == 0.5
    assert metrics["external_queries_per_image"] == 1.0
    assert metrics["relative_latency"] == 1.5
    assert counts["external_queries"] == 1


def test_export_tables_preserves_missing_cells(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "coco_chair_svo_chair.json").write_text(
        json.dumps(
            {
                "dataset": "coco_chair",
                "task": "chair",
                "method": "svo",
                "metrics": {"chairs": 0.25, "chairi": 0.125},
            }
        ),
        encoding="utf-8",
    )

    metric_files = load_metric_files(metrics_dir)
    table = build_table(selected_specs(["chair_main"])[0], metric_files, missing_value="NA")
    ours = next(row for row in table["rows"] if row["label"] == "Ours")

    assert ours["display"]["CHAIRs ↓"] == "0.25"
    assert ours["display"]["Avg. Length"] == "NA"

    written = export_table(table, tmp_path / "tables", formats=["md", "csv", "json"])
    assert {path.suffix for path in written} == {".md", ".csv", ".json"}
