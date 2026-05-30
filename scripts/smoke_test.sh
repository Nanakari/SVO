#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/smoke}"

cd "$ROOT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"/{objects,verifications,revisions,metrics,tables}

CAPTIONS="examples/smoke/smoke_captions.jsonl"
ANNOTATIONS="examples/smoke/coco_instances_smoke.json"
OBJECTS="$OUT_DIR/objects/smoke_objects.jsonl"
VERIFICATIONS="$OUT_DIR/verifications/smoke_svo.jsonl"
REVISIONS="$OUT_DIR/revisions/smoke_svo_revisions.jsonl"

echo "[1/7] Extract objects and risk scores"
"$PYTHON" scripts/extract_objects.py \
  --config configs/default.yaml \
  --method configs/methods/svo.yaml \
  --input "$CAPTIONS" \
  --output "$OBJECTS" \
  --backend rule \
  --overwrite

echo "[2/7] Verify no objects by using a high risk threshold"
"$PYTHON" scripts/verify_objects.py \
  --config configs/default.yaml \
  --method configs/methods/svo.yaml \
  --input "$OBJECTS" \
  --output "$VERIFICATIONS" \
  --risk-threshold 99 \
  --overwrite

echo "[3/7] Revise captions conservatively"
"$PYTHON" scripts/revise_captions.py \
  --config configs/default.yaml \
  --input "$VERIFICATIONS" \
  --output "$REVISIONS" \
  --overwrite

echo "[4/7] Evaluate CHAIR"
"$PYTHON" scripts/evaluate.py \
  --config configs/default.yaml \
  --dataset coco_chair \
  --task chair \
  --predictions "$REVISIONS" \
  --text-field revised_caption \
  --coco-annotations "$ANNOTATIONS" \
  --backend rule \
  --method svo \
  --output "$OUT_DIR/metrics/coco_chair_svo_chair.json"

echo "[5/7] Evaluate efficiency"
"$PYTHON" scripts/evaluate.py \
  --config configs/default.yaml \
  --dataset coco_chair \
  --task efficiency \
  --objects "$OBJECTS" \
  --verifications "$VERIFICATIONS" \
  --base-predictions "$CAPTIONS" \
  --method svo \
  --output "$OUT_DIR/metrics/coco_chair_svo_efficiency.json"

echo "[6/7] Evaluate false-correction metrics"
"$PYTHON" scripts/evaluate.py \
  --config configs/default.yaml \
  --dataset coco_chair \
  --task false_correction \
  --predictions "$REVISIONS" \
  --coco-annotations "$ANNOTATIONS" \
  --backend rule \
  --method svo \
  --output "$OUT_DIR/metrics/coco_chair_svo_false_correction.json"

echo "[7/7] Export tables from real smoke metrics"
"$PYTHON" scripts/export_results.py \
  --config configs/default.yaml \
  --metrics-dir "$OUT_DIR/metrics" \
  --out "$OUT_DIR/tables" \
  --missing-value NA

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/smoke")
objects = [json.loads(line) for line in (root / "objects/smoke_objects.jsonl").read_text(encoding="utf-8").splitlines()]
names = {obj["normalized"] for obj in objects[0]["objects"]}
assert {"person", "laptop", "bottle"}.issubset(names), names
assert all("risk" in obj for obj in objects[0]["objects"])
verifications = [json.loads(line) for line in (root / "verifications/smoke_svo.jsonl").read_text(encoding="utf-8").splitlines()]
assert verifications[0]["external_queries"] == 0
for metric_path in (root / "metrics").glob("*.json"):
    data = json.loads(metric_path.read_text(encoding="utf-8"))
    assert "metrics" in data and isinstance(data["metrics"], dict), metric_path
assert (root / "tables/manifest.json").exists()
print("Smoke assertions passed.")
PY

echo "Smoke test output: $OUT_DIR"
