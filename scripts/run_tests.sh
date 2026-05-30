#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

"$PYTHON" -m compileall -q src scripts tests
"$PYTHON" -m pytest tests
bash scripts/smoke_test.sh
