#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-SVO}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_BIN="${CONDA_BIN:-conda}"
WITH_MODELS=0
WITH_NLP=1
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_cloud.sh [options]

Create or check the SVO conda environment for Ubuntu + CUDA 12 experiments.

Options:
  --env-name NAME       Conda environment name. Default: SVO
  --python VERSION     Python version. Default: 3.10
  --with-models        Install CUDA 12 PyTorch and model dependencies.
  --with-nlp           Install NLP dependencies and the spaCy small English model. Default.
  --without-nlp        Skip NLP dependencies; use only rule-based smoke/test paths.
  --check-only         Only print environment diagnostics; do not install.
  -h, --help           Show this help.

Environment variables:
  CONDA_BIN            Conda executable. Default: conda
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --with-models)
      WITH_MODELS=1
      shift
      ;;
    --with-nlp)
      WITH_NLP=1
      shift
      ;;
    --without-nlp)
      WITH_NLP=0
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
  echo "Conda executable not found: $CONDA_BIN" >&2
  echo "Set CONDA_BIN=/path/to/conda or initialize conda before running this script." >&2
  exit 1
fi

env_exists() {
  "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

run_in_env() {
  "$CONDA_BIN" run -n "$ENV_NAME" "$@"
}

print_check() {
  echo "Project root: $ROOT_DIR"
  echo "Conda: $("$CONDA_BIN" --version)"
  if env_exists; then
    echo "Environment: $ENV_NAME"
    run_in_env python --version
    run_in_env python - <<'PY'
import importlib.util
mods = ["yaml", "numpy", "pandas", "PIL", "paper_reproduce", "torch", "transformers", "groundingdino"]
for name in mods:
    print(f"{name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
PY
  else
    echo "Environment missing: $ENV_NAME"
    return 1
  fi
}

cd "$ROOT_DIR"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  print_check
  exit $?
fi

if env_exists; then
  echo "Using existing conda environment: $ENV_NAME"
else
  echo "Creating conda environment $ENV_NAME with Python $PYTHON_VERSION"
  "$CONDA_BIN" create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

run_in_env python -m pip install -U pip setuptools wheel
run_in_env python -m pip install -e ".[eval,dev]"

if [[ "$WITH_NLP" -eq 1 ]]; then
  run_in_env python -m pip install -e ".[nlp]"
  run_in_env python -m spacy download en_core_web_sm
fi

if [[ "$WITH_MODELS" -eq 1 ]]; then
  run_in_env python -m pip install -r requirements-models-cu12.txt
fi

print_check

echo
echo "Next steps:"
echo "  conda activate $ENV_NAME"
echo "  bash scripts/prepare_data.sh"
echo "  bash scripts/download_models.sh --confirm"
echo "  bash scripts/smoke_test.sh"
