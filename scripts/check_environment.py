"""Check experiment environment dependencies and local model-cache readiness."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CORE_IMPORTS = {
    "yaml": "pyyaml",
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "pillow",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "huggingface_hub": "huggingface-hub",
    "tokenizers": "tokenizers",
    "spacy": "spacy",
    "groundingdino": "groundingdino",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate package versions, imports, CUDA visibility, and offline BERT cache."
    )
    parser.add_argument(
        "--lock",
        default="requirements-lock-cu121.txt",
        help="Pinned requirements file to compare with installed distributions.",
    )
    parser.add_argument(
        "--skip-pip-check",
        action="store_true",
        help="Skip `python -m pip check`.",
    )
    parser.add_argument(
        "--skip-bert-cache",
        action="store_true",
        help="Skip offline bert-base-uncased tokenizer/model loading.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero if torch.cuda.is_available() is false.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    if not args.skip_pip_check:
        print("== pip check ==")
        result = subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
        if result.returncode != 0:
            failures.append("pip check reported broken requirements")

    if args.lock:
        failures.extend(_check_lock(resolve_path(args.lock)))

    failures.extend(_check_imports())
    failures.extend(_check_cuda(require_cuda=args.require_cuda))
    if not args.skip_bert_cache:
        failures.extend(_check_bert_cache())

    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("\nEnvironment check passed.")


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _check_lock(lock_path: Path) -> list[str]:
    print("\n== lock comparison ==")
    if not lock_path.exists():
        return [f"lock file not found: {lock_path}"]

    failures: list[str] = []
    for requirement in _locked_requirements(lock_path):
        name = requirement["name"]
        expected = requirement.get("version")
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            failures.append(f"{name} is pinned in {lock_path.name} but is not installed")
            continue

        if expected is not None and installed != expected:
            failures.append(f"{name}=={installed} does not match locked {expected}")
            print(f"MISMATCH {name}: installed={installed} locked={expected}")
        elif expected is None:
            print(f"OK {name}: installed={installed} locked=direct-reference")
        else:
            print(f"OK {name}: {installed}")
    return failures


def _locked_requirements(lock_path: Path) -> list[dict[str, str]]:
    requirements: list[dict[str, str]] = []
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.startswith("-e ") or line.startswith("git+"):
            continue

        exact = re.match(r"^([A-Za-z0-9_.-]+)==([^;\s]+)", line)
        if exact:
            requirements.append({"name": exact.group(1), "version": exact.group(2)})
            continue

        direct = re.match(r"^([A-Za-z0-9_.-]+)\s+@", line)
        if direct:
            requirements.append({"name": direct.group(1)})
    return requirements


def _check_imports() -> list[str]:
    print("\n== import check ==")
    failures: list[str] = []
    for module_name, dist_name in CORE_IMPORTS.items():
        if importlib.util.find_spec(module_name) is None:
            failures.append(f"import missing: {module_name}")
            print(f"MISSING {module_name}")
            continue
        try:
            version = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            version = "unknown"
        print(f"OK {module_name} ({dist_name} {version})")
    return failures


def _check_cuda(*, require_cuda: bool) -> list[str]:
    print("\n== cuda check ==")
    try:
        import torch
    except ImportError:
        return ["torch import failed during CUDA check"]

    torch_version = getattr(torch, "__version__", "unknown")
    cuda_module = getattr(torch, "cuda", None)
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if cuda_module is None:
        print(f"torch={torch_version}")
        print(f"torch_cuda={torch_cuda_version}")
        print("cuda_available=false")
        return ["torch.cuda is unavailable; check the installed torch package"]

    available = cuda_module.is_available()
    print(f"torch={torch_version}")
    print(f"torch_cuda={torch_cuda_version}")
    print(f"cuda_available={available}")
    if available:
        print(f"cuda_device={cuda_module.get_device_name(0)}")
    if require_cuda and not available:
        return ["CUDA is required but torch.cuda.is_available() is false"]
    return []


def _check_bert_cache() -> list[str]:
    print("\n== offline bert-base-uncased cache check ==")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer, BertModel

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        model = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - show cache/model-load errors verbatim.
        return [f"offline bert-base-uncased load failed: {exc!r}"]

    print(f"tokenizer={tokenizer.__class__.__name__}")
    print(f"model={model.__class__.__name__}")
    return []


if __name__ == "__main__":
    main()
