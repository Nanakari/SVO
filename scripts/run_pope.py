"""Run Base POPE yes/no inference with LLaVA."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paper_reproduce.datasets import load_pope_samples
from paper_reproduce.models.llava_hf import build_generator
from paper_reproduce.utils.cli import add_common_config_args, load_cli_config, positive_int
from paper_reproduce.utils.config import resolve_path
from paper_reproduce.utils.io import append_jsonl, ensure_parent, existing_sample_ids
from paper_reproduce.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run POPE object-existence yes/no inference.")
    add_common_config_args(parser)
    parser.add_argument("--output", help="Output JSONL path. Defaults to outputs/predictions.")
    parser.add_argument(
        "--settings",
        nargs="+",
        help="POPE settings to run, e.g. random popular adversarial. Defaults to config order.",
    )
    parser.add_argument(
        "--limit-per-setting",
        type=positive_int,
        help="Limit the number of questions per POPE setting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file instead of resuming from existing sample_ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cli_config(args)
    dataset_config = config["dataset"]
    method_config = config["method"]
    dataset_name = str(dataset_config.get("name", "pope"))
    method_name = str(method_config.get("name", "base"))
    set_seed(config.get("project", {}).get("seed"))

    output_path = _default_output_path(args.output, dataset_name, method_name, "pope")
    if args.overwrite and output_path.exists():
        output_path.unlink()
    skip_ids = existing_sample_ids(output_path)

    samples = load_pope_samples(
        dataset_config,
        PROJECT_ROOT,
        settings=args.settings,
        limit_per_setting=args.limit_per_setting,
    )
    generator = build_generator(config)
    prompt_template = str(
        dataset_config.get("inference", {}).get(
            "prompt_template",
            "{question}\nAnswer the question using only Yes or No.",
        )
    )
    generation_metadata = _generation_metadata(config)

    written = 0
    for sample in samples:
        if sample.sample_id in skip_ids:
            continue
        prompt = prompt_template.format(question=sample.question)
        result = generator.generate(sample.image_path, prompt)
        record = {
            "sample_id": sample.sample_id,
            "image_id": sample.image_id,
            "image_path": str(sample.image_path),
            "dataset": dataset_name,
            "method": method_name,
            "setting": sample.setting,
            "question": sample.question,
            "prompt": prompt,
            "raw_response": result.text,
            "answer": normalise_yes_no(result.text),
            "label": sample.label,
            "generation": generation_metadata,
            "token_scores": result.token_scores,
            "latency_sec": result.latency_sec,
            "metadata": {"source": sample.raw},
        }
        append_jsonl(output_path, [record])
        written += 1

    print(f"Loaded samples: {len(samples)}")
    print(f"Skipped existing: {len(skip_ids)}")
    print(f"Wrote records: {written}")
    print(f"Output: {output_path}")


def normalise_yes_no(text: str) -> str:
    """Map generated text to yes/no/unknown without changing the raw response."""

    match = re.search(r"\b(yes|no)\b", text.strip(), flags=re.IGNORECASE)
    if not match:
        return "unknown"
    return match.group(1).lower()


def _default_output_path(
    explicit_output: str | None, dataset_name: str, method_name: str, suffix: str
) -> Path:
    if explicit_output:
        return ensure_parent(resolve_path(explicit_output, PROJECT_ROOT))
    return ensure_parent(
        PROJECT_ROOT / "outputs" / "predictions" / f"{dataset_name}_{method_name}_{suffix}.jsonl"
    )


def _generation_metadata(config: dict[str, Any]) -> dict[str, Any]:
    generation_config = config.get("generation", {})
    project_config = config.get("project", {})
    return {
        "model_family": generation_config.get("model_family"),
        "model_name_or_path": generation_config.get("model_name_or_path"),
        "max_new_tokens": generation_config.get("max_new_tokens"),
        "temperature": generation_config.get("temperature"),
        "top_p": generation_config.get("top_p"),
        "do_sample": generation_config.get("do_sample"),
        "seed": project_config.get("seed"),
    }


if __name__ == "__main__":
    main()
