"""Hugging Face LLaVA image-text generation adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from paper_reproduce.models.base import GenerationResult


class LlavaHfGenerator:
    """LLaVA-1.5 adapter built on `transformers`.

    This module does not fabricate outputs. If model dependencies or checkpoint files are
    unavailable, construction fails with an actionable error.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - depends on optional model deps
            raise RuntimeError(
                "LLaVA generation requires the CUDA model stack. Install it with "
                "`pip install -r requirements-models-cu12.txt` on CUDA 12.1 hosts, then "
                "`pip install -e . --no-deps` if the project package is not installed."
            ) from exc

        self._torch = torch
        self._image_cls = Image
        generation_config = config.get("generation", {})
        runtime_config = config.get("runtime", {})

        self.model_name_or_path = generation_config.get(
            "model_name_or_path", "llava-hf/llava-1.5-7b-hf"
        )
        self.device = runtime_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_dtype(torch, runtime_config.get("dtype", "float16"), self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.max_new_tokens = int(generation_config.get("max_new_tokens", 128))
        self.temperature = float(generation_config.get("temperature", 0.0))
        self.top_p = float(generation_config.get("top_p", 1.0))
        self.do_sample = bool(generation_config.get("do_sample", False))
        self.return_token_scores = bool(generation_config.get("return_token_scores", True))

    def generate(self, image_path: str | Path, prompt: str) -> GenerationResult:
        """Generate text for one image."""

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = self._image_cls.open(path).convert("RGB")
        formatted_prompt = _format_llava_prompt(prompt)

        start = time.perf_counter()
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_token_count = int(inputs["input_ids"].shape[-1])

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
        if self.return_token_scores:
            generation_kwargs.update({"output_scores": True, "return_dict_in_generate": True})

        with self._torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        latency_sec = time.perf_counter() - start

        if self.return_token_scores:
            sequence = outputs.sequences[0]
            generated_ids = sequence[input_token_count:]
            text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
            token_scores = self._build_token_scores(outputs, generated_ids)
        else:
            sequence = outputs[0]
            generated_ids = sequence[input_token_count:]
            text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
            token_scores = None

        return GenerationResult(text=text, latency_sec=latency_sec, token_scores=token_scores)

    def _build_token_scores(self, outputs: Any, generated_ids: Any) -> list[dict[str, Any]]:
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )[0]
        records: list[dict[str, Any]] = []
        for token_id, logprob in zip(generated_ids.tolist(), transition_scores.tolist()):
            records.append(
                {
                    "token_id": int(token_id),
                    "token": self.processor.decode([token_id], skip_special_tokens=False),
                    "logprob": float(logprob),
                    "prob": float(self._torch.exp(self._torch.tensor(logprob)).item()),
                }
            )
        return records


def build_generator(config: Mapping[str, Any]) -> LlavaHfGenerator:
    """Build the configured LVLM generator for stage-2 scripts."""

    generation_config = config.get("generation", {})
    model_family = generation_config.get("model_family", "llava")
    if model_family not in {"llava", "llava_hf"}:
        raise ValueError(f"Unsupported generation.model_family for stage 2: {model_family}")
    return LlavaHfGenerator(config)


def _format_llava_prompt(prompt: str) -> str:
    if "<image>" in prompt:
        return prompt
    return f"USER: <image>\n{prompt}\nASSISTANT:"


def _resolve_dtype(torch: Any, dtype_name: str, device: str) -> Any:
    if device == "cpu":
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported runtime.dtype: {dtype_name}")
    return mapping[dtype_name]
