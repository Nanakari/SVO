"""GroundingDINO verifier adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from paper_reproduce.utils.config import resolve_path
from paper_reproduce.verification.types import DetectionBox, VerificationResult


class GroundingDinoVerifier:
    """Open-vocabulary object verifier backed by GroundingDINO.

    The adapter follows the official Python inference API:
    `load_model`, `load_image`, and `predict` from `groundingdino.util.inference`.
    """

    detector_name = "groundingdino"

    def __init__(self, config: Mapping[str, Any], project_root: str | Path) -> None:
        verification_config = config.get("verification", {})
        grounding_config = verification_config.get("groundingdino", {})
        runtime_config = config.get("runtime", {})

        self.device = str(runtime_config.get("device", "cuda"))
        self.box_threshold = float(grounding_config.get("box_threshold", 0.35))
        self.text_threshold = float(grounding_config.get("text_threshold", 0.25))
        evidence_threshold = verification_config.get("evidence_threshold")
        self.evidence_threshold = (
            float(evidence_threshold) if evidence_threshold is not None else self.box_threshold
        )
        self.query_template = str(grounding_config.get("query_template", "{object} ."))
        self.save_boxes = bool(grounding_config.get("save_boxes", True))

        config_path = resolve_path(grounding_config.get("config_path"), project_root)
        checkpoint_path = resolve_path(grounding_config.get("checkpoint_path"), project_root)
        if config_path is None:
            raise ValueError("verification.groundingdino.config_path is required")
        if checkpoint_path is None:
            raise ValueError("verification.groundingdino.checkpoint_path is required")
        if not config_path.exists():
            raise FileNotFoundError(f"GroundingDINO config file not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint_path}")

        try:
            from groundingdino.util.inference import load_image, load_model, predict
        except ImportError as exc:  # pragma: no cover - depends on optional detector install
            raise RuntimeError(
                "GroundingDINO is not installed. Clone the official IDEA-Research/GroundingDINO "
                "repository, install it with `pip install -e .`, then set config_path and "
                "checkpoint_path in configs/default.yaml."
            ) from exc

        self._load_image = load_image
        self._predict = predict
        self.model = load_model(str(config_path), str(checkpoint_path), device=self.device)

    def verify(self, image_path: str | Path, object_name: str) -> VerificationResult:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found for visual verification: {path}")

        query = self._format_query(object_name)
        start = time.perf_counter()
        _, image = self._load_image(str(path))
        boxes, logits, phrases = self._predict(
            model=self.model,
            image=image,
            caption=query,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        latency_sec = time.perf_counter() - start

        detection_boxes = _to_detection_boxes(boxes, logits, phrases, save_boxes=self.save_boxes)
        max_score = max((box.score for box in detection_boxes), default=0.0)
        return VerificationResult(
            normalized=object_name,
            query=query,
            score=max_score,
            has_visual_evidence=max_score >= self.evidence_threshold,
            detector=self.detector_name,
            boxes=detection_boxes,
            latency_sec=latency_sec,
            metadata={
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
                "evidence_threshold": self.evidence_threshold,
                "device": self.device,
            },
        )

    def _format_query(self, object_name: str) -> str:
        query = self.query_template.format(object=object_name)
        return query if query.endswith(".") else f"{query} ."


def build_verifier(config: Mapping[str, Any], project_root: str | Path) -> GroundingDinoVerifier:
    """Build the configured visual verifier."""

    detector = str(config.get("verification", {}).get("detector", "groundingdino")).lower()
    if detector != "groundingdino":
        raise ValueError(f"Unsupported verification.detector for stage 4: {detector}")
    return GroundingDinoVerifier(config, project_root)


def _to_detection_boxes(
    boxes: Any, logits: Any, phrases: Any, *, save_boxes: bool
) -> list[DetectionBox]:
    box_values = _to_list(boxes)
    logit_values = _to_list(logits)
    phrase_values = list(phrases) if phrases is not None else []

    detections: list[DetectionBox] = []
    for index, score_value in enumerate(logit_values):
        raw_box = box_values[index] if index < len(box_values) else []
        box = [float(value) for value in raw_box] if save_boxes else []
        phrase = str(phrase_values[index]) if index < len(phrase_values) else None
        detections.append(
            DetectionBox(
                box=box,
                score=float(score_value),
                phrase=phrase,
            )
        )
    detections.sort(key=lambda item: item.score, reverse=True)
    return detections


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)
