# Result Schemas

All records are JSON Lines unless stated otherwise. Fields may grow as implementation progresses, but required identifiers should stay stable.

## captions

```json
{
  "sample_id": "string",
  "image_id": "string",
  "image_path": "string",
  "dataset": "coco_chair",
  "method": "base",
  "prompt": "Please describe this image in detail.",
  "caption": "string",
  "generation": {
    "model_name_or_path": "string",
    "max_new_tokens": 128,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42
  },
  "token_scores": null,
  "latency_sec": 0.0
}
```

## objects

```json
{
  "sample_id": "string",
  "image_id": "string",
  "caption": "string",
  "method": "svo",
  "caption_method": "base",
  "objects": [
    {
      "text": "bottle",
      "normalized": "bottle",
      "span": [0, 0],
      "object_index": 1,
      "source": "spacy_noun_chunk",
      "metadata": {},
      "risk": {
        "uncertainty": null,
        "position": 1.0,
        "prior": null,
        "total": null,
        "enabled_terms": {
          "uncertainty": true,
          "position": true,
          "prior": true
        },
        "missing_terms": []
      }
    }
  ]
}
```

## static priors

```json
{
  "source_captions": "outputs/predictions/coco_train_base_captions.jsonl",
  "coco_annotation_file": "data/coco/annotations/instances_train2017.json",
  "min_count": 5,
  "epsilon": 1e-6,
  "mean_prior": 0.0,
  "generated_total": 0,
  "hallucinated_total": 0,
  "priors": {
    "bottle": {
      "generated": 0,
      "hallucinated": 0,
      "prior": 0.0,
      "low_frequency": true
    }
  }
}
```

## verifications

```json
{
  "sample_id": "string",
  "image_id": "string",
  "method": "svo",
  "object_method": "svo",
  "verified_objects": [
    {
      "normalized": "bottle",
      "query": "bottle",
      "score": 0.0,
      "has_visual_evidence": false,
      "detector": "groundingdino",
      "boxes": [
        {
          "box": [0.0, 0.0, 0.0, 0.0],
          "score": 0.0,
          "phrase": "bottle"
        }
      ],
      "source_object": {
        "text": "bottle",
        "normalized": "bottle",
        "span": [0, 0],
        "object_index": 1,
        "risk": {}
      }
    }
  ],
  "external_queries": 1,
  "latency_sec": 0.0
}
```

## revisions

```json
{
  "sample_id": "string",
  "image_id": "string",
  "method": "svo",
  "original_caption": "string",
  "revised_caption": "string",
  "actions": [
    {
      "object": "bottle",
      "action": "delete",
      "rule": "coordination",
      "reason": "no_visual_evidence",
      "span": [0, 0],
      "replacement": "",
      "score": 0.0,
      "has_visual_evidence": false
    }
  ],
  "revision": {
    "strategy": "conservative",
    "allow_rules": [
      "coordination",
      "simple_existence",
      "simple_with_preposition"
    ]
  }
}
```

## pope revisions

```json
{
  "sample_id": "string",
  "image_id": "string",
  "question": "Is there a bottle in the image?",
  "original_answer": "yes",
  "revised_answer": "no",
  "answer": "no",
  "action": {
    "action": "yes_to_no",
    "rule": "pope_yes_to_no",
    "reason": "no_visual_evidence",
    "object": "bottle"
  },
  "verified_objects": []
}
```

## metrics

Metric files should be machine-generated JSON:

```json
{
  "dataset": "coco_chair",
  "method": "svo",
  "metrics": {
    "chairs": null,
    "chairi": null,
    "average_length": null,
    "correct_object_coverage": null,
    "verification_rate": null,
    "external_queries_per_image": null,
    "relative_latency": null,
    "hallucinated_object_removal": null,
    "correct_object_retention": null,
    "false_correction_rate": null
  },
  "source_files": [],
  "created_at": "ISO-8601 timestamp"
}
```

Exported tables must read these metric files. Missing metric files should produce empty cells or explicit missing markers, never fabricated values.

Task-specific metric keys:

- `chair`: `chairs`, `chairi`, `average_length`, `correct_object_coverage`.
- `pope`: per-setting and `overall` `accuracy`, `precision`, `recall`, `f1`, `yes_ratio`.
- `amber`: `accuracy`, `precision`, `recall`, `f1`, `yes_ratio`, `amber_object_score`.
- `efficiency`: `verification_rate`, `external_queries_per_image`, `relative_latency`, latency components.
- `false_correction`: `hallucinated_object_removal`, `correct_object_retention`, `false_correction_rate`.

## exported tables

`scripts/export_results.py` writes Markdown, CSV, and JSON tables under `outputs/tables` by default.
JSON table files keep cell provenance:

```json
{
  "name": "chair_main",
  "title": "CHAIR Main Results",
  "columns": ["Method", "CHAIRs ↓"],
  "rows": [
    {
      "label": "Ours",
      "method_candidates": ["svo", "ours", "ours_full"],
      "cells": [
        {
          "column": "CHAIRs ↓",
          "value": null,
          "display": "",
          "missing": true,
          "source_file": null,
          "metric_path": "chairs",
          "task": "chair",
          "dataset": "coco_chair"
        }
      ]
    }
  ]
}
```

Missing cells mean the corresponding metric JSON was not present or did not contain that metric.
