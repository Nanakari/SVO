"""Evaluation interfaces for hallucination, efficiency, and false-correction metrics."""

from paper_reproduce.evaluation.chair import evaluate_chair_records
from paper_reproduce.evaluation.classification import amber_object_score, evaluate_yes_no_records
from paper_reproduce.evaluation.coco_gt import load_coco_gt_objects
from paper_reproduce.evaluation.efficiency import evaluate_efficiency_records
from paper_reproduce.evaluation.export_tables import build_table, export_table, load_metric_files
from paper_reproduce.evaluation.false_correction import evaluate_false_correction_records

__all__ = [
    "amber_object_score",
    "evaluate_chair_records",
    "evaluate_efficiency_records",
    "evaluate_false_correction_records",
    "evaluate_yes_no_records",
    "build_table",
    "export_table",
    "load_metric_files",
    "load_coco_gt_objects",
]
