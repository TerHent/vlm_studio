import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from evaluator.metrics import COCO_IOU_THRESHOLDS

@dataclass(frozen=True)
class EvaluatorConfig:
    model_name: str
    dataset_path: str
    dataset_split: str = "validation"
    label_map: Dict[str, str] = field(default_factory=dict)
    iou_thresholds: List[float] = field(default_factory=lambda: list(COCO_IOU_THRESHOLDS))
    output_report_path: str = "evaluation_report.json"
    device: str = "auto"
    max_samples: Optional[int] = None
    visualize: bool = True
    visualize_dir: str = "visualizations"
    mode: str = "all"
    predictions_dir: str = "predictions"
    conf_threshold: float = 0.0
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    images_dir: Optional[str] = None

    def validate(self) -> None:
        valid_modes = ["all", "predict", "evaluate"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {valid_modes}.")
        if not self.model_name:
            raise ValueError("model_name cannot be empty.")
        if not self.dataset_path:
            raise ValueError("dataset_path cannot be empty.")
        if not isinstance(self.label_map, dict):
            raise ValueError("label_map must be a dictionary.")
        if not isinstance(self.iou_thresholds, list) or len(self.iou_thresholds) == 0:
            raise ValueError("iou_thresholds must be a non-empty list of floats.")
        for th in self.iou_thresholds:
            if not (0.0 <= th <= 1.0):
                raise ValueError(f"IoU threshold {th} must be between 0.0 and 1.0.")
        if not (0.0 <= self.conf_threshold <= 1.0):
            raise ValueError(f"conf_threshold must be between 0.0 and 1.0, got {self.conf_threshold}.")
