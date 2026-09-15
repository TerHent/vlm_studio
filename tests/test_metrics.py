import pytest
import numpy as np
from evaluator.metrics import (
    compute_iou, 
    DetectionEvaluator, 
    COCO_IOU_THRESHOLDS, 
    parse_iou_thresholds
)

def test_iou_computation() -> None:
    # Test perfect match
    box1 = [0.1, 0.1, 0.5, 0.5]
    box2 = [0.1, 0.1, 0.5, 0.5]
    assert abs(compute_iou(box1, box2) - 1.0) < 1e-6
    
    # Test no overlap
    box3 = [0.6, 0.6, 0.9, 0.9]
    assert compute_iou(box1, box3) == 0.0
    
    # Test 50% overlap (half overlap in x-axis)
    # Area box1 = 0.4 * 0.4 = 0.16
    # Area box4 = 0.4 * 0.4 = 0.16
    # Intersection = y: (0.1 to 0.5) -> 0.4, x: (0.3 to 0.5) -> 0.2. Intersection area = 0.08
    # Union = 0.16 + 0.16 - 0.08 = 0.24
    # IoU = 0.08 / 0.24 = 1/3
    box4 = [0.1, 0.3, 0.5, 0.7]
    assert abs(compute_iou(box1, box4) - (1.0 / 3.0)) < 1e-6

def test_label_mapping_alignment() -> None:
    # Configure evaluator with mappings: 'woman' -> 'person', 'car' -> 'vehicle'
    label_map = {"woman": "person", "car": "vehicle"}
    evaluator = DetectionEvaluator(label_map=label_map, iou_thresholds=[0.5])
    
    # Ground truth is standard 'person' and 'vehicle'
    gt = [
        {"bbox": [0.1, 0.1, 0.5, 0.5], "label": "person"},
        {"bbox": [0.2, 0.2, 0.6, 0.6], "label": "vehicle"}
    ]
    
    # Predictions use aliased 'woman' and 'car'
    pred = [
        {"bbox": [0.11, 0.11, 0.49, 0.49], "label": "woman", "score": 0.95},
        {"bbox": [0.21, 0.21, 0.59, 0.59], "label": "car", "score": 0.90}
    ]
    
    evaluator.update(predictions=pred, ground_truths=gt)
    metrics = evaluator.compute_metrics()
    
    # Verify mapping occurred: predictions mapped to person/vehicle should match ground truths
    # With threshold 0.5, both IoUs are > 0.5, so they should match perfectly
    iou_metrics = metrics["per_iou"]["iou_0.50"]
    assert iou_metrics["mAP"] == 1.0
    assert iou_metrics["mean_precision"] == 1.0
    assert iou_metrics["mean_recall"] == 1.0
    
    # Check that category metrics for mapped keys are computed correctly
    class_metrics = iou_metrics["class_metrics"]
    assert "person" in class_metrics
    assert "vehicle" in class_metrics
    # Since woman was mapped to person, there should be 1 prediction for person
    assert class_metrics["person"]["num_predictions"] == 1
    assert class_metrics["person"]["AP"] == 1.0
    
    # Confirms 'woman' and 'car' did not leak into output classes
    assert "woman" not in class_metrics
    assert "car" not in class_metrics

def test_ap_calculation() -> None:
    # Test AP computation with correct matching and one false positive
    evaluator = DetectionEvaluator(label_map={}, iou_thresholds=[0.5])
    
    # Image 0
    gt_0 = [{"bbox": [0.0, 0.0, 0.5, 0.5], "label": "dog"}]
    pred_0 = [
        {"bbox": [0.0, 0.0, 0.48, 0.48], "label": "dog", "score": 0.9}, # TP
        {"bbox": [0.6, 0.6, 0.9, 0.9], "label": "dog", "score": 0.8}    # FP (no GT overlap)
    ]
    evaluator.update(pred_0, gt_0)
    
    # Image 1
    gt_1 = [{"bbox": [0.0, 0.0, 0.5, 0.5], "label": "dog"}]
    pred_1 = [] # FN (Missed detection)
    evaluator.update(pred_1, gt_1)
    
    metrics = evaluator.compute_metrics()
    iou_metrics = metrics["per_iou"]["iou_0.50"]
    class_metrics = iou_metrics["class_metrics"]
    
    dog_metrics = class_metrics["dog"]
    assert dog_metrics["num_ground_truth"] == 2
    assert dog_metrics["num_predictions"] == 2
    
    # Predictions sorted by score:
    # 1. score 0.9 (TP) -> Rec = 0.5, Prec = 1.0
    # 2. score 0.8 (FP) -> Rec = 0.5, Prec = 0.5
    # Recalls: [0.5, 0.5]
    # Precisions: [1.0, 0.5]
    # All-points AP interpolation curve:
    # mrec = [0.0, 0.5, 0.5, 1.0]
    # mpre = [0.0, 1.0, 0.5, 0.0]
    # Precision envelope: mpre -> [1.0, 1.0, 0.5, 0.0]
    # AP = (0.5 - 0.0) * 1.0 + (1.0 - 0.5) * 0.0 = 0.5
    assert abs(dog_metrics["AP"] - 0.5) < 1e-6
    assert abs(dog_metrics["precision"] - 0.5) < 1e-6
    assert abs(dog_metrics["recall"] - 0.5) < 1e-6

def test_parse_iou_thresholds() -> None:
    # None / "coco" / "default"
    assert parse_iou_thresholds(None) == COCO_IOU_THRESHOLDS
    assert parse_iou_thresholds("coco") == COCO_IOU_THRESHOLDS
    assert parse_iou_thresholds("COCO") == COCO_IOU_THRESHOLDS
    assert parse_iou_thresholds("default") == COCO_IOU_THRESHOLDS

    # Range syntax
    range_10 = parse_iou_thresholds("0.5:0.95:0.05")
    assert range_10 == COCO_IOU_THRESHOLDS
    assert parse_iou_thresholds("0.5:0.95") == COCO_IOU_THRESHOLDS
    assert parse_iou_thresholds("0.5:0.7:0.1") == [0.5, 0.6, 0.7]

    # Comma-separated list
    assert parse_iou_thresholds("0.3, 0.5") == [0.3, 0.5]
    assert parse_iou_thresholds("0.5") == [0.5]

    # List of floats
    assert parse_iou_thresholds([0.5, 0.75]) == [0.5, 0.75]

    # Error conditions
    with pytest.raises(ValueError):
        parse_iou_thresholds("")
    with pytest.raises(ValueError):
        parse_iou_thresholds("1.5")  # > 1.0
    with pytest.raises(ValueError):
        parse_iou_thresholds("-0.1") # < 0.0
    with pytest.raises(ValueError):
        parse_iou_thresholds("0.5:0.9:0.0") # step <= 0
    with pytest.raises(ValueError):
        parse_iou_thresholds("0.5:0.7:0.1:0.2") # too many colons

def test_default_coco_evaluator() -> None:
    evaluator = DetectionEvaluator()
    assert len(evaluator.iou_thresholds) == 10
    assert evaluator.iou_thresholds == COCO_IOU_THRESHOLDS
    assert 0.5 in evaluator.iou_thresholds
    assert 0.75 in evaluator.iou_thresholds

def test_coco_map_metrics() -> None:
    # Evaluator with default COCO thresholds
    evaluator = DetectionEvaluator()
    
    # 2 images with perfect boxes and slight offset boxes
    gt_0 = [{"bbox": [0.1, 0.1, 0.5, 0.5], "label": "cup"}]
    pred_0 = [{"bbox": [0.1, 0.1, 0.52, 0.52], "label": "cup", "score": 0.9}]
    evaluator.update(pred_0, gt_0)

    report = evaluator.compute_metrics()
    assert report["is_coco_standard"] is True
    assert "iou_0.50" in report["per_iou"]
    assert "iou_0.75" in report["per_iou"]
    assert "iou_0.95" in report["per_iou"]
    assert report["mAP_50"] == 1.0
    assert report["mAP_75"] == 1.0
    assert report["mAP_50_95"] > 0.0
    assert "cup" in report["per_class_summary"]
    cup_summary = report["per_class_summary"]["cup"]
    assert cup_summary["AP_50"] == 1.0
    assert cup_summary["AP_75"] == 1.0
    assert cup_summary["num_ground_truth"] == 1
    assert cup_summary["num_predictions"] == 1

def test_coco_101_interpolation() -> None:
    evaluator = DetectionEvaluator(iou_thresholds=[0.5], ap_method="coco_101")
    gt = [{"bbox": [0.0, 0.0, 0.5, 0.5], "label": "dog"}]
    pred = [{"bbox": [0.0, 0.0, 0.48, 0.48], "label": "dog", "score": 0.9}]
    evaluator.update(pred, gt)
    report = evaluator.compute_metrics()
    assert report["per_iou"]["iou_0.50"]["mAP"] == 1.0

def test_conf_threshold_filtering() -> None:
    evaluator = DetectionEvaluator(iou_thresholds=[0.5], conf_threshold=0.5)
    gt = [{"bbox": [0.0, 0.0, 0.5, 0.5], "label": "dog"}]
    preds = [
        {"bbox": [0.0, 0.0, 0.48, 0.48], "label": "dog", "score": 0.8}, # Kept (0.8 >= 0.5)
        {"bbox": [0.6, 0.6, 0.9, 0.9], "label": "dog", "score": 0.3}    # Filtered out (0.3 < 0.5)
    ]
    evaluator.update(preds, gt)
    assert len(evaluator.all_predictions) == 1
    assert evaluator.all_predictions[0]["score"] == 0.8

def test_evaluator_config_validation() -> None:
    from evaluator.config import EvaluatorConfig
    # Valid config
    cfg = EvaluatorConfig(
        model_name="test-model",
        dataset_path="test-dataset",
        conf_threshold=0.25,
        api_base="http://localhost:8000/v1",
        api_key="secret"
    )
    cfg.validate()
    
    # Invalid conf_threshold > 1.0
    with pytest.raises(ValueError):
        EvaluatorConfig(model_name="test-model", dataset_path="test-dataset", conf_threshold=1.5).validate()
        
    # Invalid conf_threshold < 0.0
    with pytest.raises(ValueError):
        EvaluatorConfig(model_name="test-model", dataset_path="test-dataset", conf_threshold=-0.1).validate()

    # Invalid ap_method
    with pytest.raises(ValueError):
        EvaluatorConfig(model_name="test-model", dataset_path="test-dataset", ap_method="invalid_method").validate()

    # Valid ap_methods
    EvaluatorConfig(model_name="test-model", dataset_path="test-dataset", ap_method="all_points").validate()
    EvaluatorConfig(model_name="test-model", dataset_path="test-dataset", ap_method="coco_101").validate()

