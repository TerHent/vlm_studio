import numpy as np
from evaluator.metrics import compute_iou, DetectionEvaluator

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
