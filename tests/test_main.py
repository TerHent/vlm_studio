import os
import json
import tempfile
import pytest
from unittest.mock import patch

from evaluator.config import EvaluatorConfig
from evaluator.predictions import save_prediction_cache
from main import parse_args, load_label_map, run_evaluation, print_summary

def test_load_label_map_string() -> None:
    # Empty string
    assert load_label_map("") == {}
    
    # Valid JSON string
    mapping = load_label_map('{"Milk Container": "Milk Pitcher", "Espresso": "Cup"}')
    assert mapping == {"Milk Container": "Milk Pitcher", "Espresso": "Cup"}
    
    # Invalid JSON string raises SystemExit
    with pytest.raises(SystemExit):
        load_label_map("not-valid-json{")

def test_load_label_map_file(tmp_path) -> None:
    map_file = tmp_path / "label_map.json"
    map_file.write_text(json.dumps({"cat": "feline", "dog": "canine"}))
    
    mapping = load_label_map(str(map_file))
    assert mapping == {"cat": "feline", "dog": "canine"}

def test_parse_args_defaults() -> None:
    test_argv = ["main.py", "--dataset", "datasets/test_ds"]
    with patch("sys.argv", test_argv):
        args = parse_args()
        assert args.dataset == "datasets/test_ds"
        assert args.model == "auto"
        assert args.mode == "all"
        assert args.split == "validation"
        assert args.iou_thresholds == "coco"
        assert args.ap_method == "all_points"
        assert args.conf_threshold == 0.0
        assert args.visualize is True

def test_parse_args_custom() -> None:
    test_argv = [
        "main.py", 
        "--dataset", "datasets/custom_ds",
        "--model", "test-vlm",
        "--mode", "evaluate",
        "--split", "all",
        "--iou-thresholds", "0.3,0.5",
        "--ap-method", "coco_101",
        "--conf-threshold", "0.25",
        "--no-visualize"
    ]
    with patch("sys.argv", test_argv):
        args = parse_args()
        assert args.dataset == "datasets/custom_ds"
        assert args.model == "test-vlm"
        assert args.mode == "evaluate"
        assert args.split == "all"
        assert args.iou_thresholds == "0.3,0.5"
        assert args.ap_method == "coco_101"
        assert args.conf_threshold == 0.25
        assert args.visualize is False

def test_run_evaluation_evaluate_mode(tmp_path) -> None:
    # Set up prediction cache
    preds_dir = tmp_path / "predictions"
    preds_dir.mkdir()
    
    samples = [
        {
            "file_name": "frame_01.jpg",
            "ground_truths": [{"bbox": [0.1, 0.1, 0.5, 0.5], "label": "cup"}],
            "predictions": [{"bbox": [0.1, 0.1, 0.5, 0.5], "label": "cup", "score": 0.95}]
        }
    ]
    cache_file = preds_dir / "my_ds_my_model.json"
    save_prediction_cache(str(cache_file), samples, {"model_name": "my_model", "dataset_path": "my_ds"})
    
    config = EvaluatorConfig(
        model_name="my_model",
        dataset_path="my_ds",
        mode="evaluate",
        predictions_dir=str(preds_dir),
        visualize=False,
        ap_method="all_points"
    )
    
    report = run_evaluation(config)
    assert report is not None
    assert report["image_count"] == 1
    assert report["total_ground_truths"] == 1
    assert report["total_predictions"] == 1
    assert report["mAP_50"] == 1.0
    assert report["metadata"]["model_name"] == "my_model"
    assert report["metadata"]["ap_method"] == "all_points"

def test_print_summary(capsys) -> None:
    report = {
        "image_count": 1,
        "total_ground_truths": 1,
        "total_predictions": 1,
        "is_coco_standard": True,
        "mAP_50_95": 0.85,
        "mAP_50": 1.0,
        "mAP_75": 0.9,
        "per_iou": {
            "iou_0.50": {
                "mean_precision": 1.0,
                "mean_recall": 1.0,
                "class_metrics": {
                    "cup": {"AP": 1.0, "precision": 1.0, "recall": 1.0, "num_ground_truth": 1, "num_predictions": 1}
                }
            }
        },
        "per_class_summary": {
            "cup": {
                "AP_50_95": 0.85,
                "AP_50": 1.0,
                "AP_75": 0.9,
                "precision_50": 1.0,
                "recall_50": 1.0,
                "num_ground_truth": 1,
                "num_predictions": 1
            }
        }
    }
    
    print_summary(report)
    captured = capsys.readouterr()
    assert "EVALUATION SUMMARY REPORT" in captured.out
    assert "mAP@[.50:.95]" in captured.out
    assert "cup" in captured.out
