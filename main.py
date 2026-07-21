import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from evaluator.config import EvaluatorConfig
from evaluator.dataset import DatasetLoader
from evaluator.metrics import DetectionEvaluator
from evaluator.models import get_model_adapter
from evaluator.visualizer import create_side_by_side, save_visualization
from evaluator.predictions import save_prediction_cache, load_prediction_cache

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM Object Detection Evaluator CLI - Compare model predictions to ground truth labels.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples of usage:

1. End-to-end Inference & Evaluation (Default Mode):
   python3 main.py --model zai-org/glm-4.6v-flash --dataset datasets/Barista_workflow_small --split all --visualize

2. Run Model Inference Only (Saves prediction cache to predictions/ folder):
   python3 main.py --model zai-org/glm-4.6v-flash --dataset datasets/Barista_workflow_small --split all --mode predict

3. Offline Evaluation & Visualization (Instant execution using cached predictions):
   python3 main.py --model zai-org/glm-4.6v-flash --dataset datasets/Barista_workflow_small --split all --mode evaluate --iou-thresholds "0.3,0.5" --visualize
"""
    )
    
    # Execution Mode & Control
    mode_group = parser.add_argument_group("Execution Mode & Control")
    mode_group.add_argument(
        "--mode",
        type=str,
        choices=["all", "predict", "evaluate"],
        default="all",
        help=(
            "Execution mode:\n"
            "  all      : Runs inference, saves predictions cache, computes metrics, and visualizes (default)\n"
            "  predict  : Runs VLM inference and saves raw predictions to predictions/ cache JSON\n"
            "  evaluate : Reads pre-saved predictions cache JSON and runs offline evaluation & visualization"
        )
    )
    mode_group.add_argument(
        "--predictions-dir",
        type=str,
        default="predictions",
        help="Directory where persistent raw prediction cache JSON files are stored (default: predictions)"
    )
    mode_group.add_argument(
        "--max-samples", 
        type=int, 
        default=None,
        help="Limit execution to first N samples (useful for quick dry runs)"
    )

    # Model & Dataset Configuration
    model_group = parser.add_argument_group("Model & Dataset Configuration")
    model_group.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Hugging Face model ID or hosted model name (e.g., zai-org/glm-4.6v-flash or microsoft/Florence-2-base)"
    )
    model_group.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Hugging Face dataset identifier or local dataset folder path"
    )
    model_group.add_argument(
        "--split", 
        type=str, 
        default="validation",
        help="Dataset split to evaluate (e.g., validation, train, all) (default: validation)"
    )
    model_group.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Execution device: auto, cuda, mps, cpu (default: auto)"
    )

    # Evaluation & Metrics Parameters
    eval_group = parser.add_argument_group("Evaluation & Metrics Parameters")
    eval_group.add_argument(
        "--label-map", 
        type=str, 
        default="{}",
        help="JSON string or path to a JSON file mapping prediction categories to ground truth classes"
    )
    eval_group.add_argument(
        "--iou-thresholds", 
        type=str, 
        default="0.5",
        help="Comma-separated list of IoU thresholds for evaluation (e.g., '0.3,0.5,0.75') (default: 0.5)"
    )
    eval_group.add_argument(
        "--output", 
        type=str, 
        default="evaluation_report.json",
        help="Path where final evaluation JSON report is saved (default: results/[dataset]_[model].json)"
    )

    # Visualization Options
    vis_group = parser.add_argument_group("Visualization Options")
    vis_group.add_argument(
        "--visualize", 
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable side-by-side ground truth vs predictions image saving (default: enabled)"
    )
    vis_group.add_argument(
        "--visualize-dir", 
        type=str, 
        default="visualizations",
        help="Directory to save comparison images (default: visualizations)"
    )
    
    return parser.parse_args()

def load_label_map(label_map_arg: str) -> Dict[str, str]:
    """Parses label map argument from raw JSON string or JSON file path."""
    if not label_map_arg:
        return {}
        
    if os.path.exists(label_map_arg):
        try:
            with open(label_map_arg, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading label map file: {e}", file=sys.stderr)
            sys.exit(1)
            
    try:
        return json.loads(label_map_arg)
    except json.JSONDecodeError as e:
        print(f"Error parsing label map JSON string: {e}", file=sys.stderr)
        print("Provide either a path to a valid JSON file or a valid inline JSON string.", file=sys.stderr)
        sys.exit(1)

def run_evaluation(config: EvaluatorConfig) -> Optional[Dict[str, Any]]:
    dataset_folder = os.path.basename(config.dataset_path.rstrip("/"))
    model_folder_name = config.model_name.replace("/", "_").replace(" ", "_")
    prediction_cache_file = os.path.join(config.predictions_dir, f"{dataset_folder}_{model_folder_name}.json")

    print("----------------------------------------------------------------")
    print(f"VLM Evaluator Execution Engine")
    print(f"  Mode:            {config.mode.upper()}")
    print(f"  Model:           {config.model_name}")
    print(f"  Dataset:         {config.dataset_path} ({config.dataset_split})")
    print(f"  Prediction Cache: {prediction_cache_file}")
    print("----------------------------------------------------------------")

    samples: List[Dict[str, Any]] = []

    # 1. INFERENCE / PREDICT PHASE (Required for 'predict' and 'all' modes)
    if config.mode in ["predict", "all"]:
        print("\nLoading dataset for inference...")
        loader = DatasetLoader(dataset_path=config.dataset_path, split=config.dataset_split)
        dataset_length = len(loader)
        print(f"Successfully loaded dataset with {dataset_length} samples.")
        
        limit = config.max_samples if config.max_samples is not None else dataset_length
        run_samples = min(limit, dataset_length)
        if config.max_samples is not None:
            print(f"Restricting execution to first {run_samples} samples.")

        classes = loader.category_names
        print(f"Detected ground truth classes: {classes}")

        print(f"Loading local model adapter on device '{config.device}'...")
        adapter = get_model_adapter(
            model_name=config.model_name, 
            device=config.device, 
            classes=classes
        )

        print("\nRunning model inference...")
        for idx, (image, ground_truths, file_name) in enumerate(tqdm(loader, total=run_samples)):
            if idx >= run_samples:
                break
                
            try:
                predictions = adapter.predict(image)
            except Exception as e:
                print(f"\n[Warning] Skipped sample index {idx} due to prediction error: {e}", file=sys.stderr)
                predictions = []

            samples.append({
                "file_name": file_name,
                "ground_truths": ground_truths,
                "predictions": predictions
            })

        # Save raw predictions to cache (overwriting existing cache)
        cache_metadata = {
            "model_name": config.model_name,
            "dataset_path": config.dataset_path,
            "dataset_split": config.dataset_split,
            "total_samples": len(samples)
        }
        save_prediction_cache(prediction_cache_file, samples, cache_metadata)
        print(f"\n[OK] Successfully saved {len(samples)} prediction records to '{prediction_cache_file}'")

        if config.mode == "predict":
            print("Predict mode complete. Exiting without calculating metrics.")
            return None

    # 2. EVALUATE-ONLY PHASE (Reads from pre-saved prediction cache)
    elif config.mode == "evaluate":
        print(f"\nLoading cached predictions from '{prediction_cache_file}'...")
        samples, cache_meta = load_prediction_cache(prediction_cache_file)
        
        if config.max_samples is not None:
            samples = samples[:config.max_samples]
            print(f"Restricting evaluation to first {len(samples)} cached samples.")
            
        print(f"Loaded {len(samples)} prediction records from cache.")

    # 3. METRICS EVALUATION & VISUALIZATION PHASE
    evaluator = DetectionEvaluator(
        label_map=config.label_map, 
        iou_thresholds=config.iou_thresholds
    )

    # Optional dataset loader for loading original images during offline evaluation visualization
    dataset_loader_for_vis: Optional[DatasetLoader] = None
    image_lookup_map: Dict[str, Any] = {}
    
    if config.visualize and config.mode == "evaluate":
        try:
            print("Loading dataset images for offline visualization rendering...")
            dataset_loader_for_vis = DatasetLoader(dataset_path=config.dataset_path, split=config.dataset_split)
            for img, _, f_name in dataset_loader_for_vis:
                image_lookup_map[f_name] = img
        except Exception as e:
            print(f"[Warning] Could not load dataset images for offline visualization: {e}", file=sys.stderr)

    print("\nMatching detections and calculating metrics...")
    
    # In 'all' mode, we have access to loaded images during the loop if loader was present
    loader_ref = DatasetLoader(dataset_path=config.dataset_path, split=config.dataset_split) if (config.visualize and config.mode == "all") else None
    loader_img_map: Dict[str, Any] = {}
    if loader_ref:
        for img, _, f_name in loader_ref:
            loader_img_map[f_name] = img

    for sample in tqdm(samples):
        file_name = sample["file_name"]
        ground_truths = sample["ground_truths"]
        predictions = sample["predictions"]
        
        evaluator.update(predictions=predictions, ground_truths=ground_truths)

        if config.visualize:
            # Look up original image
            img = image_lookup_map.get(file_name) or loader_img_map.get(file_name)
            if img:
                mapped_predictions = []
                for p in predictions:
                    lbl = p["label"]
                    mapped_lbl = config.label_map.get(lbl, lbl)
                    mapped_predictions.append({
                        "bbox": p["bbox"],
                        "label": mapped_lbl,
                        "score": p.get("score", 1.0)
                    })
                comp_img = create_side_by_side(img, ground_truths, mapped_predictions)
                save_visualization(comp_img, config.visualize_dir, file_name)

    print("\nCalculating summary metrics...")
    report = evaluator.compute_metrics()
    
    report["metadata"] = {
        "model_name": config.model_name,
        "dataset_path": config.dataset_path,
        "dataset_split": config.dataset_split,
        "label_map": config.label_map,
        "iou_thresholds": config.iou_thresholds,
        "max_samples": config.max_samples,
        "prediction_cache_file": prediction_cache_file
    }
    
    return report

def print_summary(report: Dict[str, Any]) -> None:
    """Renders formatted evaluation summary metrics to stdout."""
    print("\n" + "=" * 64)
    print(" EVALUATION SUMMARY REPORT")
    print("=" * 64)
    print(f"Total Evaluated Images:    {report['image_count']}")
    print(f"Total Ground Truth Bboxes: {report['total_ground_truths']}")
    print(f"Total Model Predictions:   {report['total_predictions']}")
    print(f"mAP@[.50:.95]:            {report['mAP_50_95']:.4f}")
    if "iou_0.50" in report["per_iou"]:
        print(f"mAP@.50:                   {report['mAP_50']:.4f}")
        print(f"Mean Precision@.50:        {report['per_iou']['iou_0.50']['mean_precision']:.4f}")
        print(f"Mean Recall@.50:           {report['per_iou']['iou_0.50']['mean_recall']:.4f}")
        
    print("\nPer-Category Average Precision (AP) at IoU=0.50:")
    if "iou_0.50" in report["per_iou"]:
        class_metrics = report["per_iou"]["iou_0.50"]["class_metrics"]
        for cls, vals in class_metrics.items():
            print(f"  - {cls:<16} AP: {vals['AP']:.4f} | Prec: {vals['precision']:.4f} | Rec: {vals['recall']:.4f} (GT: {vals['num_ground_truth']}, Pred: {vals['num_predictions']})")
    print("=" * 64)

def main() -> None:
    args = parse_args()
    
    label_map = load_label_map(args.label_map)
    
    try:
        iou_thresholds = [float(x.strip()) for x in args.iou_thresholds.split(",")]
    except ValueError:
        print("IoU thresholds must be comma-separated float values (e.g. '0.5,0.75').", file=sys.stderr)
        sys.exit(1)

    dataset_folder = os.path.basename(args.dataset.rstrip("/"))
    model_folder_name = args.model.replace("/", "_").replace(" ", "_")

    visualize_dir = args.visualize_dir
    if args.visualize:
        visualize_dir = os.path.join(args.visualize_dir, f"{dataset_folder}_{model_folder_name}")

    output_report_path = args.output
    if args.output == "evaluation_report.json":
        output_report_path = os.path.join("results", f"{dataset_folder}_{model_folder_name}.json")

    config = EvaluatorConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        dataset_split=args.split,
        label_map=label_map,
        iou_thresholds=iou_thresholds,
        output_report_path=output_report_path,
        device=args.device,
        max_samples=args.max_samples,
        visualize=args.visualize,
        visualize_dir=visualize_dir,
        mode=args.mode,
        predictions_dir=args.predictions_dir
    )
    
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
        
    try:
        report = run_evaluation(config)
    except Exception as e:
        print(f"\nExecution failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    if report is not None:
        print_summary(report)
        
        try:
            output_dir = os.path.dirname(config.output_report_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(config.output_report_path, "w") as f:
                json.dump(report, f, indent=4)
            print(f"\nSaved detailed evaluation report JSON to: '{config.output_report_path}'")
        except Exception as e:
            print(f"Error saving JSON report: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
