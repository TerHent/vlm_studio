import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from evaluator.config import EvaluatorConfig
from evaluator.dataset import DatasetLoader
from evaluator.metrics import DetectionEvaluator, parse_iou_thresholds
from evaluator.models import get_model_adapter
from evaluator.visualizer import create_side_by_side, save_visualization
from evaluator.predictions import save_prediction_cache, load_prediction_cache

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM Object Detection Evaluator CLI - Compare model predictions to ground truth labels.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples of usage:

1. End-to-end Inference & Evaluation (Default Mode with standard COCO mAP@[.50:.95]):
   python3 main.py --model zai-org/glm-4.6v-flash --dataset datasets/Barista_workflow_small --split all --visualize

2. Run Model Inference Only (Saves prediction cache to predictions/ folder):
   python3 main.py --model zai-org/glm-4.6v-flash --dataset datasets/Barista_workflow_small --split all --mode predict

3. Offline Evaluation & Visualization (Instant execution using cached predictions):
   python3 main.py --model zai-org/glm-4.6v-flash --dataset datasets/Barista_workflow_small --split all --mode evaluate --visualize
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
    model_group.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional directory containing images (useful if images are stored separately from COCO annotation JSON)"
    )

    # API & Endpoint Parameters
    api_group = parser.add_argument_group("API & Endpoint Parameters (for LM Studio / vLLM / OpenAI endpoints)")
    api_group.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible VLM server (e.g., http://localhost:1234/v1 or http://localhost:8000/v1)"
    )
    api_group.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for authenticated endpoints (or set OPENAI_API_KEY / LM_STUDIO_API_KEY env vars)"
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
        "--conf-threshold",
        type=float,
        default=0.0,
        help="Minimum confidence score threshold to retain predictions for evaluation (default: 0.0)"
    )
    eval_group.add_argument(
        "--iou-thresholds", 
        type=str, 
        default="coco",
        help=(
            "IoU thresholds for evaluation (default: 'coco'):\n"
            "  'coco'           : Standard 10-step COCO thresholds [0.50:0.95:0.05]\n"
            "  '0.5:0.95:0.05'  : Range syntax (start:stop:step)\n"
            "  '0.5' or '0.3,0.5': Explicit threshold value(s)"
        )
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
    if config.images_dir:
        print(f"  Images Dir:      {config.images_dir}")
    print(f"  Prediction Cache: {prediction_cache_file}")
    print("----------------------------------------------------------------")

    samples: List[Dict[str, Any]] = []

    # 1. INFERENCE / PREDICT PHASE (Required for 'predict' and 'all' modes)
    if config.mode in ["predict", "all"]:
        print("\nLoading dataset for inference...")
        loader = DatasetLoader(
            dataset_path=config.dataset_path, 
            split=config.dataset_split,
            images_dir=config.images_dir
        )
        dataset_length = len(loader)
        format_info = f"COCO JSON ({loader.coco_json_path})" if loader.is_coco else "Hugging Face"
        print(f"Successfully loaded {format_info} dataset with {dataset_length} samples.")
        
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
            classes=classes,
            api_base=config.api_base,
            api_key=config.api_key
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

            # In 'all' mode, render and save visualization directly while image is already loaded in RAM
            if config.mode == "all" and config.visualize:
                mapped_predictions = []
                for p in predictions:
                    lbl = p["label"]
                    mapped_lbl = config.label_map.get(lbl, lbl)
                    mapped_predictions.append({
                        "bbox": p["bbox"],
                        "label": mapped_lbl,
                        "score": p.get("score", 1.0)
                    })
                comp_img = create_side_by_side(image, ground_truths, mapped_predictions)
                save_visualization(comp_img, config.visualize_dir, file_name)

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
        iou_thresholds=config.iou_thresholds,
        conf_threshold=config.conf_threshold
    )

    # Optional dataset image loading for offline evaluate mode visualization
    image_lookup_map: Dict[str, Any] = {}
    if config.visualize and config.mode == "evaluate":
        try:
            print("Loading dataset images for offline visualization rendering...")
            dataset_loader_for_vis = DatasetLoader(
                dataset_path=config.dataset_path, 
                split=config.dataset_split,
                images_dir=config.images_dir
            )
            for img, _, f_name in dataset_loader_for_vis:
                image_lookup_map[f_name] = img
        except Exception as e:
            print(f"[Warning] Could not load dataset images for offline visualization: {e}", file=sys.stderr)

    print("\nMatching detections and calculating metrics...")
    for sample in tqdm(samples):
        file_name = sample["file_name"]
        ground_truths = sample["ground_truths"]
        predictions = sample["predictions"]
        
        evaluator.update(predictions=predictions, ground_truths=ground_truths)

        # In evaluate mode, look up original image and render side-by-side comparison
        if config.visualize and config.mode == "evaluate":
            img = image_lookup_map.get(file_name)
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
        "images_dir": config.images_dir,
        "label_map": config.label_map,
        "conf_threshold": config.conf_threshold,
        "iou_thresholds": config.iou_thresholds,
        "max_samples": config.max_samples,
        "api_base": config.api_base,
        "prediction_cache_file": prediction_cache_file
    }
    
    return report

def print_summary(report: Dict[str, Any]) -> None:
    """Renders formatted evaluation summary metrics to stdout."""
    print("\n" + "=" * 76)
    print(" EVALUATION SUMMARY REPORT (COCO Object Detection Benchmark)")
    print("=" * 76)
    print(f"Total Evaluated Images:    {report['image_count']}")
    print(f"Total Ground Truth Bboxes: {report['total_ground_truths']}")
    print(f"Total Model Predictions:   {report['total_predictions']}")
    print("-" * 76)
    
    is_coco = report.get("is_coco_standard", False)
    if is_coco:
        print(f"mAP@[.50:.95] (COCO AP):   {report['mAP_50_95']:.4f}")
    else:
        print(f"Mean mAP (Evaluated IoUs): {report.get('mAP_mean', report.get('mAP_50_95', 0.0)):.4f}")
        
    if "iou_0.50" in report["per_iou"]:
        print(f"mAP@.50       (AP50):      {report['mAP_50']:.4f}")
        print(f"Mean Precision@.50:        {report['per_iou']['iou_0.50']['mean_precision']:.4f}")
        print(f"Mean Recall@.50:           {report['per_iou']['iou_0.50']['mean_recall']:.4f}")
    if "iou_0.75" in report["per_iou"]:
        print(f"mAP@.75       (AP75):      {report['mAP_75']:.4f}")
        print(f"Mean Precision@.75:        {report['per_iou']['iou_0.75']['mean_precision']:.4f}")
        print(f"Mean Recall@.75:           {report['per_iou']['iou_0.75']['mean_recall']:.4f}")
        
    print("\n" + "-" * 76)
    col_ap = "mAP@[.50:.95]" if is_coco else "Mean AP"
    print(f"{'Category':<20} {col_ap:>13} {'AP@.50':>8} {'AP@.75':>8} {'Prec@.50':>9} {'Rec@.50':>8} {'GT':>5} {'Pred':>6}")
    print("-" * 76)
    
    summary_data = report.get("per_class_summary")
    if summary_data:
        for cls, vals in summary_data.items():
            print(
                f"{cls:<20} "
                f"{vals['AP_50_95']:>13.4f} "
                f"{vals['AP_50']:>8.4f} "
                f"{vals['AP_75']:>8.4f} "
                f"{vals['precision_50']:>9.4f} "
                f"{vals['recall_50']:>8.4f} "
                f"{vals['num_ground_truth']:>5} "
                f"{vals['num_predictions']:>6}"
            )
    elif "iou_0.50" in report["per_iou"]:
        class_metrics = report["per_iou"]["iou_0.50"]["class_metrics"]
        for cls, vals in class_metrics.items():
            print(f"{cls:<20} {'N/A':>13} {vals['AP']:>8.4f} {'N/A':>8} {vals['precision']:>9.4f} {vals['recall']:>8.4f} {vals['num_ground_truth']:>5} {vals['num_predictions']:>6}")
    print("=" * 76)

def main() -> None:
    args = parse_args()
    
    label_map = load_label_map(args.label_map)
    
    try:
        iou_thresholds = parse_iou_thresholds(args.iou_thresholds)
    except Exception as e:
        print(f"Error parsing --iou-thresholds '{args.iou_thresholds}': {e}", file=sys.stderr)
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
        predictions_dir=args.predictions_dir,
        conf_threshold=args.conf_threshold,
        api_base=args.api_base,
        api_key=args.api_key,
        images_dir=args.images_dir
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
