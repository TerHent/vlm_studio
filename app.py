import os
import sys
import json
import glob
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
from PIL import Image

# Ensure repository root is in python path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from evaluator.config import EvaluatorConfig
from evaluator.dataset import DatasetLoader
from evaluator.metrics import DetectionEvaluator, COCO_IOU_THRESHOLDS, parse_iou_thresholds
from evaluator.predictions import load_prediction_cache
from evaluator.visualizer import create_side_by_side, draw_boxes
from main import run_evaluation

# Page configuration
st.set_page_config(
    page_title="VLM Studio - Object Detection Evaluator",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_available_reports() -> List[str]:
    """Finds all report JSON files in results directory."""
    results_dir = os.path.join(repo_root, "results")
    if not os.path.exists(results_dir):
        return []
    return sorted(glob.glob(os.path.join(results_dir, "*.json")))

def get_available_caches() -> List[str]:
    """Finds all prediction cache JSON files in predictions directory."""
    preds_dir = os.path.join(repo_root, "predictions")
    if not os.path.exists(preds_dir):
        return []
    return sorted(glob.glob(os.path.join(preds_dir, "*.json")))

@st.cache_resource(show_spinner=False)
def load_dataset_images(dataset_path: str, split: str = "all", images_dir: Optional[str] = None) -> Dict[str, Image.Image]:
    """Loads and caches images from dataset loader for instant visual inspection."""
    image_dict = {}
    try:
        loader = DatasetLoader(dataset_path=dataset_path, split=split, images_dir=images_dir)
        for img, _, file_name in loader:
            image_dict[file_name] = img
    except Exception as e:
        st.warning(f"Could not load images from dataset '{dataset_path}': {e}")
    return image_dict

# App Header
st.markdown('<div class="main-header">👁️ VLM Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Benchmark and inspect Vision-Language Models (VLMs) on Object Detection tasks with standard COCO metrics.</div>',
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Navigation & Configuration")

sidebar_mode = st.sidebar.radio(
    "Select Workspace:",
    [
        "📊 Evaluation Reports & Metrics",
        "🔍 Interactive Visual Inspector",
        "🚀 Run New Evaluation",
        "ℹ️ Documentation & Architecture"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Quick Info")
st.sidebar.markdown("""
- **Primary Metric:** Standard COCO `mAP@[.50:.95]`
- **IoU Steps:** 10 thresholds `[0.50 : 0.95 : 0.05]`
- **Continuous Curve:** 101-point COCO interpolation
- **Decoupled Architecture:** Offline evaluation in < 1s
""")

# ==============================================================================
# TAB 1: Evaluation Reports & Metrics
# ==============================================================================
if sidebar_mode == "📊 Evaluation Reports & Metrics":
    st.subheader("Evaluation Reports & Leaderboard")
    
    available_reports = get_available_reports()
    
    if not available_reports:
        st.info("No evaluation reports found in `results/`. Run an evaluation from the **🚀 Run New Evaluation** tab or via CLI.")
    else:
        # Multi-model Leaderboard Comparison Table
        if len(available_reports) > 1:
            st.markdown("#### 🏆 Benchmark Leaderboard")
            leaderboard_rows = []
            for rep_path in available_reports:
                try:
                    with open(rep_path, "r") as f:
                        data = json.load(f)
                    meta = data.get("metadata", {})
                    leaderboard_rows.append({
                        "Model": meta.get("model_name", os.path.basename(rep_path)),
                        "Dataset": meta.get("dataset_path", "N/A"),
                        "Split": meta.get("dataset_split", "all"),
                        "Images": data.get("image_count", 0),
                        "mAP [0.50-0.95]": f"{data.get('mAP_50_95', 0.0):.4f}",
                        "mAP@.50 (AP50)": f"{data.get('mAP_50', 0.0):.4f}",
                        "mAP@.75 (AP75)": f"{data.get('mAP_75', 0.0):.4f}",
                        "GT Boxes": data.get("total_ground_truths", 0),
                        "Preds": data.get("total_predictions", 0),
                    })
                except Exception:
                    pass
            if leaderboard_rows:
                df_leaderboard = pd.DataFrame(leaderboard_rows)
                st.dataframe(df_leaderboard, use_container_width=True, hide_index=True)
                st.markdown("---")

        # Report selector
        selected_report_path = st.selectbox(
            "Select Evaluation Report to Inspect:",
            options=available_reports,
            format_func=lambda p: os.path.basename(p)
        )
        
        if selected_report_path and os.path.exists(selected_report_path):
            with open(selected_report_path, "r") as f:
                report = json.load(f)
                
            meta = report.get("metadata", {})
            
            # Top metrics cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("COCO mAP@[.50:.95]", f"{report.get('mAP_50_95', 0.0):.4f}")
            with col2:
                st.metric("mAP@.50 (AP50)", f"{report.get('mAP_50', 0.0):.4f}")
            with col3:
                st.metric("mAP@.75 (AP75)", f"{report.get('mAP_75', 0.0):.4f}")
            with col4:
                st.metric("Evaluated Images", report.get("image_count", 0))

            # Additional run info
            with st.expander("Metadata & Configuration Details", expanded=False):
                st.json(meta)

            st.markdown("### 📈 Detection Performance Decay (IoU Curve)")
            per_iou = report.get("per_iou", {})
            if per_iou:
                iou_records = []
                for k, v in per_iou.items():
                    val_iou = float(k.replace("iou_", ""))
                    iou_records.append({
                        "IoU Threshold": val_iou,
                        "Mean AP": v.get("mAP", 0.0),
                        "Mean Precision": v.get("mean_precision", 0.0),
                        "Mean Recall": v.get("mean_recall", 0.0),
                    })
                df_iou = pd.DataFrame(iou_records).sort_values("IoU Threshold")
                st.line_chart(df_iou.set_index("IoU Threshold"))

            st.markdown("### 🏷️ Per-Category Metrics Breakdown")
            summary_data = report.get("per_class_summary", {})
            if summary_data:
                rows = []
                for cls_name, vals in summary_data.items():
                    rows.append({
                        "Category": cls_name,
                        "mAP [0.50-0.95]": round(vals.get("AP_50_95", 0.0), 4),
                        "AP@.50": round(vals.get("AP_50", 0.0), 4),
                        "AP@.75": round(vals.get("AP_75", 0.0), 4),
                        "Precision@.50": round(vals.get("precision_50", 0.0), 4),
                        "Recall@.50": round(vals.get("recall_50", 0.0), 4),
                        "Ground Truth": vals.get("num_ground_truth", 0),
                        "Predictions": vals.get("num_predictions", 0),
                    })
                df_cats = pd.DataFrame(rows).sort_values("mAP [0.50-0.95]", ascending=False)
                
                # Render chart and table
                col_chart, col_table = st.columns([1, 1])
                with col_chart:
                    st.markdown("**mAP [0.50-0.95] by Category:**")
                    st.bar_chart(df_cats.set_index("Category")["mAP [0.50-0.95]"])
                with col_table:
                    st.markdown("**Detailed Class Table:**")
                    st.dataframe(df_cats, use_container_width=True, hide_index=True)
            else:
                st.info("No per-category breakdown available in this report.")

# ==============================================================================
# TAB 2: Interactive Visual Inspector
# ==============================================================================
elif sidebar_mode == "🔍 Interactive Visual Inspector":
    st.subheader("Interactive Visual Inspector")
    st.markdown("Inspect side-by-side **Ground Truth (Left)** vs **Model Predictions (Right)** with dynamic confidence threshold filtering.")

    available_caches = get_available_caches()
    if not available_caches:
        st.warning("No prediction cache found in `predictions/`. Run an inference or evaluation first to generate cached predictions.")
    else:
        selected_cache = st.selectbox(
            "Select Prediction Cache:",
            options=available_caches,
            format_func=lambda p: os.path.basename(p)
        )

        if selected_cache:
            samples, cache_meta = load_prediction_cache(selected_cache)
            dataset_path = cache_meta.get("dataset_path", "datasets/Barista_workflow_small")
            dataset_split = cache_meta.get("dataset_split", "all")
            
            # Interactive Controls
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                sample_idx = st.slider(
                    f"Select Sample (Total: {len(samples)}):",
                    min_value=0,
                    max_value=max(0, len(samples) - 1),
                    value=0,
                    format="Sample #%d"
                )
            with c2:
                conf_slider = st.slider(
                    "Min Confidence Score:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )
            with c3:
                show_gt = st.checkbox("Show Ground Truth", value=True)
                show_pred = st.checkbox("Show Predictions", value=True)

            if samples and sample_idx < len(samples):
                current_sample = samples[sample_idx]
                f_name = current_sample["file_name"]
                all_gts = current_sample.get("ground_truths", [])
                all_preds = current_sample.get("predictions", [])

                # Filter predictions by confidence threshold
                filtered_preds = [p for p in all_preds if p.get("score", 1.0) >= conf_slider] if show_pred else []
                filtered_gts = all_gts if show_gt else []

                # Load original image
                image_cache_dict = load_dataset_images(dataset_path=dataset_path, split=dataset_split)
                img = image_cache_dict.get(f_name)

                # Fallback if image filename was stored with a relative path
                if img is None:
                    img = image_cache_dict.get(os.path.basename(f_name))

                if img is not None:
                    comp_img = create_side_by_side(img, filtered_gts, filtered_preds)
                    st.image(
                        comp_img, 
                        caption=f"File: {f_name} | Ground Truth Boxes: {len(filtered_gts)} | Active Predictions (Score >= {conf_slider:.2f}): {len(filtered_preds)}",
                        use_container_width=True
                    )
                else:
                    st.error(f"Image '{f_name}' could not be loaded from dataset path '{dataset_path}'.")

                # Show tabular detections for this image
                exp1, exp2 = st.columns(2)
                with exp1:
                    with st.expander(f"Ground Truth Detections ({len(filtered_gts)})", expanded=False):
                        if filtered_gts:
                            st.dataframe(pd.DataFrame(filtered_gts), use_container_width=True)
                        else:
                            st.write("No ground truth boxes.")
                with exp2:
                    with st.expander(f"Model Predictions ({len(filtered_preds)})", expanded=False):
                        if filtered_preds:
                            st.dataframe(pd.DataFrame(filtered_preds), use_container_width=True)
                        else:
                            st.write("No predictions above threshold.")

# ==============================================================================
# TAB 3: Run New Evaluation
# ==============================================================================
elif sidebar_mode == "🚀 Run New Evaluation":
    st.subheader("Run VLM Object Detection Evaluation")
    st.markdown("Execute end-to-end inference and evaluation, or run instant offline evaluation from saved predictions.")

    with st.form("evaluation_form"):
        col_m, col_d = st.columns(2)
        with col_m:
            model_name = st.text_input(
                "Model Identifier:",
                value="zai-org/glm-4.6v-flash",
                help="Model name or Hugging Face ID (e.g. zai-org/glm-4.6v-flash, microsoft/Florence-2-base, or gpt-4o-mini)"
            )
            device = st.selectbox("Execution Device:", ["auto", "cuda", "cpu", "mps"])
            api_base = st.text_input(
                "API Base URL (for LM Studio / vLLM / OpenAI):",
                value="http://localhost:1234/v1",
                help="Base endpoint URL for OpenAI-compatible VLM server"
            )
            api_key = st.text_input(
                "API Key (Optional):",
                value="",
                type="password",
                help="Optional API key for authenticated endpoints"
            )

        with col_d:
            dataset_path = st.text_input(
                "Dataset Path:",
                value="datasets/Barista_workflow_small",
                help="Local dataset folder, COCO JSON annotation path, or Hugging Face hub identifier"
            )
            dataset_split = st.text_input("Dataset Split:", value="all")
            images_dir = st.text_input(
                "Images Directory (Optional for COCO JSON):",
                value="",
                help="Optional directory containing images if stored separately from COCO annotation JSON"
            )
            exec_mode = st.selectbox(
                "Execution Mode:",
                ["evaluate", "all", "predict"],
                help="'evaluate' runs instantly offline using cached predictions. 'all' runs inference + evaluation."
            )

        col_opts1, col_opts2, col_opts3 = st.columns(3)
        with col_opts1:
            max_samples_input = st.number_input("Max Samples (0 for all):", min_value=0, max_value=5000, value=5)
        with col_opts2:
            conf_threshold_input = st.slider("Confidence Filter Threshold:", 0.0, 1.0, 0.0, step=0.05)
        with col_opts3:
            enable_vis = st.checkbox("Export Comparison Images", value=True)

        submitted = st.form_submit_button("⚡ Run Evaluation")

    if submitted:
        max_samples = int(max_samples_input) if max_samples_input > 0 else None
        
        dataset_folder = os.path.basename(dataset_path.rstrip("/"))
        model_folder_name = model_name.replace("/", "_").replace(" ", "_")
        output_report_path = os.path.join("results", f"{dataset_folder}_{model_folder_name}.json")
        visualize_dir = os.path.join("visualizations", f"{dataset_folder}_{model_folder_name}")

        config = EvaluatorConfig(
            model_name=model_name,
            dataset_path=dataset_path,
            dataset_split=dataset_split,
            output_report_path=output_report_path,
            device=device,
            max_samples=max_samples,
            visualize=enable_vis,
            visualize_dir=visualize_dir,
            mode=exec_mode,
            conf_threshold=conf_threshold_input,
            api_base=api_base if api_base.strip() else None,
            api_key=api_key if api_key.strip() else None,
            images_dir=images_dir.strip() if images_dir.strip() else None
        )

        try:
            config.validate()
        except ValueError as err:
            st.error(f"Configuration Validation Error: {err}")
            st.stop()

        with st.spinner(f"Executing '{exec_mode.upper()}' evaluation on '{model_name}'..."):
            try:
                report = run_evaluation(config)
                if report is not None:
                    st.success("Evaluation completed successfully!")
                    
                    # Display quick summary cards
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("COCO mAP@[.50:.95]", f"{report.get('mAP_50_95', 0.0):.4f}")
                    c2.metric("mAP@.50", f"{report.get('mAP_50', 0.0):.4f}")
                    c3.metric("mAP@.75", f"{report.get('mAP_75', 0.0):.4f}")
                    c4.metric("Samples", report.get("image_count", 0))

                    # Save report
                    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
                    with open(output_report_path, "w") as f:
                        json.dump(report, f, indent=4)
                    st.info(f"Report saved to: `{output_report_path}`")
                else:
                    st.success(f"Predict mode completed. Raw predictions cached in `{config.predictions_dir}/`.")
            except Exception as e:
                st.error(f"Execution failed: {e}")

# ==============================================================================
# TAB 4: Documentation & Architecture
# ==============================================================================
elif sidebar_mode == "ℹ️ Documentation & Architecture":
    st.subheader("About VLM Studio")
    st.markdown("""
    **VLM Studio** is an open-source evaluation framework for benchmarking Vision-Language Models on object detection.
    
    ### Key Architectural Highlights:
    1. **Decoupled 3-Mode Pipeline:**
       - `predict`: Runs VLM inference and saves persistent prediction cache JSON files.
       - `evaluate`: Evaluates pre-saved predictions offline in **< 1 second** without GPU resources.
       - `all`: Runs end-to-end inference, prediction caching, metrics calculation, and image export.
    2. **Standard COCO Benchmark Compliance:**
       - Uses official 10-step IoU thresholds `[0.50 : 0.95 : 0.05]`
       - Calculates continuous 101-point COCO interpolation for area under the Precision-Recall curve.
    3. **Dataset Agnostic:**
       - Local and Hub Hugging Face datasets (custom response formats or standard sequence columns).
       - Standard COCO format JSON (`instances_*.json`) with automatic split resolution and optional `--images-dir`.
    4. **Headless Visual Comparison Export:**
       - Saves dual-panel composite image files to `visualizations/` with stable per-class color hashing, top-edge clipping prevention, and adaptive fonts.
    """)

