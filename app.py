import os
import sys
import json
import glob
import random
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

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
from evaluator.models.lmstudio import fetch_lmstudio_models

# Page configuration
st.set_page_config(
    page_title="VLM Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# Authentic Linear Design System CSS (Clean, Cohesive, Minimalist)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* 1. Global Page Layout & Typography */
    .stApp {
        background-color: #0b0c10 !important;
        background-image: radial-gradient(900px 320px at 50% -40px, rgba(94, 106, 210, 0.09), transparent) !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #e6e8ec !important;
        letter-spacing: -0.011em;
    }

    /* Container max-width for desktop balance */
    .block-container {
        max-width: 1260px !important;
        padding-top: 1.25rem !important;
        padding-bottom: 3.5rem !important;
    }

    /* Thin, unobtrusive scrollbars */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.12); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.2); }

    /* 2. Sleek Studio Header */
    .studio-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.85rem 1.25rem;
        background: #12141c;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        margin-bottom: 1.25rem;
    }
    .studio-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .studio-mark {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #5e6ad2 0%, #434eb0 100%);
        border-radius: 6px;
        color: #ffffff;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(94, 106, 210, 0.35);
    }
    .studio-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f7f8f9;
        letter-spacing: -0.02em;
    }
    .studio-tag {
        font-size: 0.72rem;
        font-weight: 500;
        color: #8b8f98;
        background: rgba(255, 255, 255, 0.04);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .studio-meta-group {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .studio-badge {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.78rem;
        color: #8b8f98;
        background: #161822;
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 0.3rem 0.7rem;
        border-radius: 5px;
    }
    .status-dot-active {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
    }
    .status-dot-idle {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #8b8f98;
    }

    /* 3. Linear Stat / Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.85rem;
        margin-bottom: 1.25rem;
    }
    .metric-card {
        background: #12141c;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        transition: border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: rgba(255, 255, 255, 0.14);
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8b8f98;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #f7f8f9;
        line-height: 1.15;
        letter-spacing: -0.03em;
    }
    .metric-footer {
        font-size: 0.74rem;
        color: #636873;
        margin-top: 0.4rem;
    }

    /* 4. Section Headings */
    .section-title {
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #8b8f98;
        margin: 1.1rem 0 0.55rem 0;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    /* 5. Clean Streamlit Widget Refinements */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        gap: 0.5rem;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.86rem;
        font-weight: 500;
        padding: 0.6rem 1.1rem;
        color: #8b8f98 !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #f7f8f9 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #5e6ad2 !important;
    }

    /* Primary Action Buttons */
    div[data-testid="stFormSubmitButton"] > button,
    div.stButton > button[kind="primary"] {
        background: linear-gradient(180deg, #5e6ad2 0%, #4e5ac6 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border-radius: 6px !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.12s ease !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(180deg, #6c79e0 0%, #5965d0 100%) !important;
        box-shadow: 0 2px 8px rgba(94, 106, 210, 0.35) !important;
    }

    /* Standard Buttons */
    div.stButton > button {
        background: #141620 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 6px !important;
        color: #e2e4e9 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 0.85rem !important;
        transition: all 0.12s ease;
    }
    div.stButton > button:hover {
        background: #1a1d2a !important;
        border-color: rgba(255, 255, 255, 0.14) !important;
        color: #ffffff !important;
    }

    /* Card Panels */
    .panel-card {
        background: #12141c;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 1.15rem;
        margin-bottom: 0.85rem;
    }

    /* Code blocks */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #161822 !important;
        color: #c7d2fe !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 4px !important;
        font-size: 0.84em !important;
    }

    /* Image Viewport Frame */
    div[data-testid="stImage"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        overflow: hidden;
        background: #090a0d;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Helper Functions & Discovery Engines
# ==============================================================================
def get_available_reports() -> List[str]:
    """Finds all report JSON files in results directory."""
    results_dir = os.path.join(repo_root, "results")
    if not os.path.exists(results_dir):
        return []
    return sorted(glob.glob(os.path.join(results_dir, "*.json")), key=os.path.getmtime, reverse=True)


def get_available_caches() -> List[str]:
    """Finds all prediction cache JSON files in predictions directory."""
    preds_dir = os.path.join(repo_root, "predictions")
    if not os.path.exists(preds_dir):
        return []
    return sorted(glob.glob(os.path.join(preds_dir, "*.json")), key=os.path.getmtime, reverse=True)


def get_available_datasets() -> List[str]:
    """Finds all dataset directories or annotation JSON files in datasets directory."""
    datasets_dir = os.path.join(repo_root, "datasets")
    if not os.path.exists(datasets_dir):
        return []
    datasets = []
    for item in sorted(os.listdir(datasets_dir)):
        p = os.path.join(datasets_dir, item)
        if os.path.isdir(p):
            datasets.append(os.path.join("datasets", item))
        elif item.endswith(".json"):
            datasets.append(os.path.join("datasets", item))
    return datasets


@st.cache_resource(show_spinner=False)
def load_dataset_images(dataset_path: str, split: str = "all", images_dir: Optional[str] = None) -> Dict[str, Image.Image]:
    """Loads and caches images from dataset loader for visual inspection."""
    image_dict = {}
    try:
        loader = DatasetLoader(dataset_path=dataset_path, split=split, images_dir=images_dir)
        for img, _, file_name in loader:
            image_dict[file_name] = img
    except Exception:
        pass
    return image_dict


def render_metrics_table(df: pd.DataFrame) -> None:
    """Renders a clean, high-precision dark dataframe with native progress bars and zero white boxes."""
    if df.empty:
        st.caption("No records to display.")
        return

    col_config: Dict[str, Any] = {}
    for col in df.columns:
        if "mAP" in col or col in ("AP@.50", "AP@.75", "Precision@.50", "Recall@.50", "Confidence"):
            col_config[col] = st.column_config.ProgressColumn(
                col,
                format="%.4f",
                min_value=0.0,
                max_value=1.0,
            )
        elif pd.api.types.is_float_dtype(df[col]):
            col_config[col] = st.column_config.NumberColumn(
                col,
                format="%.4f",
            )
        elif pd.api.types.is_integer_dtype(df[col]):
            col_config[col] = st.column_config.NumberColumn(
                col,
                format="%d",
            )

    st.dataframe(
        df,
        column_config=col_config,
        hide_index=True,
        use_container_width=True,
    )


# ==============================================================================
# Top Navigation & Command Bar (Linear Workspace Header)
# ==============================================================================
if "api_base" not in st.session_state:
    st.session_state["api_base"] = (
        os.environ.get("LM_STUDIO_API_BASE") or 
        os.environ.get("OPENAI_API_BASE") or 
        "http://localhost:1234/v1"
    )

available_reports = get_available_reports()
available_caches = get_available_caches()
available_datasets = get_available_datasets()
lm_system_info = fetch_lmstudio_models(st.session_state["api_base"])
active_lm_model = lm_system_info.get("active_model")
loaded_lm_models = lm_system_info.get("loaded_models", [])

is_lm_online = lm_system_info.get("is_connected", False)
if is_lm_online:
    if active_lm_model:
        lm_pill_html = f'<span class="status-dot-active"></span><span>LM Studio: <strong style="color: #34d399;">{active_lm_model}</strong></span>'
    else:
        lm_pill_html = '<span class="status-dot-idle"></span><span>LM Studio: <strong style="color: #818cf8;">Connected (Idle)</strong></span>'
else:
    lm_pill_html = '<span class="status-dot-idle"></span><span>LM Studio: <span style="color: #8b8f98;">Offline</span></span>'

st.markdown(f"""
<div class="studio-header">
    <div class="studio-brand">
        <div class="studio-mark">◈</div>
        <div class="studio-title">VLM Studio</div>
        <div class="studio-tag">Detection Benchmark</div>
    </div>
    <div class="studio-meta-group">
        <div class="studio-badge">{lm_pill_html}</div>
        <div class="studio-badge">Datasets: <strong style="color: #f7f8f9;">{len(available_datasets)}</strong></div>
        <div class="studio-badge">Runs: <strong style="color: #f7f8f9;">{len(available_reports)}</strong></div>
        <div class="studio-badge">Caches: <strong style="color: #f7f8f9;">{len(available_caches)}</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Studio Tabs
tab_leaderboard, tab_inspector, tab_runner, tab_docs = st.tabs([
    "Leaderboard & Metrics",
    "Visual Inspector",
    "Run Evaluation",
    "Architecture"
])

# ==============================================================================
# TAB 1: Benchmark Leaderboard & Reports
# ==============================================================================
with tab_leaderboard:
    if not available_reports:
        st.info("No evaluation reports found in results/. Execute an evaluation via the **Run Evaluation** tab or CLI.")
    else:
        # Leaderboard Table with Search Filter
        sec_col, filter_col = st.columns([2, 1])
        with sec_col:
            st.markdown("<div class='section-title'>◈ Model Benchmark Run(s)</div>", unsafe_allow_html=True)
        with filter_col:
            search_query = st.text_input(
                "Filter Benchmark Runs",
                value="",
                placeholder="🔍 Filter runs by model or dataset...",
                label_visibility="collapsed"
            )

        leaderboard_rows = []
        filtered_reports = []
        for rep_path in available_reports:
            try:
                with open(rep_path, "r") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                m_name = meta.get("model_name", os.path.basename(rep_path))
                d_name = os.path.basename(meta.get("dataset_path", "N/A"))
                if search_query.strip():
                    q = search_query.strip().lower()
                    if q not in m_name.lower() and q not in d_name.lower():
                        continue
                filtered_reports.append(rep_path)
                leaderboard_rows.append({
                    "Model": m_name,
                    "Dataset": d_name,
                    "Split": meta.get("dataset_split", "all"),
                    "mAP [0.50-0.95]": round(float(data.get("mAP_50_95", 0.0)), 4),
                    "AP@.50": round(float(data.get("mAP_50", 0.0)), 4),
                    "AP@.75": round(float(data.get("mAP_75", 0.0)), 4),
                    "Samples": data.get("image_count", 0),
                    "GT Boxes": data.get("total_ground_truths", 0),
                    "Preds": data.get("total_predictions", 0),
                })
            except Exception:
                pass

        if leaderboard_rows:
            df_leaderboard = pd.DataFrame(leaderboard_rows).sort_values("mAP [0.50-0.95]", ascending=False)
            render_metrics_table(df_leaderboard)
        elif search_query.strip():
            st.caption(f"No benchmark runs match filter '{search_query}'.")

        # Active Report Detailed View
        st.markdown("<div class='section-title'>◈ Detailed Evaluation Report</div>", unsafe_allow_html=True)
        active_report_choices = filtered_reports if filtered_reports else available_reports
        selected_report_path = st.selectbox(
            "Select Report for Detailed Inspection:",
            options=active_report_choices,
            format_func=lambda p: os.path.basename(p),
            label_visibility="collapsed"
        )

        if selected_report_path and os.path.exists(selected_report_path):
            with open(selected_report_path, "r") as f:
                report = json.load(f)

            meta = report.get("metadata", {})
            map_coco = float(report.get("mAP_50_95", 0.0))
            map_50 = float(report.get("mAP_50", 0.0))
            map_75 = float(report.get("mAP_75", 0.0))
            img_count = report.get("image_count", 0)

            # Linear KPI Metric Cards (No tacky rainbow stripes)
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">
                        <span>COCO Primary AP</span>
                        <span style="color: #818cf8;">mAP@[.50:.95]</span>
                    </div>
                    <div class="metric-value">{map_coco:.4f}</div>
                    <div class="metric-footer">10-step ladder mean</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">
                        <span>PASCAL VOC</span>
                        <span style="color: #34d399;">AP@.50</span>
                    </div>
                    <div class="metric-value">{map_50:.4f}</div>
                    <div class="metric-footer">Precision@50: {report.get('per_iou', {}).get('iou_0.50', {}).get('mean_precision', 0.0):.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">
                        <span>Strict Loc</span>
                        <span style="color: #fbbf24;">AP@.75</span>
                    </div>
                    <div class="metric-value">{map_75:.4f}</div>
                    <div class="metric-footer">Precision@75: {report.get('per_iou', {}).get('iou_0.75', {}).get('mean_precision', 0.0):.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">
                        <span>Evaluation Set</span>
                        <span>{img_count} Images</span>
                    </div>
                    <div class="metric-value">{report.get('total_ground_truths', 0)}</div>
                    <div class="metric-footer">{report.get('total_predictions', 0)} Predictions Total</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Clean Plotly Visualizations
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                st.markdown("<div class='section-title'>◈ IoU Threshold Degradation Curve</div>", unsafe_allow_html=True)
                per_iou = report.get("per_iou", {})
                if per_iou:
                    ious, maps, precs, recs = [], [], [], []
                    for k, v in sorted(per_iou.items(), key=lambda x: float(x[0].replace("iou_", ""))):
                        ious.append(float(k.replace("iou_", "")))
                        maps.append(round(v.get("mAP", 0.0), 4))
                        precs.append(round(v.get("mean_precision", 0.0), 4))
                        recs.append(round(v.get("mean_recall", 0.0), 4))

                    fig_iou = go.Figure()
                    fig_iou.add_trace(go.Scatter(
                        x=ious, y=maps, mode='lines+markers', name='Mean AP',
                        line=dict(color='#818cf8', width=2),
                        marker=dict(size=5, color='#5e6ad2')
                    ))
                    fig_iou.add_trace(go.Scatter(
                        x=ious, y=precs, mode='lines', name='Precision',
                        line=dict(color='#10b981', width=1.5, dash='dash')
                    ))
                    fig_iou.add_trace(go.Scatter(
                        x=ious, y=recs, mode='lines', name='Recall',
                        line=dict(color='#f59e0b', width=1.5, dash='dot')
                    ))
                    fig_iou.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(18, 20, 28, 0.6)',
                        margin=dict(l=30, r=20, t=15, b=30),
                        height=240,
                        font=dict(family="Inter", size=11, color="#8a8f98"),
                        xaxis=dict(
                            title=dict(text="IoU Match Threshold", font=dict(size=10)),
                            gridcolor="rgba(255,255,255,0.04)",
                            zerolinecolor="rgba(255,255,255,0.08)"
                        ),
                        yaxis=dict(
                            range=[0, 1.05],
                            gridcolor="rgba(255,255,255,0.04)",
                            zerolinecolor="rgba(255,255,255,0.08)"
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(size=10)
                        )
                    )
                    st.plotly_chart(fig_iou, width="stretch", config={"displayModeBar": False})

            with c_chart2:
                st.markdown("<div class='section-title'>◈ Category Performance (COCO mAP)</div>", unsafe_allow_html=True)
                summary_data = report.get("per_class_summary", {})
                if summary_data:
                    cat_names, cat_maps = [], []
                    for k, v in summary_data.items():
                        cat_names.append(k)
                        cat_maps.append(round(float(v.get("AP_50_95", 0.0)), 4))

                    df_plot_cat = pd.DataFrame({"Category": cat_names, "mAP": cat_maps}).sort_values("mAP", ascending=True)

                    fig_cat = go.Figure(go.Bar(
                        x=df_plot_cat["mAP"],
                        y=df_plot_cat["Category"],
                        orientation='h',
                        marker=dict(
                            color='#5e6ad2',
                            opacity=0.88,
                            line=dict(color='#818cf8', width=1)
                        )
                    ))
                    fig_cat.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(18, 20, 28, 0.6)',
                        margin=dict(l=30, r=20, t=15, b=30),
                        height=240,
                        font=dict(family="Inter", size=11, color="#8a8f98"),
                        xaxis=dict(
                            range=[0, 1.0],
                            gridcolor="rgba(255,255,255,0.04)",
                            zerolinecolor="rgba(255,255,255,0.08)"
                        ),
                        yaxis=dict(
                            gridcolor="rgba(255,255,255,0.04)"
                        )
                    )
                    st.plotly_chart(fig_cat, width="stretch", config={"displayModeBar": False})

            # Detailed Per-Category Breakdown Table
            st.markdown("<div class='section-title'>◈ Per-Category Metric Breakdown</div>", unsafe_allow_html=True)
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
                render_metrics_table(df_cats)

            # Metadata Table
            with st.expander("Runtime Configuration & Metadata", expanded=False):
                if meta:
                    meta_df = pd.DataFrame([
                        {"Parameter": str(k), "Value": str(v)}
                        for k, v in meta.items()
                    ])
                    render_metrics_table(meta_df)


# ==============================================================================
# TAB 2: Interactive Visual Inspector
# ==============================================================================
with tab_inspector:
    if not available_caches:
        st.warning("No prediction caches found in predictions/. Run an evaluation first to populate predictions.")
    else:
        # Top Controls Row
        c_cache, c_layout = st.columns([2, 2])
        with c_cache:
            selected_cache = st.selectbox(
                "Prediction Cache:",
                options=available_caches,
                format_func=lambda p: os.path.basename(p)
            )
        with c_layout:
            view_mode = st.radio(
                "Viewport Layout:",
                ["Side-by-Side (Comparison)", "Ground Truth Only", "Predictions Only"],
                horizontal=True
            )

        if selected_cache:
            samples, cache_meta = load_prediction_cache(selected_cache)
            available_ds = get_available_datasets()
            fallback_ds = available_ds[0] if available_ds else ""
            dataset_path = cache_meta.get("dataset_path")
            if not dataset_path:
                dataset_path = fallback_ds
            elif not os.path.exists(dataset_path) and not os.path.exists(os.path.join(repo_root, dataset_path)):
                local_in_ds = os.path.join(repo_root, "datasets", os.path.basename(dataset_path))
                if os.path.exists(local_in_ds):
                    dataset_path = local_in_ds
                # Otherwise keep dataset_path as-is so DatasetLoader can load from Hugging Face Hub if applicable
            dataset_split = cache_meta.get("dataset_split", "all")

            # Session State for Indexing
            if "current_sample_idx" not in st.session_state:
                st.session_state.current_sample_idx = 0

            max_idx = max(0, len(samples) - 1)
            st.session_state.current_sample_idx = min(st.session_state.current_sample_idx, max_idx)

            # Sample Navigation Row
            s_prev, s_status, s_next, s_rand = st.columns([1, 4, 1, 1])
            with s_prev:
                if st.button("◀ Prev", width="stretch"):
                    st.session_state.current_sample_idx = max(0, st.session_state.current_sample_idx - 1)
                    st.rerun()
            with s_status:
                cur_sample = samples[st.session_state.current_sample_idx]
                f_name = cur_sample["file_name"]
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: center; height: 38px; gap: 0.6rem; background: #12141c; border: 1px solid rgba(255,255,255,0.06); border-radius: 6px;">
                    <span style="font-weight: 600; font-size: 0.84rem; color: #f7f8f9;">Sample #{st.session_state.current_sample_idx + 1} of {len(samples)}</span>
                    <span style="color: rgba(255,255,255,0.15);">|</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #818cf8;">{f_name}</span>
                </div>
                """, unsafe_allow_html=True)
            with s_next:
                if st.button("Next ▶", width="stretch"):
                    st.session_state.current_sample_idx = min(max_idx, st.session_state.current_sample_idx + 1)
                    st.rerun()
            with s_rand:
                if st.button("🎲 Random", width="stretch"):
                    st.session_state.current_sample_idx = random.randint(0, max_idx)
                    st.rerun()

            if max_idx > 0:
                slider_val = st.slider(
                    "Sample Index Navigation",
                    min_value=0,
                    max_value=max_idx,
                    value=st.session_state.current_sample_idx,
                    format="#%d",
                    label_visibility="collapsed"
                )
                if slider_val != st.session_state.current_sample_idx:
                    st.session_state.current_sample_idx = slider_val
                    st.rerun()

            # Dynamic Filter Row
            f_col_conf, f_col_class = st.columns([1, 2])
            with f_col_conf:
                conf_slider = st.slider(
                    "Confidence Filter",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    help="Suppresses predictions with confidence below this score."
                )

            # Extract distinct classes
            all_classes_in_cache = set()
            for s in samples:
                for gt in s.get("ground_truths", []):
                    all_classes_in_cache.add(gt["label"])
                for p in s.get("predictions", []):
                    all_classes_in_cache.add(p["label"])
            sorted_classes = sorted(list(all_classes_in_cache))

            with f_col_class:
                selected_classes = st.multiselect(
                    "Active Classes",
                    options=sorted_classes,
                    default=sorted_classes,
                    help="Filter displayed bounding boxes to specific object classes."
                )

            # Active Sample State
            cur_idx = st.session_state.current_sample_idx
            sample = samples[cur_idx]
            raw_gts = sample.get("ground_truths", [])
            raw_preds = sample.get("predictions", [])

            active_gts = [g for g in raw_gts if g["label"] in selected_classes]
            active_preds = [
                p for p in raw_preds
                if p.get("score", 1.0) >= conf_slider and p["label"] in selected_classes
            ]

            # Image Rendering Viewport
            image_cache_dict = load_dataset_images(dataset_path=dataset_path, split=dataset_split)
            img = image_cache_dict.get(f_name) or image_cache_dict.get(os.path.basename(f_name))

            if img is not None:
                if view_mode == "Side-by-Side (Comparison)":
                    render_img = create_side_by_side(img, active_gts, active_preds)
                elif view_mode == "Ground Truth Only":
                    render_img = draw_boxes(img, active_gts, is_prediction=False)
                else:
                    render_img = draw_boxes(img, active_preds, is_prediction=True)

                st.image(render_img, width="stretch")
            else:
                st.error(f"Image '{f_name}' could not be located in dataset '{dataset_path}'.")

            # Clean Detection Breakdown Tables (No clunky HTML list boxes)
            c_gt_col, c_pred_col = st.columns(2)
            with c_gt_col:
                st.markdown(f"<div class='section-title'>● Ground Truth Objects ({len(active_gts)})</div>", unsafe_allow_html=True)
                if active_gts:
                    gt_rows = [{
                        "Class": g["label"],
                        "Bounding Box [xmin, ymin, xmax, ymax]": f"[{b[1]:.3f}, {b[0]:.3f}, {b[3]:.3f}, {b[2]:.3f}]"
                    } for g in active_gts for b in [g["bbox"]]]
                    st.dataframe(pd.DataFrame(gt_rows), hide_index=True, use_container_width=True)
                else:
                    st.caption("No ground truth boxes matching filters.")

            with c_pred_col:
                st.markdown(f"<div class='section-title'>● Model Predictions ({len(active_preds)})</div>", unsafe_allow_html=True)
                if active_preds:
                    pred_rows = [{
                        "Class": p["label"],
                        "Confidence": round(float(p.get("score", 1.0)), 4),
                        "Bounding Box [xmin, ymin, xmax, ymax]": f"[{b[1]:.3f}, {b[0]:.3f}, {b[3]:.3f}, {b[2]:.3f}]"
                    } for p in active_preds for b in [p["bbox"]]]
                    st.dataframe(
                        pd.DataFrame(pred_rows),
                        column_config={
                            "Confidence": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0.0, max_value=1.0)
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.caption("No predictions matching filters.")


# ==============================================================================
# TAB 3: Run Model Inference & Evaluation
# ==============================================================================
with tab_runner:
    # 1. Sample Scope Quick Selector
    st.markdown("<div class='section-title'>◈ Evaluation Sample Scope</div>", unsafe_allow_html=True)
    if "eval_sample_count" not in st.session_state:
        st.session_state["eval_sample_count"] = 5

    sq1, sq2, sq3 = st.columns(3)
    with sq1:
        if st.button("⚡ Quick Test (3 Samples)", width="stretch"):
            st.session_state["eval_sample_count"] = 3
            st.rerun()
    with sq2:
        if st.button("📊 Standard Run (10 Samples)", width="stretch"):
            st.session_state["eval_sample_count"] = 10
            st.rerun()
    with sq3:
        if st.button("🎯 Full Evaluation (All Samples)", width="stretch"):
            st.session_state["eval_sample_count"] = 0
            st.rerun()

    # 2. Endpoint query and refresh controls
    col_endp, col_ref = st.columns([4, 1])
    with col_endp:
        api_base_val = st.text_input(
            "OpenAI-Compatible Endpoint URL",
            value=st.session_state.get("api_base", "http://localhost:1234/v1"),
            help="LM Studio, vLLM, or local server endpoint"
        )
        if api_base_val != st.session_state.get("api_base"):
            st.session_state["api_base"] = api_base_val
            st.rerun()
    with col_ref:
        st.write("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Models", width="stretch"):
            st.rerun()

    lm_info = fetch_lmstudio_models(st.session_state["api_base"])

    # 3. Dynamic Model Selection
    if lm_info["is_connected"]:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem; font-size: 0.78rem;">
            <span class="status-dot-active"></span>
            <span style="color: #34d399; font-weight: 500;">Connected to LM Studio:</span>
            <span style="color: #8b8f98;">{len(lm_info['loaded_models'])} resident in VRAM • {len(lm_info['vlm_models'])} vision models</span>
        </div>
        """, unsafe_allow_html=True)

        display_options = []
        model_value_map = {}

        # 1. Models loaded in VRAM
        for m in lm_info.get("loaded_models", []):
            label = f"🟢 {m} (Resident in VRAM)"
            display_options.append(label)
            model_value_map[label] = m

        # 2. Available VLMs in LM Studio
        for m in lm_info.get("vlm_models", []):
            if m not in lm_info.get("loaded_models", []):
                label = f"🟣 {m} (VLM in LM Studio)"
                display_options.append(label)
                model_value_map[label] = m

        # 3. Text LLMs in LM Studio
        for m in lm_info.get("all_models", []):
            if m not in lm_info.get("vlm_models", []) and m not in lm_info.get("loaded_models", []):
                label = f"⚪ {m} (Text LLM)"
                display_options.append(label)
                model_value_map[label] = m

        # 4. Custom entry
        custom_label = "✏️ Custom Model / Local Path..."
        display_options.append(custom_label)
        model_value_map[custom_label] = "__custom__"

        selected_label = st.selectbox(
            "Select Model (Dynamically queried from LM Studio)",
            options=display_options,
            index=0,
            help="Select any model currently available in LM Studio, or choose custom to type another ID."
        )

        if model_value_map[selected_label] == "__custom__":
            model_name = st.text_input(
                "Enter Model ID or Local Path", 
                value="", 
                placeholder="e.g. Qwen/Qwen2.5-VL-7B-Instruct or Hugging Face ID"
            )
        else:
            model_name = model_value_map[selected_label]
    else:
        st.markdown(f"""
        <div style="font-size: 0.78rem; color: #f59e0b; margin-bottom: 0.5rem;">
            ○ LM Studio offline at <code>{st.session_state['api_base']}</code>. Enter model identifier manually:
        </div>
        """, unsafe_allow_html=True)
        model_name = st.text_input(
            "Model Identifier", 
            value="", 
            placeholder="e.g. Qwen/Qwen2.5-VL-7B-Instruct or local model path"
        )

    # 4. Dynamic Dataset Discovery & Selection
    available_datasets = get_available_datasets()
    if available_datasets:
        ds_choices = list(available_datasets) + ["✏️ Custom Path / Hugging Face ID..."]
        chosen_ds = st.selectbox(
            "Select Benchmark Dataset",
            options=ds_choices,
            index=0,
            help="Discovered datasets from local datasets/ folder or custom entry"
        )
        if chosen_ds == "✏️ Custom Path / Hugging Face ID...":
            dataset_path = st.text_input(
                "Custom Dataset Path",
                value="",
                placeholder="e.g. /path/to/dataset or COCO json path",
                help="Local dataset folder, COCO JSON annotation path, or Hugging Face hub identifier"
            )
        else:
            dataset_path = chosen_ds
    else:
        dataset_path = st.text_input(
            "Dataset Path",
            value="",
            placeholder="e.g. datasets/my_dataset or COCO json path",
            help="Local dataset folder, COCO JSON annotation path, or Hugging Face hub identifier"
        )

    with st.form("evaluation_form"):
        col_m, col_d = st.columns(2)
        with col_m:
            st.markdown("<div class='section-title'>Execution Environment</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="panel-card">
                <div style="font-size: 0.7rem; color: #8b8f98; text-transform: uppercase; font-weight: 600;">Active Model Target</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #f7f8f9; font-family: 'JetBrains Mono', monospace; margin-top: 0.15rem;">{model_name if model_name else '(None specified)'}</div>
            </div>
            """, unsafe_allow_html=True)
            device = st.selectbox("Execution Device", ["auto", "cuda", "cpu", "mps"])
            api_key = st.text_input(
                "API Key (Optional)",
                value="",
                type="password",
                help="Optional API key for authenticated endpoints"
            )
            api_base = st.session_state["api_base"]

        with col_d:
            st.markdown("<div class='section-title'>Dataset Configuration</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="panel-card">
                <div style="font-size: 0.7rem; color: #8b8f98; text-transform: uppercase; font-weight: 600;">Active Dataset Target</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #f7f8f9; font-family: 'JetBrains Mono', monospace; margin-top: 0.15rem;">{dataset_path if dataset_path else '(None specified)'}</div>
            </div>
            """, unsafe_allow_html=True)
            dataset_split = st.text_input("Dataset Split", value="all")
            images_dir = st.text_input(
                "Images Directory (Optional)",
                value="",
                help="Optional directory containing images if stored separately from COCO annotation JSON"
            )
            exec_mode = st.selectbox(
                "Execution Mode",
                ["evaluate", "all", "predict"],
                help="'evaluate' runs offline using cached predictions in <1 sec. 'all' runs inference + evaluation."
            )

        st.markdown("<div class='section-title'>Evaluation Parameters</div>", unsafe_allow_html=True)
        col_opts1, col_opts2, col_opts3 = st.columns(3)
        with col_opts1:
            max_samples_input = st.number_input(
                "Max Samples (0 for all)", 
                min_value=0, 
                max_value=50000, 
                value=st.session_state.get("eval_sample_count", 5)
            )
        with col_opts2:
            conf_threshold_input = st.slider("Confidence Filter", 0.0, 1.0, 0.0, step=0.05)
        with col_opts3:
            enable_vis = st.checkbox("Export Side-by-Side Images to Disk", value=True)

        col_adv1, col_adv2, col_adv3 = st.columns([2, 1, 1])
        with col_adv1:
            label_map_input = st.text_input(
                "Label Aliases (JSON)",
                value="{}",
                help="Optional JSON dict mapping model outputs to ground truth taxonomy, e.g. {'Milk Container': 'Milk Pitcher'}"
            )
        with col_adv2:
            iou_thresholds_input = st.text_input(
                "IoU Thresholds",
                value="coco",
                help="Thresholds: 'coco' (0.50:0.95:0.05), '0.5', '0.3,0.5', or '0.5:0.95:0.05'"
            )
        with col_adv3:
            ap_method_input = st.selectbox(
                "AP Calculation Method",
                options=["all_points", "coco_101"],
                help="'all_points' (VOC style continuous AUC) or 'coco_101' (101-point COCO ladder)"
            )

        submitted = st.form_submit_button("Start Evaluation Run ↵", width="stretch")

    if submitted:
        if not model_name.strip():
            st.error("Please specify a Model Identifier.")
            st.stop()
        if not dataset_path.strip():
            st.error("Please specify a Dataset Path.")
            st.stop()

        parsed_label_map = {}
        if label_map_input.strip() and label_map_input.strip() != "{}":
            try:
                parsed_label_map = json.loads(label_map_input)
                if not isinstance(parsed_label_map, dict):
                    st.error("Label Map must be a valid JSON dictionary.")
                    st.stop()
            except json.JSONDecodeError as err:
                st.error(f"Invalid JSON in Label Map: {err}")
                st.stop()

        try:
            parsed_iou_thresholds = parse_iou_thresholds(iou_thresholds_input)
        except Exception as err:
            st.error(f"Invalid IoU Thresholds: {err}")
            st.stop()

        max_samples = int(max_samples_input) if max_samples_input > 0 else None

        dataset_folder = os.path.basename(dataset_path.rstrip("/"))
        model_folder_name = model_name.replace("/", "_").replace(" ", "_")
        output_report_path = os.path.join("results", f"{dataset_folder}_{model_folder_name}.json")
        visualize_dir = os.path.join("visualizations", f"{dataset_folder}_{model_folder_name}")

        config = EvaluatorConfig(
            model_name=model_name,
            dataset_path=dataset_path,
            dataset_split=dataset_split,
            label_map=parsed_label_map,
            iou_thresholds=parsed_iou_thresholds,
            output_report_path=output_report_path,
            device=device,
            max_samples=max_samples,
            visualize=enable_vis,
            visualize_dir=visualize_dir,
            mode=exec_mode,
            conf_threshold=conf_threshold_input,
            ap_method=ap_method_input,
            api_base=api_base if api_base.strip() else None,
            api_key=api_key if api_key.strip() else None,
            images_dir=images_dir.strip() if images_dir.strip() else None
        )

        try:
            config.validate()
        except ValueError as err:
            st.error(f"Configuration Validation Error: {err}")
            st.stop()

        with st.spinner(f"Running '{exec_mode.upper()}' evaluation for '{model_name}'..."):
            try:
                report = run_evaluation(config)
                if report is not None:
                    st.success("Evaluation completed successfully.")

                    # Linear KPI Cards
                    map_val = float(report.get("mAP_50_95", 0.0))
                    map_50_val = float(report.get("mAP_50", 0.0))
                    map_75_val = float(report.get("mAP_75", 0.0))
                    samples_val = report.get("image_count", 0)

                    st.markdown(f"""
                    <div class="metric-grid" style="margin-top: 1rem;">
                        <div class="metric-card">
                            <div class="metric-label"><span>COCO AP</span><span style="color: #818cf8;">mAP@[.50:.95]</span></div>
                            <div class="metric-value">{map_val:.4f}</div>
                            <div class="metric-footer">Primary Benchmark Metric</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label"><span>PASCAL VOC</span><span style="color: #34d399;">AP@.50</span></div>
                            <div class="metric-value">{map_50_val:.4f}</div>
                            <div class="metric-footer">IoU 0.50 Threshold</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label"><span>Strict Loc</span><span style="color: #fbbf24;">AP@.75</span></div>
                            <div class="metric-value">{map_75_val:.4f}</div>
                            <div class="metric-footer">IoU 0.75 Threshold</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label"><span>Evaluation Set</span><span>Images</span></div>
                            <div class="metric-value">{samples_val}</div>
                            <div class="metric-footer">{report.get('total_predictions', 0)} Predictions Total</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Save report
                    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
                    with open(output_report_path, "w") as f:
                        json.dump(report, f, indent=4)
                    st.info(f"Report written to `{output_report_path}`")
                else:
                    st.success("Predict mode complete. Predictions saved to `predictions/`.")
            except Exception as e:
                st.error(f"Execution failed: {e}")


# ==============================================================================
# TAB 4: Architecture & Design Decisions
# ==============================================================================
with tab_docs:
    st.markdown("""
    ### System Architecture & Pipeline Design

    **VLM Studio** is an open-source evaluation framework for Vision-Language Models on object detection tasks.

    #### 1. Decoupled 3-Mode Pipeline
    - `predict`: Connects to local PyTorch models or OpenAI-compatible VLM endpoints (e.g. LM Studio, vLLM) and dumps normalized bounding box detections to `predictions/*.json`.
    - `evaluate`: Offline evaluation that loads cached predictions and computes exact COCO metric matrices against ground truth in **< 1 second without GPU**.
    - `all`: Executes end-to-end inference, metric calculation, and side-by-side visual exports.

    #### 2. COCO Benchmark Standard
    - Strict 10-step ladder `[0.50 : 0.05 : 0.95]` with continuous 101-point PR curve interpolation.
    - Computes primary **mAP@[.50:.95]**, **PASCAL VOC AP@.50**, and **Strict AP@.75**.

    #### 3. Defensive Output Parsing
    - Auto-recovers from VLM syntax hallucinations: double closing brackets (`]]`), unescaped quotes, trailing commas, markdown fences, and relaxed key casing.
    - Support for reasoning/thinking VLMs (such as Google Gemma 4 or DeepSeek R1) that place bounding box coordinates in reasoning tokens.

    #### 4. Universal Dataset Ingestion
    - Automatic detection of local COCO annotations (`annotations.json`, `instances_default.json`).
    - Split discovery (`all`, `train`, `val`, `test`) with optional decoupled `images_dir`.
    - Direct support for Hugging Face datasets with sequence bounding boxes or text-based coordinates.
    """)
