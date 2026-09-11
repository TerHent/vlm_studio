---
title: VLM Studio
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.59.0
app_file: app.py
pinned: false
license: mit
---

# VLM Studio: Local Vision-Language Model Object Detection Evaluator

[![Tests](https://github.com/TerHent/vlm_studio/actions/workflows/test.yml/badge.svg)](https://github.com/TerHent/vlm_studio/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**VLM Studio** is a modular evaluation framework designed for benchmarking Vision-Language Models (VLMs) on object detection tasks. 

It provides an end-to-end evaluation pipeline that supports local models (via LM Studio, Florence-2, PaliGemma, Qwen, GLM-4V), structured dataset loading, offline prediction caching, static label aliasing, mathematically verified mAP metrics, and side-by-side ground truth vs prediction image exports.

---

## 🌟 Key Features

* **Modular 3-Mode Execution Architecture (`--mode`):**
  * `all`: Runs VLM inference $\rightarrow$ caches raw predictions $\rightarrow$ calculates metrics $\rightarrow$ saves comparison images to disk.
  * `predict`: Runs VLM inference and saves raw predictions to `predictions/` (ideal for long/overnight runs).
  * `evaluate`: Loads cached predictions and evaluates metrics & saves comparison images **instantly offline (under 1 sec)** without touching GPU/LLM resources.
* **Defensive Output Parsing & Recovery:**
  * Handles JSON syntax quirks, `bbox_2d` key shifts, duplicate closing brackets (`]]`), trailing commas, single quotes, and truncated tokens.
* **Flexible Endpoints & Authentication (`--api-base`, `--api-key`):**
  * Connect directly to LM Studio, vLLM, Ollama, or remote OpenAI-compatible endpoints with custom URLs and authorization headers.
* **Side-by-Side Comparison Image Export (`--visualize`):**
  * Headless and non-interactive: generates and saves composite image files directly to the `visualizations/` folder on disk (no GUI or on-screen display).
  * Draws **Ground Truth (Left Panel)** vs. **Model Predictions (Right Panel)** with stable category colors, adaptive resolution-scaled fonts, and top-edge clipping protection for offline review in an external viewer.
* **Label Mapping & Confidence Filtering (`--label-map`, `--conf-threshold`):**
  * Map model synonym outputs to dataset taxonomy and filter out low-confidence predictions.
* **Standard COCO Benchmark Evaluation by Default (`--iou-thresholds`):**
  * Evaluates models by default across standard 10-step COCO IoU thresholds (`[0.50:0.95:0.05]`), reporting primary **COCO mAP@[.50:.95]**, **mAP@.50 (AP50)**, **mAP@.75 (AP75)**, and per-category AP breakdowns.
  * Supports `'coco'` (default), range syntax (e.g., `'0.5:0.95:0.05'`), or custom thresholds (e.g. `'0.3,0.5'`).
* **Flexible Dataset Loading (COCO JSON & Hugging Face):**
  * Supports standard COCO JSON format files (`instances_*.json`, `_annotations.coco.json`) and standard COCO directory layouts with automatic split detection.
  * Optional `--images-dir` to specify image location if stored separately from annotations.
  * Supports Hugging Face datasets (local and Hub) with both standard sequence columns and text-based response coordinates.
* **Fully Tested:** 
  * Comprehensive test suite using `pytest` verifying metrics calculations, adapter parsing, COCO and Hugging Face dataset loading, image visualizer math, and prediction cache I/O.

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/your-username/vlm_studio.git
cd vlm_studio
pip install -r requirements.txt
```

### 2. Launch Interactive Web Dashboard
Run the web dashboard locally in your browser:
```bash
streamlit run app.py
```

### 3. View CLI Help Documentation
```bash
python3 main.py --help
```

---

## 💡 Usage Examples

### End-to-End Evaluation (`--mode all`)
Runs model inference, computes metrics, and generates side-by-side visualizations:
```bash
python3 main.py \
  --model zai-org/glm-4.6v-flash \
  --dataset datasets/Barista_workflow_small \
  --split all \
  --label-map '{"Milk Container": "Milk Pitcher", "Espresso Machine": "Group Head"}' \
  --iou-thresholds "0.3,0.5" \
  --mode all
```

### Model Inference Only (`--mode predict`)
Runs inference on LM Studio and saves raw predictions to `predictions/`:
```bash
python3 main.py \
  --model zai-org/glm-4.6v-flash \
  --dataset datasets/Barista_workflow_small \
  --split all \
  --mode predict
```

### Offline Evaluation & Image Export (`--mode evaluate`)
Reads pre-saved predictions from `predictions/`, computes metrics, and exports comparison images in **under 1 second**:
```bash
python3 main.py \
  --model zai-org/glm-4.6v-flash \
  --dataset datasets/Barista_workflow_small \
  --split all \
  --mode evaluate \
  --iou-thresholds "0.3,0.5" \
  --visualize
```

### Evaluating on Standard COCO Datasets
Supports direct JSON file paths or COCO directory hierarchies with optional separate images directory:
```bash
python3 main.py \
  --model zai-org/glm-4.6v-flash \
  --dataset path/to/annotations/instances_val2017.json \
  --images-dir path/to/val2017 \
  --mode all
```

---

## 🧪 Running Unit Tests

To run the offline test suite:
```bash
pytest
```

---

## 📁 Output Directory Structure

* `predictions/`: Contains raw JSON prediction caches (`predictions/[dataset]_[model].json`).
* `results/`: Contains detailed summary metrics JSON reports (`results/[dataset]_[model].json`).
* `visualizations/`: Contains saved side-by-side comparison image files (`visualizations/[dataset]_[model]/`).

---

## 📄 License
MIT License.
