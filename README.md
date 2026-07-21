# VLM Studio: Local Vision-Language Model Object Detection Evaluator

**VLM Studio** is a modular evaluation framework designed for benchmarking Vision-Language Models (VLMs) on object detection tasks. 

It provides an end-to-end evaluation pipeline that supports local models (via LM Studio, Florence-2, PaliGemma, Qwen, GLM-4V), structured dataset loading, offline prediction caching, static label aliasing, mathematically verified mAP metrics, and dual-panel visual comparisons.

---

## 🌟 Key Features

* **Modular 3-Mode Execution Architecture (`--mode`):**
  * `all`: Runs VLM inference $\rightarrow$ caches raw predictions $\rightarrow$ calculates metrics $\rightarrow$ renders visualizations.
  * `predict`: Runs VLM inference and saves raw predictions to `predictions/` (ideal for long/overnight runs).
  * `evaluate`: Loads cached predictions and evaluates metrics & visualizations **instantly offline (under 1 sec)** without touching GPU/LLM resources.
* **Defensive Output Parsing & Recovery:**
  * Handles JSON syntax quirks, `bbox_2d` key shifts, duplicate closing brackets (`]]`), trailing commas, single quotes, and truncated tokens.
* **Side-by-Side Dual-Panel Visualizer:**
  * Renders composite comparison JPEGs displaying **Ground Truth (Left)** vs. **Model Predictions (Right)** with stable category colors and panel headers.
* **Label Mapping & Synonym Aliasing (`--label-map`):**
  * Map model synonym outputs (e.g. `"Milk Container": "Milk Pitcher"`) to dataset taxonomy without modifying ground truth files.
* **Flexible IoU Thresholding (`--iou-thresholds`):**
  * Evaluate models at custom overlap thresholds (e.g. `0.30` for noisy/POV video datasets or `0.50` for standard benchmarks).
* **Fully Tested:** 
  * Comprehensive test suite using `pytest` verifying metrics calculations, adapter parsing, image visualizer math, and prediction cache I/O.

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/your-username/vlm_studio.git
cd vlm_studio
pip install -r requirements.txt
```

### 2. View CLI Help Documentation
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

### Offline Evaluation & Visualization (`--mode evaluate`)
Reads pre-saved predictions from `predictions/` and computes metrics & visualizations in **under 1 second**:
```bash
python3 main.py \
  --model zai-org/glm-4.6v-flash \
  --dataset datasets/Barista_workflow_small \
  --split all \
  --mode evaluate \
  --iou-thresholds "0.3,0.5" \
  --visualize
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
* `visualizations/`: Contains dual-panel comparison JPEGs (`visualizations/[dataset]_[model]/`).

---

## 📄 License
MIT License.
