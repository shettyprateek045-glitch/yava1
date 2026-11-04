# Advanced Version — Audit & Product Build-Out

This document records the engineering audit performed on top of the four-week
Computer Vision internship shape classification pipeline, and details every production feature
and enhancement built to transform it into an enterprise-grade Computer Vision platform.

---

## 1. What the Audit Found

Going into this pass, the repository contained the core deliverables across the four internship weeks:
- **Week 1:** Flowchart diagrams and design report.
- **Week 2:** A hand-crafted NumPy CNN (`ShapeCNN`), synthetic dataset generator, training script, and checkpoint.
- **Week 3:** Magnitude weight pruning (40%) and post-training int8 symmetric quantization scripts.
- **Week 4:** A basic FastAPI file (`week4_integration/api.py`) with `/predict` and `/health`, and a test client.

However, several critical gaps remained for a modern, enterprise-ready machine learning product:
1. **No User-Facing Interactive Surface:** There was no graphical user interface. Testing the model required running python scripts or sending curl commands with image files on disk.
2. **Minimal Serving Layer:** The existing API lacked CORS headers, couldn't accept base64 image strings directly from HTML canvas elements, and didn't log predictions.
3. **No Computer Vision Explainability (XAI):** There was no way to visualize *why* the CNN classified a shape, verify that it wasn't overfitting to background noise, or view intermediate convolutional layer activations.
4. **No Live Multi-Model Comparison:** Although Week 3 benchmarked pruning and quantization on a static test set, there was no runtime capability to evaluate a single image against FP32, Pruned, and Quantized variants simultaneously to inspect latency and confidence differences.
5. **No Robustness & Perturbation Analysis:** No automated framework existed to measure how sensitive the trained model was to additive sensor noise or geometric rotation variations.
6. **No Containerization or CI/CD:** The repository lacked a `Dockerfile`, `docker-compose.yml`, and GitHub Actions workflow for automated testing and deployment.
7. **No Standard PyTest Test Suite:** Only a single procedural test script was present in Week 4; unit tests for dataset generation, model layers, and optimization bounds were missing.

---

## 2. What Was Added

### 2.1 Interactive Web Studio (`static/index.html`)
A responsive, dark-mode single-page application built with Tailwind CSS, Lucide icons, and Chart.js:
- **Interactive HTML5 Drawing Canvas:** Draw circles, squares, triangles, or stars with adjustable brush thickness, clear, and invert capabilities.
- **Procedural Shape Presets:** 1-click generation of randomized synthetic shapes (with noise and rotation) to immediately test the model.
- **Drag-and-Drop Image Uploader:** Upload any external PNG or JPG file for inference.
- **Real-Time Classification Card:** Displays predicted shape class, confidence percentage, execution latency in milliseconds, and animated softmax probability distribution bars across all 4 classes.

### 2.2 Production Product API (`api.py`)
A root-level FastAPI application superseding `week4_integration/api.py` (which remains intact as the Week 4 assignment deliverable):
- **CORS Middleware:** Configured for cross-origin frontend requests.
- **Dual Input Modes:** Accepts both `multipart/form-data` file uploads and base64 JSON payloads.
- **Multi-Variant Model Manager:** Lazily initializes and caches FP32 Baseline, 40% Pruned, and Int8 Quantized models in memory.
- **SQLite Audit Trail (`data/predictions.db`):** Automatically logs timestamp, predicted class, confidence, model variant, latency, and source for every request.
- **REST Endpoints:**
  - `GET /`: Serves the Interactive Studio console.
  - `GET /health`: Service health and model variant statuses.
  - `POST /api/predict`: Runs inference using selected model variant.
  - `POST /api/explain`: Computes Saliency Heatmap and intermediate Conv feature maps.
  - `POST /api/compare`: Executes all 3 models on the same input and reports latency, size, and confidence side-by-side.
  - `POST /api/robustness`: Evaluates performance under varying noise and rotation.
  - `GET /api/sample/{shape}`: Returns a synthetic test shape.
  - `GET /api/benchmark`: Returns official Week 3 performance optimization benchmarks.
  - `GET /api/history` & `DELETE /api/history`: Manages audit records.
  - `GET /api/stats`: Real-time aggregate statistics.

### 2.3 Computer Vision Explainability (XAI)
- **Gradient Pixel Saliency Map:** Backpropagates class score gradient to input pixels (\(\nabla_X y_c\)) using zero learning rate and momentum. Renders a normalized Jet/Turbo pseudo-color heatmap showing exact contour vertices.
- **Conv1 & Conv2 Feature Map Visualizer:** Captures activation tensors after non-linearities for all 8 channels of Conv1 and 16 channels of Conv2, displaying them in a grid.

### 2.4 Multi-Model Comparison & Robustness Testing
- **Side-by-Side Scoring:** Benchmarks FP32 (262.7 KB), 40% Pruned (67.2k params), and Int8 Quantized (65.9 KB) side-by-side, computing storage savings (-74.9%) and speedup.
- **Perturbation Sweeps:** Tests predictions across 7 noise standard deviations (\(\sigma \in [0.0, 0.5]\)) and 8 geometric orientations (\(\theta \in [0^\circ, 315^\circ]\)), rendered as interactive line and bar charts.

### 2.5 Alternative Streamlit Dashboard (`app.py`)
A standalone Streamlit application providing interactive canvas and preset testing for environments where Streamlit is preferred.

### 2.6 Master Pipeline Runner (`run_all.py`)
An automated orchestrator that reproduces the entire four-week pipeline in sequence (diagram generation &rarr; dataset verification &rarr; optimization benchmarks &rarr; integration testing &rarr; pytest suite) and formats a clean terminal report.

### 2.7 Automated PyTest Test Suite (`tests/`)
22 unit and system tests covering:
- Dataset shape, dtype, class balance, noise injection (`test_dataset.py`).
- NumPy CNN layer mechanics (Conv2D, ReLU, MaxPool2D, Dense) forward and backward shapes, parameter counts, checkpoint integrity (`test_model.py`).
- Pruning sparsity thresholds, int8 quantization bounds and byte reduction (`test_optimization.py`).
- API health, predict (file & base64), explain, compare, robustness, error handling (`test_api.py`).

### 2.8 DevOps, CI/CD, and Documentation
- **`Dockerfile` & `docker-compose.yml`:** Containerized multi-environment deployment.
- **`.github/workflows/ci.yml`:** Automated GitHub Actions pipeline executing full pipeline verification and test suite on push and PR.
- **`MODEL_CARD.md`:** Detailed specification of ShapeCNN according to standard model card practices.
- **`TECHNICAL_IMPLEMENTATION.md`:** Deep mathematical reference covering vectorized convolution, backprop derivations, and quantization formulas.
- **`LICENSE`:** MIT open-source license.
