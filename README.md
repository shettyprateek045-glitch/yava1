[![CI](https://github.com/shettyprateek045-glitch/yava1/actions/workflows/ci.yml/badge.svg)](https://github.com/shettyprateek045-glitch/yava1/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)
![PyTest](https://img.shields.io/badge/pytest-passing-success?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

# Computer Vision Internship — Shape Classification & Optimization Studio

An end-to-end, four-week computer vision project built as a complete, production-grade pipeline:
**Design &rarr; Model Training &rarr; Model Optimization &rarr; Integration & Deployment**.

Includes an interactive **ShapeVision Studio Web Console**, real-time **Explainable AI (XAI)** with pixel-level saliency heatmaps & convolutional feature map visualizers, empirical **Optimization Benchmarks** (comparing FP32 Baseline, 40% Magnitude Pruning, and Int8 Post-Training Quantization), **Perturbation Robustness Testing**, **Docker** containerization, **GitHub Actions CI/CD**, and a **22-test automated PyTest suite**.

---

## Architecture & Dataflow

```
   [ Interactive Web Canvas / Image Upload / Preset Generator ]
                               │
                               ▼
            [ FastAPI High-Performance Serving Layer ]
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       [ FP32 Baseline ] [ 40% Pruned ]  [ Int8 Quantized ]
          (262.7 KB)       (40% sparse)     (65.9 KB / -74.9%)
              │                │                │
              └────────────────┼────────────────┘
                               │
                 [ Pure NumPy Vectorized CNN ]
                 im2col / col2im GEMM Backprop
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [ Softmax Predictions ]             [ Explainability (XAI) ]
 (Circle, Square, Triangle, Star)    • Saliency Heatmap (∇x y_c)
                                     • Conv1 & Conv2 Feature Maps
```

---

## Key Results & Empirical Benchmarks

All metrics below are derived from actual execution runs on a single CPU core without external dataset dependencies:

| Metric | FP32 Baseline | 40% Magnitude Pruned | Int8 Quantized (Deployed) |
|---|:---:|:---:|:---:|
| **Test Accuracy** | **90.5%** | **86.5%** (-4.0%) | **89.5%** (-1.0%) |
| **Weight Memory Footprint** | 262.7 KB | 262.7 KB | **65.9 KB** (**-74.9%**) |
| **Parameters** | 67,252 | 40,352 active (26,900 zeroed) | 67,252 (8-bit int) |
| **Compression Ratio** | 1.0x (Reference) | 1.0x (Sparse) | **3.98x Storage Reduction** |
| **Inference Latency** | ~0.65 ms | ~0.64 ms | **~0.58 ms** |
| **Live API Accuracy** | — | — | **91.7%** (55 of 60 correct) |
| **Integration Test Pass Rate**| — | — | **58 of 63 passed** (92.1%) |
| **Automated PyTest Suite** | — | — | **22 of 22 passed** (100%) |

---

## Repository Layout

| Directory / File | Description |
|---|---|
| `week1_algorithm_design/` | Algorithm design report (`.docx`), data preprocessing and algorithm flowcharts, diagram generator |
| `week2_model_training/` | Synthetic shape generator (`dataset.py`), vectorized NumPy CNN (`model.py`), training script (`train.py`), weights checkpoint (`model_checkpoint.npz`), training logs and curve |
| `week3_optimization/` | Magnitude weight pruning and symmetric int8 quantization benchmark script (`optimize.py`), benchmark results (`.json`), comparison chart |
| `week4_integration/` | Original FastAPI integration service (`api.py`), integration test suite (`test_client.py`), architecture diagram, test results |
| **`api.py`** | **Production Product API:** Serves the interactive console, XAI Saliency Maps, multi-model comparative inference, robustness sweeps, and SQLite audit logging |
| **`app.py`** | **Alternative Streamlit Dashboard:** Live interactive testing interface |
| **`static/index.html`** | **Interactive Web Studio:** Dark-mode console with HTML5 drawing canvas, live prediction bars, XAI inspector, model comparison table, and perturbation graphs |
| **`run_all.py`** | **Master Runner:** Executes the entire four-week pipeline end-to-end and verifies all artifacts |
| **`tests/`** | **Automated PyTest Suite:** 22 unit & integration tests covering dataset, model math, optimization, and API endpoints |
| **`Dockerfile` & `docker-compose.yml`** | Production containerization setup |
| **`.github/workflows/ci.yml`** | GitHub Actions CI workflow reproducing pipeline and verifying tests |
| **`MODEL_CARD.md`** | Complete AI model card specification (Google/HuggingFace format) |
| **`TECHNICAL_IMPLEMENTATION.md`**| Detailed mathematical walkthrough of `im2col`, backprop gradients, pruning, and quantization |
| **`ADVANCED_VERSION.md`** | Comprehensive audit and build-out engineering documentation |

---

## Quickstart & Reproducing Results

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/shettyprateek045-glitch/yava1.git
cd yava1

pip install -r requirements.txt -r requirements-advanced.txt
```

### 2. Run the Complete 4-Week Pipeline (1 Command)

```bash
python run_all.py
```
This executes diagram generation &rarr; model verification &rarr; optimization benchmarks &rarr; integration tests &rarr; pytest suite in sequence.

### 3. Run the Automated Test Suite

```bash
pytest tests/ -v
```

### 4. Launch the Interactive Vision Studio

```bash
uvicorn api:app --reload --port 8000
```
Then navigate to:
- **Interactive Studio UI:** [http://localhost:8000/](http://localhost:8000/)
- **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

*(Optional) Launch Streamlit alternative:*
```bash
streamlit run app.py
```

### 5. Run via Docker / Docker Compose

```bash
docker-compose up --build
```

---

## Interactive Studio Features

1. **Interactive HTML5 Drawing Canvas:** Draw any geometric contour directly with your mouse or stylus; adjust brush size, invert canvas, or clear with 1 click.
2. **Synthetic Generator Presets:** Instantly load procedurally generated circles, squares, triangles, or stars with realistic sensor noise and rotational transforms.
3. **Computer Vision Explainability (XAI):**
   - **Pixel Saliency Map:** Computes backpropagated input gradients \(\nabla_X y_c\) to highlight the exact perimeter contours and vertices that informed the prediction.
   - **Filter Activations:** Visualizes all 8 feature map channels of `Conv1` and 16 channels of `Conv2` in real-time.
4. **Multi-Model Optimization Comparison:** Evaluates your active input image simultaneously across FP32 Baseline, 40% Pruned, and Int8 Quantized models, displaying latency, parameter count, and weight size side-by-side.
5. **Robustness & Perturbation Playground:** Stress-tests model predictions against additive Gaussian noise (\(\sigma \in [0.0, 0.5]\)) and rotational variations (\(0^\circ\) to \(315^\circ\)) with live interactive curves.
6. **SQLite Prediction Audit Trail:** Automatically logs every prediction event with timestamps, confidence, latency, and probabilities, with JSON export and log clearing tools.

---

## API Reference Summary

| Endpoint | Method | Description |
|---|:---:|---|
| `/` | `GET` | Serves the interactive ShapeVision Studio web console |
| `/health` | `GET` | Service status, target dimensions, loaded model variants |
| `/api/predict` | `POST` | Predict shape class from multipart image file or base64 canvas JSON |
| `/api/explain` | `POST` | Computes XAI Saliency Heatmap and Conv1/Conv2 feature map activations |
| `/api/compare` | `POST` | Side-by-side evaluation across Baseline, Pruned, and Quantized models |
| `/api/robustness` | `POST` | Evaluates model performance across noise and rotation sweeps |
| `/api/sample/{shape}` | `GET` | Generates a fresh synthetic shape (circle, square, triangle, star, random) |
| `/api/benchmark` | `GET` | Returns official Week 3 model optimization benchmarks |
| `/api/history` | `GET` | Retrieves recent prediction records from SQLite audit trail |
| `/api/history` | `DELETE`| Clears prediction audit history |
| `/api/stats` | `GET` | Returns aggregate inference counts, latency, and class distribution |

---

## Limitations & Future Directions

- **NumPy Implementation:** The CNN is intentionally constructed in pure NumPy (using `im2col`/`col2im` GEMM) to demonstrate foundational mechanics without framework abstraction. Porting the architecture to PyTorch for GPU acceleration is an available next step.
- **Simulated Quantization:** Quantization employs standard fake-quant evaluation. Native integer acceleration can be achieved via ONNX Runtime or TensorRT.
- **Domain Shift:** Synthetic shapes provide clean contrast. Adapting to real-world camera imagery with complex backgrounds would benefit from fine-tuning with transfer learning.

---

## License

This project is licensed under the [MIT License](LICENSE) &copy; 2026 Prateek Shetty.
