# Model Card — ShapeCNN (Geometric Shape Classifier)

Following the standard Model Card format (Mitchell et al., 2019), this document provides
a comprehensive specification of the **ShapeCNN** model, its training data, quantitative performance,
optimization variants, and production operational characteristics.

---

## 1. Model Details

- **Model Name:** ShapeCNN
- **Model Type:** Convolutional Neural Network (CNN) implemented from scratch in pure NumPy
- **Architecture Overview:**
  - Input Layer: `(3, 32, 32)` normalized RGB tensor
  - Layer 1: `Conv2D(in=3, out=8, k=3, pad=1)` + `ReLU` + `MaxPool2D(2x2, stride=2)` &rarr; `(8, 16, 16)`
  - Layer 2: `Conv2D(in=8, out=16, k=3, pad=1)` + `ReLU` + `MaxPool2D(2x2, stride=2)` &rarr; `(16, 8, 8)`
  - Layer 3: `Flatten(1024)` + `Dense(1024 -> 64)` + `ReLU` &rarr; `(64)`
  - Layer 4: `Dense(64 -> 4)` + `Softmax` &rarr; `(4)`
- **Total Parameters:** `67,252` parameters
  - Conv1: 224 parameters
  - Conv2: 1,168 parameters
  - FC1: 65,600 parameters
  - FC2: 260 parameters
- **Framework:** Pure NumPy (vectorized via `im2col`/`col2im` GEMM), Pillow for synthetic rasterization.
- **License:** MIT License
- **Author:** Prateek Shetty

---

## 2. Intended Use

- **Primary Intended Use:** Real-time geometric contour classification on resource-constrained embedded edge devices, automated quality inspection, and educational demonstration of CNN internals without framework abstraction.
- **Target Classes:** 4 distinct 2D geometric shapes:
  1. `circle`
  2. `square`
  3. `triangle`
  4. `star`
- **Out-of-Scope Uses:** Real-world photographic object recognition without domain adaptation; non-geometric object detection.

---

## 3. Training Data & Preprocessing

- **Dataset:** Synthetic Shape Dataset generated via PIL procedural graphics with controlled randomness.
- **Data Augmentation & Randomization:**
  - Dynamic centroids: \(cx, cy \in [22, 42]\)
  - Scale variation: Radius / half-size \(r \in [14, 22]\)
  - Arbitrary geometric rotation: \(\theta \in [0, \pi)\)
  - Color palette: Randomized RGB tuples \((60 \dots 255)\)
  - Sensor noise: Additive Gaussian noise \(\mathcal{N}(\mu=0, \sigma=12)\)
- **Input Preprocessing:**
  1. Downscaled from 64x64 to 32x32 bilinear resolution.
  2. Normalized from uint8 \([0, 255]\) to float32 \([0.0, 1.0]\).
  3. Transposed from HWC `(32, 32, 3)` to CHW `(3, 32, 32)`.

---

## 4. Quantitative Evaluation & Benchmark Results

Evaluated on an independent, held-out test split of 200 synthetic images (50 per class) with noise:

### Performance Comparison Across Model Variants

| Metric | FP32 Baseline | 40% Magnitude Pruned | Int8 Post-Training Quantized (Deployed) |
|---|:---:|:---:|:---:|
| **Test Accuracy** | **90.5%** | **86.5%** (-4.0%) | **89.5%** (-1.0%) |
| **Weight Storage** | 262.7 KB | 262.7 KB | **65.9 KB** (**-74.9%**) |
| **Active Parameters** | 67,252 | 40,352 (26,900 zeroed) | 67,252 (8-bit int) |
| **Sparsity** | 0.0% | **40.0%** | 0.0% |
| **Single-Image Latency (CPU)** | ~0.65 ms | ~0.64 ms | **~0.58 ms** |
| **Live API Accuracy (Week 4)** | — | — | **91.7%** (55/60 correct) |

### Per-Class Evaluation Summary (Int8 Model)

| Class | Precision | Recall | F1-Score | Notes |
|---|:---:|:---:|:---:|---|
| **Circle** | 0.94 | 0.96 | 0.95 | High rotational invariance |
| **Square** | 0.88 | 0.86 | 0.87 | Minor confusion with rotated diamonds |
| **Triangle** | 0.90 | 0.92 | 0.91 | Stable 3-vertex identification |
| **Star** | 0.87 | 0.84 | 0.85 | Sensitive to heavy noise at concavities |
| **Overall Macro Avg** | **0.90** | **0.90** | **0.90** | **Robust multi-class performance** |

---

## 5. Explainability (XAI) & Interpretability

- **Pixel Saliency Attribution:** Computed via input gradient backpropagation \(\frac{\partial y_{\text{pred}}}{\partial X}\). Gradient norms accurately highlight geometric vertices, outer contours, and concavities.
- **Filter Visualizations:**
  - `Conv1` (8 channels): Low-level edge-detection kernels (horizontal, vertical, diagonal, and contrast boundaries).
  - `Conv2` (16 channels): High-level corner and intersection detectors that distinguish triangles, squares, and stars.

---

## 6. Environmental Impact

- **Compute Infrastructure:** Low-power single CPU core.
- **Training Time:** ~26 seconds total training time (20 epochs).
- **Carbon Footprint:** Estimated < 0.0001 kg \(\text{CO}_2\)eq.
- **Edge Deployment Footprint:** Weighs only **65.9 KB**, easily deployable to MicroPython, Arduino, ESP32, or embedded automotive IoT sensors.

---

## 7. Limitations & Recommendations

1. **Domain Shift:** Synthetic shapes have clean contrast and background. Camera glare, shadows, or background clutter in photographic images will degrade accuracy unless fine-tuned on real images.
2. **Quantization Format:** The current quantization uses simulated fake-quant roundtripping. On hardware with native INT8 SIMD (AVX-512 VNNI or ARM NEON), further 2x–3x integer acceleration is possible.
