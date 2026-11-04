# Technical Implementation & Mathematical Formulations

This document provides a rigorous mathematical and algorithmic explanation of the
Shape Classification Pipeline implemented in pure NumPy without deep learning frameworks.

---

## 1. Vectorized Convolution via `im2col` and `col2im`

Standard nested loop implementations of 2D cross-correlation are \(O(N \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot C_{\text{in}} \cdot k_H \cdot k_W)\), which is prohibitively slow in Python. We formulate convolution as a single General Matrix Multiply (GEMM) using the `im2col` algorithm.

### 1.1 The Forward Pass
Given an input batch \(X \in \mathbb{R}^{N \times C_{\text{in}} \times H \times W}\) padded with padding \(P\), each sliding \(k \times k\) receptive field is unrolled into a row vector of length \(C_{\text{in}} \cdot k^2\).

1. **Unrolling Input:**
   $$\text{cols} = \text{im2col}(X) \in \mathbb{R}^{(N \cdot H_{\text{out}} \cdot W_{\text{out}}) \times (C_{\text{in}} \cdot k^2)}$$
   where:
   $$H_{\text{out}} = \left\lfloor \frac{H + 2P - k}{S} \right\rfloor + 1, \quad W_{\text{out}} = \left\lfloor \frac{W + 2P - k}{S} \right\rfloor + 1$$

2. **Weight Flattening:**
   $$W_{\text{col}} = W.\text{reshape}(C_{\text{out}}, C_{\text{in}} \cdot k^2)^T \in \mathbb{R}^{(C_{\text{in}} \cdot k^2) \times C_{\text{out}}}$$

3. **Matrix Multiplication:**
   $$\text{Out}_{\text{flat}} = \text{cols} \cdot W_{\text{col}} + b \in \mathbb{R}^{(N \cdot H_{\text{out}} \cdot W_{\text{out}}) \times C_{\text{out}}}$$

4. **Reshaping:** Transpose and reshape \(\text{Out}_{\text{flat}}\) to \((N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})\).

### 1.2 Backpropagation Gradients
Given the upstream gradient \(\delta = \frac{\partial \mathcal{L}}{\partial \text{Out}}\):
1. **Weight Gradient:**
   $$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{N} \left( \text{cols}^T \cdot \delta_{\text{col}} \right)^T$$
2. **Bias Gradient:**
   $$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i, h, w} \delta_{i, :, h, w}$$
3. **Input Gradient:**
   $$\delta_{\text{in, col}} = \delta_{\text{col}} \cdot W_{\text{col}}^T \implies \frac{\partial \mathcal{L}}{\partial X} = \text{col2im}(\delta_{\text{in, col}})$$

---

## 2. Loss Function: Softmax Cross-Entropy

For logits \(z \in \mathbb{R}^{N \times K}\) and target class index \(y_i\):
$$\hat{p}_{i, k} = \frac{e^{z_{i, k} - \max_j z_{i, j}}}{\sum_{j=1}^K e^{z_{i, j} - \max_m z_{i, m}}}$$
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log(\hat{p}_{i, y_i} + \epsilon)$$

The gradient with respect to logits \(z\) is simply:
$$\frac{\partial \mathcal{L}}{\partial z_{i, k}} = \frac{1}{N} (\hat{p}_{i, k} - \mathbb{I}(y_i = k))$$

---

## 3. Real-Time Model Optimization Techniques

### 3.1 Magnitude-Based Unstructured Weight Pruning
Given a pruning ratio \(\alpha \in [0, 1]\) (here \(\alpha = 0.40\)):
1. Collect absolute weights for layer \(l\): \(A = |W^{(l)}|\).
2. Compute the \(k\)-th order statistic where \(k = \lfloor \alpha \cdot |W^{(l)}| \rfloor\):
   $$\tau = \text{partition}(A, k)[k]$$
3. Generate binary sparse mask:
   $$M_{i, j} = \begin{cases} 1 & \text{if } |W^{(l)}_{i, j}| \ge \tau \\ 0 & \text{if } |W^{(l)}_{i, j}| < \tau \end{cases}$$
4. Apply mask: \(\tilde{W}^{(l)} = W^{(l)} \odot M\).

### 3.2 Post-Training Symmetric Int8 Quantization
To compress full 32-bit floating-point weights to 8-bit signed integers:
1. **Scale Factor Determination:**
   $$s_l = \frac{\max |W^{(l)}|}{127.0}$$
2. **Quantization:**
   $$Q^{(l)} = \text{clip}\left( \left\lfloor \frac{W^{(l)}}{s_l} \right\rceil, -127, 127 \right) \in \text{int8}$$
3. **Storage:**
   $$S_{\text{int8}} = \sum_l |W^{(l)}| \times 1\text{ byte} + \sum_l |b^{(l)}| \times 4\text{ bytes}$$
   Yields a **74.9% reduction** in memory footprint (from 262.7 KB down to 65.9 KB).
4. **Dequantization for Inference (Fake-Quant Simulation):**
   $$\hat{W}^{(l)} = Q^{(l)} \cdot s_l \in \text{float32}$$

---

## 4. Computer Vision Explainability (XAI)

### 4.1 Gradient Saliency Map
To find which input pixels most influenced the model's prediction:
1. Given input \(X\) and predicted class \(c = \arg\max \hat{p}\):
   $$\delta_z = [0, \dots, 0, \underbrace{1}_{\text{position } c}, 0, \dots, 0]$$
2. Execute backpropagation with \(\text{lr}=0, \text{momentum}=0\) down to the input layer:
   $$\nabla_X = \frac{\partial z_c}{\partial X} \in \mathbb{R}^{3 \times 32 \times 32}$$
3. Aggregate color channels via \(L_\infty\) norm:
   $$S(h, w) = \max_{c \in \{R, G, B\}} |\nabla_X(c, h, w)|$$
4. Normalize to \([0, 1]\) and map to Jet/Turbo pseudo-color heatmap.

### 4.2 Intermediate Feature Map Visualizations
Activations after non-linearities are captured:
- **Conv1 Activations:** \(\phi_1 = \text{ReLU}(\text{Conv1}(X)) \in \mathbb{R}^{8 \times 32 \times 32}\)
- **Conv2 Activations:** \(\phi_2 = \text{ReLU}(\text{Conv2}(\text{MaxPool}(\phi_1))) \in \mathbb{R}^{16 \times 16 \times 16}\)
Each channel is normalized to \([0, 255]\) and displayed in the studio console.
