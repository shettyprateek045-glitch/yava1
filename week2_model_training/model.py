"""
Compact Convolutional Neural Network - NumPy implementation
-------------------------------------------------------------
Architecture:
    Conv2D(3->8, 3x3, pad=1) -> ReLU -> MaxPool(2x2)
    Conv2D(8->16, 3x3, pad=1) -> ReLU -> MaxPool(2x2)
    Flatten -> Dense(1024->64) -> ReLU -> Dense(64->NUM_CLASSES) -> Softmax

Implemented directly in NumPy (im2col-vectorized convolution) rather than
PyTorch/TensorFlow, because this sandboxed environment's network policy only
allows installing the CUDA-bundled `torch` build from PyPI (multi-GB, no
CPU-only wheel index is reachable), which does not fit this task's time/
resource budget. The architecture, training loop and backprop below follow
the same mechanics a PyTorch `nn.Module` would use. Porting the identical
architecture to PyTorch for GPU training is a listed future step.
"""
import numpy as np


def im2col(x, kh, kw, stride=1, pad=1):
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    cols = np.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for y in range(kh):
        y_max = y + stride * out_h
        for xk in range(kw):
            x_max = xk + stride * out_w
            cols[:, :, y, xk, :, :] = xp[:, :, y:y_max:stride, xk:x_max:stride]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return cols, out_h, out_w


def col2im(cols, x_shape, kh, kw, stride=1, pad=1):
    N, C, H, W = x_shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    cols = cols.reshape(N, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    for y in range(kh):
        y_max = y + stride * out_h
        for xk in range(kw):
            x_max = xk + stride * out_w
            xp[:, :, y:y_max:stride, xk:x_max:stride] += cols[:, :, y, xk, :, :]
    return xp[:, :, pad:pad + H, pad:pad + W]


class Conv2D:
    def __init__(self, in_c, out_c, k=3, pad=1, stride=1):
        scale = np.sqrt(2.0 / (in_c * k * k))
        self.W = np.random.randn(out_c, in_c, k, k).astype(np.float32) * scale
        self.b = np.zeros(out_c, dtype=np.float32)
        self.k, self.pad, self.stride = k, pad, stride
        self.in_c, self.out_c = in_c, out_c
        self.vW, self.vb = np.zeros_like(self.W), np.zeros_like(self.b)

    def forward(self, x):
        self.x_shape = x.shape
        N = x.shape[0]
        cols, oh, ow = im2col(x, self.k, self.k, self.stride, self.pad)
        self.cols = cols
        Wcol = self.W.reshape(self.out_c, -1).T
        out = cols @ Wcol + self.b
        out = out.reshape(N, oh, ow, self.out_c).transpose(0, 3, 1, 2)
        self.out_shape = (oh, ow)
        return out

    def backward(self, dout, lr, momentum=0.9):
        N = dout.shape[0]
        dout_col = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_c)
        dW = (self.cols.T @ dout_col).T.reshape(self.W.shape) / N
        db = dout_col.sum(axis=0) / N
        Wcol = self.W.reshape(self.out_c, -1)
        dcols = dout_col @ Wcol
        dx = col2im(dcols, self.x_shape, self.k, self.k, self.stride, self.pad)
        self.vW = momentum * self.vW - lr * dW
        self.vb = momentum * self.vb - lr * db
        self.W += self.vW
        self.b += self.vb
        return dx


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout, lr=None, momentum=None):
        return dout * self.mask


class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size, self.stride = size, stride

    def forward(self, x):
        N, C, H, W = x.shape
        s, st = self.size, self.stride
        oh, ow = H // st, W // st
        x_reshaped = x[:, :, :oh * st, :ow * st].reshape(N, C, oh, st, ow, st)
        out = x_reshaped.max(axis=(3, 5))
        self.x_shape = x.shape
        self.mask = (x_reshaped == out[:, :, :, None, :, None])
        return out

    def backward(self, dout, lr=None, momentum=None):
        N, C, H, W = self.x_shape
        s, st = self.size, self.stride
        oh, ow = H // st, W // st
        dout_exp = dout[:, :, :, None, :, None]
        dx_reshaped = self.mask * dout_exp
        dx = np.zeros((N, C, oh * st, ow * st), dtype=dout.dtype)
        dx[:, :, :oh * st, :ow * st] = dx_reshaped.transpose(0, 1, 2, 3, 4, 5).reshape(N, C, oh * st, ow * st)
        full = np.zeros(self.x_shape, dtype=dout.dtype)
        full[:, :, :oh * st, :ow * st] = dx[:, :, :oh * st, :ow * st]
        return full


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout, lr=None, momentum=None):
        return dout.reshape(self.shape)


class Dense:
    def __init__(self, in_f, out_f):
        scale = np.sqrt(2.0 / in_f)
        self.W = np.random.randn(in_f, out_f).astype(np.float32) * scale
        self.b = np.zeros(out_f, dtype=np.float32)
        self.vW, self.vb = np.zeros_like(self.W), np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, lr, momentum=0.9):
        N = dout.shape[0]
        dW = self.x.T @ dout / N
        db = dout.sum(axis=0) / N
        dx = dout @ self.W.T
        self.vW = momentum * self.vW - lr * dW
        self.vb = momentum * self.vb - lr * db
        self.W += self.vW
        self.b += self.vb
        return dx


def softmax_cross_entropy(logits, labels):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(probs[np.arange(N), labels] + 1e-9).mean()
    dlogits = probs.copy()
    dlogits[np.arange(N), labels] -= 1
    dlogits /= N
    return loss, dlogits, probs


class ShapeCNN:
    """Conv(3->8)-ReLU-Pool -> Conv(8->16)-ReLU-Pool -> Flatten -> Dense(64) -> ReLU -> Dense(num_classes)"""

    def __init__(self, num_classes=4, img_size=32):
        self.conv1 = Conv2D(3, 8, k=3, pad=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()
        self.conv2 = Conv2D(8, 16, k=3, pad=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()
        self.flatten = Flatten()
        flat_dim = 16 * (img_size // 4) * (img_size // 4)
        self.fc1 = Dense(flat_dim, 64)
        self.relu3 = ReLU()
        self.fc2 = Dense(64, num_classes)
        self.layers = [self.conv1, self.relu1, self.pool1, self.conv2, self.relu2,
                       self.pool2, self.flatten, self.fc1, self.relu3, self.fc2]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dlogits, lr, momentum=0.9):
        dout = dlogits
        for layer in reversed(self.layers):
            if hasattr(layer, "backward"):
                sig = layer.backward
                try:
                    dout = sig(dout, lr, momentum)
                except TypeError:
                    dout = sig(dout)
        return dout

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(axis=1)

    def num_params(self):
        total = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                total += layer.W.size + layer.b.size
        return total
