import sys
from pathlib import Path
import numpy as np
import pytest

WEEK2_DIR = Path(__file__).resolve().parent.parent / "week2_model_training"
sys.path.insert(0, str(WEEK2_DIR))

from model import ShapeCNN, Conv2D, ReLU, MaxPool2D, Flatten, Dense


def test_conv2d_forward_backward():
    conv = Conv2D(in_c=3, out_c=4, k=3, pad=1, stride=1)
    x = np.random.randn(2, 3, 16, 16).astype(np.float32)
    out = conv.forward(x)
    assert out.shape == (2, 4, 16, 16)

    dout = np.random.randn(*out.shape).astype(np.float32)
    dx = conv.backward(dout, lr=0.01, momentum=0.9)
    assert dx.shape == x.shape


def test_relu_forward_backward():
    relu = ReLU()
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    out = relu.forward(x)
    np.testing.assert_allclose(out, [[0.0, 2.0], [3.0, 0.0]])

    dout = np.ones_like(x)
    dx = relu.backward(dout)
    np.testing.assert_allclose(dx, [[0.0, 1.0], [1.0, 0.0]])


def test_maxpool2d_forward_backward():
    pool = MaxPool2D(size=2, stride=2)
    x = np.random.randn(2, 4, 16, 16).astype(np.float32)
    out = pool.forward(x)
    assert out.shape == (2, 4, 8, 8)

    dout = np.random.randn(*out.shape).astype(np.float32)
    dx = pool.backward(dout)
    assert dx.shape == x.shape


def test_dense_forward_backward():
    dense = Dense(in_f=32, out_f=8)
    x = np.random.randn(4, 32).astype(np.float32)
    out = dense.forward(x)
    assert out.shape == (4, 8)

    dout = np.random.randn(4, 8).astype(np.float32)
    dx = dense.backward(dout, lr=0.01)
    assert dx.shape == x.shape


def test_shape_cnn_architecture_and_params():
    net = ShapeCNN(num_classes=4, img_size=32)
    num_params = net.num_params()
    # Expected:
    # Conv1: 3*8*3*3 + 8 = 224
    # Conv2: 8*16*3*3 + 16 = 1,168
    # FC1: (16*8*8)*64 + 64 = 1024*64 + 64 = 65,600
    # FC2: 64*4 + 4 = 260
    # Total = 224 + 1168 + 65600 + 260 = 67,252
    assert num_params == 67252

    x = np.random.randn(3, 3, 32, 32).astype(np.float32)
    logits = net.forward(x)
    assert logits.shape == (3, 4)

    preds = net.predict(x)
    assert preds.shape == (3,)
    assert all(0 <= p < 4 for p in preds)


def test_checkpoint_loading():
    ckpt_path = WEEK2_DIR / "model_checkpoint.npz"
    assert ckpt_path.exists(), "Model checkpoint must exist"

    data = np.load(ckpt_path)
    required_keys = ["conv1_W", "conv1_b", "conv2_W", "conv2_b", "fc1_W", "fc1_b", "fc2_W", "fc2_b"]
    for k in required_keys:
        assert k in data, f"Missing key {k} in checkpoint"
