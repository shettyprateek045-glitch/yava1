import sys
from pathlib import Path
import numpy as np
import pytest

WEEK2_DIR = Path(__file__).resolve().parent.parent / "week2_model_training"
WEEK3_DIR = Path(__file__).resolve().parent.parent / "week3_optimization"
sys.path.insert(0, str(WEEK2_DIR))
sys.path.insert(0, str(WEEK3_DIR))

from model import ShapeCNN
from optimize import load_model, prune_weights, quantize_weights_int8, model_weight_bytes


def test_pruning_sparsity():
    net = load_model()
    sparsity_report = prune_weights(net, amount=0.40)
    for layer_name, sparsity in sparsity_report.items():
        assert abs(sparsity - 0.40) < 0.05, f"Layer {layer_name} sparsity {sparsity} not near 0.40"

    # Forward pass after pruning still works
    x = np.random.randn(2, 3, 32, 32).astype(np.float32)
    logits = net.forward(x)
    assert logits.shape == (2, 4)


def test_quantization_int8_size():
    net = load_model()
    fp32_bytes = model_weight_bytes(net, dtype_bytes=4)
    int8_bytes = quantize_weights_int8(net)

    # Int8 weight size should be ~25% of FP32 (weights 1 byte, biases 4 bytes)
    ratio = int8_bytes / fp32_bytes
    assert 0.24 < ratio < 0.28, f"Unexpected compression ratio {ratio}"

    # Forward pass after quantization works
    x = np.random.randn(2, 3, 32, 32).astype(np.float32)
    logits = net.forward(x)
    assert logits.shape == (2, 4)
