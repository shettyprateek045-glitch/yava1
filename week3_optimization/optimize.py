"""
Week 3 - Real-Time Performance Optimization of Computer Vision Algorithms
----------------------------------------------------------------------------
Loads the CNN trained in Week 2, then applies and MEASURES (not estimates):
  1) Magnitude-based weight pruning
  2) Post-training weight quantization (float32 -> int8, simulated round-trip)
Every number in benchmark_results.json / optimization_log.txt comes from an
actual run on this machine - none are hand-written.
"""
import sys
import os
import time
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2_model_training"))
from dataset import generate_dataset, CLASSES  # noqa: E402
from model import ShapeCNN, Conv2D, Dense  # noqa: E402

TARGET_SIZE = 32
CKPT_PATH = os.path.join(os.path.dirname(__file__), "..", "week2_model_training", "model_checkpoint.npz")


def resize_batch(images, size=TARGET_SIZE):
    out = np.zeros((len(images), size, size, 3), dtype=np.float32)
    for i, im in enumerate(images):
        out[i] = np.array(Image.fromarray(im).resize((size, size)), dtype=np.float32)
    out /= 255.0
    return out.transpose(0, 3, 1, 2)


def load_model():
    net = ShapeCNN(num_classes=len(CLASSES), img_size=TARGET_SIZE)
    d = np.load(CKPT_PATH)
    net.conv1.W, net.conv1.b = d["conv1_W"], d["conv1_b"]
    net.conv2.W, net.conv2.b = d["conv2_W"], d["conv2_b"]
    net.fc1.W, net.fc1.b = d["fc1_W"], d["fc1_b"]
    net.fc2.W, net.fc2.b = d["fc2_W"], d["fc2_b"]
    return net


def accuracy(net, X, y, batch=64):
    correct = 0
    for i in range(0, len(X), batch):
        correct += (net.predict(X[i:i + batch]) == y[i:i + batch]).sum()
    return correct / len(X)


def model_weight_bytes(net, dtype_bytes=4):
    total = 0
    for layer in [net.conv1, net.conv2, net.fc1, net.fc2]:
        total += layer.W.size * dtype_bytes + layer.b.size * dtype_bytes
    return total


def benchmark_inference(net, X, n_repeats=30, batch=32):
    xb = X[:batch]
    # warm-up
    net.forward(xb)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        net.forward(xb)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return times.mean(), times.std()


def prune_weights(net, amount):
    """Zero out the `amount` fraction of smallest-magnitude weights per layer."""
    sparsity_report = {}
    for name, layer in [("conv1", net.conv1), ("conv2", net.conv2), ("fc1", net.fc1), ("fc2", net.fc2)]:
        flat = np.abs(layer.W).flatten()
        k = int(len(flat) * amount)
        if k > 0:
            threshold = np.partition(flat, k)[k]
            mask = np.abs(layer.W) >= threshold
            layer.W = layer.W * mask
        actual_sparsity = (layer.W == 0).mean()
        sparsity_report[name] = float(actual_sparsity)
    return sparsity_report


def quantize_weights_int8(net):
    """Simulated post-training quantization: per-layer symmetric int8 quantization,
    then dequantize back to float32 for the forward pass (standard fake-quant
    evaluation approach). Returns the quantized int8 byte size actually used."""
    int8_bytes = 0
    for layer in [net.conv1, net.conv2, net.fc1, net.fc2]:
        scale = np.abs(layer.W).max() / 127.0 if np.abs(layer.W).max() > 0 else 1.0
        q = np.clip(np.round(layer.W / scale), -127, 127).astype(np.int8)
        layer.W = (q.astype(np.float32)) * scale  # dequantized weights used for forward pass
        int8_bytes += q.size * 1 + layer.b.size * 4  # biases kept at fp32
    return int8_bytes


def main():
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("Week 3: Real-Time Performance Optimization of Computer Vision Algorithms")
    log("=" * 70)

    log("\n[1/4] Loading Week 2 trained model and generating test set ...")
    test_imgs, test_labels = generate_dataset(50, seed=3, add_noise=True)
    X_test = resize_batch(test_imgs)
    y_test = test_labels

    baseline = load_model()
    baseline_acc = accuracy(baseline, X_test, y_test)
    baseline_bytes = model_weight_bytes(baseline, dtype_bytes=4)
    mean_t, std_t = benchmark_inference(baseline, X_test, n_repeats=30)
    log(f"  Baseline  accuracy={baseline_acc:.3f}  weight_size={baseline_bytes/1024:.1f} KB  "
        f"inference={mean_t*1000:.3f} ms +/- {std_t*1000:.3f} ms  (batch=32, mean of 30 runs)")

    log("\n[2/4] Applying magnitude-based weight pruning (target 40% sparsity) ...")
    pruned = load_model()
    sparsity_report = prune_weights(pruned, amount=0.40)
    pruned_acc = accuracy(pruned, X_test, y_test)
    mean_tp, std_tp = benchmark_inference(pruned, X_test, n_repeats=30)
    for layer_name, sp in sparsity_report.items():
        log(f"    {layer_name}: {sp*100:.1f}% weights zeroed")
    log(f"  Pruned    accuracy={pruned_acc:.3f}  inference={mean_tp*1000:.3f} ms +/- {std_tp*1000:.3f} ms")
    log("  Note: this NumPy implementation performs dense matrix multiplication, so zeroed")
    log("  weights reduce accuracy risk and would enable real speedups with a sparse-matrix")
    log("  kernel (e.g. scipy.sparse / hardware sparse acceleration); the dense forward pass")
    log("  timed above does not itself skip the zeroed multiplications.")

    log("\n[3/4] Applying post-training int8 weight quantization ...")
    quantized = load_model()
    int8_bytes = quantize_weights_int8(quantized)
    quantized_acc = accuracy(quantized, X_test, y_test)
    mean_tq, std_tq = benchmark_inference(quantized, X_test, n_repeats=30)
    log(f"  Quantized accuracy={quantized_acc:.3f}  weight_size={int8_bytes/1024:.1f} KB  "
        f"inference={mean_tq*1000:.3f} ms +/- {std_tq*1000:.3f} ms")

    log("\n[4/4] Summary")
    log("-" * 70)
    log(f"{'Variant':<12}{'Accuracy':<12}{'Weight Size':<16}{'Inference (ms)':<18}")
    log(f"{'Baseline':<12}{baseline_acc:<12.3f}{baseline_bytes/1024:<16.1f}{mean_t*1000:<18.3f}")
    log(f"{'Pruned':<12}{pruned_acc:<12.3f}{baseline_bytes/1024:<16.1f}{mean_tp*1000:<18.3f}")
    log(f"{'Quantized':<12}{quantized_acc:<12.3f}{int8_bytes/1024:<16.1f}{mean_tq*1000:<18.3f}")
    size_reduction = (1 - int8_bytes / baseline_bytes) * 100
    log(f"\nQuantization reduced weight storage by {size_reduction:.1f}% "
        f"({baseline_bytes/1024:.1f} KB -> {int8_bytes/1024:.1f} KB)")
    log(f"Accuracy change, baseline -> pruned:    {(pruned_acc-baseline_acc)*100:+.1f} pts")
    log(f"Accuracy change, baseline -> quantized: {(quantized_acc-baseline_acc)*100:+.1f} pts")

    results = {
        "baseline": {"accuracy": baseline_acc, "weight_size_kb": baseline_bytes / 1024,
                     "inference_ms_mean": mean_t * 1000, "inference_ms_std": std_t * 1000},
        "pruned_40pct": {"accuracy": pruned_acc, "weight_size_kb": baseline_bytes / 1024,
                         "inference_ms_mean": mean_tp * 1000, "inference_ms_std": std_tp * 1000,
                         "layer_sparsity": sparsity_report},
        "quantized_int8": {"accuracy": quantized_acc, "weight_size_kb": int8_bytes / 1024,
                           "inference_ms_mean": mean_tq * 1000, "inference_ms_std": std_tq * 1000,
                           "size_reduction_pct": size_reduction},
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("optimization_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    print("\nSaved: benchmark_results.json, optimization_log.txt")


if __name__ == "__main__":
    main()
