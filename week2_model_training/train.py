"""
Week 2 - Deep Learning Model Implementation and Training
----------------------------------------------------------
Trains the NumPy CNN (model.py) on the synthetic shape-classification
dataset (dataset.py). Produces a real training log, a loss/accuracy curve,
and a saved model checkpoint - all generated from an actual run, not
hand-written placeholder numbers.
"""
import time
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import generate_dataset, CLASSES, IMG_SIZE
from model import ShapeCNN, softmax_cross_entropy

np.random.seed(42)
TARGET_SIZE = 32
N_TRAIN_PER_CLASS = 200
N_VAL_PER_CLASS = 50
N_TEST_PER_CLASS = 50
EPOCHS = 25
BATCH_SIZE = 32
LR = 0.08


def resize_batch(images, size=TARGET_SIZE):
    out = np.zeros((len(images), size, size, 3), dtype=np.float32)
    for i, im in enumerate(images):
        out[i] = np.array(Image.fromarray(im).resize((size, size)), dtype=np.float32)
    out /= 255.0
    return out.transpose(0, 3, 1, 2)  # NCHW


def accuracy(net, X, y, batch=64):
    correct = 0
    for i in range(0, len(X), batch):
        preds = net.predict(X[i:i + batch])
        correct += (preds == y[i:i + batch]).sum()
    return correct / len(X)


def main():
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("Week 2: Deep Learning Model Implementation and Training")
    log("Task: 4-class synthetic shape image classification (circle, square, triangle, star)")
    log("=" * 70)

    t0 = time.time()
    log("\n[1/4] Generating synthetic dataset ...")
    train_imgs, train_labels = generate_dataset(N_TRAIN_PER_CLASS, seed=1, add_noise=True)
    val_imgs, val_labels = generate_dataset(N_VAL_PER_CLASS, seed=2, add_noise=True)
    test_imgs, test_labels = generate_dataset(N_TEST_PER_CLASS, seed=3, add_noise=True)
    log(f"  Train: {len(train_imgs)}  Val: {len(val_imgs)}  Test: {len(test_imgs)}  (raw size {IMG_SIZE}x{IMG_SIZE})")

    log("\n[2/4] Preprocessing (resize -> {0}x{0}, normalize to [0,1], NCHW) ...".format(TARGET_SIZE))
    X_train = resize_batch(train_imgs)
    y_train = train_labels
    X_val = resize_batch(val_imgs)
    y_val = val_labels
    X_test = resize_batch(test_imgs)
    y_test = test_labels
    log(f"  X_train shape: {X_train.shape}")

    log("\n[3/4] Building model (Conv-ReLU-Pool x2 -> Dense(64) -> Dense(4)) ...")
    net = ShapeCNN(num_classes=len(CLASSES), img_size=TARGET_SIZE)
    log(f"  Total trainable parameters: {net.num_params():,}")

    log("\n[4/4] Training ...")
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    n = len(X_train)
    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(n)
        X_train, y_train = X_train[perm], y_train[perm]
        epoch_losses = []
        correct = 0
        for i in range(0, n, BATCH_SIZE):
            xb = X_train[i:i + BATCH_SIZE]
            yb = y_train[i:i + BATCH_SIZE]
            logits = net.forward(xb)
            loss, dlogits, probs = softmax_cross_entropy(logits, yb)
            net.backward(dlogits, lr=LR)
            epoch_losses.append(loss)
            correct += (logits.argmax(1) == yb).sum()
        train_loss = float(np.mean(epoch_losses))
        train_acc = correct / n

        val_logits = net.forward(X_val)
        val_loss, _, _ = softmax_cross_entropy(val_logits, y_val)
        val_acc = (val_logits.argmax(1) == y_val).mean()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        log(f"  Epoch {epoch:2d}/{EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    test_acc = accuracy(net, X_test, y_test)
    elapsed = time.time() - t0
    log("\n" + "=" * 70)
    log(f"FINAL TEST ACCURACY: {test_acc:.3f}  ({int(test_acc*len(X_test))}/{len(X_test)} correct)")
    log(f"Total wall-clock time: {elapsed:.1f} seconds")
    log("=" * 70)

    # Per-class accuracy on test set
    log("\nPer-class test accuracy:")
    preds = net.predict(X_test)
    for ci, cname in enumerate(CLASSES):
        idx = y_test == ci
        acc_c = (preds[idx] == ci).mean() if idx.sum() > 0 else float("nan")
        log(f"  {cname:10s}: {acc_c:.3f}  (n={idx.sum()})")

    # Save artifacts
    np.savez("model_checkpoint.npz",
              conv1_W=net.conv1.W, conv1_b=net.conv1.b,
              conv2_W=net.conv2.W, conv2_b=net.conv2.b,
              fc1_W=net.fc1.W, fc1_b=net.fc1.b,
              fc2_W=net.fc2.W, fc2_b=net.fc2.b)
    with open("training_history.json", "w") as f:
        json.dump({"history": history, "test_accuracy": test_acc, "elapsed_sec": elapsed,
                    "num_params": net.num_params()}, f, indent=2)
    with open("training_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", color="#2C5F2D")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", color="#E8A33D")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss Curve")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["epoch"], history["train_acc"], label="Train Acc", color="#2C5F2D")
    axes[1].plot(history["epoch"], history["val_acc"], label="Val Acc", color="#E8A33D")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Accuracy Curve")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    print("\nSaved: model_checkpoint.npz, training_history.json, training_log.txt, training_curve.png")


if __name__ == "__main__":
    main()
