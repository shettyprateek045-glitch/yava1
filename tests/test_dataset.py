import sys
from pathlib import Path
import numpy as np
import pytest

WEEK2_DIR = Path(__file__).resolve().parent.parent / "week2_model_training"
sys.path.insert(0, str(WEEK2_DIR))

from dataset import CLASSES, IMG_SIZE, NUM_CLASSES, generate_image, generate_dataset


def test_classes_definition():
    assert len(CLASSES) == 4
    assert CLASSES == ["circle", "square", "triangle", "star"]
    assert NUM_CLASSES == 4


def test_generate_single_image_shape_and_dtype():
    for cls in CLASSES:
        img = generate_image(cls, add_noise=False)
        assert img.shape == (IMG_SIZE, IMG_SIZE, 3)
        assert img.dtype == np.uint8
        # Ensure image has non-zero pixels
        assert img.sum() > 0


def test_generate_single_image_with_noise():
    img_clean = generate_image("circle", add_noise=False)
    img_noisy = generate_image("circle", add_noise=True)
    assert img_clean.shape == img_noisy.shape == (64, 64, 3)
    assert img_clean.dtype == img_noisy.dtype == np.uint8


def test_generate_dataset_balance_and_shapes():
    n_per_class = 5
    images, labels = generate_dataset(n_per_class, seed=42)
    assert len(images) == n_per_class * NUM_CLASSES
    assert len(labels) == n_per_class * NUM_CLASSES
    assert images.shape == (20, 64, 64, 3)
    assert labels.shape == (20,)

    # Verify per-class balance
    counts = np.bincount(labels)
    assert len(counts) == 4
    assert all(c == n_per_class for c in counts)
