"""
Synthetic Shape Classification Dataset
---------------------------------------
Generates labeled 64x64 RGB images of four geometric shapes (circle, square,
triangle, star) with randomized position, size, rotation, color and additive
noise. Used as a stand-in for a public image dataset so the full pipeline
(data generation -> preprocessing -> augmentation -> training -> evaluation)
can run end-to-end without external downloads.
"""
import numpy as np
from PIL import Image, ImageDraw
import random
import math

IMG_SIZE = 64
CLASSES = ["circle", "square", "triangle", "star"]
NUM_CLASSES = len(CLASSES)


def _random_color():
    return tuple(random.randint(60, 255) for _ in range(3))


def _draw_star(draw, cx, cy, r_outer, r_inner, rotation, fill):
    points = []
    for i in range(10):
        angle = rotation + i * math.pi / 5
        r = r_outer if i % 2 == 0 else r_inner
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=fill)


def _draw_triangle(draw, cx, cy, size, rotation, fill):
    points = []
    for i in range(3):
        angle = rotation + i * 2 * math.pi / 3
        points.append((cx + size * math.cos(angle), cy + size * math.sin(angle)))
    draw.polygon(points, fill=fill)


def generate_image(label, add_noise=True, rng=None):
    """Generate one synthetic RGB image (as a uint8 numpy array) for a class label."""
    rng = rng or random
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    cx = rng.randint(22, IMG_SIZE - 22)
    cy = rng.randint(22, IMG_SIZE - 22)
    size = rng.randint(14, 22)
    rotation = rng.uniform(0, math.pi)
    color = _random_color()

    if label == "circle":
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
    elif label == "square":
        half = size
        pts = [(-half, -half), (half, -half), (half, half), (-half, half)]
        rpts = [(cx + x * math.cos(rotation) - y * math.sin(rotation),
                  cy + x * math.sin(rotation) + y * math.cos(rotation)) for x, y in pts]
        draw.polygon(rpts, fill=color)
    elif label == "triangle":
        _draw_triangle(draw, cx, cy, size + 6, rotation, color)
    elif label == "star":
        _draw_star(draw, cx, cy, size + 6, (size + 6) * 0.45, rotation, color)

    arr = np.array(img).astype(np.float32)
    if add_noise:
        noise = np.random.normal(0, 12, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
    return arr.astype(np.uint8)


def generate_dataset(n_per_class, seed=None, add_noise=True):
    """Returns (images [N,H,W,3] uint8, labels [N] int)."""
    rng = random.Random(seed)
    images, labels = [], []
    for class_idx, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            images.append(generate_image(cls, add_noise=add_noise, rng=rng))
            labels.append(class_idx)
    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)
    # shuffle
    perm = rng.sample(range(len(labels)), len(labels))
    return images[perm], labels[perm]


if __name__ == "__main__":
    imgs, labels = generate_dataset(5, seed=0)
    print("Generated", imgs.shape, "images, label counts:", np.bincount(labels))
