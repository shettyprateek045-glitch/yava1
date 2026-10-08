import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

PRIMARY = "#2C5F2D"
SECONDARY = "#97BC62"
ACCENT = "#E8A33D"
TEXT = "#1A1D29"


def draw_flowchart(steps, filename, title, box_colors=None, figsize=(5.4, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    n = len(steps)
    box_h = 0.8
    gap = 0.45
    total_h = n * box_h + (n - 1) * gap
    y_top = total_h
    xs = -2.4

    for i, label in enumerate(steps):
        y = y_top - i * (box_h + gap) - box_h
        color = box_colors[i] if box_colors else (PRIMARY if i in (0, n - 1) else "#EFF4EC")
        text_color = "white" if color != "#EFF4EC" else TEXT
        is_terminal = i in (0, n - 1)
        if is_terminal:
            patch = mpatches.FancyBboxPatch((xs, y), 4.8, box_h, boxstyle="round,pad=0.02,rounding_size=0.4",
                                             linewidth=0, facecolor=color)
        else:
            patch = mpatches.FancyBboxPatch((xs, y), 4.8, box_h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                             linewidth=0, facecolor=color)
        ax.add_patch(patch)
        ax.text(xs + 2.4, y + box_h / 2, label, ha="center", va="center", fontsize=10.5,
                color=text_color, fontweight="bold" if is_terminal else "normal", wrap=True)
        if i < n - 1:
            arrow = FancyArrowPatch((xs + 2.4, y), (xs + 2.4, y - gap),
                                     arrowstyle="-|>", mutation_scale=14, color="#6B7280", linewidth=1.3)
            ax.add_patch(arrow)

    ax.set_xlim(xs - 0.6, xs + 5.4)
    ax.set_ylim(-0.3, total_h + 0.6)
    ax.set_title(title, fontsize=13, fontweight="bold", color=PRIMARY, pad=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=160, bbox_inches="tight")
    plt.close()


# Diagram 1: Data preprocessing & augmentation pipeline
pipeline_steps = [
    "Raw Input Image",
    "Resize to fixed\ninput dimensions (32x32)",
    "Pixel normalization\n(scale to [0, 1])",
    "Data augmentation\n(rotation, noise injection)",
    "Train / Validation / Test\nsplit",
    "Model-ready tensor\n(N, C, H, W)",
]
draw_flowchart(pipeline_steps, "diagrams/data_preprocessing_flowchart.png",
                "Data Preprocessing Pipeline")

# Diagram 2: CNN algorithm / architecture flow
algo_steps = [
    "Input Image (32x32x3)",
    "Conv2D (3->8, 3x3) + ReLU",
    "Max Pooling (2x2)",
    "Conv2D (8->16, 3x3) + ReLU",
    "Max Pooling (2x2)",
    "Flatten",
    "Dense (1024->64) + ReLU",
    "Dense (64->4) + Softmax",
    "Predicted Class",
]
draw_flowchart(algo_steps, "diagrams/algorithm_flowchart.png",
                "Proposed CNN Algorithm Flow", figsize=(5.4, 12))

print("Diagrams saved.")
