"""
Streamlit Interactive Dashboard — ShapeVision Studio
=====================================================
Alternative interactive console for the Shape Classification Pipeline, providing
live model predictions, explainability heatmaps (XAI), and benchmark comparisons.

Run:
    streamlit run app.py
"""

import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import streamlit as st
except ImportError:
    st = None

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week2_model_training"))
sys.path.insert(0, str(ROOT / "week3_optimization"))

from dataset import CLASSES, generate_image
from optimize import load_model, quantize_weights_int8, prune_weights, model_weight_bytes
from api import compute_saliency_map, apply_colormap_jet, extract_feature_maps, softmax, preprocess_pil_image


def main():
    if st is None:
        print("Streamlit not installed. Install with `pip install streamlit` or use the FastAPI console at http://localhost:8000")
        return

    st.set_page_config(page_title="ShapeVision Studio", page_icon="🔍", layout="wide")
    st.title("🔍 ShapeVision Studio — CV Optimization & Retention Console")
    st.markdown("### 4-Week Computer Vision Shape Classification & Model Optimization Pipeline")

    # Sidebar: Model Selection
    st.sidebar.header("⚙️ Model Configuration")
    variant = st.sidebar.selectbox(
        "Select Model Variant:",
        ["Int8 Quantized (Recommended, 65.9 KB)", "FP32 Baseline (262.7 KB)", "40% Pruned (67.2k params)"]
    )

    # Initialize model
    if "model_variant" not in st.session_state or st.session_state.get("current_variant") != variant:
        net = load_model()
        if "Quantized" in variant:
            quantize_weights_int8(net)
        elif "Pruned" in variant:
            prune_weights(net, 0.40)
        st.session_state["model"] = net
        st.session_state["current_variant"] = variant

    net = st.session_state["model"]

    # Input Mode
    st.subheader("1. Input Shape Image")
    col1, col2 = st.columns([1, 1])

    with col1:
        input_mode = st.radio("Choose Input Method:", ["Generate Synthetic Preset", "Upload Image File"])
        active_img = None

        if input_mode == "Generate Synthetic Preset":
            shape_choice = st.selectbox("Select Geometric Shape:", CLASSES)
            if st.button("Generate Fresh Shape"):
                arr = generate_image(shape_choice, add_noise=True)
                active_img = Image.fromarray(arr)
                st.session_state["active_img"] = active_img
        else:
            uploaded = st.file_uploader("Upload an image (PNG, JPG):", type=["png", "jpg", "jpeg"])
            if uploaded:
                active_img = Image.open(uploaded).convert("RGB")
                st.session_state["active_img"] = active_img

        if "active_img" in st.session_state:
            st.image(st.session_state["active_img"], caption="Active Input Image", width=160)

    with col2:
        st.subheader("2. Real-Time Classification")
        if "active_img" in st.session_state:
            img = st.session_state["active_img"]
            arr = preprocess_pil_image(img)
            logits = net.forward(arr)
            probs = softmax(logits)[0]
            pred_idx = int(probs.argmax())

            st.success(f"**Predicted Class:** `{CLASSES[pred_idx].upper()}` ({probs[pred_idx]*100:.1f}% confidence)")

            # Probability Bars
            for i, cls in enumerate(CLASSES):
                st.progress(float(probs[i]), text=f"{cls.capitalize()}: {probs[i]*100:.1f}%")

            # XAI Saliency
            st.subheader("3. Explainability (Saliency Map)")
            saliency = compute_saliency_map(net, arr)
            sal_norm = Image.fromarray((saliency * 255).astype(np.uint8), mode="L").resize((128, 128), Image.Resampling.BILINEAR)
            sal_colored = apply_colormap_jet(np.array(sal_norm, dtype=np.float32) / 255.0)
            st.image(sal_colored, caption="Pixel Gradient Saliency Heatmap", width=160)
        else:
            st.info("Select or upload an image on the left to begin inference.")


if __name__ == "__main__":
    main()
