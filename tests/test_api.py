import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api import app, manager, array_to_base64_png
from week2_model_training.dataset import generate_image, CLASSES

# Ensure models and DB are initialized for testing
manager.initialize()
client = TestClient(app)


def get_sample_png_bytes(shape="circle"):
    arr = generate_image(shape, add_noise=False)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_check():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["models_loaded"]["quantized"] is True
    assert data["models_loaded"]["baseline"] is True
    assert data["models_loaded"]["pruned"] is True


def test_predict_file_upload():
    buf = get_sample_png_bytes("circle")
    files = {"file": ("test_circle.png", buf, "image/png")}
    r = client.post("/api/predict?variant=quantized", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_class"] in CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 4
    assert data["model_variant"] == "quantized"
    assert data["latency_ms"] >= 0.0


def test_predict_base64_payload():
    arr = generate_image("square", add_noise=False)
    b64_url = array_to_base64_png(arr)
    payload = {
        "image_base64": b64_url,
        "model_variant": "baseline",
        "source": "pytest"
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_class"] in CLASSES
    assert data["model_variant"] == "baseline"


def test_explain_endpoint():
    arr = generate_image("triangle", add_noise=False)
    b64_url = array_to_base64_png(arr)
    payload = {
        "image_base64": b64_url,
        "model_variant": "quantized"
    }
    r = client.post("/api/explain", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert "saliency_heatmap" in data
    assert data["saliency_heatmap"].startswith("data:image/png;base64,")
    assert data["feature_map_counts"]["conv1"] == 8
    assert data["feature_map_counts"]["conv2"] == 16


def test_compare_endpoint():
    arr = generate_image("star", add_noise=False)
    b64_url = array_to_base64_png(arr)
    payload = {"image_base64": b64_url}
    r = client.post("/api/compare", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "baseline" in data["comparisons"]
    assert "pruned" in data["comparisons"]
    assert "quantized" in data["comparisons"]
    assert data["compression_ratio"] >= 3.5
    assert data["memory_savings_pct"] >= 70.0


def test_robustness_endpoint():
    arr = generate_image("circle", add_noise=False)
    b64_url = array_to_base64_png(arr)
    payload = {"image_base64": b64_url, "model_variant": "quantized"}
    r = client.post("/api/robustness", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["noise_analysis"]) > 3
    assert len(data["rotation_analysis"]) == 8


def test_sample_generation():
    for shape in CLASSES:
        r = client.get(f"/api/sample/{shape}")
        assert r.status_code == 200
        data = r.json()
        assert data["shape"] == shape
        assert data["image_base64"].startswith("data:image/png;base64,")


def test_benchmark_endpoint():
    r = client.get("/api/benchmark")
    assert r.status_code == 200


def test_history_and_stats():
    # History
    r_hist = client.get("/api/history")
    assert r_hist.status_code == 200
    assert "history" in r_hist.json()

    # Stats
    r_stats = client.get("/api/stats")
    assert r_stats.status_code == 200
    assert "total_inferences" in r_stats.json()


def test_error_handling_non_image():
    bad_file = {"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")}
    r = client.post("/api/predict", files=bad_file)
    assert r.status_code == 400
