"""
Dashboard End-to-End System Verification
=========================================
Tests all frontend-connected endpoints and features:
1. GET / — serves dashboard HTML
2. GET /health — liveness check & model statuses
3. GET /api/sample/{shape} — synthetic shape generation
4. POST /api/predict — canvas & file classification
5. POST /api/explain — saliency heatmap & 24 Conv feature maps
6. POST /api/compare — multi-model comparative scoring
7. POST /api/robustness — noise and rotation perturbation sweeps
8. GET /api/history & GET /api/stats — SQLite audit logs and metrics
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api import app, manager

manager.initialize()
client = TestClient(app)


def test_dashboard_full_flow():
    print("=== 1. Testing GET / (Dashboard HTML) ===")
    r = client.get("/")
    assert r.status_code == 200, f"Status {r.status_code}"
    assert "ShapeVision Studio" in r.text
    assert "drawing-canvas" in r.text
    print(f"GET / SUCCESS: HTML dashboard loaded ({len(r.text)} bytes)")

    print("\n=== 2. Testing GET /health ===")
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["models_loaded"]["quantized"] is True
    assert data["models_loaded"]["baseline"] is True
    assert data["models_loaded"]["pruned"] is True
    print(f"GET /health SUCCESS: {data['models_loaded']}")

    print("\n=== 3. Testing GET /api/sample/circle ===")
    r = client.get("/api/sample/circle")
    assert r.status_code == 200
    sample_data = r.json()
    assert sample_data["shape"] == "circle"
    assert sample_data["image_base64"].startswith("data:image/png;base64,")
    print("GET /api/sample SUCCESS: circle sample generated")

    print("\n=== 4. Testing POST /api/predict (Interactive Studio Classification) ===")
    payload = {
        "image_base64": sample_data["image_base64"],
        "model_variant": "quantized",
        "source": "studio_verification"
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    pred_data = r.json()
    conf_pct = pred_data["confidence"] * 100
    print(f"POST /api/predict SUCCESS: predicted='{pred_data['predicted_class']}', confidence={conf_pct:.1f}%, latency={pred_data['latency_ms']:.2f}ms")

    print("\n=== 5. Testing POST /api/explain (XAI Saliency & Conv Features) ===")
    r = client.post("/api/explain", json={"image_base64": sample_data["image_base64"], "model_variant": "quantized"})
    assert r.status_code == 200
    explain_data = r.json()
    assert explain_data["status"] == "success"
    assert explain_data["saliency_heatmap"].startswith("data:image/png;base64,")
    c1_count = len(explain_data["feature_maps"]["conv1_feature_maps"])
    c2_count = len(explain_data["feature_maps"]["conv2_feature_maps"])
    assert c1_count == 8
    assert c2_count == 16
    print(f"POST /api/explain SUCCESS: Saliency heatmap generated; extracted {c1_count} Conv1 and {c2_count} Conv2 feature maps")

    print("\n=== 6. Testing POST /api/compare (Multi-Model 3-Variant Benchmarking) ===")
    r = client.post("/api/compare", json={"image_base64": sample_data["image_base64"]})
    assert r.status_code == 200
    comp_data = r.json()
    assert "baseline" in comp_data["comparisons"]
    assert "pruned" in comp_data["comparisons"]
    assert "quantized" in comp_data["comparisons"]
    print(f"POST /api/compare SUCCESS: Compression ratio={comp_data['compression_ratio']}x, Memory savings={comp_data['memory_savings_pct']}%")

    print("\n=== 7. Testing POST /api/robustness (Stress Testing Sweeps) ===")
    r = client.post("/api/robustness", json={"image_base64": sample_data["image_base64"], "model_variant": "quantized"})
    assert r.status_code == 200
    rob_data = r.json()
    assert len(rob_data["noise_analysis"]) == 7
    assert len(rob_data["rotation_analysis"]) == 8
    print(f"POST /api/robustness SUCCESS: {len(rob_data['noise_analysis'])} noise levels & {len(rob_data['rotation_analysis'])} rotation angles evaluated")

    print("\n=== 8. Testing GET /api/history & GET /api/stats (Audit Trail) ===")
    r_hist = client.get("/api/history")
    assert r_hist.status_code == 200
    hist_data = r_hist.json()
    assert hist_data["count"] > 0
    print(f"GET /api/history SUCCESS: {hist_data['count']} audit log records retrieved")

    r_stats = client.get("/api/stats")
    assert r_stats.status_code == 200
    stats_data = r_stats.json()
    print(f"GET /api/stats SUCCESS: total_inferences={stats_data['total_inferences']}, avg_latency={stats_data['average_latency_ms']}ms, distribution={stats_data['class_distribution']}")

    print("\n>>> ALL DASHBOARD SYSTEMS AND APIS ARE FULLY VERIFIED AND WORKING PROPERLY! <<<")


if __name__ == "__main__":
    test_dashboard_full_flow()
