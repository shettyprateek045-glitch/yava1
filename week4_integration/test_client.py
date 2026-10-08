"""
Week 4 - Integration Testing
-------------------------------
Runs real end-to-end system tests against the FastAPI service (api.py) using
FastAPI's TestClient (in-process ASGI transport - no network socket needed,
but it exercises the full HTTP request/response/validation path exactly as a
real client would). Every result below is captured from an actual run.
"""
import io
import os
import sys
import json

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2_model_training"))
from dataset import generate_image, CLASSES  # noqa: E402
from api import app  # noqa: E402

client = TestClient(app)


def img_to_bytes(arr):
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf


def main():
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("Week 4: Integration Testing - FastAPI Shape Classifier Service")
    log("=" * 70)

    results = {"health_check": None, "prediction_cases": [], "error_handling": []}

    # --- Test 1: health check ---
    log("\n[Test 1] GET /health")
    r = client.get("/health")
    ok = r.status_code == 200 and r.json().get("status") == "ok"
    log(f"  status_code={r.status_code}  body={r.json()}  -> {'PASS' if ok else 'FAIL'}")
    results["health_check"] = {"status_code": r.status_code, "body": r.json(), "pass": ok}

    # --- Test 2: prediction correctness across N synthetic samples per class ---
    log("\n[Test 2] POST /predict - synthetic samples per class")
    n_per_class = 15
    rng = np.random.RandomState(123)
    correct = 0
    total = 0
    import random as _random
    py_rng = _random.Random(7)
    for cls in CLASSES:
        for i in range(n_per_class):
            img_arr = generate_image(cls, add_noise=True, rng=py_rng)
            files = {"file": ("test.png", img_to_bytes(img_arr), "image/png")}
            r = client.post("/predict", files=files)
            body = r.json()
            passed = r.status_code == 200 and body.get("predicted_class") == cls
            total += 1
            correct += int(passed)
            results["prediction_cases"].append({
                "expected": cls, "predicted": body.get("predicted_class"),
                "confidence": body.get("confidence"), "status_code": r.status_code, "pass": passed,
            })
    acc = correct / total
    log(f"  {correct}/{total} correct predictions via the live API  (accuracy={acc:.3f})")

    # --- Test 3: error handling - invalid (non-image) file ---
    log("\n[Test 3] POST /predict with a non-image file (expects HTTP 400)")
    bad_file = {"file": ("not_an_image.txt", io.BytesIO(b"this is not an image"), "text/plain")}
    r = client.post("/predict", files=bad_file)
    ok3 = r.status_code == 400
    log(f"  status_code={r.status_code}  detail={r.json().get('detail')}  -> {'PASS' if ok3 else 'FAIL'}")
    results["error_handling"].append({"case": "non_image_upload", "status_code": r.status_code, "pass": ok3})

    # --- Test 4: error handling - corrupted image bytes with image content-type ---
    log("\n[Test 4] POST /predict with corrupted image bytes (expects HTTP 400)")
    corrupt_file = {"file": ("corrupt.png", io.BytesIO(b"\x89PNG\r\n not a real png"), "image/png")}
    r = client.post("/predict", files=corrupt_file)
    ok4 = r.status_code == 400
    log(f"  status_code={r.status_code}  detail={r.json().get('detail')}  -> {'PASS' if ok4 else 'FAIL'}")
    results["error_handling"].append({"case": "corrupted_image_bytes", "status_code": r.status_code, "pass": ok4})

    # --- Summary ---
    log("\n" + "=" * 70)
    total_tests = 1 + len(results["prediction_cases"]) + len(results["error_handling"])
    total_pass = (int(ok) + sum(c["pass"] for c in results["prediction_cases"]) +
                  sum(c["pass"] for c in results["error_handling"]))
    log(f"TOTAL: {total_pass}/{total_tests} test cases passed")
    log(f"Live API prediction accuracy on {total} synthetic samples: {acc:.3f}")
    log("=" * 70)

    results["summary"] = {"total_tests": total_tests, "total_pass": total_pass,
                           "api_prediction_accuracy": acc}
    with open("integration_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("integration_test_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    print("\nSaved: integration_test_results.json, integration_test_log.txt")


if __name__ == "__main__":
    main()
