"""
Run the Complete Four-Week Computer Vision Shape Classification Pipeline
========================================================================
Reproduces all supporting technical artifacts, verifies model training checkpoints,
executes magnitude pruning & int8 quantization benchmarks, and runs integration tests.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STEPS = [
    {
        "week": "Week 1",
        "title": "Algorithm Design & Diagram Generation",
        "cmd": [PYTHON, str(ROOT / "week1_algorithm_design" / "make_diagrams.py")],
        "cwd": ROOT / "week1_algorithm_design"
    },
    {
        "week": "Week 2",
        "title": "Dataset Generation & Verification",
        "cmd": [PYTHON, str(ROOT / "week2_model_training" / "dataset.py")],
        "cwd": ROOT / "week2_model_training"
    },
    {
        "week": "Week 3",
        "title": "Magnitude Pruning & Int8 Quantization Optimization Benchmarks",
        "cmd": [PYTHON, str(ROOT / "week3_optimization" / "optimize.py")],
        "cwd": ROOT / "week3_optimization"
    },
    {
        "week": "Week 4",
        "title": "FastAPI Integration Test Suite",
        "cmd": [PYTHON, str(ROOT / "week4_integration" / "test_client.py")],
        "cwd": ROOT / "week4_integration"
    },
    {
        "week": "PyTest",
        "title": "Automated Unit & System Test Suite",
        "cmd": [PYTHON, "-m", "pytest", "tests/", "-v", "--tb=short"],
        "cwd": ROOT
    }
]


def print_banner(text):
    width = 75
    print("\n" + "=" * width)
    print(f" {text}".center(width))
    print("=" * width + "\n")


def main():
    start_time = time.time()
    print_banner("SHAPE VISION STUDIO — 4-WEEK PIPELINE EXECUTION")
    
    passed = 0
    total = len(STEPS)

    for i, step in enumerate(STEPS, 1):
        print(f"\n[{i}/{total}] >>> {step['week']}: {step['title']}")
        print(f"    Working Directory: {step['cwd']}")
        print(f"    Command: {' '.join(step['cmd'])}")
        t0 = time.time()
        try:
            res = subprocess.run(step["cmd"], cwd=step["cwd"], check=True)
            elapsed = time.time() - t0
            print(f"    SUCCESS [Elapsed: {elapsed:.2f}s]")
            passed += 1
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            print(f"    FAILED [Code: {e.returncode}, Elapsed: {elapsed:.2f}s]")
            return 1

    total_time = time.time() - start_time
    print_banner(f"ALL {passed}/{total} PIPELINE STAGES PASSED ({total_time:.2f}s)")
    print("Artifacts summary:")
    print("  - Week 1: diagrams/ (algorithm_flowchart.png, data_preprocessing_flowchart.png)")
    print("  - Week 2: model_checkpoint.npz, training_curve.png, training_history.json")
    print("  - Week 3: benchmark_comparison.png, benchmark_results.json, optimization_log.txt")
    print("  - Week 4: integration_test_results.json, integration_test_log.txt")
    print("  - Advanced: api.py, static/index.html, tests/, Dockerfile, docker-compose.yml")
    print("\nLaunch Interactive Console:")
    print(f"  {PYTHON} -m uvicorn api:app --reload --port 8000")
    print("  Then open: http://localhost:8000/\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
