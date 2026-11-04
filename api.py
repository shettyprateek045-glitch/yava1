"""
Advanced Product API — Shape Vision Studio Control Center
==========================================================
Super-powers the Computer Vision Shape Classification Pipeline with a production-grade
REST API, real-time explainability (XAI Saliency Maps & Feature Map Activations),
multi-model optimization benchmarking (FP32 Baseline vs 40% Pruned vs Int8 Quantized),
robustness perturbation testing, and an interactive web console.

Endpoints:
  GET  /                        -> Interactive Vision Studio Web Console
  GET  /health                  -> Service health & loaded model variants
  POST /api/predict             -> Predict shape from image file or base64 canvas
  POST /api/explain             -> XAI Saliency Map & intermediate Conv feature maps
  POST /api/compare             -> Multi-model comparative inference (FP32 vs Pruned vs Int8)
  POST /api/robustness          -> Stress testing under noise & rotation perturbations
  GET  /api/sample/{shape}      -> Generate fresh synthetic test shape
  GET  /api/benchmark           -> Benchmark metrics & memory trade-offs
  GET  /api/history             -> SQLite audit trail of recent predictions
  DELETE /api/history           -> Clear prediction audit history
  GET  /api/stats               -> Real-time inference stats & class distribution
"""

import base64
import io
import json
import logging
import math
import os
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

# Import existing core modules
WEEK2_DIR = Path(__file__).resolve().parent / "week2_model_training"
WEEK3_DIR = Path(__file__).resolve().parent / "week3_optimization"
sys.path.insert(0, str(WEEK2_DIR))
sys.path.insert(0, str(WEEK3_DIR))

from dataset import CLASSES, IMG_SIZE, generate_image, generate_dataset
from model import ShapeCNN
from optimize import load_model as base_load_model, quantize_weights_int8, prune_weights, model_weight_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("shape_vision_api")

TARGET_SIZE = 32
DB_PATH = Path(__file__).resolve().parent / "data" / "predictions.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"


# =============================================================================
# 1) Database Audit Trail
# =============================================================================

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_variant TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                probabilities TEXT NOT NULL,
                source TEXT NOT NULL
            )
        """)
        conn.commit()

# Ensure database and data directory are initialized immediately
init_db()


def log_prediction(predicted_class: str, confidence: float, model_variant: str,
                   latency_ms: float, probabilities: dict, source: str = "upload"):
    try:
        init_db()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO prediction_history (timestamp, predicted_class, confidence, model_variant, latency_ms, probabilities, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                predicted_class,
                float(confidence),
                model_variant,
                float(latency_ms),
                json.dumps(probabilities),
                source
            ))
            conn.commit()
    except Exception as e:
        logger.warning("Failed to log prediction to DB: %s", e)


# =============================================================================
# 2) Model Manager (FP32 Baseline, Pruned, Int8 Quantized)
# =============================================================================

class ModelManager:
    def __init__(self):
        self.models: Dict[str, ShapeCNN] = {}
        self.metadata: Dict[str, dict] = {}

    def initialize(self):
        logger.info("Initializing ShapeCNN model variants...")
        # 1. Baseline FP32
        baseline = base_load_model()
        self.models["baseline"] = baseline
        self.metadata["baseline"] = {
            "name": "FP32 Baseline",
            "weight_bytes": model_weight_bytes(baseline, 4),
            "size_kb": round(model_weight_bytes(baseline, 4) / 1024, 2),
            "parameters": baseline.num_params(),
            "sparsity_pct": 0.0,
            "description": "Standard full-precision floating-point weights"
        }

        # 2. Pruned (40% magnitude pruning)
        pruned = base_load_model()
        prune_weights(pruned, amount=0.40)
        self.models["pruned"] = pruned
        self.metadata["pruned"] = {
            "name": "40% Magnitude Pruned",
            "weight_bytes": model_weight_bytes(pruned, 4),
            "size_kb": round(model_weight_bytes(pruned, 4) / 1024, 2),
            "parameters": pruned.num_params(),
            "sparsity_pct": 40.0,
            "description": "40% smallest magnitude weights zeroed out per layer"
        }

        # 3. Quantized Int8
        quantized = base_load_model()
        int8_bytes = quantize_weights_int8(quantized)
        self.models["quantized"] = quantized
        self.metadata["quantized"] = {
            "name": "Int8 Post-Training Quantized",
            "weight_bytes": int8_bytes,
            "size_kb": round(int8_bytes / 1024, 2),
            "parameters": quantized.num_params(),
            "sparsity_pct": 0.0,
            "description": "Symmetric post-training int8 weight quantization (74.9% memory savings)"
        }
        logger.info("Model variants initialized successfully.")

    def get_model(self, variant: str = "quantized") -> ShapeCNN:
        variant = variant.lower()
        if variant not in self.models:
            variant = "quantized"
        return self.models[variant]


manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    manager.initialize()
    yield


app = FastAPI(
    title="ShapeVision Studio API",
    description="Enterprise Computer Vision Shape Classification & Model Optimization Suite",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 3) Image Processing & Computer Vision Explainability (XAI)
# =============================================================================

def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """Preprocess PIL Image into normalized (1, 3, 32, 32) float32 array."""
    img = img.convert("RGB").resize((TARGET_SIZE, TARGET_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]  # (1, 3, 32, 32)
    return arr


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def array_to_base64_png(arr: np.ndarray) -> str:
    """Converts a uint8 (H, W, 3) or (H, W) numpy array to base64 data URI."""
    if arr.ndim == 2:
        im = Image.fromarray(arr, mode="L")
    else:
        im = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def apply_colormap_jet(val_2d: np.ndarray) -> np.ndarray:
    """Simple, fast vector colormap mapping normalized [0, 1] to RGB (approx JET/Turbo)."""
    val = np.clip(val_2d, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * val - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * val - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * val - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def compute_saliency_map(net: ShapeCNN, x: np.ndarray, target_class: Optional[int] = None) -> np.ndarray:
    """
    Computes Saliency Map via backpropagating target class gradient to input pixels.
    Uses zero learning rate and momentum to keep weights strictly pristine.
    """
    logits = net.forward(x)
    pred_class = int(logits.argmax(axis=1)[0]) if target_class is None else target_class

    dlogits = np.zeros_like(logits)
    dlogits[0, pred_class] = 1.0

    # Backprop with lr=0 ensures zero weight update
    dx = net.backward(dlogits, lr=0.0, momentum=0.0)

    # Saliency is maximum absolute gradient across RGB color channels
    saliency = np.max(np.abs(dx[0]), axis=0)  # (32, 32)
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)
    return saliency


def extract_feature_maps(net: ShapeCNN, x: np.ndarray) -> Dict[str, List[str]]:
    """
    Extracts intermediate activation feature maps for Conv1 and Conv2 layers.
    Returns base64 PNG data URLs for visual inspection.
    """
    # Conv1 + ReLU
    c1 = net.relu1.forward(net.conv1.forward(x))  # (1, 8, 32, 32)
    # Pool1 + Conv2 + ReLU
    p1 = net.pool1.forward(c1)
    c2 = net.relu2.forward(net.conv2.forward(p1))  # (1, 16, 16, 16)

    def process_channel_maps(tensor):
        maps = []
        num_channels = tensor.shape[1]
        for c in range(num_channels):
            chan = tensor[0, c]
            cmin, cmax = chan.min(), chan.max()
            if cmax > cmin:
                norm = ((chan - cmin) / (cmax - cmin) * 255.0).astype(np.uint8)
            else:
                norm = np.zeros_like(chan, dtype=np.uint8)
            # Upscale for clear display
            im = Image.fromarray(norm, mode="L").resize((64, 64), Image.Resampling.NEAREST)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            maps.append(f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}")
        return maps

    return {
        "conv1_feature_maps": process_channel_maps(c1),
        "conv2_feature_maps": process_channel_maps(c2)
    }


def parse_image_input(file: Optional[UploadFile], base64_data: Optional[str]) -> Image.Image:
    """Parses image either from multipart UploadFile or base64 data string."""
    if file is not None:
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        try:
            contents = file.file.read()
            return Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode image file")
    elif base64_data:
        try:
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(base64_data)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode base64 image data")
    else:
        raise HTTPException(status_code=400, detail="No image provided (upload a file or provide image_base64)")


# =============================================================================
# 4) Pydantic Models
# =============================================================================

class PredictRequest(BaseModel):
    image_base64: Optional[str] = None
    model_variant: Optional[str] = "quantized"
    source: Optional[str] = "canvas"


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    model_variant: str
    latency_ms: float
    model_metadata: dict


class ExplainRequest(BaseModel):
    image_base64: Optional[str] = None
    model_variant: Optional[str] = "quantized"
    target_class: Optional[str] = None


class RobustnessRequest(BaseModel):
    image_base64: Optional[str] = None
    model_variant: Optional[str] = "quantized"


# =============================================================================
# 5) API Routes
# =============================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ShapeVision Studio API",
        "version": "2.0.0",
        "models_loaded": {
            "baseline": "baseline" in manager.models,
            "pruned": "pruned" in manager.models,
            "quantized": "quantized" in manager.models,
        },
        "target_image_size": [TARGET_SIZE, TARGET_SIZE],
        "classes": CLASSES
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_compat(file: UploadFile = File(...)):
    """Legacy compatibility endpoint matching Week 4 API signature."""
    return await handle_prediction(file=file, base64_data=None, variant="quantized", source="legacy_upload")


@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
    variant: Optional[str] = Query("quantized")
):
    """
    Predicts geometric shape from uploaded image or JSON payload with base64 data.
    Supports model variants: 'quantized', 'baseline', 'pruned'.
    """
    base64_str = None
    source = "upload" if file else "canvas"

    if file is None:
        try:
            body = await request.json()
            base64_str = body.get("image_base64")
            variant = body.get("model_variant", variant)
            source = body.get("source", "canvas")
        except Exception:
            pass

    return await handle_prediction(file=file, base64_data=base64_str, variant=variant, source=source)


async def handle_prediction(file: Optional[UploadFile], base64_data: Optional[str],
                            variant: str, source: str) -> PredictionResponse:
    img = parse_image_input(file, base64_data)
    arr = preprocess_pil_image(img)

    net = manager.get_model(variant)
    t0 = time.perf_counter()
    logits = net.forward(arr)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = softmax(logits)[0]
    pred_idx = int(probs.argmax())
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])
    prob_dict = {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}

    log_prediction(pred_class, confidence, variant, latency_ms, prob_dict, source)

    return PredictionResponse(
        predicted_class=pred_class,
        confidence=confidence,
        probabilities=prob_dict,
        model_variant=variant,
        latency_ms=round(latency_ms, 3),
        model_metadata=manager.metadata.get(variant, {})
    )


@app.post("/api/explain")
async def api_explain(req: ExplainRequest):
    """
    Explainability (XAI): Returns Saliency Heatmap and intermediate Conv layer feature maps.
    """
    img = parse_image_input(None, req.image_base64)
    arr = preprocess_pil_image(img)
    net = manager.get_model(req.model_variant or "quantized")

    # Target class index
    target_idx = None
    if req.target_class and req.target_class in CLASSES:
        target_idx = CLASSES.index(req.target_class)

    # 1. Saliency Map
    saliency = compute_saliency_map(net, arr, target_class=target_idx)
    # Upscale saliency map to 128x128 for clear viewing
    sal_img = Image.fromarray((saliency * 255).astype(np.uint8), mode="L").resize((128, 128), Image.Resampling.BILINEAR)
    sal_norm = np.array(sal_img, dtype=np.float32) / 255.0
    saliency_colormap = apply_colormap_jet(sal_norm)
    saliency_b64 = array_to_base64_png(saliency_colormap)

    # 2. Intermediate Feature Maps
    fmaps = extract_feature_maps(net, arr)

    return {
        "status": "success",
        "saliency_heatmap": saliency_b64,
        "feature_maps": fmaps,
        "feature_map_counts": {
            "conv1": len(fmaps["conv1_feature_maps"]),
            "conv2": len(fmaps["conv2_feature_maps"])
        },
        "target_class": CLASSES[target_idx] if target_idx is not None else "auto_detected"
    }


@app.post("/api/compare")
async def api_compare(req: PredictRequest):
    """
    Executes Baseline FP32, 40% Pruned, and Int8 Quantized models side-by-side
    on the exact same input image, returning latency, size, sparsity, and confidence.
    """
    img = parse_image_input(None, req.image_base64)
    arr = preprocess_pil_image(img)

    results = {}
    for var in ["baseline", "pruned", "quantized"]:
        net = manager.get_model(var)
        times = []
        for _ in range(5):  # multiple passes for accurate timing
            t0 = time.perf_counter()
            logits = net.forward(arr)
            times.append(time.perf_counter() - t0)

        probs = softmax(logits)[0]
        p_idx = int(probs.argmax())
        results[var] = {
            "variant": var,
            "metadata": manager.metadata[var],
            "predicted_class": CLASSES[p_idx],
            "confidence": round(float(probs[p_idx]), 4),
            "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))},
            "latency_ms": round(float(np.mean(times) * 1000.0), 3)
        }

    base_bytes = results["baseline"]["metadata"]["weight_bytes"]
    quant_bytes = results["quantized"]["metadata"]["weight_bytes"]
    compression_ratio = round(base_bytes / quant_bytes, 2)
    memory_savings_pct = round((1.0 - quant_bytes / base_bytes) * 100, 1)

    return {
        "comparisons": results,
        "compression_ratio": compression_ratio,
        "memory_savings_pct": memory_savings_pct,
        "speedup_vs_baseline": round(results["baseline"]["latency_ms"] / max(results["quantized"]["latency_ms"], 1e-4), 2)
    }


@app.post("/api/robustness")
async def api_robustness(req: RobustnessRequest):
    """
    Stress-tests the model against additive Gaussian noise and rotational perturbations.
    """
    img = parse_image_input(None, req.image_base64)
    net = manager.get_model(req.model_variant or "quantized")

    # 1. Noise Stress Test
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    noise_results = []
    base_arr = preprocess_pil_image(img)[0]  # (3, 32, 32)

    for sigma in noise_levels:
        noisy = base_arr.copy()
        if sigma > 0:
            noise = np.random.normal(0, sigma, noisy.shape).astype(np.float32)
            noisy = np.clip(noisy + noise, 0.0, 1.0)
        logits = net.forward(noisy[None, ...])
        probs = softmax(logits)[0]
        p_idx = int(probs.argmax())
        noise_results.append({
            "noise_level": sigma,
            "predicted_class": CLASSES[p_idx],
            "confidence": round(float(probs[p_idx]), 4)
        })

    # 2. Rotation Stress Test
    rotations = [0, 45, 90, 135, 180, 225, 270, 315]
    rotation_results = []
    for deg in rotations:
        rotated_img = img.rotate(deg, resample=Image.Resampling.BILINEAR, fillcolor=(20, 20, 20))
        arr = preprocess_pil_image(rotated_img)
        logits = net.forward(arr)
        probs = softmax(logits)[0]
        p_idx = int(probs.argmax())
        rotation_results.append({
            "rotation_deg": deg,
            "predicted_class": CLASSES[p_idx],
            "confidence": round(float(probs[p_idx]), 4)
        })

    return {
        "noise_analysis": noise_results,
        "rotation_analysis": rotation_results
    }


@app.get("/api/sample/{shape_name}")
def api_sample(shape_name: str):
    """Generates a freshly rendered synthetic shape (circle, square, triangle, star, or random)."""
    shape_name = shape_name.lower()
    if shape_name == "random" or shape_name not in CLASSES:
        import random
        shape_name = random.choice(CLASSES)

    arr = generate_image(shape_name, add_noise=True)
    b64_url = array_to_base64_png(arr)
    return {
        "shape": shape_name,
        "image_base64": b64_url,
        "size": [IMG_SIZE, IMG_SIZE]
    }


@app.get("/api/benchmark")
def api_benchmark():
    """Returns official Week 3 model optimization benchmarks."""
    benchmark_file = WEEK3_DIR / "benchmark_results.json"
    if benchmark_file.exists():
        with open(benchmark_file, "r") as f:
            data = json.load(f)
        return data
    return {"message": "Benchmark results not yet generated."}


@app.get("/api/history")
def api_history(limit: int = 50):
    """Retrieves recent prediction audit records."""
    try:
        init_db()
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = [dict(r) for r in cursor.fetchall()]
            for r in rows:
                try:
                    r["probabilities"] = json.loads(r["probabilities"])
                except Exception:
                    pass
            return {"history": rows, "count": len(rows)}
    except Exception as e:
        return {"history": [], "count": 0, "error": str(e)}


@app.delete("/api/history")
def api_clear_history():
    """Clears prediction audit history."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM prediction_history")
        conn.commit()
    return {"status": "cleared"}


@app.get("/api/stats")
def api_stats():
    """Returns aggregate stats on API usage and predictions."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            total = conn.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0]
            avg_lat = conn.execute("SELECT AVG(latency_ms) FROM prediction_history").fetchone()[0] or 0.0
            cursor = conn.execute("SELECT predicted_class, COUNT(*) FROM prediction_history GROUP BY predicted_class")
            dist = {row[0]: row[1] for row in cursor.fetchall()}
            return {
                "total_inferences": total,
                "average_latency_ms": round(float(avg_lat), 2),
                "class_distribution": dist
            }
    except Exception as e:
        return {"total_inferences": 0, "average_latency_ms": 0.0, "class_distribution": {}}


# Mount static directory for the console UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ShapeVision Studio API is live. Visit /docs for OpenAPI specs."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
