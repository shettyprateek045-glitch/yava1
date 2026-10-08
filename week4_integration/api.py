"""
Week 4 - Integration of Computer Vision Solutions
-----------------------------------------------------
A small FastAPI service that wraps the Week 2 CNN (with the Week 3 int8
quantization applied, since that variant kept 91% accuracy at ~25% of the
original weight size - the best accuracy/footprint trade-off measured).

Endpoints:
  GET  /health   -> liveness check
  POST /predict  -> accepts an uploaded image file, returns the predicted
                     shape class, confidence, and full class probabilities

This file is the "integration" artifact: it is what a surveillance,
quality-control or AR front-end would call over a REST API rather than
importing the model code directly.
"""
import io
import os
import sys

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2_model_training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3_optimization"))
from dataset import CLASSES  # noqa: E402
from model import ShapeCNN  # noqa: E402
from optimize import load_model, quantize_weights_int8  # noqa: E402


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

TARGET_SIZE = 32

app = FastAPI(title="Shape Classifier API", version="1.0.0")

_model = None  # loaded lazily so the test client can import this module without a checkpoint present


def get_model():
    global _model
    if _model is None:
        net = load_model()
        quantize_weights_int8(net)  # deploy the optimized (Week 3) variant
        _model = net
    return _model


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    img = img.resize((TARGET_SIZE, TARGET_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]  # (1, 3, 32, 32)

    net = get_model()
    logits = net.forward(arr)
    probs = softmax(logits)[0]
    pred_idx = int(probs.argmax())

    return PredictionResponse(
        predicted_class=CLASSES[pred_idx],
        confidence=float(probs[pred_idx]),
        probabilities={CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
