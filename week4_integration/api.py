"""
Week 4 - Integration of Computer Vision Solutions
-----------------------------------------------------
FastAPI shape classifier service supporting both the standalone Week 4 integration
contract and the full ShapeVision Studio console.

Endpoints:
  GET  /         -> Serves ShapeVision Studio interactive console
  GET  /health   -> liveness check
  POST /predict  -> accepts an uploaded image file, returns predicted class
  (Also mounts full /api/* suite from root api.py for unified capability)
"""

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week2_model_training"))
sys.path.insert(0, str(ROOT / "week3_optimization"))

# Import full production app from root api.py
try:
    from api import app, get_model, manager, PredictionResponse, handle_prediction, CLASSES, TARGET_SIZE
except Exception:
    import numpy as np
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel
    from PIL import Image
    from dataset import CLASSES
    from model import ShapeCNN
    from optimize import load_model, quantize_weights_int8

    def softmax(logits):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    TARGET_SIZE = 32
    app = FastAPI(title="Shape Classifier API", version="2.0.0")
    _model = None

    def get_model():
        global _model
        if _model is None:
            net = load_model()
            quantize_weights_int8(net)
            _model = net
        return _model

    class PredictionResponse(BaseModel):
        predicted_class: str
        confidence: float
        probabilities: dict

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": _model is not None}

    @app.get("/")
    @app.get("/index.html")
    @app.get("/dashboard")
    def index():
        index_file = ROOT / "index.html"
        if not index_file.exists():
            index_file = ROOT / "static" / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "ShapeVision Studio API is live. Visit /docs"}

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
        arr = arr.transpose(2, 0, 1)[None, ...]

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
