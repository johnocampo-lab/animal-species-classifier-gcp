"""Cloud Run inference service for the cat/dog image classifier."""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/image_classifier.h5")
IMG_SIZE = (256, 256)
# Matches image_dataset_from_directory alphabetical labels: cats=0, dogs=1
CLASS_NAMES = ("cat", "dog")

model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    model = load_model(MODEL_PATH)
    yield


app = FastAPI(title="Cat vs Dog Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


def preprocess(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        batch = preprocess(raw)
    except Exception as exc:  # noqa: BLE001 - return client-safe error
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    score = float(model.predict(batch, verbose=0)[0][0])
    label = CLASS_NAMES[1] if score > 0.5 else CLASS_NAMES[0]
    confidence = score if label == "dog" else 1.0 - score
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "raw_score": round(score, 4),
    }
