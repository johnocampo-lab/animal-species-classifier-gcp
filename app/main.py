"""
app/main.py
FastAPI production application for Animal Species Recognition.
Uses ONNX Runtime CPU for low-latency, zero-PyTorch-dependency inference.
"""

import io
import os
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import onnxruntime as ort

app = FastAPI(
    title="Animal Species Classifier",
    description="PMLE Demo: Serverless Animal Species Recognition on Google Cloud Run"
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.onnx")
CLASSES_PATH = os.getenv("CLASSES_PATH", "models/classes.json")

session = None
class_names = []

@app.on_event("startup")
def load_model():
    global session, class_names
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        print(f"Loading ONNX model from {MODEL_PATH}...")
        session = ort.InferenceSession(MODEL_PATH)
        with open(CLASSES_PATH) as f:
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} classes: {class_names}")
    else:
        print("Warning: Model artifacts not found yet. Predictions will return mock until model is trained.")
        class_names = ["bird", "cat", "deer", "dog", "frog", "horse"]

def preprocess_image(image_bytes: bytes, img_size: int = 224) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        # Transpose HWC -> CHW and add batch dimension (1, 3, 224, 224)
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": session is not None, "classes": class_names}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    inp = preprocess_image(content)

    if session is not None:
        input_name = session.get_inputs()[0].name
        raw_outputs = session.run(None, {input_name: inp})[0][0]
        probs = softmax(raw_outputs)
    else:
        # Fallback dummy distribution if model hasn't been trained yet
        probs = np.random.dirichlet(np.ones(len(class_names)))

    # Sort descending
    top_indices = np.argsort(probs)[::-1][:3]
    predictions = [
        {
            "rank": rank + 1,
            "species": class_names[idx].capitalize(),
            "confidence": float(round(probs[idx] * 100, 2))
        }
        for rank, idx in enumerate(top_indices)
    ]

    return {
        "filename": file.filename,
        "predicted_species": predictions[0]["species"],
        "confidence": predictions[0]["confidence"],
        "top_predictions": predictions
    }

@app.get("/", response_class=HTMLResponse)
def index():
    template_path = "app/templates/index.html"
    if os.path.exists(template_path):
        with open(template_path) as f:
            return f.read()
    return "<h1>Animal Species Classifier API is Running!</h1><p>Visit /docs for API documentation.</p>"
