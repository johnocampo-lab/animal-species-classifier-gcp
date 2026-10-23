from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert len(body["classes"]) == 6


def test_predict_accepts_image():
    client = TestClient(app)
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_species" in data
    assert len(data["top_predictions"]) == 3
