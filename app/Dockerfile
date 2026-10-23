# Ultra-lightweight container for Cloud Run serving (<150 MB)
FROM python:3.12-slim

# Avoid buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Install production requirements (ONNX Runtime, FastAPI, etc. - no PyTorch!)
COPY app/requirements-prod.txt requirements-prod.txt
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy model artifacts and application code
COPY models/ models/
COPY app/ app/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
