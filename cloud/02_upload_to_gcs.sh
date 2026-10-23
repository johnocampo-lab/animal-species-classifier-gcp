#!/bin/bash
# 02_upload_to_gcs.sh
# Uploads prepared datasets and/or model artifacts to GCS in sbox-johnny-001.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

echo "=========================================================="
echo "Syncing data and models to gs://${BUCKET_NAME}..."
echo "=========================================================="

if [ -d "data/processed" ]; then
    echo "Uploading processed dataset..."
    gcloud storage rsync -r data/processed "gs://${BUCKET_NAME}/data/processed"
else
    echo "Notice: data/processed not found locally. Run 'python data/download_data.py' first if you wish to upload data."
fi

if [ -d "models" ]; then
    echo "Uploading model checkpoints and ONNX artifacts..."
    gcloud storage rsync -r models "gs://${BUCKET_NAME}/models"
fi

echo "Sync completed successfully."
