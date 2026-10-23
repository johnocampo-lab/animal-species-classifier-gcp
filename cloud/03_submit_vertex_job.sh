#!/bin/bash
# 03_submit_vertex_job.sh
# Submits an ephemeral Vertex AI Custom Job to train MobileNetV3 in sbox-johnny-001.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

JOB_NAME="animal-species-training-$(date +%Y%m%d-%H%M%S)"

echo "=========================================================="
echo "Submitting Ephemeral Vertex AI Custom Job: ${JOB_NAME}"
echo "Project: ${PROJECT_ID} | Region: ${REGION}"
echo "=========================================================="

gcloud ai custom-jobs create \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --config="${SCRIPT_DIR}/vertex_job_spec.yaml"

echo ""
echo "Custom job submitted successfully!"
echo "Check progress in GCP Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "Once finished, download the trained model with:"
echo "  gcloud storage cp gs://${BUCKET_NAME}/models/model.onnx models/"
echo "  gcloud storage cp gs://${BUCKET_NAME}/models/classes.json models/"
