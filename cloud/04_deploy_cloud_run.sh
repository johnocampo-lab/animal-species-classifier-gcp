#!/bin/bash
# 04_deploy_cloud_run.sh
# Builds the production container using Cloud Build and deploys to Cloud Run in sbox-johnny-001.
# Enforces scale-to-zero (--min-instances 0) to ensure zero idle cost and stay under budget.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:latest"

echo "=========================================================="
echo "Building Container Image with Google Cloud Build..."
echo "Image URI: ${IMAGE_URI}"
echo "=========================================================="

gcloud builds submit \
    --project="${PROJECT_ID}" \
    --tag="${IMAGE_URI}" \
    -f app/Dockerfile .

echo "=========================================================="
echo "Deploying to Cloud Run: ${SERVICE_NAME}"
echo "Settings: min-instances=0 (Zero idle cost), max-instances=2, 1 CPU, 1Gi RAM"
echo "=========================================================="

gcloud run deploy "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --platform="managed" \
    --allow-unauthenticated \
    --cpu=1 \
    --memory=1Gi \
    --min-instances=0 \
    --max-instances=2 \
    --port=8080

echo ""
echo "=========================================================="
echo "Deployment Complete!"
echo "Check your live service URL in the output above."
echo "=========================================================="
