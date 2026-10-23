#!/bin/bash
# 01_setup_gcp.sh
# Enables APIs, creates GCS bucket and Artifact Registry repository in sbox-johnny-001.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

echo "=========================================================="
echo "Configuring GCP Project: ${PROJECT_ID} (${REGION})"
echo "=========================================================="
gcloud config set project "${PROJECT_ID}"

echo "Enabling GCP Services (Vertex AI, Cloud Run, Cloud Build, Artifact Registry, GCS)..."
gcloud services enable \
    aiplatform.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com

echo "Creating Cloud Storage bucket: gs://${BUCKET_NAME}..."
gcloud storage buckets create "gs://${BUCKET_NAME}" --location="${REGION}" --uniform-bucket-level-access || true

echo "Creating Artifact Registry repository: ${AR_REPO}..."
gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Repository for animal species classifier containers" || true

echo "=========================================================="
echo "GCP Setup Complete for ${PROJECT_ID}!"
echo "=========================================================="
