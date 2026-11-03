# Cat vs Dog Classifier (local train → GCP Cloud Run)

Train a small Keras CNN locally, then serve it on **Cloud Run** as an HTTPS API. Training stays on your machine; GCP only stores the model and runs inference (scales to zero when idle).

## What’s in the repo

| Path | Purpose |
|------|---------|
| `image_classifier.ipynb` | Train / evaluate the model locally |
| `gcp_deploy.ipynb` | Upload model → deploy Cloud Run → call `/predict` |
| `serving/` | FastAPI app + Dockerfile used by Cloud Run |
| `cat_test.jpg` / `dog_test.jpg` | Sample images for API tests |

Training images (`data/`, `holdout/`) and the `.h5` weights are **not** in git (too large). Keep them locally; the deploy notebook copies the model into the container build.

## Prerequisites

- Python 3.12+ with TensorFlow (local training)
- [`gcloud`](https://cloud.google.com/sdk) logged into a GCP project with billing
- APIs: Cloud Storage, Cloud Run, Cloud Build, Artifact Registry

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## 1. Train locally

1. Put images in `data/cats/` and `data/dogs/`.
2. Run `image_classifier.ipynb` end-to-end.
3. Confirm `models/image_classifier.h5` exists.

Optional: score never-seen images in `holdout/` (the notebook includes a holdout cell).

## 2. Deploy to Cloud Run

Run `gcp_deploy.ipynb` (edit `PROJECT` / `BUCKET` in the first code cell):

1. **Upload** `.h5` to GCS (model only — not the full dataset).
2. **Deploy** from `serving/` with `--min-instances=0` (idle ≈ $0).
3. **Call** `/health` and `/predict`.

Or from a terminal (after copying the model into the build context):

```bash
cp models/image_classifier.h5 serving/models/image_classifier.h5

gcloud run deploy cat-dog-classifier \
  --source=serving \
  --region=us-central1 \
  --allow-unauthenticated \
  --min-instances=0 --max-instances=2 \
  --memory=1Gi --cpu=1
```

## 3. Test the API

Cloud Run gives you an **API URL**, not a website UI. Opening the base URL in a browser will not show a page. Use `/health` or POST an image to `/predict`:

```bash
URL=$(gcloud run services describe cat-dog-classifier \
  --region=us-central1 --format='value(status.url)')

curl -sS "$URL/health"
curl -sS -F "file=@dog_test.jpg;type=image/jpeg" "$URL/predict"
```

Example response:

```json
{"label":"dog","confidence":0.72,"raw_score":0.72}
```

## Cost notes

- Prefer **local training**; avoid Vertex Workbench / GPUs for this demo.
- Cloud Run **`min-instances=0`** → no charge when idle.
- Store the **model** (and maybe a small holdout) in GCS — not the full training set unless you need it.
- Set a **budget alert** on the project (e.g. $50 with 20/50/80% thresholds).

## Tear down

```bash
gcloud run services delete cat-dog-classifier --region=us-central1
# optional: gcloud storage rm -r gs://YOUR_BUCKET
```
