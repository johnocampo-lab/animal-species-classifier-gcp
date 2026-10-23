# Animal Species Recognition Demo (Google Cloud PMLE Portfolio)

An end-to-end Machine Learning demo showcasing transfer learning, model evaluation, and serverless deployment on Google Cloud Platform, engineered to stay **well under a $50 budget** (projected cost: **< $3.00**).

---

## 🏗️ Architecture & Technology Stack

```
[ Dataset (STL-10 Animal Subset) ]
               │
               ▼
 [ Stratified Splitting (Train 70% / Val 15% / Test 15%) ]
               │
               ▼
[ Transfer Learning (MobileNetV3-Small) ] ──► Trained locally or on Ephemeral Vertex AI Job
               │
               ▼
[ Model Evaluation & Diagnostics ] ─────────► Confusion Matrix Heatmap, Per-Class F1
               │
               ▼
    [ Model Export to ONNX ] ──────────────► <150MB container footprint, sub-second latency
               │
               ▼
[ Serverless Serving (Cloud Run) ] ─────────► Scale-to-Zero (--min-instances 0), $0 idle cost
```

* **Target GCP Project**: `sbox-johnny-001`
* **Default Region**: `us-central1`
* **Model**: MobileNetV3-Small fine-tuned for animal species (Bird, Cat, Deer, Dog, Frog, Horse)
* **Inference Engine**: ONNX Runtime CPU (no heavy PyTorch dependency in container)
* **Serving**: FastAPI + Drag-and-drop Web UI hosted on Google Cloud Run

---

## 💰 Budget & Cost Breakdown

| Component | GCP Service | Estimated Cost |
| :--- | :--- | :--- |
| **Data & Artifact Storage** | Cloud Storage (GCS) (~500MB) | `< $0.02` |
| **Container Images** | Artifact Registry | `< $0.10` |
| **Training (Optional Cloud)**| Ephemeral Vertex AI Job (1x T4 GPU, ~15 mins) | `~$0.15 - $0.50` |
| **Serving (Scale-to-Zero)**  | Cloud Run (2M free requests/month) | **$0.00** (Free Tier) |
| **Total Cost** | | **$0.27 – $3.00** |

> [!IMPORTANT]
> **Why Cloud Run instead of Vertex AI Endpoints?**
> A persistent Vertex AI Prediction endpoint runs dedicated VMs 24/7, costing **~$100–$180/month** even with zero traffic. Deploying to Cloud Run with `--min-instances 0` guarantees the service scales to zero when idle, keeping costs well under $50.

---

## 📚 Official Google Cloud Documentation References

* **Cloud Storage**: [Creating and Managing Buckets](https://cloud.google.com/storage/docs/creating-buckets)
* **Vertex AI Custom Training**: [Overview and Pre-built Containers](https://cloud.google.com/vertex-ai/docs/training/overview)
* **Vertex AI Custom Jobs CLI**: [Creating Custom Jobs with gcloud](https://cloud.google.com/vertex-ai/docs/training/create-custom-job)
* **Vertex AI Model Registry**: [Model Registry Concepts](https://cloud.google.com/vertex-ai/docs/model-registry/introduction)
* **Cloud Run**: [Deploying Python Services](https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service)
* **Cloud Run Pricing**: [Always-Free Tier Limits](https://cloud.google.com/run/pricing)
* **Artifact Registry**: [Docker Container Repositories](https://cloud.google.com/artifact-registry/docs/docker/store-docker-container-images)

---

## 🚀 Step-by-Step Instructions

### Step 0: Environment & GCP Authentication
Ensure your CLI is authenticated and active project is set to `sbox-johnny-001`:
```bash
gcloud auth login
gcloud config set project sbox-johnny-001
gcloud auth application-default login
gcloud auth application-default set-quota-project sbox-johnny-001
```

Activate the Python virtual environment:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Step 1: Provision GCP Infrastructure
Run the automated setup script to enable required APIs, create the regional GCS bucket, and set up Artifact Registry:
```bash
bash cloud/01_setup_gcp.sh
```

---

### Step 2: Download & Prepare the Animal Dataset
Download the animal species dataset and strictly split it into `train/` (70%), `val/` (15%), and `test/` (15%):
```bash
# Option A: Fast demo sample dataset for instant validation
python data/download_data.py --demo-sample

# Option B: Full STL-10 animal dataset (bird, cat, deer, dog, frog, horse)
python data/download_data.py
```

---

### Step 3: Model Training & ONNX Export
Fine-tune MobileNetV3-Small using transfer learning:
```bash
python -m training.train --epochs 5 --batch-size 32 --lr 0.001
```
*Outputs:*
* PyTorch weights: `models/model.pt`
* Class labels: `models/classes.json`
* Production ONNX graph: `models/model.onnx`

*(Optional)* If you wish to submit an ephemeral training job to Vertex AI instead of training locally:
```bash
bash cloud/02_upload_to_gcs.sh
bash cloud/03_submit_vertex_job.sh
```

---

### Step 4: Model Evaluation & Diagnostics
Evaluate the model against the held-out test split (evaluates accuracy, per-class F1-scores, and generates a confusion matrix):
```bash
python -m training.evaluate
```
*Outputs:*
* Classification report in terminal
* Metrics summary: `evaluation_results/metrics.json`
* Confusion matrix plot: `evaluation_results/confusion_matrix.png`

---

### Step 5: Test Web App Locally
Run the FastAPI serving application locally:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```
Open [http://localhost:8080](http://localhost:8080) in your browser to test drag-and-drop animal species classification.

---

### Step 6: Deploy to Google Cloud Run
Build the lightweight container using Google Cloud Build and deploy to Cloud Run:
```bash
bash cloud/04_deploy_cloud_run.sh
```
Once deployment finishes, `gcloud` will output your live, public HTTPS URL.
