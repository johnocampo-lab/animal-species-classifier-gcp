# Animal Species Recognition Demo (Google Cloud PMLE Portfolio)

An end-to-end Machine Learning demo showcasing transfer learning, model evaluation, and serverless deployment on Google Cloud Platform, engineered to stay **well under a $50 budget** (projected cost: **< $3.00**).

---

## 🏗️ Architecture & Technology Stack

```
[ Dataset (CIFAR-10 Animal Subset; optional STL-10) ]
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

## 🧪 Train, Validate, and Test

Data under `data/processed/` has three splits (stratified 70% / 15% / 15%):

| Split | Source | Used for |
| :--- | :--- | :--- |
| **Train** | CIFAR-10 train fold (animal classes) | Weight updates each epoch (with augmentations) |
| **Validation** | Held-out slice of CIFAR-10 train fold | Epoch metrics during training; `best_model.pt` |
| **Test** | **Official CIFAR-10 test fold** (animals only) | `training.evaluate` only — never used in training |

With the default dataset, splits are loaded via `data/processed/dataset_meta.json` (no JPEG export). STL-10 / `--demo-sample` still use `train/`, `val/`, `test/` folders on disk.

### 1. Train

Fine-tune MobileNetV3-Small on the **train** split. After each epoch, the script runs **validation** on `val/` (no augmentation) and prints train/val loss and accuracy. The best validation accuracy is saved as `models/best_model.pt`; the last epoch is saved as `models/model.pt` and exported to ONNX.

```bash
python -m training.train \
  --data-dir data/processed \
  --epochs 5 \
  --batch-size 32 \
  --lr 0.001 \
  --output-dir models
```

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--data-dir` | `data/processed` | Root containing `train/`, `val/`, `test/` |
| `--epochs` | `5` | Full passes over the training set |
| `--batch-size` | `32` | Mini-batch size |
| `--lr` | `0.001` | AdamW learning rate (classifier head only) |
| `--output-dir` | `models` | Weights, `classes.json`, and `model.onnx` |

**Artifacts:** `models/model.pt`, `models/best_model.pt`, `models/classes.json`, `models/model.onnx`

**Typical runtime:** ~1–2 min (`--demo-sample` smoke test); ~5–15 min (CIFAR-10 animals, 5 epochs on Apple Silicon).

### 2. Validate (during training)

Validation is **built into** `training.train` — there is no separate validate command. Watch the per-epoch lines:

```text
Epoch [01/05] Train Loss: ... | Train Acc: ... | Val Loss: ... | Val Acc: ...
```

Use **val accuracy** to spot overfitting (train acc rises while val acc stalls or drops). For cloud training, the same script runs inside the Vertex custom job (`cloud/vertex_job_spec.yaml`).

### 3. Test (held-out evaluation)

Run **once** after training on the **test** split (no augmentation, same transforms as val). This produces metrics and plots for reporting; do not tune hyperparameters based on test results.

```bash
python -m training.evaluate \
  --data-dir data/processed \
  --model-dir models \
  --output-dir evaluation_results
```

**Outputs:**

* Terminal classification report (precision, recall, F1 per species)
* `evaluation_results/metrics.json`
* `evaluation_results/confusion_matrix.png`

`evaluate` loads `models/model.pt` (final epoch). To score `best_model.pt` instead, copy or symlink it to `model.pt` before running evaluate.

### Quick local loop (no GCP)

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m data.download_data --demo-sample   # synthetic only; default is real CIFAR-10
python -m training.train --epochs 5
python -m training.evaluate
uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```

---

## ✅ Testing

```bash
pip install -r requirements-dev.txt
python -m pytest                  # fast unit/smoke (default)
python -m pytest -m integration   # 1-epoch train + evaluate smoke
```

| Documentation area | Test module |
| :--- | :--- |
| CIFAR-10 meta & splits | `tests/test_animal_data.py` |
| Loaders & augmentations | `tests/test_dataset.py` |
| ONNX export | `tests/test_model_onnx.py` |
| Cloud Run API (`/health`, `/predict`) | `tests/test_app.py` |
| Full train → eval loop | `tests/test_integration.py` |

GitHub Actions runs both suites on push/PR to `main` (see `.github/workflows/test.yml`).

---

## 🗺️ GCP AI/ML portfolio map (beyond this demo)

| GCP area | Products | Fit for this repo |
| :--- | :--- | :--- |
| **Current (Arc A)** | Custom training, GCS, Cloud Run, optional Vertex custom job | Implemented |
| **MLOps (Arc B)** | Vertex Pipelines, Experiments, Model Registry | Register ONNX; keep inference on Cloud Run |
| **Batch (Arc C)** | Vertex Batch Prediction, BigQuery ML, Dataflow | Score at scale from warehouse/GCS |
| **GenAI (Arc D)** | Gemini, embeddings + Vector Search | Compare API vs fine-tuned MobileNet |
| **Kubernetes (Arc E)** | GKE, GPU node pools | Same `app/Dockerfile`; not in repo today |

**Recommended Arc B (budget-friendly):** log metrics to **Vertex Experiments**, wrap prepare → train → evaluate in a **Pipeline**, register `model.onnx` in **Model Registry**, deploy winner to **Cloud Run** — no always-on Vertex Endpoints.

**TFX / TFRecords:** Google’s TensorFlow MLOps path; this project is **PyTorch + ONNX**. Add only if explicitly required.

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
Default: **CIFAR-10** animal classes (real photos). Train/val are split from CIFAR’s official training fold; **test uses CIFAR-10’s official test images only** (not used during training).

```bash
# Recommended — real animal photos (bird, cat, deer, dog, frog, horse)
python -m data.download_data

# Optional: STL-10 animals (96×96) if Stanford hosting is available
python -m data.download_data --dataset stl10

# Smoke test only — synthetic colored tiles, not animals
python -m data.download_data --demo-sample
```

---

### Step 3: Train & Validate
See **[Train, Validate, and Test](#-train-validate-and-test)** for flags and split semantics. Summary:

```bash
python -m training.train --epochs 5 --batch-size 32 --lr 0.001
```

*(Optional)* Ephemeral Vertex AI training (same train/val logic in the container):

```bash
bash cloud/02_upload_to_gcs.sh
bash cloud/03_submit_vertex_job.sh
```

---

### Step 4: Test (held-out evaluation)
```bash
python -m training.evaluate
```

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
