# Deepfake Face Detector

A binary classifier that detects deepfake face images using EfficientNet-B0,
served via FastAPI, with a Streamlit UI and Grad-CAM explainability.

## Results

| Metric         | Score  |
|----------------|--------|
| Test Accuracy  | 99.05% |
| Test AUC       | 0.9995 |
| Fake Recall    | 0.9940 |
| Fake F1        | 0.9905 |

> Note: Results reflect performance on the 140k Real and Fake Faces Kaggle dataset,
> which contains GAN-generated images with consistent artifacts. Performance on
> in-the-wild deepfakes may differ.

## Architecture

User → Streamlit (port 8501) → FastAPI (port 8000) → EfficientNet-B0 → prediction + Grad-CAM

## Quick start

The trained model (`model.pt`) is stored with [Git LFS](https://git-lfs.com), so install
it once before cloning:

```bash
git lfs install
git clone https://github.com/taghoutii/deepfake-detector
cd deepfake-detector
docker-compose up
```

Then open http://localhost:8501

> If you clone without Git LFS installed, `model.pt` will be a small text pointer
> file instead of the real weights, and the API container will fail to load it.
> Run `git lfs pull` after installing Git LFS to fix an existing clone.

## Experiment tracking (MLflow)

All training runs are logged to the MLflow server started by docker-compose, so the
UI at http://localhost:5000 always shows every run, whether you train locally or in Docker.

```bash
docker-compose up -d mlflow        # start the tracking server first
python -m src.preprocessing        # run from the repo root (note: -m, not src/x.py)
python -m src.train                # logs to http://localhost:5000
```

- `src/train.py` exits with a clear message if the server isn't reachable.
- Override the server with `MLFLOW_TRACKING_URI` (use `http://mlflow:5000` from inside
  the compose network).
- Run metadata lives in the `deepfake-detector-mlflow-db` Docker volume; artifacts are
  written to `./mlruns` on the host.

## Stack

| Component       | Tool                        |
|-----------------|-----------------------------|
| Model           | PyTorch · EfficientNet-B0   |
| Augmentation    | Albumentations              |
| Explainability  | Grad-CAM                    |
| Experiment tracking | MLflow                  |
| Backend         | FastAPI                     |
| Frontend        | Streamlit                   |
| Testing         | pytest                      |
| Deployment      | Docker · docker-compose     |

## Project structure

src/          — dataset loader, model, training, Grad-CAM
api/          — FastAPI backend
streamlit_app/ — Streamlit frontend
tests/        — pytest test suite
docker/       — Dockerfiles