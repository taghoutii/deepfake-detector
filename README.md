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

```bash
git clone https://github.com/taghoutii/deepfake-detector
cd deepfake-detector
docker-compose up
```

Then open http://localhost:8501

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