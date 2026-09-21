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

## Web UI

A single-page Streamlit flow: upload an image, it is analysed, and the result appears as one report card.
The card shows a verdict (LIKELY SYNTHETIC, LIKELY AUTHENTIC, or INCONCLUSIVE below 70% confidence), the
confidence score, a short reasoning line and named signal rows (Grad-CAM attention region, file metadata
finding, model scope), next to the Grad-CAM heatmap with an opacity slider. The round-trip analysis time
and a separate file-metadata card are shown too. Nothing persists between analyses: a new upload replaces
the result, and removing the file clears it.

- One light theme (`streamlit_app/theme.py`, palette values in `PALETTE`); blue marks controls and section
  headers, and the verdict colours are reserved for verdict states. All text uses a sans-serif stack.
- The attention-region row is derived in the frontend from the overlay the API already returns; it
  describes where the model looked, not evidence of manipulation. File metadata is a supplementary
  signal only: the model does not use it and it can be stripped or forged.
- Images are analysed by the model in your own API container (no third-party AI service) and are held in
  session memory only; nothing is written to disk. `.streamlit/config.toml` sets
  `browser.gatherUsageStats = false`, and a browser session (page load, analysis, upload) was measured
  making no requests to external hosts. Re-check this if you change the Streamlit config or add web
  fonts/analytics. This describes the app, not your deployment: if you host it, whoever runs the server can
  see uploads.
- The custom CSS targets Streamlit's internal markup. It was checked on Streamlit 1.39 (the Docker pin)
  and 1.56; re-check the look after upgrading Streamlit. The drag-over highlight on the upload box is
  Streamlit's own (blue on 1.56); 1.39 has no drag-over styling to customise.

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
streamlit_app/ — Streamlit frontend (app, theme, components, metadata helpers)
.streamlit/   — Streamlit config (light theme, blue primary, telemetry off, 10 MB upload cap)
tests/        — pytest test suite
docker/       — Dockerfiles