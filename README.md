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

## Model Performance & Generalization

### Training data
- Base: StyleGAN-generated faces (140k Real and Fake Faces, Kaggle)
- v2 retrain adds: Stable Diffusion 1.5 / 2.1 / XL generated faces (GRAVEX-200k subset;
  all 9,000 SD fakes, resolution 512/768/1024px — the GRAVEX filenames don't carry an
  explicit SD-version label, so version is inferred from resolution). 17,600 train /
  4,400 val / 8,000 test images, balanced real/fake. The "real" half of the added data
  is GRAVEX's own re-encoded copies of the 140k Faces reals (not new photos, and not the
  original 140k JPEGs) — matching the SD fakes' encoding pipeline so the model can't just
  learn "re-encoded = fake" instead of learning the actual generator artifacts. Built via
  `src/build_gravex_split.py`; manifest at `data/processed_v2/manifest.csv`.

### Known limitation, found and fixed
The original model (trained only on StyleGAN) was evaluated against Stable Diffusion-generated faces it had never seen:

| Metric | v1 (StyleGAN only) | v2 (StyleGAN + SD) |
|---|---|---|
| SD fake recall | 13.1% | 100.0% |
| SD AUC | 0.27 (worse than random) | 0.9999 (≈1.0) |
| Original StyleGAN test recall | 99.5% | 99.1% (retained) |

v1 didn't just miss Stable Diffusion fakes — it actively ranked them as *more* authentic than real photos (AUC below 0.5), confirming that a detector trained on one generator family doesn't automatically generalize to another. Retraining on a mixed StyleGAN + Stable Diffusion dataset closed this gap without degrading performance on the original generator family. Evaluated via `src/eval_gravex.py`.

### Scope and honest limitations
This model is trained to detect **fully AI-generated faces** (GAN and diffusion-based). It is **not** trained on face-swap manipulation (e.g. DeepFakes, FaceSwap on real video frames) and does not reliably detect it — measured as a held-out probe (never trained on) to be transparent about the boundary of what the model can and can't do:

| Source | Fake recall | Real FPR | AUC |
|---|---|---|---|
| FaceForensics++ | 95.0% | 96.3% | 0.45 |
| DFDC | 7.7% | 10.7% | 0.51 |
| Celeb-DF | 4.7% | 8.0% | 0.49 |

The near-0.5 AUCs are expected — the model was deliberately not trained on face-swap data.
But "no signal" looks different per source. On DFDC and Celeb-DF it's close to a genuine
coin flip: both recall and false-positive rate are low. On FaceForensics++ it isn't neutral:
it flags 95% of manipulated frames *and* 96.3% of real FF++ frames as fake — it reflexively
calls FF++ content "fake" almost regardless of the true label, which is why the AUC still
lands near 0.5 despite the high recall. So on FF++, "no signal" means "biased toward fake,"
not "coin flip" — worth knowing if FF++-style footage is ever run through this model.

### Generalization to unseen generator versions

To test whether the model learned a genuine "AI-generated" concept rather than memorizing specific generator fingerprints, a separate model was trained with one Stable Diffusion resolution/version (1024px) deliberately withheld from training, then evaluated exclusively on that held-out version:

| Metric | Held-out 1024px (never seen in training) |
|---|---|
| Fake recall | 95.0% |
| Real FPR | 0.0% |
| AUC | 0.9997 |

The model correctly identified fakes from a generator version it had never encountered during training, suggesting it learned generalizable artifact patterns rather than overfitting to specific generator fingerprints. This is evaluated separately from the production model (`model.pt`) and is not deployed — it exists purely to measure generalization.

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