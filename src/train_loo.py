"""Leave-one-SD-version-out training run: same architecture and hyperparameters as the
v2 retrain (src/train.py), same StyleGAN data, but with one Stable Diffusion version
held out entirely from train/val and evaluated as a never-seen generator.

Writes model_loo_<version>.pt (NOT model.pt) and logs to MLflow as a separate, clearly
tagged run ("leave_one_out_v1") in the same "deepfake-detection" experiment — it never
touches the production run or model.pt.

Requires `python -m src.build_loo_split plan/materialize --held-out <version>` first.
Run with: DATA_PROCESSED_DIR=data/processed_loo python -m src.train_loo --held-out 1024px
"""
import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import mlflow
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import DeepfakeDataset
from src.eval_gravex import auc, rate, score
from src.model import build_model

ROOT = Path(__file__).resolve().parent.parent
EPOCHS, BATCH_SIZE, LR = 8, 32, 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def setup_mlflow():
    if MLFLOW_TRACKING_URI.startswith(("http://", "https://")):
        try:
            requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=5).raise_for_status()
        except requests.exceptions.RequestException as e:
            raise SystemExit(f"MLflow server not reachable at {MLFLOW_TRACKING_URI} ({e}).")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += ((torch.sigmoid(outputs) > 0.5).float() == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", required=True, choices=["512px", "768px", "1024px"])
    a = ap.parse_args()
    held_out = a.held_out
    weights_path = ROOT / f"model_loo_{held_out.replace('px', '')}.pt"

    expected_dir = str(ROOT / "data/processed_loo")
    actual_dir = os.getenv("DATA_PROCESSED_DIR", "data/processed")
    if Path(actual_dir).resolve() != Path(expected_dir).resolve():
        raise SystemExit(
            f"DATA_PROCESSED_DIR is '{actual_dir}', expected '{expected_dir}'. "
            "Run with DATA_PROCESSED_DIR=data/processed_loo to avoid training on the wrong data."
        )

    print(f"Using device: {DEVICE}")
    print(f"Held-out SD version: {held_out}")
    setup_mlflow()

    train_set = DeepfakeDataset(split="train")
    val_set = DeepfakeDataset(split="val")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {len(train_set)} images | Val: {len(val_set)} images")

    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    mlflow.set_experiment("deepfake-detection")
    with mlflow.start_run(run_name="leave_one_out_v1"):
        mlflow.set_tag("experiment_type", "leave_one_out")
        mlflow.set_tag("held_out_sd_version", held_out)
        mlflow.set_tag("purpose", "portfolio/README generalization experiment - not production")
        mlflow.log_params({
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "model": "efficientnet_b0",
            "train_size": len(train_set), "val_size": len(val_set), "held_out_sd_version": held_out,
        })

        best_val_loss, best_state, best_epoch = float("inf"), None, -1
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            scheduler.step()

            avg_train_loss = train_loss / len(train_set)
            val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
            print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            mlflow.log_metrics({"train_loss": avg_train_loss, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"Loaded best model from epoch {best_epoch + 1} (val_loss={best_val_loss:.4f})")
            mlflow.log_params({"best_epoch": best_epoch + 1, "best_val_loss": best_val_loss})

        model.eval()

        manifest = list(csv.DictReader(open(ROOT / f"data/processed_loo/manifest_{held_out}.csv", newline="")))
        orig = [r for r in manifest if r["role"] == "orig_test"]
        hf = [r for r in manifest if r["role"] == "heldout_fake"]
        hr = [r for r in manifest if r["role"] == "heldout_real"]
        assert len(hf) == 3000 and len(hr) == 3000 and len(orig) == 2000

        print(f"\nScoring {len(orig) + len(hf) + len(hr)} eval images ...")
        L = score(model, [r["src_path"] for r in orig + hf + hr], BATCH_SIZE)
        fake_flag = lambda l: l <= 0   # logit <= 0 <=> P(real) <= 0.5 -> classified fake

        of = [L[r["src_path"]] for r in orig if r["label"] == "fake"]
        orl = [L[r["src_path"]] for r in orig if r["label"] == "real"]
        orig_test_res = dict(fake_recall=rate([fake_flag(l) for l in of]),
                              real_fpr=rate([fake_flag(l) for l in orl]), auc=auc(of, orl))

        hf_l = [L[r["src_path"]] for r in hf]
        hr_l = [L[r["src_path"]] for r in hr]
        heldout_res = dict(held_out_version=held_out,
                            fake_recall=rate([fake_flag(l) for l in hf_l]),
                            real_fpr=rate([fake_flag(l) for l in hr_l]),
                            auc=auc(hf_l, hr_l))

        mlflow.log_metrics({
            "orig_test_fake_recall": orig_test_res["fake_recall"]["rate"],
            "orig_test_real_fpr": orig_test_res["real_fpr"]["rate"],
            "orig_test_auc": orig_test_res["auc"],
            "heldout_sd_fake_recall": heldout_res["fake_recall"]["rate"],
            "heldout_sd_real_fpr": heldout_res["real_fpr"]["rate"],
            "heldout_sd_auc": heldout_res["auc"],
        })

        pct = lambda d: f"{100 * d['rate']:5.1f}% [{100 * d['ci95'][0]:.1f}-{100 * d['ci95'][1]:.1f}] (n={d['n']})"
        print(f"\n=== leave_one_out_v1 (held out {held_out}, trained on the other two SD versions) ===")
        print(f"orig test (StyleGAN/FFHQ, sanity)  fake recall {pct(orig_test_res['fake_recall'])} | "
              f"real FPR {pct(orig_test_res['real_fpr'])} | AUC {orig_test_res['auc']:.4f}")
        print(f"held-out {held_out:<8}              fake recall {pct(heldout_res['fake_recall'])} | "
              f"real FPR {pct(heldout_res['real_fpr'])} | AUC {heldout_res['auc']:.4f}")

        os.makedirs(ROOT / "outputs", exist_ok=True)
        report = {"held_out_version": held_out, "orig_test": orig_test_res, "heldout": heldout_res}
        out_json = ROOT / f"outputs/eval_loo_{held_out}.json"
        out_json.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(out_json))

        torch.save(model.state_dict(), weights_path)
        mlflow.log_artifact(str(weights_path))
        print(f"\n{weights_path.name} saved (model.pt untouched) and logged to MLflow run 'leave_one_out_v1'.")


if __name__ == "__main__":
    main()
