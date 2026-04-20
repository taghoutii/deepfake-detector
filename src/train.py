import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
from src.model import build_model
from src.dataset import DeepfakeDataset

EPOCHS     = 8
BATCH_SIZE = 32
LR         = 1e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    print(f"Using device: {DEVICE}")

    # No path argument needed — dataset.py handles it
    train_set = DeepfakeDataset(split="train")
    val_set   = DeepfakeDataset(split="val")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_set)} images | Val: {len(val_set)} images")

    model     = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    mlflow.set_experiment("deepfake-detection")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": EPOCHS, "lr": LR,
            "batch_size": BATCH_SIZE, "model": "efficientnet_b0",
            "train_size": len(train_set), "val_size": len(val_set)
        })

        for epoch in range(EPOCHS):
            # --- Training ---
            model.train()
            train_loss = 0.0
            for imgs, labels in train_loader:
                imgs   = imgs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

             # --- Test set evaluation ---
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import (
            confusion_matrix, classification_report,
            precision_score, recall_score, f1_score, roc_auc_score
        )
        import os

        test_set    = DeepfakeDataset(split="test")
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Check class balance and log it
        from collections import Counter
        label_counts = Counter(test_set.dataset.targets)
        # ImageFolder: fake=0, real=1 (alphabetical)
        print(f"\nTest set class balance:")
        print(f"  Fake (0): {label_counts[0]} images")
        print(f"  Real (1): {label_counts[1]} images")
        mlflow.log_params({
            "test_fake_count": label_counts[0],
            "test_real_count": label_counts[1]
        })

        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs    = imgs.to(DEVICE)
                outputs = torch.sigmoid(model(imgs)).cpu().numpy()
                test_preds.extend(outputs.flatten())
                test_labels.extend(labels.numpy())

        test_binary = [1 if p > 0.5 else 0 for p in test_preds]
        test_labels_int = [int(l) for l in test_labels]

        # Core metrics
        test_auc       = roc_auc_score(test_labels_int, test_preds)
        test_acc       = sum(p == l for p, l in zip(test_binary, test_labels_int)) / len(test_labels_int)

        # Per-class metrics — zero_division=0 avoids warnings if a class is missing
        precision_fake = precision_score(test_labels_int, test_binary, pos_label=0, zero_division=0)
        recall_fake    = recall_score(test_labels_int, test_binary, pos_label=0, zero_division=0)
        f1_fake        = f1_score(test_labels_int, test_binary, pos_label=0, zero_division=0)

        precision_real = precision_score(test_labels_int, test_binary, pos_label=1, zero_division=0)
        recall_real    = recall_score(test_labels_int, test_binary, pos_label=1, zero_division=0)
        f1_real        = f1_score(test_labels_int, test_binary, pos_label=1, zero_division=0)

        print(f"\nTest Results:")
        print(f"  AUC:           {test_auc:.4f}")
        print(f"  Accuracy:      {test_acc:.4f}")
        print(f"  Fake — Precision: {precision_fake:.4f} | Recall: {recall_fake:.4f} | F1: {f1_fake:.4f}")
        print(f"  Real — Precision: {precision_real:.4f} | Recall: {recall_real:.4f} | F1: {f1_real:.4f}")

        # Full classification report
        report = classification_report(
            test_labels_int, test_binary,
            target_names=["fake", "real"]
        )
        print(f"\nClassification Report:\n{report}")

        # Log all metrics to MLflow
        mlflow.log_metrics({
            "test_auc":        test_auc,
            "test_acc":        test_acc,
            "fake_precision":  precision_fake,
            "fake_recall":     recall_fake,
            "fake_f1":         f1_fake,
            "real_precision":  precision_real,
            "real_recall":     recall_real,
            "real_f1":         f1_real,
        })

        # Save classification report as text artifact
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/classification_report.txt", "w") as f:
            f.write(f"AUC: {test_auc:.4f}\n")
            f.write(f"Accuracy: {test_acc:.4f}\n\n")
            f.write(report)
        mlflow.log_artifact("outputs/classification_report.txt")

        # Confusion matrix
        cm  = confusion_matrix(test_labels_int, test_binary)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["fake", "real"],
            yticklabels=["fake", "real"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — AUC: {test_auc:.4f}")
        fig.tight_layout()
        fig.savefig("outputs/confusion_matrix.png")
        mlflow.log_artifact("outputs/confusion_matrix.png")
        plt.close()
        print("Artifacts saved to outputs/")

        torch.save(model.state_dict(), "model.pt")
        mlflow.log_artifact("model.pt")
        mlflow.pytorch.log_model(model, "model")
        print("model.pt saved and logged to MLflow.")

if __name__ == "__main__":
    train() 