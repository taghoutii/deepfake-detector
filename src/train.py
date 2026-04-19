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

            # --- Validation ---
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs    = imgs.to(DEVICE)
                    outputs = torch.sigmoid(model(imgs)).cpu().numpy()
                    all_preds.extend(outputs.flatten())
                    all_labels.extend(labels.numpy())

            auc          = roc_auc_score(all_labels, all_preds)
            preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
            acc          = sum(p == l for p, l in zip(preds_binary, all_labels)) / len(all_labels)
            avg_loss     = train_loss / len(train_loader)

            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | AUC: {auc:.4f} | Acc: {acc:.4f}")
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_auc":    auc,
                "val_acc":    acc
            }, step=epoch)

        torch.save(model.state_dict(), "model.pt")
        mlflow.pytorch.log_model(model, "model")
        print("\nTraining complete. model.pt saved.")

if __name__ == "__main__":
    train() 