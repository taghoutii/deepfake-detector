import torch
import torch.nn as nn
from torchvision import models

def build_model(pretrained=True):
    model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1)   # binary: real vs fake
    )
    return model