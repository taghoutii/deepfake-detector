import torch
from torch.utils.data import Dataset
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_transforms(split="train"):
    if split == "train":
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class DeepfakeDataset(Dataset):
    def __init__(self, split="train"):
        # Always reads from data/processed — never from raw
        self.dataset = datasets.ImageFolder(root=f"data/processed/{split}")
        self.transform = get_transforms(split)
        self.classes = self.dataset.classes  # ['fake', 'real'] — alphabetical

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return augmented["image"], torch.tensor(label, dtype=torch.float32)