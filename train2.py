"""
train2.py  —  ResNet18 with SpecAugment, time-shift, weight_decay, boosted rare-class weights
Saves best-by-macro-recall to model2.pth. Does NOT touch train.py or model.pth.
Usage:
  .venv\\Scripts\\python.exe train2.py
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix

# ── constants ────────────────────────────────────────────────────────────────
SPEC_DIR     = Path("spectrograms")
MODEL_PATH   = Path("model2.pth")
BATCH_SIZE   = 32
LR           = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 20
SEED         = 42

# ImageFolder class order (alphabetical): both, crackle, normal, wheeze
TRAIN_COUNTS = {"both": 351, "crackle": 1196, "normal": 2780, "wheeze": 645}
# Rare-class boost: both and wheeze get 2x weight before normalisation
BOOST        = {"both": 2.0, "crackle": 1.0, "normal": 1.0, "wheeze": 2.0}


# ── SpecAugment + time-shift (tensor-level, training only) ──────────────────
class SpecAugment:
    """
    Applied after ToTensor+Normalize so masked regions fill with 0.0
    (approx normalised mean). Only added to the train transform.
    """
    def __init__(self,
                 freq_mask_max: int = 8,
                 time_mask_max: int = 15,
                 n_freq_masks:  int = 2,
                 n_time_masks:  int = 2,
                 shift_max:     int = 10):
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.n_freq_masks  = n_freq_masks
        self.n_time_masks  = n_time_masks
        self.shift_max     = shift_max

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor: [C, H, W] = [3, 64, 157]
        _, H, W = tensor.shape

        # Random time shift: roll along time axis
        shift = random.randint(-self.shift_max, self.shift_max)
        tensor = torch.roll(tensor, shift, dims=2)

        # Frequency masking: 1-2 horizontal bars over mel bins
        for _ in range(random.randint(1, self.n_freq_masks)):
            f  = random.randint(1, self.freq_mask_max)
            f0 = random.randint(0, H - f)
            tensor[:, f0:f0 + f, :] = 0.0

        # Time masking: 1-2 vertical bars over time frames
        for _ in range(random.randint(1, self.n_time_masks)):
            t  = random.randint(1, self.time_mask_max)
            t0 = random.randint(0, W - t)
            tensor[:, :, t0:t0 + t] = 0.0

        return tensor


# ── transforms ───────────────────────────────────────────────────────────────
_base = [
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
]

_train_transform = transforms.Compose(_base + [SpecAugment()])
_test_transform  = transforms.Compose(_base)


def make_loaders():
    train_ds = datasets.ImageFolder(SPEC_DIR / "train", transform=_train_transform)
    test_ds  = datasets.ImageFolder(SPEC_DIR / "test",  transform=_test_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)
    return train_loader, test_loader, train_ds.classes


# ── model ────────────────────────────────────────────────────────────────────
def make_model() -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 4)
    return model


# ── class weights: 1/count * boost, normalised ───────────────────────────────
def class_weights(classes: list) -> torch.Tensor:
    raw = torch.tensor([BOOST[c] / TRAIN_COUNTS[c] for c in classes])
    return raw / raw.sum()


# ── metrics ──────────────────────────────────────────────────────────────────
def per_class_recall(y_true, y_pred, n: int) -> list:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    recalls = []
    for i in range(n):
        denom = cm[i].sum()
        recalls.append(float(cm[i, i] / denom) if denom > 0 else 0.0)
    return recalls


# ── epoch helpers ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, n_classes: int):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    acc     = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    recalls = per_class_recall(all_labels, all_preds, n_classes)
    return acc, recalls, float(np.mean(recalls))


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device("cpu")

    train_loader, test_loader, classes = make_loaders()
    n = len(classes)

    weights = class_weights(classes)
    print(f"Classes: {classes}", flush=True)
    print(f"Train: {len(train_loader.dataset)}  Test: {len(test_loader.dataset)}", flush=True)
    print("Class weights: " + "  ".join(f"{c}={w:.4f}" for c, w in zip(classes, weights)),
          flush=True)
    print(f"Device: {device}  Epochs: {EPOCHS}  LR: {LR}  weight_decay: {WEIGHT_DECAY}",
          flush=True)
    print(flush=True)

    model     = make_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_macro = -1.0
    t_total    = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss          = train_epoch(model, train_loader, criterion, optimizer, device)
        acc, recalls, macro = evaluate(model, test_loader, device, n)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"loss={train_loss:.4f}  acc={acc:.3f}  "
              f"macro-recall={macro:.3f}  [{elapsed:.0f}s]", flush=True)
        print("  recalls: " + "  ".join(f"{c}={r:.3f}" for c, r in zip(classes, recalls)),
              flush=True)

        if macro > best_macro:
            best_macro = macro
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> model2.pth saved (best macro-recall={macro:.3f})", flush=True)
        print(flush=True)

    wall = time.time() - t_total
    print(f"Finished. Wall time: {wall/60:.1f} min  "
          f"Best macro-recall: {best_macro:.3f}  Model: {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
