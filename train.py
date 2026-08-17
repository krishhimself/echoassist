"""
train.py  —  ResNet18 on mel spectrogram PNGs
Usage:
  .venv\\Scripts\\python.exe train.py --epochs 1   (timing check)
  .venv\\Scripts\\python.exe train.py              (full 10 epochs)
"""

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
SPEC_DIR   = Path("spectrograms")
MODEL_PATH = Path("model.pth")
BATCH_SIZE = 32
LR         = 1e-4
EPOCHS     = 10
SEED       = 42

# ImageFolder sorts classes alphabetically: both, crackle, normal, wheeze
TRAIN_COUNTS = {"both": 351, "crackle": 1196, "normal": 2780, "wheeze": 645}

# ── data ─────────────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # RGBA -> RGB, no resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def make_loaders():
    train_ds = datasets.ImageFolder(SPEC_DIR / "train", transform=_transform)
    test_ds  = datasets.ImageFolder(SPEC_DIR / "test",  transform=_transform)
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


# ── class weights  1/count, normalised ──────────────────────────────────────
def class_weights(classes: list[str]) -> torch.Tensor:
    raw = torch.tensor([1.0 / TRAIN_COUNTS[c] for c in classes])
    return raw / raw.sum()


# ── metrics ──────────────────────────────────────────────────────────────────
def per_class_recall(y_true, y_pred, n: int) -> list[float]:
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
def main(n_epochs: int) -> None:
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    train_loader, test_loader, classes = make_loaders()
    n = len(classes)

    print(f"Classes (idx 0-{n-1}): {classes}")
    print(f"Train: {len(train_loader.dataset)}  Test: {len(test_loader.dataset)}")
    weights = class_weights(classes)
    print("Class weights: " + "  ".join(f"{c}={w:.4f}" for c, w in zip(classes, weights)))
    print(f"Device: {device}  Epochs: {n_epochs}\n")

    model     = make_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_macro_recall = -1.0
    t_total = time.time()

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss          = train_epoch(model, train_loader, criterion, optimizer, device)
        acc, recalls, macro = evaluate(model, test_loader, device, n)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{n_epochs}  "
              f"loss={train_loss:.4f}  acc={acc:.3f}  "
              f"macro-recall={macro:.3f}  [{elapsed:.0f}s]")
        print("  recalls: " + "  ".join(f"{c}={r:.3f}" for c, r in zip(classes, recalls)))

        if macro > best_macro_recall:
            best_macro_recall = macro
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> model.pth saved (best macro-recall={macro:.3f})")
        print()

    wall = time.time() - t_total
    print(f"Finished. Wall time: {wall/60:.1f} min  "
          f"Best macro-recall: {best_macro_recall:.3f}  "
          f"Model: {MODEL_PATH}")


if __name__ == "__main__":
    n_epochs = EPOCHS
    if "--epochs" in sys.argv:
        n_epochs = int(sys.argv[sys.argv.index("--epochs") + 1])
    main(n_epochs)
