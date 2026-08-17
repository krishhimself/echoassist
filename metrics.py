"""
metrics.py  --  evaluate model2.pth on the held-out test set
Outputs:
    confusion_matrix.png   heat-map with counts
    metrics_bar.png        per-class precision / recall / F1 bar chart
Run: .venv\\Scripts\\python.exe metrics.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

SPEC_DIR   = Path("spectrograms")
MODEL_PATH = Path("model2.pth")
BATCH_SIZE = 64
CLASSES    = ["both", "crackle", "normal", "wheeze"]

_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model() -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 4)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    m.eval()
    return m


def get_predictions(model, loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            all_preds.extend(model(imgs).argmax(dim=1).tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, save_path: Path) -> None:
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised %)")

    thresh = 0.5
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.0%})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_pct[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_metrics_bar(precision, recall, f1, save_path: Path) -> None:
    x     = np.arange(len(CLASSES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width, precision, width, label="Precision", color="#2196F3", alpha=0.9)
    ax.bar(x,         recall,    width, label="Recall",    color="#4CAF50", alpha=0.9)
    ax.bar(x + width, f1,        width, label="F1",        color="#FF9800", alpha=0.9)

    for bars, vals in [(x - width, precision), (x, recall), (x + width, f1)]:
        for bx, v in zip(bars, vals):
            ax.text(bx, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1  (model2.pth, test set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main() -> None:
    test_ds = datasets.ImageFolder(SPEC_DIR / "test", transform=_transform)
    loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Test samples : {len(test_ds)}")
    print(f"Model        : {MODEL_PATH}")
    print()

    model          = load_model()
    y_true, y_pred = get_predictions(model, loader)

    acc = (y_true == y_pred).mean()
    print(f"Overall accuracy : {acc:.3f}\n")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0
    )

    print(f"{'class':<10} {'prec':>7} {'rec':>7} {'f1':>7} {'support':>9}")
    print("-" * 46)
    for i, cls in enumerate(CLASSES):
        print(f"{cls:<10} {precision[i]:>7.3f} {recall[i]:>7.3f} "
              f"{f1[i]:>7.3f} {int(support[i]):>9}")
    print("-" * 46)
    print(f"{'macro':<10} {precision.mean():>7.3f} {recall.mean():>7.3f} "
          f"{f1.mean():>7.3f} {int(support.sum()):>9}")

    print("\n--- sklearn classification_report ---")
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    plot_confusion_matrix(y_true, y_pred, Path("confusion_matrix.png"))
    plot_metrics_bar(precision, recall, f1, Path("metrics_bar.png"))


if __name__ == "__main__":
    main()
