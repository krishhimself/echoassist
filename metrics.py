"""
metrics.py  —  confusion matrix + per-class precision/recall/F1 on the test set

Usage:
  .venv\\Scripts\\python.exe metrics.py
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import make_model, _transform, SPEC_DIR, MODEL_PATH

BATCH_SIZE = 32
OUT_DIR    = Path("metrics_out")


def evaluate_test_set():
    device = torch.device("cpu")
    test_ds     = datasets.ImageFolder(SPEC_DIR / "test", transform=_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    classes = test_ds.classes

    model = make_model().to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs.to(device)).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return np.array(all_labels), np.array(all_preds), classes


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    thresh = cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    y_true, y_pred, classes = evaluate_test_set()
    n = len(classes)

    cm     = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    report = classification_report(y_true, y_pred, target_names=classes,
                                    digits=3, zero_division=0)

    print(f"Test set size: {len(y_true)}")
    print(f"Classes: {classes}\n")

    print("Confusion matrix (rows=true, cols=predicted):")
    print("        " + "  ".join(f"{c:>8}" for c in classes))
    for i, row in enumerate(cm):
        print(f"{classes[i]:>8}  " + "  ".join(f"{v:>8}" for v in row))
    print()

    print(report)

    plot_path = OUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, classes, plot_path)
    print(f"Saved confusion matrix plot -> {plot_path}")


if __name__ == "__main__":
    main()
