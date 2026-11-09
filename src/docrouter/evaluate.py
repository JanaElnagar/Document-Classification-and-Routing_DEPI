"""Metrics shared by every model, so results are comparable (FR16)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score


def compute_metrics(y_true, y_pred, labels):
    """Per-class precision/recall/F1 plus macro and weighted averages, flat for MLflow."""
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "accuracy": report["accuracy"],
    }
    for label in labels:
        key = label.lower()
        metrics[f"{key}_precision"] = report[label]["precision"]
        metrics[f"{key}_recall"] = report[label]["recall"]
        metrics[f"{key}_f1"] = report[label]["f1-score"]
    return metrics


def save_confusion_matrix(y_true, y_pred, labels, path):
    """Save a confusion-matrix PNG and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax,
                                            xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
