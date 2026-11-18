from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_multiclass_roc(y_true, y_prob, class_names, out_path: str):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    num_classes = len(class_names)
    onehot = np.eye(num_classes)[y_true]

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, class_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(onehot[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")
        except Exception:
            continue
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(loc="lower right")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
