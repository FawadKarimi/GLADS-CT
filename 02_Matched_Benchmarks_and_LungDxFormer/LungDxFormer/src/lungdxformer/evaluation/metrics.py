from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

def classification_metrics(y_true, y_pred, y_prob=None, num_classes: int = 3):
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_per, rec_per, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    auc_macro = None
    if y_prob is not None:
        try:
            y_true_onehot = np.eye(num_classes)[np.array(y_true)]
            auc_macro = roc_auc_score(y_true_onehot, y_prob, average="macro", multi_class="ovr")
        except Exception:
            auc_macro = None

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_per_class": prec_per.tolist(),
        "recall_per_class": rec_per.tolist(),
        "f1_per_class": f1_per.tolist(),
        "support_per_class": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "auc_macro_ovr": None if auc_macro is None else float(auc_macro),
    }
