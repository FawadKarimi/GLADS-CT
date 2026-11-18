"""Common three-class diagnostics accumulated by identity, never by batch mean."""
import math

from .bindings import CLASS_ORDER
from .reuse import canonical, digest
from .selection import ValidationEpoch

METRIC_DEFINITION = {
    "class_order": list(CLASS_ORDER), "prediction": "argmax_first_tie",
    "confusion": "rows_true_columns_predicted", "macro_f1": "mean_all_three",
    "zero_denominator": 0, "secondary": "accuracy_equals_micro_f1",
    "diagnostic_ce": "unweighted_log_softmax_float32_sum_float64_divide_n",
}


def metric_definition_sha256():
    return digest(canonical(METRIC_DEFINITION))


class DiagnosticAccumulator:
    def __init__(self, inputs, *, split):
        self.inputs, self.split = inputs, split
        self.expected = inputs.ids(split)
        self._rows = {}

    def add(self, ids, logits, labels):
        import torch
        ids = tuple(ids)
        if (not ids or len(set(ids)) != len(ids) or any(i in self._rows for i in ids)
                or not set(ids) <= set(self.expected)):
            raise ValueError("duplicate, missing or foreign metric identity")
        wanted = self.inputs.supervision(ids, split=self.split)["labels"]
        if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64
                or labels.device.type != "cpu" or not torch.equal(labels, wanted)
                or not isinstance(logits, torch.Tensor) or logits.device.type != "cpu"
                or logits.dtype != torch.float32 or tuple(logits.shape) != (len(ids), 3)
                or not torch.isfinite(logits).all()):
            raise ValueError("bound labels and finite CPU float32 diagnostic logits required")
        with torch.no_grad():
            predictions = logits.argmax(dim=1).tolist()
            ce = -logits.log_softmax(dim=1).gather(1, labels[:, None]).flatten()
            if not torch.isfinite(ce).all():
                raise ValueError("nonfinite diagnostic CE")
            staged = {i: (int(y), int(p), float(c)) for i, y, p, c in
                      zip(ids, labels.tolist(), predictions, ce.tolist())}
        self._rows.update(staged)

    def summary(self):
        if set(self._rows) != set(self.expected):
            raise ValueError("complete frozen split required; no partial metrics for selection")
        confusion = [[0] * 3 for _ in range(3)]
        for y, p, _ in self._rows.values():
            confusion[y][p] += 1
        per_class = {}
        for c, name in enumerate(CLASS_ORDER):
            tp = confusion[c][c]
            support = sum(confusion[c])
            predicted = sum(row[c] for row in confusion)
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[name] = dict(precision=precision, recall=recall, f1=f1, support=support)
        n = len(self._rows)
        accuracy = sum(confusion[c][c] for c in range(3)) / n
        return {"split": self.split, "n": n, "confusion_matrix": confusion,
                "per_class": per_class, "macro_f1": math.fsum(v["f1"] for v in per_class.values()) / 3,
                "accuracy": accuracy, "micro_f1": accuracy,
                "diagnostic_ce": math.fsum(self._rows[i][2] for i in self.expected) / n,
                "input_lock_sha256": self.inputs.sha256,
                "prediction_identity_sha256": digest(canonical(list(self.expected))),
                "metric_definition_sha256": metric_definition_sha256()}

    def selection_record(self, epoch):
        if self.split != "val":
            raise ValueError("only validation metrics can select a checkpoint")
        summary = self.summary()
        return ValidationEpoch(epoch, summary["macro_f1"], summary["diagnostic_ce"])
