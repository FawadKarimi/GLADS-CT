from __future__ import annotations

LABEL_TO_ID = {"benign": 0, "indeterminate": 1, "malignant": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

def normalize_label(label):
    if isinstance(label, str):
        lab = label.strip().lower()
        if lab in LABEL_TO_ID:
            return LABEL_TO_ID[lab]
        if lab.isdigit():
            return int(lab)
    return int(label)

def map_malignancy_score_to_class(score: float) -> int:
    """
    Default 3-class mapping decision:
    1-2 -> benign
    3   -> indeterminate
    4-5 -> malignant
    """
    score = float(score)
    if score <= 2:
        return 0
    if score == 3:
        return 1
    return 2
