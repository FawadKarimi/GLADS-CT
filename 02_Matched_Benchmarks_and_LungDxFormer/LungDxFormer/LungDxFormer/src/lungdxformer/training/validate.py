from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from lungdxformer.evaluation.metrics import classification_metrics

@torch.no_grad()
def evaluate_model(model, loader: DataLoader, criterion, device: str, num_classes: int = 3):
    model.eval()
    total_loss = 0.0
    all_y, all_pred, all_prob = [], [], []

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        out = model(x)
        logits = out["logits"]
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        all_y.extend(y.detach().cpu().tolist())
        all_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_prob.extend(torch.softmax(logits, dim=1).detach().cpu().numpy().tolist())

    total_loss /= max(1, len(loader.dataset))
    metrics = classification_metrics(all_y, all_pred, np.array(all_prob), num_classes=num_classes)
    metrics["loss"] = float(total_loss)
    return metrics
