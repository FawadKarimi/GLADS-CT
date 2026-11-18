from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lungdxformer.training.losses import build_loss
from lungdxformer.training.scheduler import build_scheduler
from lungdxformer.training.early_stopping import EarlyStopping
from lungdxformer.evaluation.metrics import classification_metrics
from lungdxformer.utils.checkpoints import save_checkpoint

class Trainer:
    def __init__(self, model, optimizer, config: dict, device: str, logger):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.logger = logger
        self.scheduler = build_scheduler(optimizer, config["training"])
        self.early_stopping = EarlyStopping(patience=config["training"]["early_stopping_patience"], mode="min")
        self.criterion = build_loss(
            class_weights=config["training"].get("_computed_class_weights"),
            label_smoothing=config["training"].get("label_smoothing", 0.0),
            device=device,
        )
        for p in config["paths"].values():
            Path(p).mkdir(parents=True, exist_ok=True)

    def _run_epoch(self, loader: DataLoader, train: bool = True):
        self.model.train(train)
        epoch_loss = 0.0
        all_y, all_pred, all_prob, all_paths = [], [], [], []

        iterator = tqdm(loader, leave=False, desc="train" if train else "val")
        for batch in iterator:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            with torch.set_grad_enabled(train):
                out = self.model(x)
                logits = out["logits"]
                loss = self.criterion(logits, y)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip_norm = self.config["training"].get("grad_clip_norm", None)
                    if clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                    self.optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred = logits.argmax(dim=1).detach().cpu().numpy()

            all_y.extend(y.detach().cpu().tolist())
            all_pred.extend(pred.tolist())
            all_prob.extend(prob.tolist())
            all_paths.extend(batch["path"])

        epoch_loss /= max(1, len(loader.dataset))
        metrics = classification_metrics(all_y, all_pred, np.array(all_prob), num_classes=self.config["dataset"]["num_classes"])
        metrics["loss"] = float(epoch_loss)
        return metrics, all_y, all_pred, all_prob, all_paths

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        best_metric = None
        history = []

        for epoch in range(1, self.config["training"]["epochs"] + 1):
            train_metrics, *_ = self._run_epoch(train_loader, train=True)
            val_metrics, y_true, y_pred, y_prob, paths = self._run_epoch(val_loader, train=False)

            if self.scheduler is not None:
                self.scheduler.step(val_metrics["loss"])

            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(record)

            self.logger.info(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1_macro']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1_macro']:.4f} val_auc={val_metrics['auc_macro_ovr']}"
            )

            improved = self.early_stopping.step(val_metrics["loss"])
            if improved:
                best_metric = val_metrics["f1_macro"]
                ckpt_path = str(Path(self.config["paths"]["checkpoint_dir"]) / "best_model.pt")
                save_checkpoint(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.config,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    ckpt_path,
                )
                pred_df = pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred})
                pred_df.to_csv(Path(self.config["paths"]["predictions_dir"]) / "best_val_predictions.csv", index=False)

            if self.early_stopping.should_stop:
                self.logger.info("Early stopping triggered.")
                break

        with open(Path(self.config["paths"]["metrics_dir"]) / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return history, best_metric
