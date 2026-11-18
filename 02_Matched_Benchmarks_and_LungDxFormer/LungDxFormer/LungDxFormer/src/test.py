from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[0]))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from lungdxformer.utils.config import load_yaml_config
from lungdxformer.utils.logger import get_logger
from lungdxformer.utils.checkpoints import load_checkpoint
from lungdxformer.data.dataset import load_metadata, create_splits_from_df_or_csv_labels, LungNoduleDataset
from lungdxformer.models.lungdxformer import LungDxFormer
from lungdxformer.evaluation.metrics import classification_metrics
from lungdxformer.evaluation.confusion_matrix import plot_confusion_matrix
from lungdxformer.evaluation.roc_auc import plot_multiclass_roc

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None)
    return parser

@torch.no_grad()
def main():
    args = build_argparser().parse_args()
    config = load_yaml_config(args.config)
    if args.metadata_csv:
        config["dataset"]["metadata_csv"] = args.metadata_csv
    if args.image_root:
        config["dataset"]["image_root"] = args.image_root

    logger = get_logger(log_file="outputs/test.log")
    device = "cuda" if torch.cuda.is_available() and config.get("device", "auto") == "auto" else "cpu"

    df = load_metadata(config["dataset"]["metadata_csv"], config["dataset"]["image_root"])
    splits = create_splits_from_df_or_csv_labels(
        df,
        train_ratio=config["dataset"]["train_ratio"],
        val_ratio=config["dataset"]["val_ratio"],
        test_ratio=config["dataset"]["test_ratio"],
        random_state=config["seed"],
    )
    test_ds = LungNoduleDataset(splits.test_df, config["dataset"]["image_size"], config["dataset"]["in_channels"], augmenter=None)
    test_loader = DataLoader(test_ds, batch_size=config["dataset"]["batch_size"], shuffle=False, num_workers=config["dataset"]["num_workers"])

    model = LungDxFormer(
        image_size=config["dataset"]["image_size"],
        in_channels=config["dataset"]["in_channels"],
        num_classes=config["dataset"]["num_classes"],
        cnn_channels=tuple(config["model"]["cnn_channels"]),
        embed_dim=config["model"]["embed_dim"],
        transformer_heads=config["model"]["transformer_heads"],
        transformer_layers=config["model"]["transformer_layers"],
        transformer_mlp_ratio=config["model"]["transformer_mlp_ratio"],
        dropout=config["model"]["dropout"],
        use_transformer=config["model"]["use_transformer"],
        use_positional_encoding=config["model"]["use_positional_encoding"],
        use_spatial_attention=config["model"]["use_spatial_attention"],
        fusion_type=config["model"]["fusion_type"],
        use_raw_transformer_in_fusion=config["model"]["use_raw_transformer_in_fusion"],
        classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, y_pred, y_prob, paths = [], [], [], []
    for batch in test_loader:
        x = batch["image"].to(device)
        out = model(x)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(batch["label"].tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())
        paths.extend(batch["path"])

    metrics = classification_metrics(y_true, y_pred, np.array(y_prob), num_classes=config["dataset"]["num_classes"])
    Path(config["paths"]["metrics_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(config["paths"]["metrics_dir"]) / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred}).to_csv(
        Path(config["paths"]["predictions_dir"]) / "test_predictions.csv", index=False
    )

    class_names = config["dataset"]["class_names"]
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, str(Path(config["paths"]["plots_dir"]) / "confusion_matrix.png"), normalize=False)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, str(Path(config["paths"]["plots_dir"]) / "confusion_matrix_normalized.png"), normalize=True)
    plot_multiclass_roc(y_true, y_prob, class_names, str(Path(config["paths"]["plots_dir"]) / "roc_curves.png"))

    logger.info(f"Test metrics: {metrics}")

if __name__ == "__main__":
    main()
