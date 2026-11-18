from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(str(Path(__file__).resolve().parents[0]))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from lungdxformer.utils.config import load_yaml_config, parse_override_items, deep_update
from lungdxformer.utils.seed import set_seed
from lungdxformer.utils.logger import get_logger
from lungdxformer.data.dataset import (
    load_metadata,
    create_splits_from_df_or_csv_labels,
    LungNoduleDataset,
)
from lungdxformer.data.augmentation import BasicAugmenter
from lungdxformer.models.lungdxformer import LungDxFormer
from lungdxformer.training.trainer import Trainer

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    return parser

def main():
    args = build_argparser().parse_args()
    config = load_yaml_config(args.config)
    if args.override:
        config = deep_update(config, parse_override_items(args.override))

    set_seed(config["seed"])
    logger = get_logger(log_file="outputs/train.log")

    device = "cuda" if torch.cuda.is_available() and config.get("device", "auto") == "auto" else "cpu"
    logger.info(f"Using device: {device}")

    df = load_metadata(config["dataset"]["metadata_csv"], config["dataset"]["image_root"])
    splits = create_splits_from_df_or_csv_labels(
        df,
        train_ratio=config["dataset"]["train_ratio"],
        val_ratio=config["dataset"]["val_ratio"],
        test_ratio=config["dataset"]["test_ratio"],
        random_state=config["seed"],
    )

    train_aug = BasicAugmenter(**config["augmentation"]["train"], enabled=True)
    train_ds = LungNoduleDataset(splits.train_df, config["dataset"]["image_size"], config["dataset"]["in_channels"], augmenter=train_aug)
    val_ds = LungNoduleDataset(splits.val_df, config["dataset"]["image_size"], config["dataset"]["in_channels"], augmenter=None)

    train_loader = DataLoader(train_ds, batch_size=config["dataset"]["batch_size"], shuffle=True, num_workers=config["dataset"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=config["dataset"]["batch_size"], shuffle=False, num_workers=config["dataset"]["num_workers"])

    class_weights = None
    if config["training"]["class_weights"] == "auto":
        y_train = splits.train_df["label_id"].astype(int).values
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(config["dataset"]["num_classes"]),
            y=y_train,
        ).tolist()
    elif isinstance(config["training"]["class_weights"], list):
        class_weights = config["training"]["class_weights"]
    config["training"]["_computed_class_weights"] = class_weights

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

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    trainer = Trainer(model, optimizer, config, device, logger)
    history, best_metric = trainer.fit(train_loader, val_loader)
    logger.info(f"Training complete. Best val F1_macro: {best_metric}")

if __name__ == "__main__":
    main()
