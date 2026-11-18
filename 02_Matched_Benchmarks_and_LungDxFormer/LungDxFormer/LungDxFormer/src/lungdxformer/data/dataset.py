from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from lungdxformer.data.label_mapping import normalize_label

@dataclass
class DatasetSplits:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

def _read_image(path: str, image_size: int) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        img = np.load(p)
    else:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.resize(img.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    if img.max() > 1.0:
        img = img / 255.0
    return img.astype(np.float32)

def load_metadata(metadata_csv: str, image_root: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    required = {"patient_id", "image_path", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"metadata_csv missing columns: {missing}")
    if image_root:
        root = Path(image_root)
        df["image_path"] = df["image_path"].apply(lambda p: str((root / p).resolve()) if not Path(p).is_absolute() else p)
    df["label_id"] = df["label"].apply(normalize_label)
    return df

def create_patient_level_splits(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int = 42,
) -> DatasetSplits:
    patients = df["patient_id"].astype(str).unique()
    train_p, temp_p = train_test_split(patients, test_size=(1 - train_ratio), random_state=random_state)
    rel_test = test_ratio / (val_ratio + test_ratio)
    val_p, test_p = train_test_split(temp_p, test_size=rel_test, random_state=random_state)

    train_df = df[df["patient_id"].astype(str).isin(train_p)].reset_index(drop=True)
    val_df = df[df["patient_id"].astype(str).isin(val_p)].reset_index(drop=True)
    test_df = df[df["patient_id"].astype(str).isin(test_p)].reset_index(drop=True)
    return DatasetSplits(train_df=train_df, val_df=val_df, test_df=test_df)

def create_splits_from_df_or_csv_labels(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int = 42,
) -> DatasetSplits:
    if "split" in df.columns:
        return DatasetSplits(
            train_df=df[df["split"].str.lower() == "train"].reset_index(drop=True),
            val_df=df[df["split"].str.lower() == "val"].reset_index(drop=True),
            test_df=df[df["split"].str.lower() == "test"].reset_index(drop=True),
        )
    return create_patient_level_splits(df, train_ratio, val_ratio, test_ratio, random_state)

class LungNoduleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, in_channels: int = 1, augmenter=None):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.in_channels = in_channels
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        img = _read_image(row["image_path"], self.image_size)
        if self.augmenter is not None:
            img = self.augmenter(img)
        if self.in_channels == 1:
            tensor = torch.from_numpy(img).unsqueeze(0)
        else:
            tensor = torch.from_numpy(np.stack([img]*self.in_channels, axis=0))
        label = int(row["label_id"])
        sample = {
            "image": tensor.float(),
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": str(row["patient_id"]),
            "path": str(row["image_path"]),
        }
        return sample
