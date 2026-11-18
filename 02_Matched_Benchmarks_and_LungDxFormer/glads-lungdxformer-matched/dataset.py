
from __future__ import annotations
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2

class LIDCSliceDataset(Dataset):
    """
    Protocol-matched LIDC Slice Dataset.
    Loads 64x64 float32 slices and applies paper-matched augmentations.
    """
    def __init__(self, manifest_df: pd.DataFrame, is_train: bool = False, config: dict = None):
        self.df = manifest_df.reset_index(drop=True)
        self.is_train = is_train
        self.config = config or {}

    def __len__(self):
        return len(self.df)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # img is [64, 64] float32 in [0, 1]
        # 1. Random Horizontal Flip (50%)
        if np.random.rand() > 0.5:
            img = np.fliplr(img)

        # 2. Random Vertical Flip (50%)
        if np.random.rand() > 0.5:
            img = np.flipud(img)

        # 3. Random Rotation (+- 10 deg) & Scale (0.9 to 1.1)
        angle = np.random.uniform(-10.0, 10.0)
        scale = np.random.uniform(0.9, 1.1)
        h, w = img.shape
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # 4. Small Translation (+- 5%)
        tx = np.random.uniform(-0.05, 0.05) * w
        ty = np.random.uniform(-0.05, 0.05) * h
        T = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, T, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # 5. Intensity Jitter (+- 10%)
        jitter = np.random.uniform(0.9, 1.1)
        img = np.clip(img * jitter, 0.0, 1.0)

        # 6. Gaussian Noise (std = 0.01)
        noise = np.random.normal(0.0, 0.01, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

        return img

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        
        # Load .npy
        img = np.load(img_path).astype(np.float32)
        
        # Ensure 64x64
        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
            
        # Ensure minmax normalized in [0, 1]
        mn, mx = img.min(), img.max()
        if mx - mn > 1e-6:
            img = (img - mn) / (mx - mn)
        else:
            img = np.zeros_like(img)

        if self.is_train:
            img = self._augment(img)

        # Tensor shape: [1, 64, 64]
        tensor_img = torch.from_numpy(img).unsqueeze(0).float()
        label = int(row["label"])
        
        # Reader target soft distribution
        rt = torch.tensor([
            float(row["reader_target_0"]),
            float(row["reader_target_1"]),
            float(row["reader_target_2"])
        ], dtype=torch.float32)
        
        # Inter-reader disagreement
        disagreement = float(row.get("disagreement", 0.0))

        return {
            "image": tensor_img,
            "label": torch.tensor(label, dtype=torch.long),
            "reader_target": rt,
            "disagreement": torch.tensor(disagreement, dtype=torch.float32),
            "slice_id": str(row["slice_id"]),
            "patient_id": str(row["patient_id"])
        }
