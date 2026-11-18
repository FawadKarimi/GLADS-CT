from __future__ import annotations
import numpy as np
from lungdxformer.data.preprocessing import resize_image

def crop_centered_roi(slice_img: np.ndarray, center_x: int, center_y: int, patch_size: int = 96) -> np.ndarray:
    h, w = slice_img.shape[:2]
    half = patch_size // 2
    x1, x2 = max(0, center_x - half), min(w, center_x + half)
    y1, y2 = max(0, center_y - half), min(h, center_y + half)
    roi = slice_img[y1:y2, x1:x2]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        raise ValueError("Empty ROI crop produced.")
    return resize_image(roi, patch_size)

def crop_bbox_roi(slice_img: np.ndarray, x1: int, y1: int, x2: int, y2: int, patch_size: int = 96) -> np.ndarray:
    roi = slice_img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        raise ValueError("Empty ROI crop produced.")
    return resize_image(roi, patch_size)
