from __future__ import annotations
import numpy as np
import cv2

def hu_clip(volume: np.ndarray, min_hu: int = -1000, max_hu: int = 400) -> np.ndarray:
    return np.clip(volume, min_hu, max_hu)

def minmax_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mn, mx = image.min(), image.max()
    if mx - mn < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return (image - mn) / (mx - mn)

def zscore_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    std = image.std()
    if std < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return (image - image.mean()) / std

def denoise_image(image: np.ndarray, method: str = "median") -> np.ndarray:
    image = image.astype(np.float32)
    if method == "median":
        return cv2.medianBlur(image, 3)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (3, 3), 0)
    return image

def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)

def segment_lung_simple(slice_img: np.ndarray) -> np.ndarray:
    """
    Lightweight heuristic lung-region mask for 2D slices.
    Not a substitute for production-grade medical segmentation.
    """
    img = minmax_normalize(slice_img)
    img_u8 = (img * 255).astype(np.uint8)
    _, th = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    return (th > 0).astype(np.uint8)

def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return image * mask
