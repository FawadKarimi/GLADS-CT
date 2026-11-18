from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def save_image_grid(images, titles, out_path: str, cmap: str = "gray") -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    image = image.astype(np.float32)
    image = (image - image.min()) / max(1e-8, image.max() - image.min())
    heatmap = heatmap.astype(np.float32)
    heatmap = (heatmap - heatmap.min()) / max(1e-8, heatmap.max() - heatmap.min())
    heat_rgb = plt.cm.jet(heatmap)[..., :3]
    if image.ndim == 2:
        image_rgb = np.stack([image]*3, axis=-1)
    else:
        image_rgb = image
    overlay = (1-alpha) * image_rgb + alpha * heat_rgb
    overlay = np.clip(overlay, 0, 1)
    return overlay
