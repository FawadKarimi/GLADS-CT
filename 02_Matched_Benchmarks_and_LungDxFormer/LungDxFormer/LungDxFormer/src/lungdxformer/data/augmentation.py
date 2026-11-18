from __future__ import annotations
import random
import numpy as np
import cv2

class BasicAugmenter:
    def __init__(
        self,
        hflip: bool = True,
        vflip: bool = False,
        rotation_deg: float = 15,
        translation_frac: float = 0.05,
        intensity_jitter: float = 0.05,
        gaussian_noise_std: float = 0.01,
        enabled: bool = True,
    ) -> None:
        self.hflip = hflip
        self.vflip = vflip
        self.rotation_deg = rotation_deg
        self.translation_frac = translation_frac
        self.intensity_jitter = intensity_jitter
        self.gaussian_noise_std = gaussian_noise_std
        self.enabled = enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image.astype(np.float32)

        img = image.astype(np.float32).copy()
        h, w = img.shape[:2]

        if self.hflip and random.random() < 0.5:
            img = np.fliplr(img)
        if self.vflip and random.random() < 0.5:
            img = np.flipud(img)

        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        tx = random.uniform(-self.translation_frac, self.translation_frac) * w
        ty = random.uniform(-self.translation_frac, self.translation_frac) * h
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        M[:, 2] += [tx, ty]
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        scale = 1.0 + random.uniform(-self.intensity_jitter, self.intensity_jitter)
        shift = random.uniform(-self.intensity_jitter, self.intensity_jitter)
        img = img * scale + shift

        noise = np.random.normal(0, self.gaussian_noise_std, size=img.shape).astype(np.float32)
        img = img + noise
        img = np.clip(img, 0.0, 1.0)
        return img.astype(np.float32)
