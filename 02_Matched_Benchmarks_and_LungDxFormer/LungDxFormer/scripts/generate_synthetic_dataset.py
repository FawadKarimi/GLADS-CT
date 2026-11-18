from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

def draw_circle(img, center, radius, value):
    cv2.circle(img, center, radius, value, thickness=-1)

def draw_irregular(img, center, radius, value):
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    points = []
    for a in angles:
        r = radius * np.random.uniform(0.7, 1.3)
        x = int(center[0] + r * np.cos(a))
        y = int(center[1] + r * np.sin(a))
        points.append([x, y])
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(img, [pts], value)

def make_sample(label_id: int, size: int = 96) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    noise = np.random.normal(0.15, 0.05, size=(size, size)).astype(np.float32)
    img += noise
    center = (np.random.randint(28, size-28), np.random.randint(28, size-28))

    if label_id == 0:  # benign
        draw_circle(img, center, np.random.randint(10, 16), np.random.uniform(0.6, 0.8))
    elif label_id == 1:  # indeterminate
        draw_circle(img, center, np.random.randint(12, 18), np.random.uniform(0.55, 0.75))
        cv2.GaussianBlur(img, (5, 5), 0, dst=img)
    else:  # malignant
        draw_irregular(img, center, np.random.randint(12, 20), np.random.uniform(0.7, 0.95))
        for _ in range(np.random.randint(4, 8)):
            angle = np.random.uniform(0, 2*np.pi)
            length = np.random.randint(8, 16)
            x2 = int(center[0] + length*np.cos(angle))
            y2 = int(center[1] + length*np.sin(angle))
            cv2.line(img, center, (x2, y2), np.random.uniform(0.7, 0.95), 1)

    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=120)
    parser.add_argument("--image_size", type=int, default=96)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    class_names = ["benign", "indeterminate", "malignant"]

    for i in range(args.num_samples):
        label_id = i % 3
        label_name = class_names[label_id]
        patient_id = f"patient_{i:04d}"
        patient_dir = out_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        image = make_sample(label_id, size=args.image_size)
        file_name = f"nodule_{i:04d}.png"
        rel_path = str(Path(patient_id) / file_name)
        cv2.imwrite(str(patient_dir / file_name), image)

        split = "train" if i < int(0.7 * args.num_samples) else "val" if i < int(0.85 * args.num_samples) else "test"
        records.append({
            "patient_id": patient_id,
            "image_path": rel_path,
            "label": label_name,
            "split": split,
        })

    pd.DataFrame(records).to_csv(out_dir / "metadata.csv", index=False)
    print(f"Created synthetic dataset at: {out_dir}")

if __name__ == "__main__":
    main()
