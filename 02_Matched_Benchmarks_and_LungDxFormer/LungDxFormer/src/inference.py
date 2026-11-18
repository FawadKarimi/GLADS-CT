from __future__ import annotations
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[0]))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from lungdxformer.utils.config import load_yaml_config
from lungdxformer.utils.checkpoints import load_checkpoint
from lungdxformer.utils.visualization import overlay_heatmap, save_image_grid
from lungdxformer.models.lungdxformer import LungDxFormer
from lungdxformer.explainability.gradcam import GradCAM
from lungdxformer.explainability.attention_maps import upscale_attention_map

def load_image(path: str, image_size: int):
    p = Path(path)
    if p.suffix.lower() == ".npy":
        img = np.load(p).astype(np.float32)
    else:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    if img.max() > 1.0:
        img = img / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    return img, tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="outputs/predictions/inference")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    device = "cuda" if torch.cuda.is_available() and config.get("device", "auto") == "auto" else "cpu"

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

    image, tensor = load_image(args.input, config["dataset"]["image_size"])
    tensor = tensor.to(device)
    with torch.set_grad_enabled(True):
        out = model(tensor)
        probs = out["probs"][0].detach().cpu().numpy()
        pred = int(probs.argmax())
        class_name = config["dataset"]["class_names"][pred]

        attn = upscale_attention_map(out["attention_map"], image.shape[:2])[0, 0].detach().cpu().numpy()
        cam = GradCAM(model).generate(tensor, target_class=pred).numpy()

    attn_overlay = overlay_heatmap(image, attn)
    cam_overlay = overlay_heatmap(image, cam)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image_grid(
        [image, attn_overlay, cam_overlay],
        [f"Input | Pred: {class_name}", "Dynamic Attention", "Grad-CAM"],
        str(out_dir / "inference_visualization.png"),
        cmap="gray",
    )
    np.save(out_dir / "probabilities.npy", probs)
    with open(out_dir / "prediction.txt", "w", encoding="utf-8") as f:
        f.write(f"Predicted class: {class_name}\n")
        for name, p in zip(config["dataset"]["class_names"], probs):
            f.write(f"{name}: {float(p):.6f}\n")

if __name__ == "__main__":
    main()
