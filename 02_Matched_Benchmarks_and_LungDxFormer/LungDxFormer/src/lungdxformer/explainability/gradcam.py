from __future__ import annotations
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Minimal Grad-CAM on the fused feature map stored by LungDxFormer.
    """
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def _normalize(self, cam):
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def generate(self, image_tensor: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        out = self.model(image_tensor)
        logits = out["logits"]
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        feature_map = self.model.last_feature_map
        if feature_map is None:
            raise RuntimeError("No feature map captured for Grad-CAM.")

        feature_map.retain_grad()
        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        grads = feature_map.grad
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feature_map).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = self._normalize(cam[0, 0].detach().cpu())
        return cam
