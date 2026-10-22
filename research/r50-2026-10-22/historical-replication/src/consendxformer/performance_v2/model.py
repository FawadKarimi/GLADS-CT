"""Specified R50 hybrid; random CPU construction only, without a weight loader."""
import torch
from torch import nn
from torch.nn import functional as F

from .rng import initialization_scope, stream_seed

MODEL_ID = "performance_v2_r50_hybrid_v2"
OBJECTIVES = ("HARD", "LEGACY", "SHARED-010", "SEPARATE-010", "CAPACITY-010")


def batch_norm(channels):
    return nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1,
                          affine=True, track_running_stats=True)


class Position(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, 49, 128))
        nn.init.trunc_normal_(self.position, mean=0, std=0.02, a=-2, b=2)

    def forward(self, tokens):
        return tokens + self.position


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(128, eps=1e-5, elementwise_affine=True)
        self.attention = nn.MultiheadAttention(
            128, 4, dropout=0.0, bias=True, add_bias_kv=False,
            add_zero_attn=False, kdim=128, vdim=128, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(128, eps=1e-5, elementwise_affine=True)
        self.mlp = nn.Sequential(nn.Linear(128, 512, bias=True),
                                 nn.GELU(approximate="none"),
                                 nn.Linear(512, 128, bias=True))

    def forward(self, tokens):
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized,
                                     need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(128, 1, 1, bias=True)

    def forward(self, features):
        logits = self.projection(features)
        attention = logits.flatten(2).softmax(dim=-1).reshape_as(logits)
        return features * attention, attention


class DeepFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_projection = nn.Conv2d(2048, 128, 1, bias=True)
        self.fusion = nn.Sequential(nn.Conv2d(256, 128, 1, bias=False),
                                    batch_norm(128), nn.ReLU(inplace=False))

    def forward(self, c3, attended):
        return self.fusion(torch.cat((self.cnn_projection(c3), attended), dim=1))


class GatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lateral_projection = nn.Sequential(
            nn.Conv2d(channels, 128, 1, bias=False), batch_norm(128), nn.ReLU(inplace=False))
        self.gate = nn.Sequential(nn.Conv2d(256, 128, 1, bias=True), nn.Sigmoid())
        self.refine = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                    batch_norm(128), nn.ReLU(inplace=False))

    def forward(self, deep, lateral):
        deep = F.interpolate(deep, size=lateral.shape[-2:], mode="bilinear",
                             align_corners=False, antialias=False)
        lateral = self.lateral_projection(lateral)
        gate = self.gate(torch.cat((deep, lateral), dim=1))
        return self.refine(deep + gate * lateral)


def _check_backbone(backbone):
    """Check constructor semantics without forwarding tensors or reading files."""
    stem = backbone.conv1
    if (stem.in_channels, stem.out_channels, stem.kernel_size, stem.stride,
        stem.padding, stem.bias) != (3, 64, (7, 7), (2, 2), (3, 3), None):
        raise ValueError("unexpected R50 stem")
    pool = backbone.maxpool
    if (pool.kernel_size, pool.stride, pool.padding, pool.ceil_mode) != (3, 2, 1, False):
        raise ValueError("unexpected R50 pool")
    for index, count in enumerate((3, 4, 6, 3), 1):
        stage = getattr(backbone, f"layer{index}")
        if len(stage) != count:
            raise ValueError("unexpected R50 stage depth")
        for number, block in enumerate(stage):
            stride = (2, 2) if index > 1 and number == 0 else (1, 1)
            width = 64 * 2**(index - 1)
            if (block.conv1.out_channels != width or block.conv3.out_channels != 4 * width
                    or block.conv1.stride != (1, 1) or block.conv2.stride != stride
                    or block.conv2.kernel_size != (3, 3) or block.conv2.padding != (1, 1)
                    or block.conv3.stride != (1, 1)):
                raise ValueError("unexpected R50 bottleneck")
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            if module.groups != 1 or module.dilation != (1, 1) or module.bias is not None:
                raise ValueError("unexpected R50 convolution")
        if isinstance(module, nn.BatchNorm2d):
            if (module.eps != 1e-5 or module.momentum != 0.1 or not module.affine
                    or not module.track_running_stats):
                raise ValueError("unexpected R50 normalization")


class R50Hybrid(nn.Module):
    model_id = MODEL_ID
    pilot_execution_authorized = False

    def __init__(self, *, seed=42, objective="HARD", initialization="RANDOM",
                 capacity_control=False):
        super().__init__()
        stream_seed(seed, "encoder_initialization")
        if initialization != "RANDOM":
            raise ValueError("pretrained integration remains blocked; no weight loader exists")
        if objective not in OBJECTIVES or (objective == "CAPACITY-010" and not capacity_control):
            raise ValueError("unreviewed objective or inactive conditional capacity control")
        from torchvision.models import resnet50
        self.objective = objective
        self.seed = seed
        self.initialization = initialization
        with initialization_scope(seed, "encoder_initialization"):
            self.backbone = resnet50(
                weights=None, zero_init_residual=False, groups=1, width_per_group=64,
                replace_stride_with_dilation=[False, False, False], norm_layer=batch_norm,
            )
            _check_backbone(self.backbone)
            # Source constructor's fc consumes only the isolated backbone stream.
            self.backbone.fc = nn.Identity()
            self.backbone.avgpool = nn.Identity()
        with initialization_scope(seed, "shared_fusion_initialization"):
            self.token_projection = nn.Conv2d(2048, 128, 1, bias=True)
            self.position = Position()
            self.transformer = nn.Sequential(TransformerBlock(), TransformerBlock())
            self.dynamic_attention = SpatialAttention()
            self.deep_fusion = DeepFusion()
            self.c2_fusion = GatedFusion(1024)
            self.c1_fusion = GatedFusion(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        with initialization_scope(seed, "diagnostic_head_initialization"):
            self.diagnostic_head = nn.Linear(128, 3, bias=True)
        self.reader_head = None
        if objective in ("SEPARATE-010", "CAPACITY-010"):
            with initialization_scope(seed, "auxiliary_head_initialization"):
                self.reader_head = nn.Linear(128, 3, bias=True)

    def _forward_impl(self, images, include_reader):
        if (images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224)
                or images.shape[0] < 1 or images.dtype != torch.float32
                or not torch.isfinite(images).all()):
            raise ValueError("expected finite float32 [B,3,224,224] normalized images")
        b = self.backbone
        x = b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(images)))))
        c1 = b.layer2(x)
        c2 = b.layer3(c1)
        c3 = b.layer4(c2)
        tokens = self.token_projection(c3).flatten(2).transpose(1, 2)
        tokens = self.transformer(self.position(tokens))
        features = tokens.transpose(1, 2).reshape(images.shape[0], 128, 7, 7)
        attended, attention = self.dynamic_attention(features)
        deep = self.deep_fusion(c3, attended)
        fused = self.c1_fusion(self.c2_fusion(deep, c2), c1)
        pooled = self.pool(fused).flatten(1)
        logits = self.diagnostic_head(pooled)
        return {"logits": logits, "probs": logits.softmax(dim=-1),
                "reader_logits": None if self.reader_head is None or not include_reader else self.reader_head(pooled),
                "c1_features": c1, "c2_features": c2, "c3_features": c3,
                "transformer_features": features, "attention_map": attention,
                "deep_fused_features": deep, "fused_features": fused,
                "pooled_features": pooled}

    def forward(self, images):
        return self._forward_impl(images, include_reader=True)

    def diagnostic_probabilities(self, images):
        """Diagnostic-only API. A reader output never enters classification."""
        if self.training:
            raise ValueError("diagnosis requires eval mode")
        return self._forward_impl(images, include_reader=False)["probs"]
