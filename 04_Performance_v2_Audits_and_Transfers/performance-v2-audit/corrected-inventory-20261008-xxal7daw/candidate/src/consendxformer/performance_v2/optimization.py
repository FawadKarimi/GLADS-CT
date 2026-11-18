"""Exhaustive four-group preparation. No optimizer step or training loop."""
import math
from torch import nn

ENCODERS = {"backbone", "token_projection", "position", "transformer",
            "dynamic_attention", "deep_fusion", "c2_fusion", "c1_fusion"}
HEADS = {"diagnostic_head", "reader_head"}


def optimizer_groups(model):
    """Reject unknown and aliased parameters before constructing an optimizer."""
    groups = [dict(name=name, lr=lr, weight_decay=decay, params=[])
              for name, lr, decay in (
                  ("encoder_weights", 1e-4, 1e-4),
                  ("encoder_bias_normalization", 1e-4, 0.0),
                  ("head_weights", 1e-3, 1e-4),
                  ("head_bias_normalization", 1e-3, 0.0))]
    all_names = list(model.named_parameters(remove_duplicate=False))
    ids = [id(p) for _, p in all_names]
    if len(ids) != len(set(ids)):
        raise ValueError("aliased parameter identities")
    modules = dict(model.named_modules())
    ledger = []
    for name, param in all_names:
        if not param.requires_grad:
            raise ValueError("reviewed topology requires all parameters trainable")
        root = name.split(".")[0]
        if root not in ENCODERS | HEADS:
            raise ValueError("unclassified parameter scope: " + name)
        parent, leaf = name.rsplit(".", 1)
        owner = modules[parent]
        if name == "position.position":
            if tuple(param.shape) != (1, 49, 128):
                raise ValueError("unexpected position table")
            decay = False
        elif isinstance(owner, (nn.BatchNorm2d, nn.LayerNorm)) and leaf in ("weight", "bias"):
            decay = False
        elif isinstance(owner, (nn.Conv2d, nn.Linear)) and leaf in ("weight", "bias"):
            decay = leaf == "weight"
        elif isinstance(owner, nn.MultiheadAttention) and leaf in ("in_proj_weight", "in_proj_bias"):
            decay = leaf == "in_proj_weight"
        else:
            raise ValueError("unclassified parameter: " + name)
        index = (2 if root in HEADS else 0) + (0 if decay else 1)
        groups[index]["params"].append(param)
        ledger.append({"name": name, "shape": list(param.shape), "group": groups[index]["name"],
                       "lr": groups[index]["lr"], "weight_decay": groups[index]["weight_decay"]})
    if not all(g["params"] for g in groups):
        raise ValueError("missing reviewed parameter group")
    assigned = [id(p) for g in groups for p in g["params"]]
    if len(assigned) != len(ids) or set(assigned) != set(ids):
        raise ValueError("incomplete parameter coverage")
    return groups, ledger


def make_adamw(model):
    from torch.optim import AdamW
    groups, ledger = optimizer_groups(model)
    return AdamW(groups, betas=(0.9, 0.999), eps=1e-8), ledger


def epoch_multiplier(epoch):
    if type(epoch) is not int or not 1 <= epoch <= 100:
        raise ValueError("epoch must be in reviewed range 1..100")
    if epoch <= 5:
        return 0.1 + 0.9 * (epoch - 1) / 4
    return 0.01 + 0.99 * (1 + math.cos(math.pi * (epoch - 6) / 94)) / 2


def set_epoch_learning_rates(optimizer, epoch):
    expected = {"encoder_weights": (1e-4, 1e-4), "encoder_bias_normalization": (1e-4, 0),
                "head_weights": (1e-3, 1e-4), "head_bias_normalization": (1e-3, 0)}
    multiplier = epoch_multiplier(epoch)
    names = [g.get("name") for g in optimizer.param_groups]
    if len(names) != 4 or set(names) != set(expected):
        raise ValueError("optimizer group identity mismatch")
    for group in optimizer.param_groups:
        if group["weight_decay"] != expected[group["name"]][1]:
            raise ValueError("optimizer decay drift")
    for group in optimizer.param_groups:
        group["lr"] = expected[group["name"]][0] * multiplier
