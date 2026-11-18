"""Image-only frozen B1/CDX inference for exploratory attention evidence."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .baseline_evaluation import CLASS_ORDER
from .baseline_source import verify_baseline_source, snapshot_source_tree, assert_source_snapshot_unchanged as assert_baseline_unchanged
from .hashing import sha256_file
from .phase3_checkpoints import validate_checkpoint_record, select_best_validation_epoch
from .phase3_execution import _load_persisted_checkpoint, _module_is_within_source, _reject_cached_baseline_modules
from .phase4_evaluation_runner import PROJECT_ROOT, SavedPredictionSource, load_b1_prediction_source, assert_source_snapshot_unchanged
from .phase4_posthoc import FrozenFinalModel, FrozenAttentionLoader, _validated_attention_source_cohort
from .phase4_attention_evidence import IDENTITY, supplementary_protocol


@contextmanager
def pinned_b1_api(cache_dir: Path):
    """Import the actual pinned class, isolating cache lookup as Phase 3 does."""
    root = PROJECT_ROOT / "external" / "LungDxFormer"
    report = verify_baseline_source(root, PROJECT_ROOT / "locks" / "baseline_sources.yaml")
    snapshot = snapshot_source_tree(root)
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "baseline_common.yaml").read_text(encoding="utf-8"))
    source_path = (root / config["baseline"]["project_subdir"] / "src").resolve()
    _reject_cached_baseline_modules()
    old_path, old_bytecode, old_prefix = list(sys.path), sys.dont_write_bytecode, sys.pycache_prefix
    old_env = {name: os.environ.get(name) for name in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")}
    sys.path.insert(0, str(source_path))
    sys.dont_write_bytecode, sys.pycache_prefix = True, str(cache_dir.resolve())
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPYCACHEPREFIX"] = sys.pycache_prefix
    try:
        module = importlib.import_module("lungdxformer.models.lungdxformer")
        for name, imported in list(sys.modules.items()):
            if name == "lungdxformer" or name.startswith("lungdxformer."):
                if not _module_is_within_source(imported, source_path):
                    raise ValueError("B1 component imported outside the pinned source")
        yield module.LungDxFormer, report.to_dict()
        assert_baseline_unchanged(snapshot, snapshot_source_tree(root))
        if verify_baseline_source(root, PROJECT_ROOT / "locks" / "baseline_sources.yaml").to_dict() != report.to_dict():
            raise ValueError("Pinned B1 source changed during evidence inference")
    finally:
        for name in list(sys.modules):
            if name == "lungdxformer" or name.startswith("lungdxformer."):
                del sys.modules[name]
        sys.path[:] = old_path
        sys.dont_write_bytecode, sys.pycache_prefix = old_bytecode, old_prefix
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def construct_b1(model_class: Any, config: dict[str, Any]) -> Any:
    """The exact constructor arguments used by execute_locked_lungdxformer."""
    dataset, model = config["dataset"], config["model"]
    keys = ("embed_dim", "transformer_heads", "transformer_layers", "transformer_mlp_ratio",
            "dropout", "use_transformer", "use_positional_encoding", "use_spatial_attention",
            "fusion_type", "use_raw_transformer_in_fusion", "classifier_hidden_dim")
    return model_class(
        **{name: dataset[name] for name in ("image_size", "in_channels", "num_classes")},
        cnn_channels=tuple(model["cnn_channels"]), **{name: model[name] for name in keys},
    )


@dataclass(frozen=True)
class FrozenB1:
    model: Any
    source: SavedPredictionSource
    checkpoint_sha256: str
    config: dict[str, Any]


def load_frozen_b1(source_dir: Path, model_class: Any, *, device: str) -> FrozenB1:
    import torch

    source = load_b1_prediction_source(source_dir)
    identity = json.loads((source.source_dir / "config" / "experiment_identity.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((source.source_dir / "config" / "phase1_effective.yaml").read_text(encoding="utf-8"))
    phase3_config = yaml.safe_load((source.source_dir / "config" / "phase3_effective.yaml").read_text(encoding="utf-8"))
    config["phase3_reproducibility"] = phase3_config["determinism"]
    path = source.source_dir / "checkpoints" / "best.pt"
    record = validate_checkpoint_record(_load_persisted_checkpoint(torch, path, device=device), expected_kind="best")
    history = json.loads((source.source_dir / "history" / "epoch_history.json").read_text(encoding="utf-8"))
    selection = select_best_validation_epoch(history)
    expected = {
        "experiment_id": identity["experiment_id"], "seed": identity["seed"], "baseline_variant": "B1",
        "config_sha256": identity["combined_config_sha256"],
        "baseline_source_commit": identity["baseline_source_commit"], "project_commit": identity["project_commit"],
        "metadata_sha256": identity["metadata_sha256"], "cohort_sha256": identity["cohort_sha256"],
        "ordered_sample_sha256": identity["ordered_sample_sha256"], "frozen_split_sha256": identity["frozen_split_bundle_sha256"],
        "class_order": list(CLASS_ORDER), "effective_baseline_config": config,
        "epoch": selection["epoch"], "best_epoch": selection["epoch"], "best_validation_loss": selection["value"],
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"Persisted B1 best checkpoint disagrees with {field}")
    if record.get("validation_metrics", {}).get("loss") != selection["value"]:
        raise ValueError("B1 best validation loss differs from history")
    model = construct_b1(model_class, config)
    model.load_state_dict(record["model_state"], strict=True)
    model.to(device).eval().requires_grad_(False)
    assert_source_snapshot_unchanged(source.source_snapshot)
    return FrozenB1(model, source, sha256_file(path), config)


def _saved_replay(predictions: pd.DataFrame, identities: pd.DataFrame, logits: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    by_nodule = predictions.set_index("nodule_id", verify_integrity=True)
    aligned = by_nodule.loc[identities["nodule_id"]].reset_index()
    if aligned[list(IDENTITY)].to_dict("records") != identities[list(IDENTITY)].to_dict("records"):
        raise ValueError("Replayed classification rows differ from saved identities/labels")
    tolerance = supplementary_protocol()["prediction_replay"]
    reports = {}
    details = []
    for name, actual, prefix in (("logits", logits, "logit"), ("probabilities", probabilities, "prob")):
        saved = aligned[[f"{prefix}_{label}" for label in CLASS_ORDER]].to_numpy(dtype=float)
        if actual.shape != saved.shape or not np.isfinite(actual).all():
            raise ValueError("Invalid reconstructed classification outputs")
        close = np.isclose(actual, saved, atol=tolerance["atol"], rtol=tolerance["rtol"])
        reports[name] = {"consistent": bool(close.all()), "max_absolute_difference": float(np.abs(actual - saved).max()),
                         "differing_values": int((~close).sum())}
        for i, j in np.argwhere(~close):
            details.append({"nodule_id": identities.iloc[i]["nodule_id"], "kind": name, "class": CLASS_ORDER[j],
                            "saved": float(saved[i, j]), "reconstructed": float(actual[i, j])})
    return {"tolerance": tolerance, **reports, "discrepancies": details,
            "consistent": all(value["consistent"] for value in reports.values())}


def infer_matched_attention(cdx: FrozenFinalModel, b1: FrozenB1, loader: FrozenAttentionLoader) -> dict[str, Any]:
    """Identical image tensors, no masks in either forward API, no gradients."""
    import torch

    if not isinstance(loader, FrozenAttentionLoader):
        raise TypeError("An input-sealed frozen attention loader is required")
    expected, cohort = _validated_attention_source_cohort(cdx)
    if loader.ordered_test_cohort_sha256 != cohort["ordered_test_cohort_sha256"]:
        raise ValueError("Loader does not bind the exact frozen test cohort")
    if b1.source.descriptor["seed"] != cdx.source.descriptor["seed"]:
        raise ValueError("B1/CDX frozen model seeds differ")
    for model in (b1.model, cdx.model):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Attention inference requires frozen eval-mode models")
    assert_source_snapshot_unchanged(loader.input_snapshot)
    result: dict[str, list] = {name: [] for name in ("ct", "masks", "identities", "b1_attention", "cdx_attention", "b1_logits", "cdx_logits", "b1_probabilities", "cdx_probabilities")}
    actual_cohort = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(cdx.device)
            count = len(images)
            result["ct"].append(images.cpu().numpy().copy())
            result["masks"].append(batch["soft_consensus_mask"].cpu().numpy().copy())
            actual_cohort.extend({field: batch[field][i] for field in expected[0]} for i in range(count))
            result["identities"].extend({field: batch["true_class" if field == "true_class" else field][i] for field in IDENTITY} for i in range(count))
            for name, model in (("b1", b1.model), ("cdx", cdx.model)):
                output = model(images.clone())
                attention = output["attention_map"].detach().cpu().numpy()
                if attention.ndim != 4 or attention.shape[:2] != (count, 1):
                    raise ValueError("Expected actual native attention shape [N,1,H,W]")
                result[f"{name}_attention"].append(attention.copy())
                result[f"{name}_logits"].append(output["logits"].detach().cpu().numpy().copy())
                result[f"{name}_probabilities"].append(output["probs"].detach().cpu().numpy().copy())
    if actual_cohort != expected:
        raise ValueError("Inference membership/order differs from frozen input rows")
    arrays = {name: np.concatenate(values) for name, values in result.items() if name != "identities"}
    if arrays["b1_attention"].shape != arrays["cdx_attention"].shape:
        raise ValueError("B1 and ConsenDxFormer native attention grids differ; no grid adaptation permitted")
    identities = pd.DataFrame(result["identities"])
    for source in (cdx.source, b1.source):
        assert_source_snapshot_unchanged(source.source_snapshot)
    assert_source_snapshot_unchanged(loader.input_snapshot)
    replay = {name: _saved_replay(source.predictions, identities, arrays[f"{name}_logits"], arrays[f"{name}_probabilities"])
              for name, source in (("b1", b1.source), ("cdx", cdx.source))}
    return {"arrays": arrays, "identities": identities, "replay": replay, "source_cohort": cohort,
            "actual_attention_shapes": {name: list(arrays[f"{name}_attention"].shape) for name in ("b1", "cdx")}}
