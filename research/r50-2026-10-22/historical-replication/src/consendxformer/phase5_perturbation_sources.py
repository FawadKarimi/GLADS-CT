"""Frozen six-source/archive validation and image-only runtime for Phase 5."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hashing import sha256_file, sha256_json
from .phase4_ablation_contracts import identity_rows
from .phase4_ablation_runner import reference_record, frozen_bundle, revalidate_inputs, guard_image_paths
from .phase4_attention_evidence import IDENTITY, array_record, supplementary_protocol
from .phase4_attention_runtime import _saved_replay, pinned_b1_api, load_frozen_b1
from .phase4_evaluation_metrics import _area_resize
from .phase4_evaluation_runner import (PROJECT_ROOT, _safe_source_member, load_saved_prediction_inputs,
    relative_source_snapshot, snapshot_source_files, assert_source_snapshot_unchanged)
from .phase4_posthoc import load_frozen_final_best_model, build_frozen_attention_loader
from .phase5_descriptive import identities
from .phase5_perturbation import (KEYS, SEEDS, CLASS_ORDER, rank_native, validate_images,
                                  perturbation_protocol, validate_probabilities)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_snapshot(target, incoming):
    for path, digest in incoming.items():
        if path in target and target[path] != digest:
            raise ValueError("Conflicting source/input snapshot")
        target[path] = digest


@dataclass(frozen=True)
class PerturbationInputs:
    ids: pd.DataFrame
    images: np.ndarray
    maps: dict
    prior: np.ndarray
    predictions: dict
    sources: dict
    source_records: dict
    archive_inputs: dict
    evidence_provenance: dict
    snapshot: dict
    source_directories: tuple


def load_perturbation_inputs(full_dirs, b1_dirs, attention_dir, attention_manifest_sha256):
    """No model construction, forwards, metrics, prior fitting or source repairs."""
    paths = [Path(p).resolve() for p in (*full_dirs, *b1_dirs)]
    if (len(full_dirs), len(b1_dirs)) != (3, 3) or len(set(paths)) != 6:
        raise ValueError("Exactly six distinct complete/B1 source directories required")
    root = Path(attention_dir).resolve()
    if (not isinstance(attention_manifest_sha256, str) or len(attention_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in attention_manifest_sha256)
            or sha256_file(root / "artifact_manifest.json") != attention_manifest_sha256):
        raise ValueError("Explicit matched-attention manifest SHA-256 differs")
    inputs = load_saved_prediction_inputs(full_dirs, b1_dirs)
    sources = {(m, s): (inputs.phase4 if m == "ConsenDxFormer" else inputs.b1)[s] for m, s in KEYS}
    records = {key: {"model": key[0], "seed": key[1], **reference_record(source),
                     "source_snapshot": relative_source_snapshot(source)} for key, source in sources.items()}
    snapshot = dict(inputs.source_snapshot)
    manifest, marker = read_json(root / "artifact_manifest.json"), read_json(root / "COMPLETED.json")
    eid = perturbation_protocol()["attention_reference_id"]
    if (manifest.get("evaluation_id") != eid or marker.get("evaluation_id") != eid
            or marker.get("artifact_manifest_sha256") != attention_manifest_sha256
            or marker.get("replay_consistent") is not True
            or marker.get("comparison_interpretation_permitted") is not True
            or manifest.get("replay_consistent") is not True
            or not isinstance(manifest.get("files"), dict)
            or manifest.get("bundle_sha256") != sha256_json(manifest["files"])):
        raise ValueError("Completed replay-consistent matched attention evidence required")
    merge_snapshot(snapshot, snapshot_source_files([root / "artifact_manifest.json", root / "COMPLETED.json"]))
    if snapshot[str((root / "artifact_manifest.json").resolve())] != attention_manifest_sha256:
        raise ValueError("Explicit attention manifest changed during source preflight")
    for name, digest in manifest["files"].items():
        canonical, member = _safe_source_member(root, name)
        if canonical != name or sha256_file(member) != digest:
            raise ValueError("Matched evidence checksum/path differs")
        merge_snapshot(snapshot, {str(member): digest})
    archive_names = {"test_inputs.npz", "location_prior.npz", *(f"seed_{s}_model_arrays.npz" for s in SEEDS)}
    required = archive_names | {"evaluation.json", "provenance/provenance.json", "provenance/supplementary_protocol.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("Required matched evidence member is not manifest-bound")
    provenance, evaluation = read_json(root / "provenance/provenance.json"), read_json(root / "evaluation.json")
    if (evaluation.get("protocol") != supplementary_protocol()
            or read_json(root / "provenance/supplementary_protocol.json") != supplementary_protocol()
            or evaluation.get("replay_consistent") is not True
            or evaluation.get("comparison_interpretation_permitted") is not True
            or sha256_json(provenance["input_file_snapshot"]) != provenance["input_file_snapshot_sha256"]):
        raise ValueError("Matched evidence policy/replay/input-snapshot differs")
    archived_records = provenance["source_experiments"]
    seen = set()
    for record in archived_records:
        key = (record["model"], record["seed"])
        if type(record["seed"]) is not int or key not in sources or key in seen:
            raise ValueError("Archived source model/seed is missing, duplicate or unexpected")
        seen.add(key)
        if record["descriptor"] != records[key]["descriptor"] or record["source_snapshot"] != records[key]["source_snapshot"]:
            raise ValueError("Archived model/checkpoint/prediction binding differs")
    if seen != set(KEYS):
        raise ValueError("Archive must bind all six exact sources")
    archives = {}
    for record in provenance["archives"]:
        name = record["file"]
        _, member = _safe_source_member(root, name)
        if name in archives or name not in archive_names or snapshot.get(str(member)) != record["sha256"]:
            raise ValueError("Archive ledger is not exact and manifest-bound")
        with np.load(member, allow_pickle=False) as saved:
            if set(saved.files) != set(record["arrays"]):
                raise ValueError("Saved archive keys differ")
            arrays = {k: saved[k].copy() for k in saved.files}
        if {k: array_record(v) for k, v in arrays.items()} != record["arrays"]:
            raise ValueError("Saved archive dtype/shape/array hash differs")
        archives[name] = arrays
    if set(archives) != archive_names:
        raise ValueError("Incomplete native evidence archives")
    saved = archives["test_inputs.npz"]
    ids = pd.DataFrame({k: saved[k] for k in IDENTITY})
    rows = identities(ids)
    if len(ids) != 183 or ids.patient_id.nunique() != 99:
        raise ValueError("Frozen test membership must contain 183 nodules / 99 patients")
    expected = identity_rows(inputs.supervision.loc[inputs.supervision.split == "test"])
    if sorted(rows, key=lambda r: r["nodule_id"]) != sorted(expected, key=lambda r: r["nodule_id"]):
        raise ValueError("Archive membership/labels differ from frozen supervision")
    predictions = {}
    for key, source in sources.items():
        identities(source.predictions)
        frame = source.predictions.set_index("nodule_id", verify_integrity=True).loc[ids.nodule_id].reset_index()
        if len(source.predictions) != 183 or identities(frame) != rows or not frame.predicted_class.isin(CLASS_ORDER).all():
            raise ValueError("Saved prediction identities/labels differ")
        predictions[key] = frame
    images = validate_images(saved["ct_crops"])
    if images.shape != (183, 1, 64, 64):
        raise ValueError("Frozen CT archive shape differs")
    soft, native, targets = saved["soft_masks_input"], saved["soft_masks_native_grid"], saved["target_distributions"]
    if soft.shape != images.shape or native.shape != (183, 8, 8) or targets.shape != native.shape:
        raise ValueError("Frozen mask/target grids differ")
    for value in (soft, native, targets):
        if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError("Invalid archived target values")
    if not np.array_equal(native, np.stack([_area_resize(m[0], (8, 8)) for m in soft])):
        raise ValueError("Native masks are not the unnormalized archived area averages")
    # Masks/targets are validated evidence only; they never leave this boundary for ranking/forwards.
    prior = archives["location_prior.npz"]["prior"]
    rank_native(prior)
    training = sorted(identity_rows(inputs.supervision.loc[inputs.supervision.split == "train"]),
                      key=lambda r: tuple(r[k] for k in IDENTITY))
    if (provenance["prior"]["array"] != array_record(prior)
            or provenance["prior"]["test_masks_used"] is not False
            or provenance["prior"]["n_nodules"] != 807 or len(training) != 807
            or provenance["prior"]["training_identities"] != training
            or provenance["prior"]["training_identities_sha256"] != sha256_json(training)
            or set(r["patient_id"] for r in training) & set(ids.patient_id)):
        raise ValueError("Existing training-only prior provenance/membership differs")
    maps = {}
    for model, seed in KEYS:
        prefix = "cdx" if model == "ConsenDxFormer" else "b1"
        arrays = archives[f"seed_{seed}_model_arrays.npz"]
        value = arrays[f"{prefix}_attention"]
        if value.shape != (183, 1, 8, 8):
            raise ValueError("Native DSA archive grid differs; no adaptation allowed")
        for item in value[:, 0]:
            rank_native(item)
        logits, probabilities = arrays[f"{prefix}_logits"], arrays[f"{prefix}_probabilities"]
        if logits.dtype != np.float32 or probabilities.dtype != np.float32:
            raise ValueError("Original reconstructed classifications must be float32")
        validate_probabilities(probabilities)
        replay = _saved_replay(predictions[(model, seed)], ids, logits, probabilities)
        if not replay["consistent"] or [CLASS_ORDER[j] for j in probabilities.argmax(axis=1)] != predictions[(model, seed)].predicted_class.tolist():
            raise ValueError("Archived classification disagrees with saved predictions/targets")
        maps[(model, seed)] = value
    merge_snapshot(snapshot, provenance["input_file_snapshot"])
    assert_source_snapshot_unchanged(snapshot)
    return PerturbationInputs(ids, images, maps, prior, predictions, sources, records, saved,
                              provenance, snapshot, tuple([*paths, root]))


class TorchImagePredictor:
    """One frozen image-only call, without gradient recording or autocast."""
    def __init__(self, model, device):
        self.model, self.device = model, device

    def __call__(self, images):
        import torch

        validate_images(images)
        if len(images) != 1 or self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("Batch-one frozen eval model required")
        with torch.no_grad(), torch.autocast(device_type=self.device, enabled=False):
            output = self.model(torch.from_numpy(images.copy()).to(self.device))
        result = {}
        for key in ("logits", "probs"):
            value = output[key]
            if value.dtype != torch.float32:
                raise ValueError("Model classifications must remain raw float32")
            result[key] = value.detach().cpu().numpy().copy()
        return result


def _model_state_hash(model):
    return sha256_json({k: array_record(v.detach().cpu().numpy()) for k, v in model.state_dict().items()})


@contextmanager
def frozen_predictors(inputs, *, device, cache_dir):
    """Reuse frozen-best/readiness/pinned-source contracts, never call a model here."""
    import torch

    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda")
    bundle, lock = frozen_bundle()
    _, readiness, live = revalidate_inputs(bundle.phase4_config, lock)
    guard_image_paths(live)
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    old_benchmark, old_cudnn = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    models, state_hashes, loaded_records, input_snapshot = {}, {}, [], {}
    try:
        with pinned_b1_api(Path(cache_dir)) as (model_class, baseline_report):
            for seed in SEEDS:
                complete = load_frozen_final_best_model(inputs.sources[("ConsenDxFormer", seed)].source_dir, device=device)
                b1 = load_frozen_b1(inputs.sources[("B1", seed)].source_dir, model_class, device=device)
                for name, frozen in (("ConsenDxFormer", complete), ("B1", b1)):
                    key = (name, seed)
                    if (reference_record(frozen.source) != reference_record(inputs.sources[key])
                            or frozen.checkpoint_sha256 != inputs.source_records[key]["best_checkpoint_sha256"]
                            or relative_source_snapshot(frozen.source) != inputs.source_records[key]["source_snapshot"]):
                        raise ValueError("Reloaded frozen-best source differs from archived binding")
                    models[key] = frozen.model
                    state_hashes[key] = _model_state_hash(frozen.model)
                    loaded_records.append({"model": name, "seed": seed, "state_sha256": state_hashes[key],
                                           "best_checkpoint_sha256": frozen.checkpoint_sha256})
                loader = build_frozen_attention_loader(complete)
                merge_snapshot(input_snapshot, loader.input_snapshot)
                # Only data reading: verify live unaugmented inputs against saved arrays before replay.
                position = 0
                for batch in loader:
                    n = len(batch["image"])
                    rows = [{k: batch[k][i] for k in IDENTITY} for i in range(n)]
                    if (rows != inputs.ids.iloc[position:position+n].to_dict("records")
                            or not np.array_equal(batch["image"].numpy(), inputs.images[position:position+n])
                            or not np.array_equal(batch["soft_consensus_mask"].numpy(), inputs.archive_inputs["soft_masks_input"][position:position+n])):
                        raise ValueError("Live frozen inputs/identities differ from archive")
                    position += n
                if position != 183:
                    raise ValueError("Live frozen loader did not retain all 183 nodules")
            merge_snapshot(input_snapshot, snapshot_source_files([
                PROJECT_ROOT / "configs/phase4_consensdxformer.yaml", PROJECT_ROOT / "configs/phase4_input_lock.json",
                PROJECT_ROOT / "locks/baseline_sources.yaml"]))
            assert_source_snapshot_unchanged(inputs.snapshot)
            assert_source_snapshot_unchanged(input_snapshot)
            metadata = {"baseline_source": baseline_report, "loaded_models": loaded_records,
                        "readiness": readiness.to_dict(), "live_input_snapshot": input_snapshot,
                        "deterministic_algorithms": True, "deterministic_warn_only": False,
                        "cudnn_benchmark": False, "cudnn_deterministic": True, "autocast": False}
            yield {k: TorchImagePredictor(m, device) for k, m in models.items()}, metadata
            if any(_model_state_hash(m) != state_hashes[k] for k, m in models.items()):
                raise ValueError("Frozen in-memory model state changed")
            assert_source_snapshot_unchanged(inputs.snapshot)
            assert_source_snapshot_unchanged(input_snapshot)
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old_benchmark, old_cudnn
