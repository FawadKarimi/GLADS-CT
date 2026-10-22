"""Saved-prediction trust boundary; no models, scoring or attention evidence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .hashing import sha256_file
from .phase4_ablation_contracts import identity_rows
from .phase4_ablation_runner import load_ablation_source, reference_record
from .phase4_evaluation_runner import (
    load_saved_prediction_inputs, assert_source_snapshot_unchanged,
    relative_source_snapshot,
)
from .phase5_descriptive import IDENTITY, MODELS, SEEDS, VARIANTS, canonical_predictions, identities


def strict_path(value):
    path = Path(value).absolute()
    if path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink/redirected paths are not accepted")
    return path


def merge_snapshot(target, incoming):
    for name, digest in incoming.items():
        if name in target and target[name] != digest:
            raise ValueError("Conflicting source snapshot")
        target[name] = digest


def assert_inference_snapshot_unchanged(snapshot):
    for name in snapshot:
        strict_path(name)
    assert_source_snapshot_unchanged(snapshot)


@dataclass(frozen=True)
class InferenceInputs:
    frames: dict
    expected_identities: list
    records: list
    snapshot: dict
    directories: tuple


def prediction_records(frames):
    return [{"model": model, "seed": seed,
             "rows": frames[(model, seed)][[*IDENTITY, "predicted_class"]].to_dict("records")}
            for model in MODELS for seed in SEEDS]


def validate_records(records, expected):
    """Validate exported hard-prediction identities without computing correctness."""
    from .phase5_bounded_mean import NodulePrediction
    if not isinstance(records, list) or len(records) != 18:
        raise ValueError("Exactly 18 prediction groups required")
    expected_frame = pd.DataFrame(expected)
    identities(expected_frame)
    if len(expected_frame) != 183 or expected_frame.patient_id.nunique() != 99:
        raise ValueError("Frozen cohort requires 183 nodules / 99 patients")
    expected_set = {tuple(row[k] for k in IDENTITY) for row in expected}
    groups = {}
    for record in records:
        model, seed = record["model"], record["seed"]
        if model not in MODELS or type(seed) is not int or seed not in SEEDS or (model, seed) in groups:
            raise ValueError("Wrong/duplicate model or seed")
        rows = record["rows"]
        if any(set(row) != {*IDENTITY, "predicted_class"} for row in rows):
            raise ValueError("Prediction export fields differ")
        parsed = tuple(NodulePrediction(**row, seed=seed) for row in rows)
        if (len(parsed) != 183 or len({r.nodule_id for r in parsed}) != 183
                or {r.identity for r in parsed} != expected_set):
            raise ValueError("Missing/extra/unpaired identities or labels")
        groups[(model, seed)] = parsed
    if set(groups) != {(m, s) for m in MODELS for s in SEEDS}:
        raise ValueError("Missing model/seed group")
    return groups


def load_inference_inputs(full_dirs, b1_dirs, ablation_dirs):
    if (len(full_dirs), len(b1_dirs), len(ablation_dirs)) != (3, 3, 12):
        raise ValueError("Exactly 3 complete, 3 B1 and 12 OFF sources required")
    paths = tuple(strict_path(p) for p in (*full_dirs, *b1_dirs, *ablation_dirs))
    if len(set(paths)) != 18:
        raise ValueError("Exactly 18 distinct sources required")
    for path in paths:
        if not path.is_dir():
            raise ValueError("Explicit completed source directory missing")
        # Check members before historical loaders open them, not just the two
        # explicitly bound checkpoint/prediction files after loading.
        for member in path.rglob("*"):
            strict_path(member)
    # This calls the distinct full and B1 loaders and their unchanged historical
    # checkpoint, prediction, supervision, fingerprint and seed-set guards.
    saved = load_saved_prediction_inputs(full_dirs, b1_dirs)
    snapshot, frames, records = dict(saved.source_snapshot), {}, []
    for model, sources, requested in ((MODELS[0], saved.phase4, paths[:3]),
                                      (MODELS[1], saved.b1, paths[3:6])):
        if set(sources) != set(SEEDS) or {s.source_dir for s in sources.values()} != set(requested):
            raise ValueError("Loaded full/B1 sources differ from explicit paths/seeds")
        for seed in SEEDS:
            source = sources[seed]
            if type(source.descriptor["seed"]) is not int or source.descriptor["seed"] != seed:
                raise ValueError("Source seed mismatch")
            frames[(model, seed)] = source.predictions.copy(deep=True)
            records.append({"model": model, "seed": seed, **reference_record(source),
                            "source_snapshot": relative_source_snapshot(source)})
    ablations = {}
    for path in paths[6:]:
        source = load_ablation_source(path)
        variant, seed = source.identity["variant"], source.identity["seed"]
        key = (variant, seed)
        if (source.source_dir != path or variant not in VARIANTS or type(seed) is not int
                or seed not in SEEDS or key in ablations):
            raise ValueError("Wrong/duplicate ablation variant/seed/source")
        if source.identity["full_model_reference"] != reference_record(saved.phase4[seed]):
            raise ValueError("Ablation same-seed full reference mismatch")
        if identity_rows(source.supervision) != identity_rows(saved.supervision):
            raise ValueError("Ablation supervision identities/labels/splits mismatch")
        ablations[key] = source
        frames[(f"{variant}_off", seed)] = source.predictions["test"].copy(deep=True)
        merge_snapshot(snapshot, source.source_snapshot)
        records.append({"model": f"{variant}_off", "seed": seed, "source_dir": str(path),
                        "identity": source.identity,
                        "best_checkpoint_sha256": sha256_file(path / "checkpoints/best.pt"),
                        "source_snapshot": relative_source_snapshot(source)})
    if set(ablations) != {(v, s) for v in VARIANTS for s in SEEDS}:
        raise ValueError("All four OFF variants and three seeds required")
    frames = canonical_predictions(frames)
    expected = identity_rows(saved.supervision.loc[saved.supervision.split == "test"])
    validate_records(prediction_records(frames), expected)
    # Explicitly bind the original prediction/checkpoint bytes, never summaries.
    for record in records:
        path = Path(record["source_dir"])
        for field, member in (("prediction_sha256", "predictions/test.csv"),
                              ("best_checkpoint_sha256", "checkpoints/best.pt")):
            target = strict_path(path / member)
            digest = sha256_file(target)
            if snapshot.get(str(target)) != digest:
                raise ValueError("Prediction/checkpoint missing from trusted snapshot")
            if field in record and record[field] != digest:
                raise ValueError("Checkpoint identity mismatch")
            record[field] = digest
    assert_inference_snapshot_unchanged(snapshot)
    return InferenceInputs(frames, expected, sorted(records, key=lambda r: (r["model"], r["seed"])),
                           snapshot, paths)
