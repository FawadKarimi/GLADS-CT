"""Exploratory supplementary attention evidence; existing metrics are reused."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd

from .hashing import sha256_file, sha256_json
from .phase3_artifacts import atomic_write_bytes
from .phase4_evaluation_metrics import (
    ATTENTION_ALIGNMENT_DEFINITION, _area_resize, _normalized_consensus,
    _validate_consensus_masks, attention_alignment_metrics,
    validate_attention_normalization,
)

METRICS = ("kl_consensus_to_attention", "spatial_correlation", "distribution_overlap", "dice", "iou")
IDENTITY = ("patient_id", "nodule_id", "series_instance_uid", "split", "true_class")
FLAGGED_NODULE = "LIDC-IDRI-0052__nodule-49c6132bfb77f3f1"


def supplementary_protocol() -> dict[str, Any]:
    return {
        "schema_version": 1, "protocol": "phase4_attention_evidence_v1",
        "analysis_status": "exploratory_post_hoc_proposed_after_original_attention_results",
        "preregistered": False, "tuning_permitted": False, "training_permitted": False,
        "source_final_model_commit": "e1b174490245c616e355ef7c0c6f7c3c239d3acd",
        "seeds": [42, 43, 44], "test_nodules": 183, "test_patients": 99,
        "training_nodules": 807, "selected_lambdas": {"ordinal": 0.5, "attention": 0.05},
        "alignment_definition": deepcopy(ATTENTION_ALIGNMENT_DEFINITION),
        "prediction_replay": {"atol": 1e-5, "rtol": 1e-4, "dtype": "float32_no_autocast"},
        "previous_scalar_replay": {"atol": 1e-8, "rtol": 1e-6},
        "exported_map_scalar_replay": {"atol": 1e-12, "rtol": 1e-12},
        "prior": "training-only: area-resize each mask, normalize each to unit mass, arithmetic nodule mean, normalize numerical drift once",
        "bootstrap": {"seed": 20260914, "resamples": 10000, "confidence_level": 0.95,
                      "rng": "PCG64", "interval": "percentile", "quantile_method": "linear",
                      "unit": "patient", "estimand": "nodule-mean paired alignment difference per trained seed",
                      "missing_correlation": "pairwise defined nodules only; preserve full patient draws, omit empty-defined replicates and report counts"},
        "difference_direction": ["ConsenDxFormer_minus_B1", "ConsenDxFormer_minus_prior", "B1_minus_prior"],
        "panel": {"seed": 42, "cases": "flagged plus low/lower-median/high mean CDX Dice across all three seeds; ascending value then nodule_id; deduplicate with reasons",
                  "flagged_nodule": FLAGGED_NODULE, "display": "CT clipped to [0,1]; consensus [0,1]; attention/prior divided by each map maximum for display only; nearest grid upsampling; alpha=0.5"},
        "interpretation": "spatial correspondence only; not causal explanation, not an attention-loss ablation, no superiority or population-wide significance claim",
    }


def array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError("Object arrays are forbidden in evidence archives")
    header = {"shape": list(array.shape), "dtype": array.dtype.str}
    digest = hashlib.sha256(sha256_json(header).encode("ascii") + array.tobytes()).hexdigest()
    return {**header, "array_sha256": digest}


def write_map_archive(path: str | Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Deterministic safe NPZ, atomically create-only; verify with pickle disabled."""
    buffer = io.BytesIO()
    records = {}
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(arrays.items()):
            if not name or not name.replace("_", "").isalnum():
                raise ValueError("Unsafe evidence array key")
            array = np.asarray(value)
            records[name] = array_record(array)
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, member.getvalue())
    destination = atomic_write_bytes(path, buffer.getvalue())
    with np.load(destination, allow_pickle=False) as loaded:
        if set(loaded.files) != set(records) or any(array_record(loaded[name]) != records[name] for name in records):
            raise ValueError("Evidence archive failed round-trip validation")
    return {"file": destination.name, "sha256": sha256_file(destination), "arrays": records}


def build_location_prior(
    training_rows: pd.DataFrame, masks: np.ndarray, grid: tuple[int, int], *, expected_count: int = 807,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Accept training rows only; the runtime separately binds the exact frozen cohort."""
    if len(training_rows) != expected_count or expected_count <= 0:
        raise ValueError("Location prior requires the exact frozen training row count")
    if not training_rows.columns.is_unique or not set(IDENTITY).issubset(training_rows):
        raise ValueError("Training prior lacks unique identity fields")
    for field in IDENTITY:
        if not training_rows[field].map(lambda v: isinstance(v, str) and bool(v) and not v.isspace()).all():
            raise ValueError("Training prior has invalid identity text")
    if not training_rows["split"].eq("train").all():
        raise ValueError("Only training rows may enter the location prior")
    if training_rows["nodule_id"].duplicated().any():
        raise ValueError("Duplicate training nodule in location prior")
    masks = _validate_consensus_masks(masks)
    if len(masks) != len(training_rows) or len(grid) != 2 or any(type(n) is not int or n <= 0 for n in grid):
        raise ValueError("Training mask count/grid mismatch")
    # Canonical order fixes floating-point reduction independent of row arrival.
    order = training_rows.reset_index(drop=True).sort_values(list(IDENTITY), kind="mergesort").index.to_numpy()
    distributions = np.stack([_normalized_consensus(_area_resize(masks[i], grid)) for i in order])
    prior = distributions.mean(axis=0)
    prior /= prior.sum()
    identities = training_rows.iloc[order].loc[:, list(IDENTITY)].to_dict("records")
    return prior, {"definition": supplementary_protocol()["prior"], "training_identities": identities,
                   "training_identities_sha256": sha256_json(identities), "n_nodules": len(identities),
                   "test_masks_used": False, "array": array_record(prior)}


def map_evidence(attention: np.ndarray, masks: np.ndarray, identities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Preserve raw maps; count foreground on the same drift-normalized maps as metrics."""
    raw = validate_attention_normalization(attention)
    masks = _validate_consensus_masks(masks)
    metrics = attention_alignment_metrics(raw, masks, identifiers=identities)
    resized = np.stack([_area_resize(mask, raw.shape[1:]) for mask in masks])
    targets = np.stack([_normalized_consensus(mask) for mask in resized])
    normalized = raw / raw.sum(axis=(1, 2), keepdims=True)
    rows = []
    for index, result in enumerate(metrics["per_sample"]):
        foreground = normalized[index] >= result["attention_threshold"]
        consensus = resized[index] >= result["consensus_threshold"]
        intersection = int((foreground & consensus).sum())
        target_count = int(consensus.sum())
        rows.append({**identities[index], **result,
                     "attention_foreground_count": int(foreground.sum()),
                     "consensus_foreground_count": target_count,
                     "intersection": intersection, "union": int((foreground | consensus).sum()),
                     "attention_constant": bool(np.ptp(normalized[index]) == 0),
                     "target_constant": bool(np.ptp(targets[index]) == 0),
                     "correlation_undefined": result["spatial_correlation"] is None,
                     "zero_overlap_reason": "empty_target_foreground" if target_count == 0 else (
                         "disjoint_nonempty_foregrounds" if intersection == 0 else "overlap"),
                     "native_shape": list(raw.shape[1:])})
    return {"metrics": metrics, "rows": rows, "resized_soft_masks": resized, "target_distributions": targets}


def paired_alignment_bootstrap(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    """Same patient draw on both sides; never pool trained seeds as patients."""
    columns = list(IDENTITY)
    sides = []
    for frame in (left, right):
        if frame.empty or not frame.columns.is_unique or not set((*IDENTITY, *METRICS)).issubset(frame):
            raise ValueError("Incomplete paired alignment rows")
        for field in IDENTITY:
            if not frame[field].map(lambda v: isinstance(v, str) and bool(v) and not v.isspace()).all():
                raise ValueError("Invalid paired identity")
        if not frame["split"].eq("test").all() or frame["nodule_id"].duplicated().any():
            raise ValueError("Paired rows must be unique test nodules")
        sides.append(frame.sort_values(columns, kind="mergesort").reset_index(drop=True))
    left, right = sides
    if left[columns].to_dict("records") != right[columns].to_dict("records"):
        raise ValueError("Paired alignment cohort mismatch")
    policy = supplementary_protocol()["bootstrap"]
    patients = sorted(left["patient_id"].unique())
    rng = np.random.Generator(np.random.PCG64(policy["seed"]))
    # Cluster multiplicities are equivalent to expanding every nodule of each
    # patient occurrence, while permitting vectorized 10,000-replicate means.
    draws = rng.integers(0, len(patients), size=(policy["resamples"], len(patients)))
    counts = np.zeros((policy["resamples"], len(patients)), dtype=np.int64)
    np.add.at(counts, (np.arange(len(draws))[:, None], draws), 1)
    results = {}
    for metric in METRICS:
        a, b = left[metric].to_numpy(dtype=float), right[metric].to_numpy(dtype=float)
        if np.isinf(a).any() or np.isinf(b).any() or (metric != "spatial_correlation" and (np.isnan(a).any() or np.isnan(b).any())):
            raise ValueError("Only undefined correlation may be missing")
        defined = np.isfinite(a) & np.isfinite(b)
        delta = a - b
        cluster_sums = np.array([np.where(defined & left["patient_id"].eq(p).to_numpy(), delta, 0).sum() for p in patients])
        cluster_n = np.array([(defined & left["patient_id"].eq(p).to_numpy()).sum() for p in patients])
        denominator = counts @ cluster_n
        values = (counts @ cluster_sums)[denominator > 0] / denominator[denominator > 0]
        lower, upper = np.quantile(values, [0.025, 0.975], method="linear") if len(values) else (None, None)
        results[metric] = {"mean_difference": float(delta[defined].mean()) if defined.any() else None,
                           "lower": None if lower is None else float(lower), "upper": None if upper is None else float(upper),
                           "n_pairwise_defined": int(defined.sum()), "n_missing_pairs": int((~defined).sum()),
                           "valid_replicates": len(values), "empty_defined_replicates": int((denominator == 0).sum())}
    return {"policy": policy, "n_patients": len(patients), "n_nodules": len(left),
            "patient_order": patients, "patient_draw_sha256": array_record(draws)["array_sha256"],
            "metrics": results, "significance_claimed": False}


def select_panel_cases(rows: pd.DataFrame, *, flagged: str = FLAGGED_NODULE) -> list[dict[str, Any]]:
    if set(rows["seed"]) != {42, 43, 44} or rows.duplicated(["seed", "nodule_id"]).any():
        raise ValueError("Panel selection needs each nodule once per final seed")
    if not rows.groupby("nodule_id")["seed"].nunique().eq(3).all():
        raise ValueError("Panel selection must retain all seeds per nodule")
    means = rows.groupby("nodule_id", sort=True)["dice"].mean().reset_index().sort_values(["dice", "nodule_id"])
    if flagged not in set(means["nodule_id"]):
        raise ValueError("The flagged nodule is absent from the frozen test cohort")
    choices = [(flagged, "flagged_zero_alignment_case")]
    for position, reason in ((0, "low"), ((len(means) - 1) // 2, "lower_median"), (len(means) - 1, "high")):
        choices.append((means.iloc[position]["nodule_id"], reason))
    selected: dict[str, list[str]] = {}
    for nodule, reason in choices:
        selected.setdefault(nodule, []).append(reason)
    return [{"nodule_id": nodule, "reasons": reasons, "display_seed": 42} for nodule, reasons in selected.items()]


def write_overlay_panel(path: str | Path, cases: Sequence[Mapping[str, Any]], identities: pd.DataFrame,
                        ct: np.ndarray, soft: np.ndarray, b1: np.ndarray, cdx: np.ndarray, prior: np.ndarray) -> Path:
    """Render only actual input arrays; no generated or substitute heatmaps."""
    import cv2

    tile = 144
    panel = np.full((len(cases) * (tile + 52), tile * 5, 3), 255, dtype=np.uint8)
    for row_number, case in enumerate(cases):
        matches = np.flatnonzero(identities["nodule_id"].eq(case["nodule_id"]).to_numpy())
        if len(matches) != 1:
            raise ValueError("Panel identity must be exact and unique")
        i = int(matches[0])
        image = cv2.resize(np.clip(ct[i].squeeze(), 0, 1), (tile, tile), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_GRAY2BGR)
        top = row_number * (tile + 52)
        cv2.putText(panel, f"{case['nodule_id']} | {','.join(case['reasons'])} | seed 42", (4, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (0, 0, 0), 1)
        for column, (name, values) in enumerate((("CT", None), ("Consensus", soft[i]), ("B1", b1[i]), ("ConsenDxFormer", cdx[i]), ("Train-only prior", prior))):
            output = gray
            if values is not None:
                scale = 1.0 if column == 1 else float(values.max())
                heat = cv2.resize(values / scale, (tile, tile), interpolation=cv2.INTER_NEAREST)
                colors = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_VIRIDIS)
                output = cv2.addWeighted(gray, 0.5, colors, 0.5, 0)
            cv2.putText(panel, name, (column * tile + 3, top + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 1)
            panel[top + 44:top + 44 + tile, column * tile:(column + 1) * tile] = output
    okay, png = cv2.imencode(".png", panel)
    if not okay:
        raise ValueError("Could not encode overlay panel")
    return atomic_write_bytes(path, png.tobytes())
