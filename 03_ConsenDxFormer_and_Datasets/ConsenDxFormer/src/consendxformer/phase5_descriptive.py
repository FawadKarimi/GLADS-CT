"""Pure, exploratory pointing/accuracy metrics; no models or formal tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_evaluation import CLASS_ORDER
from .hashing import sha256_json
from .phase4_attention_evidence import IDENTITY, FLAGGED_NODULE, array_record
from .phase4_bootstrap import validate_paired_prediction_rows

SEEDS = (42, 43, 44)
VARIANTS = ("attention_alignment", "disagreement_weighting", "ordinal_auxiliary", "hierarchical_fusion")
MODELS = ("ConsenDxFormer", "B1", *(f"{v}_off" for v in VARIANTS))
SCOPE = "pointing_and_accuracy_only"
PROPOSAL_SHA256 = "4d3f7500071a5c352b2a918c0d26ffabc90b4e11af5982cdfdef7880e1054620"


def unresolved_hypotheses():
    prefixes = ("B1", "attention_off", "disagreement_off", "ordinal_off", "fusion_off")
    return [{"hypothesis_id": f"C_{prefix}_{seed}", "seed": seed,
             "comparator": model, "endpoint": "secondary_patient_weighted_correctness",
             "status": "unresolved_not_computed", "reason": "method_calibration_unresolved"}
            for prefix, model in zip(prefixes, MODELS[1:]) for seed in SEEDS]


def descriptive_protocol():
    return {
        "schema_version": 1, "protocol": "phase5_descriptive_v1", "scope": SCOPE,
        "proposal_sha256_lf": PROPOSAL_SHA256, "analysis_status": "exploratory_post_hoc",
        "preregistered": False, "seeds": list(SEEDS), "models": list(MODELS),
        "test_nodules": 183, "test_patients": 99, "native_shape": [8, 8],
        "model_forwards": 0, "training_permitted": False,
        "pointing": {"mask": "unnormalized_archived_native_soft_mask", "threshold": 0.5,
                     "ties": "exact_maximum_then_first_row_major_index", "empty_target": "miss_retained",
                     "prior": "existing_training_only_archive_once", "flagged_nodule": FLAGGED_NODULE,
                     "denominator": "all_183_nodules", "causal_interpretation": False},
        "accuracy": {"individual_intervals": 18, "paired_intervals": 15, "resamples": 10000,
                     "seed": 20260914, "rng": "PCG64", "unit": "patient",
                     "draws": "one_common_ledger_all_models_and_seeds",
                     "estimand": "nodule_weighted_accuracy", "difference": "complete_minus_comparator",
                     "quantiles": [0.025, 0.975], "quantile_method": "linear",
                     "conditional_on_fit": True, "descriptive": True, "multiplicity_adjusted": False,
                     "simultaneous_coverage": False},
        "excluded": {"insertion_deletion": "not_executed", "formal_testing": "unresolved",
                     "multiplicity": "unresolved", "patient_weighted_correctness": "not_computed",
                     "calibration_campaign": "not_executed"},
        "hypotheses": unresolved_hypotheses(), "full_supplemental_completion": False,
    }


def load_protocol(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if sha256_json(value) != sha256_json(descriptive_protocol()):
        raise ValueError("Pointing/accuracy protocol differs from reviewed v1")
    return value


def identities(frame):
    """Validate exact identities without stripping, coercion or row filtering."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("Nonempty, unique-column identity table required")
    if not set(IDENTITY).issubset(frame):
        raise ValueError("Missing identity fields")
    records = frame[list(IDENTITY)].to_dict("records")
    for row in records:
        if any(not isinstance(v, str) or not v or v.strip() != v for v in row.values()):
            raise ValueError("Missing or malformed exact identity")
        if row["split"] != "test" or row["true_class"] not in CLASS_ORDER:
            raise ValueError("Only frozen three-class test identities are accepted")
    if frame.nodule_id.duplicated().any():
        raise ValueError("Duplicate nodule identity")
    return records


def canonical_predictions(frames):
    expected = {(m, s) for m in MODELS for s in SEEDS}
    if set(frames) != expected or any(type(s) is not int for _, s in frames):
        raise ValueError("Exactly 18 model/seed prediction sources required")
    reference = frames[(MODELS[0], SEEDS[0])]
    identities(reference)
    result = {}
    for key in [(m, s) for m in MODELS for s in SEEDS]:
        identities(frames[key])
        _, matched, _ = validate_paired_prediction_rows(reference, frames[key])
        result[key] = matched
    return result


def validate_native_inputs(frame, attention, soft_masks):
    rows = identities(frame)
    maps, soft = np.asarray(attention), np.asarray(soft_masks)
    if (maps.ndim != 4 or maps.shape[:2] != (len(rows), 1)
            or 0 in maps.shape or soft.shape != (len(rows), *maps.shape[2:])):
        raise ValueError("Expected matched native [N,1,H,W] maps and [N,H,W] soft masks")
    for values in (maps, soft):
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Native maps/masks must be finite values in [0,1]")
    if not np.allclose(maps.sum(axis=(1, 2, 3), dtype=np.float64), 1, rtol=0, atol=1e-6):
        raise ValueError("Native attention must already have unit spatial mass")
    return rows, maps, soft


def pointing_records(frame, attention, soft_masks, *, model, seed):
    """Native-grid peak hits, including every empty target. Never normalize masks."""
    rows, maps, soft = validate_native_inputs(frame, attention, soft_masks)
    records = []
    for row, value, mask in zip(rows, maps[:, 0], soft):
        flat = value.reshape(-1)
        tied = np.flatnonzero(flat == flat.max())
        index = int(tied[0])
        r, c = divmod(index, value.shape[1])
        foreground = mask >= 0.5
        count, hit = int(foreground.sum()), int(foreground[r, c])
        records.append({**row, "model": model, "seed": seed,
                        "map_sha256": array_record(value)["array_sha256"],
                        "soft_mask_sha256": array_record(mask)["array_sha256"],
                        "native_height": value.shape[0], "native_width": value.shape[1],
                        "peak_row": r, "peak_column": c, "peak_flat_index": index,
                        "peak_value": float(flat[index]), "tie_count": len(tied),
                        "maximizing_indices": json.dumps(tied.tolist(), separators=(",", ":")),
                        "constant_map": bool(len(tied) == len(flat)), "foreground_count": count,
                        "empty_target": count == 0, "hit": hit,
                        "reason": "empty_native_binary_target" if not count else ("hit" if hit else "peak_outside_target")})
    return records


def pointing_tables(frame, maps, soft, prior):
    if set(maps) != {(m, s) for m in MODELS[:2] for s in SEEDS}:
        raise ValueError("Pointing requires both native model maps for every seed")
    prior = np.asarray(prior)
    if prior.shape != np.asarray(soft).shape[1:]:
        raise ValueError("Prior and target native grids differ")
    records = []
    for model in MODELS[:2]:
        for seed in SEEDS:
            records += pointing_records(frame, maps[(model, seed)], soft, model=model, seed=seed)
    records += pointing_records(frame, np.broadcast_to(prior, (len(frame), 1, *prior.shape)),
                                soft, model="prior", seed=None)
    raw = pd.DataFrame(records)
    raw["seed"] = pd.array(raw.seed, dtype="Int64")
    summary, paired, paired_rows = [], [], []
    hits = {}
    for model, seed in [*((m, s) for m in MODELS[:2] for s in SEEDS), ("prior", None)]:
        group = raw.loc[(raw.model == model) & (raw.seed.isna() if seed is None else raw.seed.eq(seed))]
        hits[(model, seed)] = group.hit.to_numpy(dtype=np.int64)
        summary.append({"model": model, "seed": seed, "hits": int(group.hit.sum()),
                        "n_nodules": len(group), "n_patients": int(group.patient_id.nunique()),
                        "empty_targets": int(group.empty_target.sum()),
                        "nonempty_targets": int((~group.empty_target).sum()), "score": float(group.hit.mean())})
    for seed in SEEDS:
        for left, right in (("ConsenDxFormer", "B1"), ("ConsenDxFormer", "prior"), ("B1", "prior")):
            delta = hits[(left, seed)] - hits[(right, None if right == "prior" else seed)]
            contrast = f"{left}_minus_{right}"
            paired.append({"contrast": contrast, "seed": seed, "n_nodules": len(frame),
                           "n_patients": int(frame.patient_id.nunique()), "sum_difference": int(delta.sum()),
                           "mean_difference": float(delta.mean())})
            paired_rows += [{**row, "contrast": contrast, "seed": seed, "hit_difference": int(d)}
                            for row, d in zip(identities(frame), delta)]
    aggregate = [{"model": model, "mean_of_seed_means": float(np.mean(values)),
                  "sample_sd_of_seed_means": float(np.std(values, ddof=1)), "n_seeds": 3}
                 for model in MODELS[:2]
                 for values in [[r["score"] for r in summary if r["model"] == model]]]
    summaries = pd.DataFrame(summary)
    summaries["seed"] = pd.array(summaries.seed, dtype="Int64")
    return {"pointing_per_nodule.csv": raw, "pointing_per_seed.csv": summaries,
            "pointing_aggregate.csv": pd.DataFrame(aggregate),
            "pointing_paired_descriptive.csv": pd.DataFrame(paired),
            "pointing_paired_per_nodule.csv": pd.DataFrame(paired_rows),
            "flagged_case.csv": raw.loc[raw.nodule_id == FLAGGED_NODULE].reset_index(drop=True)}


def accuracy_intervals(frames):
    """Separate accuracy-only bootstrap; all endpoints share one patient ledger."""
    frames = canonical_predictions(frames)
    keys = [(m, s) for m in MODELS for s in SEEDS]
    reference = frames[keys[0]]
    patients = sorted(reference.patient_id.unique().tolist())
    patient_index = {p: i for i, p in enumerate(patients)}
    indices = np.array([patient_index[p] for p in reference.patient_id], dtype=np.int64)
    policy = descriptive_protocol()["accuracy"]
    draws = np.random.Generator(np.random.PCG64(policy["seed"])).integers(
        0, len(patients), size=(policy["resamples"], len(patients)), dtype=np.int64)
    multiplicities = np.zeros((len(draws), len(patients)), dtype=np.int64)
    np.add.at(multiplicities, (np.arange(len(draws))[:, None], draws), 1)
    sizes = np.bincount(indices, minlength=len(patients))
    denominator = multiplicities @ sizes
    correct = np.column_stack([frames[key].true_class.eq(frames[key].predicted_class).to_numpy(dtype=np.int64) for key in keys])
    cluster_correct = np.zeros((len(patients), len(keys)), dtype=np.int64)
    np.add.at(cluster_correct, indices, correct)
    estimates = (multiplicities @ cluster_correct) / denominator[:, None]
    points = correct.mean(axis=0)
    records, endpoint_ids, columns = [], [], []
    for index, (model, seed) in enumerate(keys):
        endpoint_ids.append(f"accuracy/{model}/{seed}")
        columns.append(estimates[:, index])
        records.append({"endpoint_id": endpoint_ids[-1], "kind": "individual", "model": model,
                        "comparator": None, "seed": seed, "point": float(points[index])})
    for comparator in MODELS[1:]:
        for seed in SEEDS:
            a, b = keys.index((MODELS[0], seed)), keys.index((comparator, seed))
            endpoint_ids.append(f"accuracy_difference/ConsenDxFormer_minus_{comparator}/{seed}")
            columns.append(estimates[:, a] - estimates[:, b])
            records.append({"endpoint_id": endpoint_ids[-1], "kind": "paired_difference",
                            "model": MODELS[0], "comparator": comparator, "seed": seed,
                            "point": float(points[a] - points[b])})
    replicates = np.column_stack(columns)
    bounds = np.quantile(replicates, policy["quantiles"], axis=0, method="linear")
    draw_hash = array_record(draws)["array_sha256"]
    for i, record in enumerate(records):
        record.update(lower=float(bounds[0, i]), upper=float(bounds[1, i]),
                      n_nodules=len(reference), n_patients=len(patients), resamples=len(draws),
                      patient_draw_sha256=draw_hash, descriptive=True, multiplicity_adjusted=False,
                      conditional_on_fit=True, simultaneous_coverage=False)
    raw = pd.concat([frames[key][list(IDENTITY) + ["predicted_class"]].assign(
        model=key[0], seed=key[1], correct=correct[:, i]) for i, key in enumerate(keys)], ignore_index=True)
    arrays = {"patient_ids": np.asarray(patients, dtype=np.str_), "draw_indices": draws,
              "nodule_denominators": denominator, "endpoint_ids": np.asarray(endpoint_ids, dtype=np.str_),
              "replicate_estimates": replicates}
    return pd.DataFrame(records), arrays, raw
