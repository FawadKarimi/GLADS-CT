"""Deterministic complete exports and inference-free numerical read-back."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hashing import sha256_json
from .phase4_attention_evidence import IDENTITY, array_record, write_map_archive
from .phase5_descriptive import identities, unresolved_hypotheses
from .phase5_descriptive_runner import _write_csv
from .phase5_perturbation import (CLASS_ORDER, KEYS, ORDERS, DIRECTIONS, PerturbationResult,
    hash_order, scheduled_budget, validate_probabilities, summarize_curves, response_arrays)


def validate_result(result, *, expected_nodules=183, expected_patients=99):
    units, curves = result.units, result.curves
    count = expected_nodules * 6
    if len(units) != count or units.unit_id.duplicated().any():
        raise ValueError("Incomplete/duplicate perturbation units")
    if set(zip(units.model, units.seed)) != set(KEYS):
        raise ValueError("Perturbation model/seed set differs")
    reference = None
    for model, seed in KEYS:
        group = units.loc[(units.model == model) & (units.seed == seed)]
        rows = identities(group)
        if len(group) != expected_nodules or group.patient_id.nunique() != expected_patients:
            raise ValueError("Perturbation cohort accounting differs")
        if reference is None:
            reference = rows
        if rows != reference or group.archive_row.tolist() != list(range(expected_nodules)):
            raise ValueError("Perturbation pairing/order differs")
    if units[["model", "seed"]].to_records(index=False).tolist() != [key for key in KEYS for _ in range(expected_nodules)]:
        raise ValueError("Scheduled unit order differs")
    targets = units.target_index.to_numpy()
    if targets.dtype.kind not in "iu" or (targets < 0).any() or (targets > 2).any() or units.target_class.tolist() != [CLASS_ORDER[i] for i in targets]:
        raise ValueError("Fixed target class/index differs")
    shape = (count, 3, 2, 65)
    if set(curves) != {"logits", "probabilities", "call_index", "input_sha256"}:
        raise ValueError("Incomplete raw curve evidence")
    for field in ("logits", "probabilities"):
        if curves[field].shape != (*shape, 3) or curves[field].dtype != np.float32 or not np.isfinite(curves[field]).all():
            raise ValueError("Missing step/nonfinite/wrong-dtype raw curve output")
    validate_probabilities(curves["probabilities"].reshape(-1, 3))
    if (curves["call_index"].shape != shape or curves["call_index"].dtype != np.int64
            or curves["input_sha256"].shape != shape or curves["input_sha256"].dtype != np.dtype("S64")):
        raise ValueError("Curve call/input-hash grid differs")
    permutations = result.orders["permutations"]
    if (set(result.orders) != {"unit_id", "permutations", "hash_digests_by_flat_index"}
            or permutations.shape != (count, 3, 64) or permutations.dtype != np.int64
            or not np.all(np.sort(permutations, axis=-1) == np.arange(64))
            or result.orders["hash_digests_by_flat_index"].shape != (count, 64, 32)
            or result.orders["hash_digests_by_flat_index"].dtype != np.uint8
            or result.orders["unit_id"].tolist() != units.unit_id.tolist()):
        raise ValueError("Incomplete orders/digests or invalid permutation")
    if not np.all(permutations[:, 1] == permutations[0, 1]):
        raise ValueError("Fixed prior ordering changed across units")
    controls = [hash_order(r) for r in reference]
    for u, row in units.iterrows():
        order, digests = controls[int(row.archive_row)]
        if not np.array_equal(permutations[u, 2], order) or not np.array_equal(result.orders["hash_digests_by_flat_index"][u], digests):
            raise ValueError("Hash-order control/digests differ")
    for endpoint, cache in (("unperturbed", result.replay), ("baseline", result.baseline)):
        expected = {"unit_id", "logits", "probabilities"} | ({"target_index"} if endpoint == "unperturbed" else set())
        if set(cache) != expected or cache["unit_id"].tolist() != units.unit_id.tolist():
            raise ValueError("Endpoint cache identity differs")
        if endpoint == "unperturbed" and (not np.array_equal(cache["target_index"], targets)
                or not np.array_equal(cache["probabilities"].argmax(axis=1), targets)):
            raise ValueError("Fixed target changed from replayed predicted class")
        for field in ("logits", "probabilities"):
            if cache[field].shape != (count, 3) or cache[field].dtype != np.float32 or not np.isfinite(cache[field]).all():
                raise ValueError("Endpoint cache output differs")
            for d, step in (((0, 64), (1, 0)) if endpoint == "unperturbed" else ((0, 0), (1, 64))):
                if not np.array_equal(curves[field][:, :, d, step], np.repeat(cache[field][:, None], 3, axis=1)):
                    raise ValueError("Endpoint cache not reused exactly")
        validate_probabilities(cache["probabilities"])
    for u, row in units.iterrows():
        expected = np.empty((3, 2, 65), dtype=np.int64)
        expected[:, 0, 64] = expected[:, 1, 0] = u + 1
        baseline_call = count + u * 379 + 1
        expected[:, 0, 0] = expected[:, 1, 64] = baseline_call
        expected[:, :, 1:64] = np.arange(baseline_call + 1, baseline_call + 379).reshape(3, 2, 63)
        if not np.array_equal(curves["call_index"][u], expected):
            raise ValueError("Exact endpoint-only call accounting differs")
        for d, step, field in ((0, 64, "input_sha256"), (1, 0, "input_sha256"),
                               (0, 0, "baseline_sha256"), (1, 64, "baseline_sha256")):
            if not np.all(curves["input_sha256"][u, :, d, step] == row[field].encode("ascii")):
                raise ValueError("Endpoint input hash differs")
    budget = scheduled_budget(expected_nodules)
    expected_calls = {k: budget[k] for k in ("unperturbed", "baseline", "interior")}
    ledger = result.ledger
    if (ledger.attempted != expected_calls or ledger.accepted != expected_calls or ledger.failure is not None
            or not ledger.barrier_passed or len(ledger.replays) != count
            or [r["unit_id"] for r in ledger.replays] != units.unit_id.tolist()
            or not all(r["consistent"] is True for r in ledger.replays)):
        raise ValueError("Unsuccessful global replay barrier/call ledger cannot complete")


def curve_table(result):
    n = len(result.units)
    u, o, d, k = (a.reshape(-1) for a in np.indices((n, 3, 2, 65)))
    columns = ["unit_id", *IDENTITY, "model", "seed", "target_class", "target_index"]
    table = result.units.iloc[u][columns].reset_index(drop=True)
    table["order"], table["direction"] = np.asarray(ORDERS)[o], np.asarray(DIRECTIONS)[d]
    table["step"], table["fraction"] = k, k.astype(np.float64)/64
    table["changed_tile_index"] = np.where(k == 0, -1, result.orders["permutations"][u, o, np.maximum(k-1, 0)])
    table["call_index"] = result.curves["call_index"].reshape(-1)
    table["perturbed_input_sha256"] = result.curves["input_sha256"].reshape(-1).astype(str)
    cache = np.full(len(table), "none", dtype="U11")
    cache[((d == 0) & (k == 64)) | ((d == 1) & (k == 0))] = "unperturbed"
    cache[((d == 0) & (k == 0)) | ((d == 1) & (k == 64))] = "baseline"
    table["endpoint_cache"] = cache
    table["cache_reference"] = np.where(cache == "none", "none", table.unit_id.to_numpy() + ":" + cache)
    for field, prefix in (("logits", "logit"), ("probabilities", "prob")):
        raw = result.curves[field].reshape(-1, 3)
        for j, name in enumerate(CLASS_ORDER):
            # Promote for CSV representation only; NPZ retains exact float32 bytes.
            table[f"{prefix}_{name}"] = raw[:, j].astype(np.float64)
    probs = result.curves["probabilities"].reshape(-1, 3)
    table["predicted_class"] = np.asarray(CLASS_ORDER)[probs.argmax(axis=1)]
    table["target_response"] = response_arrays(result).reshape(-1)
    table["raw_row_sum"] = probs.astype(np.float64).sum(axis=1, dtype=np.float64)
    table["raw_mass_deviation"] = table.raw_row_sum - 1
    table["raw_absolute_mass_deviation"] = np.abs(table.raw_mass_deviation)
    return table


def expected_tables(result):
    return {"units.csv": result.units, "perturbation_curves.csv": curve_table(result),
            **summarize_curves(result), "hypothesis_register.csv": pd.DataFrame(unresolved_hypotheses())}


def write_result(output, result):
    validate_result(result)
    archives = {"orders.npz": result.orders, "curves.npz": result.curves, "baseline_cache.npz": result.baseline}
    records = [write_map_archive(output / name, arrays) for name, arrays in archives.items()]
    for name, table in expected_tables(result).items():
        _write_csv(output / "tables" / name, table)
    return records


def compare_csv(path, expected, *, atol=0):
    text = {c: str for c in expected if expected[c].dtype == object}
    actual = pd.read_csv(path, dtype=text, keep_default_na=False, float_precision="round_trip")
    if list(actual) != list(expected) or len(actual) != len(expected):
        raise ValueError("Exported table columns/row accounting differ")
    for column in expected:
        a, b = actual[column].to_numpy(), expected[column].to_numpy()
        if expected[column].dtype.kind in "fiu":
            equal = np.allclose(a, b, rtol=0, atol=atol) if expected[column].dtype.kind == "f" else np.array_equal(a, b)
        else:
            equal = np.array_equal(a, b)
        if not equal:
            raise ValueError(f"Exported table value differs: {path.name}/{column}")


def validate_result_exports(directory, archive_records, ledger, *, expected_nodules=183, expected_patients=99):
    arrays = {}
    for record in archive_records:
        name = record["file"]
        if name in arrays or name not in {"orders.npz", "curves.npz", "baseline_cache.npz", "replay_cache.npz"}:
            raise ValueError("Unexpected/duplicate perturbation archive")
        with np.load(directory / name, allow_pickle=False) as saved:
            if set(saved.files) != set(record["arrays"]):
                raise ValueError("Perturbation archive keys differ")
            arrays[name] = {k: saved[k].copy() for k in saved.files}
        if {k: array_record(v) for k, v in arrays[name].items()} != record["arrays"]:
            raise ValueError("Perturbation array dtype/shape/hash differs")
    if set(arrays) != {"orders.npz", "curves.npz", "baseline_cache.npz", "replay_cache.npz"}:
        raise ValueError("Incomplete perturbation archives")
    units = pd.read_csv(directory / "tables/units.csv", dtype={k: str for k in (*IDENTITY, "unit_id")},
                        keep_default_na=False, float_precision="round_trip")
    result = PerturbationResult(units, arrays["orders.npz"], arrays["curves.npz"], arrays["replay_cache.npz"],
                                arrays["baseline_cache.npz"], ledger)
    validate_result(result, expected_nodules=expected_nodules, expected_patients=expected_patients)
    for name, expected in expected_tables(result).items():
        compare_csv(directory / "tables" / name, expected,
                    atol=1e-12 if name.startswith("perturbation_") and name != "perturbation_curves.csv" else 0)
    return result
