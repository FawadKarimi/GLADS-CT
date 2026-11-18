"""Synthetic CPU mathematics/scheduling only; never construct research models."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consendxformer.phase4_attention_evidence import array_record, write_map_archive
from consendxformer.phase5_perturbation import (
    CLASS_ORDER, KEYS, CallLedger, curve_auc, execute_curves, hash_order, perturbation_protocol,
    rank_native, response_arrays, scheduled_budget, summarize_curves, tile_input,
    validate_outputs, validate_probabilities,
)
from consendxformer.phase5_perturbation_exports import (
    expected_tables, validate_result, validate_result_exports,
)
from consendxformer.phase5_descriptive_runner import _write_csv
from consendxformer.phase5_perturbation_sources import TorchImagePredictor


def toy_output(images):
    # Known signed tile contributions and a fixed three-class primary softmax.
    weights = np.tile(np.array([1, -0.25], dtype=np.float32), 32).reshape(8, 8)
    tile_means = images[0, 0].reshape(8, 8, 8, 8).mean(axis=(1, 3))
    signal = float((tile_means * weights).sum() / 24)
    logits = np.array([[2*signal, .6, -signal]], dtype=np.float32)
    mass = np.exp(logits-logits.max(axis=1, keepdims=True))
    return {"logits": logits, "probs": (mass / mass.sum(axis=1, keepdims=True)).astype(np.float32)}


def tiny_inputs(n=1, *, zero=False):
    ids = pd.DataFrame({"patient_id": [f"p{i}" for i in range(n)], "nodule_id": [f"n{i}" for i in range(n)],
                        "series_instance_uid": [f"s{i}" for i in range(n)], "split": ["test"]*n,
                        "true_class": ["malignant"]*n})
    image = np.linspace(.1, .9, 4096, dtype=np.float32).reshape(1, 64, 64)
    images = np.repeat(image[None], n, axis=0)
    if zero: images[:] = 0
    predictions, records = {}, {}
    for key in KEYS:
        frame = ids.copy()
        outputs = [toy_output(images[i:i+1]) for i in range(n)]
        frame["predicted_class"] = [CLASS_ORDER[int(o["probs"].argmax())] for o in outputs]
        for j, name in enumerate(CLASS_ORDER):
            frame[f"logit_{name}"] = [o["logits"][0, j] for o in outputs]
            frame[f"prob_{name}"] = [o["probs"][0, j] for o in outputs]
        predictions[key] = frame
        records[key] = {"model": key[0], "seed": key[1], "best_checkpoint_sha256": "a"*64}
    dsa = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    dsa /= dsa.sum()
    maps = {key: np.broadcast_to(dsa, (n, 1, 8, 8)).copy() for key in KEYS}
    return dict(ids=ids, images=images, maps=maps, prior=np.full((8, 8), 1/64),
                predictions=predictions, predictors={k: toy_output for k in KEYS},
                source_binding="b"*64, source_records=records)


def test_tile_coverage_exact_ties_controls_and_independent_hash_goldens():
    order = rank_native(np.full((8, 8), 1/64))
    np.testing.assert_array_equal(order, np.arange(64))
    values = np.zeros((8, 8)); values.flat[[0, 63]] = .5
    assert rank_native(values).tolist() == [0, 63, *range(1, 63)]
    image = np.ones((1, 64, 64), dtype=np.float32)
    for k in range(65):
        inserted = tile_input(image, order, "insertion", k)
        deleted = tile_input(image, order, "deletion", k)
        assert inserted.sum() == k*64 and deleted.sum() == (64-k)*64
        np.testing.assert_array_equal(inserted+deleted, image)
        if k:
            previous = tile_input(image, order, "insertion", k-1)
            assert np.count_nonzero(inserted != previous) == 64
    identity = dict(patient_id="patient-a", nodule_id="nodule-a", series_instance_uid="series-a", true_class="benign")
    control, digests = hash_order(identity)
    # Independently enumerated with .NET SHA256 over literal ASCII JSON, not this helper.
    assert control.tolist() == [34,11,48,41,1,7,33,4,28,29,44,62,22,37,57,19,8,30,12,24,15,52,43,60,35,63,58,20,5,38,21,56,42,46,39,0,16,17,23,49,9,13,2,51,3,10,45,59,25,18,53,32,26,55,27,54,50,40,14,6,31,47,36,61]
    assert digests[0].tobytes().hex() == "9490357baa39570a4215cfaeeab7ece55a42a569ea646b87ab9841c0ef1caf83"
    assert digests[1].tobytes().hex() == "150f2f14fe6e57048a101172df72d6692bdf7ad83ab009128d4fd20d9ab91289"
    assert digests[2].tobytes().hex() == "acb024653f758f94c29f50b3f3ddf2ff9139c0b4e87d6a10faa4ba5296c9d210"
    changed = {**identity, "true_class": "malignant", "model": "irrelevant", "seed": 44}
    np.testing.assert_array_equal(hash_order(changed)[0], control)
    with pytest.raises(ValueError): tile_input(image, np.zeros(64, dtype=int), "insertion", 1)
    with pytest.raises(ValueError): tile_input(image, order, "insertion", 65)


def test_raw_float32_probability_policy_no_normalization_and_hand_auc():
    raw = np.array([[.2, .3, .5]], dtype=np.float32)
    copied, mass, deviation = validate_probabilities(raw)
    assert 1e-8 < abs(deviation[0]) < 1e-6
    assert copied.tobytes() == raw.tobytes() and not np.shares_memory(copied, raw)
    assert mass[0] == raw.astype(np.float64).sum()
    assert curve_auc(np.full(65, raw[0, 1], dtype=np.float32)) == float(raw[0, 1])
    assert curve_auc(np.linspace(0, 1, 65)) == .5
    assert curve_auc(np.tile([0., 1.], 33)[:65]) == .5  # nonmonotonic kept intact
    from consendxformer.phase4_evaluation_metrics import predictive_entropy
    with pytest.raises(ValueError): predictive_entropy(raw.astype(np.float64))
    with pytest.raises(ValueError): curve_auc(np.ones(64))
    with pytest.raises(ValueError): validate_outputs({"probs": raw, "logits": np.full((1, 3), np.nan, np.float32)})


@pytest.mark.parametrize("value", [
    np.array([[.2, .3, .50001]], np.float32), np.zeros((1, 3), np.float32),
    np.array([[-.1, .6, .5]], np.float32), np.array([[1.1, 0, 0]], np.float32),
    np.array([[np.nan, .5, .5]], np.float32), np.array([[np.inf, 0, 0]], np.float32),
    np.empty((0, 3), np.float32), np.ones((1, 2), np.float32),
    np.ones((1, 1, 3), np.float32), np.array([.2, .3, .5], np.float32),
    np.array([[.2, .3, .5]], np.float64),
])
def test_invalid_probabilities_fail_closed(value):
    with pytest.raises(ValueError): validate_probabilities(value)


def test_schedule_barrier_fixed_targets_rankings_and_raw_immutability():
    inputs = tiny_inputs()
    originals = {"images": inputs["images"].copy(), "prior": inputs["prior"].copy(),
                 "maps": {k: v.copy() for k, v in inputs["maps"].items()},
                 "predictions": {k: v.copy(deep=True) for k, v in inputs["predictions"].items()}}
    calls = []
    def image_only(image):
        calls.append(image.copy())
        return {**toy_output(image), "attention_map": np.full((1, 1, 8, 8), len(calls))}
    inputs["predictors"] = {key: image_only for key in KEYS}
    def barrier(cache, binding, ledger):
        assert len(calls) == 6 and ledger.attempted == {"unperturbed": 6, "baseline": 0, "interior": 0}
        assert binding["arrays"]["probabilities"] == array_record(cache["probabilities"])
        assert not cache["probabilities"].flags.writeable
    result = execute_curves(**inputs, on_barrier=barrier)
    validate_result(result, expected_nodules=1, expected_patients=1)
    assert len(calls) == 2280 and result.ledger.barrier_passed
    assert scheduled_budget()["total"] == 417240 and scheduled_budget()["interior"] == 415044
    assert scheduled_budget()["curve_positions"] == 428220 and scheduled_budget()["curves"] == 6588
    assert len(set(result.curves["call_index"].reshape(-1))) == 2280
    np.testing.assert_array_equal(np.stack(calls[:6])[:, 0], np.repeat(inputs["images"], 6, axis=0))
    # Every actual interior tensor is independently reconstructed from the frozen permutation.
    for o, order in enumerate(result.orders["permutations"][0]):
        for d, direction in enumerate(("insertion", "deletion")):
            for k in range(1, 64):
                call = result.curves["call_index"][0, o, d, k]
                expected = tile_input(inputs["images"][0], order, direction, k)
                np.testing.assert_array_equal(calls[call-1][0], expected)
    response = response_arrays(result)
    target = result.units.iloc[0].target_index
    assert result.baseline["probabilities"][0].argmax() != target
    np.testing.assert_array_equal(response[0], result.curves["probabilities"][0, ..., target].astype(np.float64))
    assert len(summarize_curves(result)["perturbation_per_seed.csv"]) == 36
    assert len(summarize_curves(result)["perturbation_paired_descriptive.csv"]) == 24
    for k in ("images", "prior"): np.testing.assert_array_equal(inputs[k], originals[k])
    for k in KEYS:
        np.testing.assert_array_equal(inputs["maps"][k], originals["maps"][k])
        pd.testing.assert_frame_equal(inputs["predictions"][k], originals["predictions"][k])
    # Evaluation masks cannot enter this API. Altering a separate mask changes no tensor/order.
    import inspect
    assert "mask" not in inspect.signature(execute_curves).parameters
    assert list(inspect.signature(image_only).parameters) == ["image"]


@pytest.mark.parametrize("kind", ["logit", "near_tie_argmax"])
def test_late_replay_failure_has_zero_baseline_and_interior_calls(kind):
    inputs, ledger = tiny_inputs(n=2), CallLedger()
    calls = []
    if kind == "near_tie_argmax":
        for frame in inputs["predictions"].values():
            frame["predicted_class"] = "benign"
            for j, c in enumerate(CLASS_ORDER):
                frame[f"prob_{c}"] = [.5, .5, 0][j]
                frame[f"logit_{c}"] = [0, 0, -100][j]
    def predictor(image):
        calls.append(True)
        output = toy_output(image) if kind == "logit" else {
            "probs": np.array([[.5, .5, 0]], np.float32), "logits": np.array([[0, 0, -100]], np.float32)}
        if len(calls) == 12:
            if kind == "logit": output["logits"][0, 0] += 1
            else: output["probs"][0, 0] = np.nextafter(np.float32(.5), np.float32(0))
        return output
    inputs["predictors"] = {k: predictor for k in KEYS}
    with pytest.raises(ValueError, match="Global unperturbed replay/argmax"):
        execute_curves(**inputs, ledger=ledger, on_barrier=lambda *a: pytest.fail("Barrier passed despite late failure"))
    assert len(calls) == 12 and ledger.attempted == {"unperturbed": 12, "baseline": 0, "interior": 0}
    assert not ledger.barrier_passed and ledger.failure["error_type"] == "ReplayMismatch"
    assert not ledger.replays[-1]["consistent"]


def test_zero_image_flat_curves_no_dedup_and_incomplete_curve_rejected():
    result = execute_curves(**tiny_inputs(zero=True))
    assert result.ledger.record()["total_attempted"] == 2280
    assert result.units.baseline_identical_input.all() and result.units.constant_input.all()
    raw = summarize_curves(result)["perturbation_auc_per_nodule.csv"]
    assert raw.flat_curve.all() and raw.equal_all_order_curves.all()
    result.curves["probabilities"] = result.curves["probabilities"][..., :64, :]
    with pytest.raises(ValueError, match="Missing step"):
        validate_result(result, expected_nodules=1, expected_patients=1)


def test_changing_only_evaluation_masks_leaves_every_order_and_call_tensor_unchanged():
    from types import SimpleNamespace
    inputs = tiny_inputs()
    source = SimpleNamespace(**inputs, archive_inputs={"soft_masks_input": np.zeros((1, 1, 64, 64), np.float32)})
    def execute_image_only_source():
        tensors = []
        def predictor(image):
            tensors.append(array_record(image)["array_sha256"])
            return toy_output(image)
        # The same explicit image-only argument boundary used by the runner;
        # archived evaluation targets are kept separately, never passed to it.
        result = execute_curves(source.ids, source.images, source.maps, source.prior, source.predictions,
            {k: predictor for k in KEYS}, source_binding=source.source_binding, source_records=source.source_records)
        return result.orders["permutations"], tensors, result.units.target_index.to_list()
    a = execute_image_only_source()
    source.archive_inputs["soft_masks_input"][:] = 1
    b = execute_image_only_source()
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1:] == b[1:]


def test_image_only_frozen_torch_adapter_disables_gradients_autocast():
    import torch
    class Tiny(torch.nn.Module):
        def forward(self, image):
            assert image.shape == (1, 1, 64, 64) and image.dtype == torch.float32
            assert not torch.is_grad_enabled() and not torch.is_autocast_enabled("cpu")
            logits = torch.stack((image.mean(), image.mean()*0+.1, image.mean()*0)).reshape(1, 3)
            image.add_(.01)  # private input copy; source is unaffected
            return {"logits": logits, "probs": torch.softmax(logits, dim=1)}
    model = Tiny().eval().requires_grad_(False)
    images = np.full((1, 1, 64, 64), .5, np.float32)
    before = images.copy()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = TorchImagePredictor(model, "cpu")(images)
    validate_outputs(output)
    np.testing.assert_array_equal(images, before)
    model.train()
    with pytest.raises(ValueError, match="frozen eval"): TorchImagePredictor(model, "cpu")(images)


def test_golden_exports_deterministic_raw_roundtrip_and_independent_auc_oracle(tmp_path):
    inputs = tiny_inputs()
    result = execute_curves(**inputs)
    archives = {"orders.npz": result.orders, "curves.npz": result.curves,
                "baseline_cache.npz": result.baseline, "replay_cache.npz": result.replay}
    manifests = []
    for root in (tmp_path / "a", tmp_path / "b"):
        records = [write_map_archive(root / name, arrays) for name, arrays in archives.items()]
        for name, table in expected_tables(result).items(): _write_csv(root / "tables" / name, table)
        loaded = validate_result_exports(root, records, result.ledger, expected_nodules=1, expected_patients=1)
        values = response_arrays(loaded)
        # Independent scalar trapezoid sum, not the production AUC helper.
        oracle = np.array([sum((float(v[k-1])+float(v[k]))/128 for k in range(1, 65))
                           for v in values.reshape(-1, 65)])
        table = pd.read_csv(root / "tables/perturbation_auc_per_nodule.csv", float_precision="round_trip")
        np.testing.assert_allclose(table.auc, oracle, atol=1e-12, rtol=0)
        from consendxformer.hashing import sha256_file
        manifests.append({p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()})
    assert manifests[0] == manifests[1]
    curve_path = tmp_path / "a/tables/perturbation_curves.csv"
    frame = pd.read_csv(curve_path); frame.loc[0, "target_response"] += .01; frame.to_csv(curve_path, index=False)
    with pytest.raises(ValueError, match="table value differs"):
        validate_result_exports(tmp_path / "a", records, result.ledger, expected_nodules=1, expected_patients=1)


def test_raw_small_mass_errors_survive_every_curve_and_diagnostic_export():
    inputs = tiny_inputs()
    raw = np.array([[.2, .3, .5]], np.float32)
    logits = np.log(raw).astype(np.float32)
    for frame in inputs["predictions"].values():
        frame["predicted_class"] = "malignant"
        for j, c in enumerate(CLASS_ORDER):
            frame[f"prob_{c}"], frame[f"logit_{c}"] = float(raw[0, j]), float(logits[0, j])
    inputs["predictors"] = {k: lambda image: {"logits": logits, "probs": raw} for k in KEYS}
    result = execute_curves(**inputs)
    tables = expected_tables(result)
    assert tables["perturbation_curves.csv"].raw_absolute_mass_deviation.between(1e-8, 1e-6).all()
    np.testing.assert_array_equal(result.curves["probabilities"], np.broadcast_to(raw, result.curves["probabilities"].shape))
    assert tables["perturbation_auc_per_nodule.csv"].auc.eq(.5).all()
    policy = perturbation_protocol()
    assert policy["probability"]["normalize"] is False and len(policy["hypotheses"]) == 15


def test_mutating_predictor_cannot_alter_sealed_original_rankings():
    inputs = tiny_inputs()
    ledger = CallLedger()
    def mutate_once(image):
        inputs["maps"][KEYS[0]][0, 0] = 1/64
        return toy_output(image)
    inputs["predictors"][KEYS[-1]] = mutate_once
    with pytest.raises(ValueError, match="Original inputs/maps/prior/rankings changed"):
        execute_curves(**inputs, ledger=ledger)
    assert ledger.attempted == {"unperturbed": 6, "baseline": 0, "interior": 0}
