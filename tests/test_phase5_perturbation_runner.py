"""Synthetic saved bundles only. Historical source-loader stubs are explicit."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.phase3_artifacts import atomic_write_json
from consendxformer.phase4_attention_evidence import array_record
from consendxformer.phase5_perturbation import CLASS_ORDER, KEYS, perturbation_protocol
import consendxformer.phase5_perturbation_runner as runner
import consendxformer.phase5_perturbation_sources as sources_api
from test_phase5_descriptive_runner import _setup as saved_fixture, _seal


def replace_archive(evidence, name, change):
    with np.load(evidence / name, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    change(arrays)
    np.savez(evidence / name, **arrays)
    path = evidence / "provenance/provenance.json"
    provenance = runner.read_json(path)
    for record in provenance["archives"]:
        if record["file"] == name:
            record.update(sha256=sha256_file(evidence / name), arrays={k: array_record(v) for k, v in arrays.items()})
    atomic_write_json(path, provenance, overwrite=True)
    _seal(evidence)


def setup_sources(tmp_path, monkeypatch):
    # Reuse synthetic 183/99/807 identity and manifest fixtures, not an old evaluator.
    _, all_sources, evidence = saved_fixture(tmp_path, monkeypatch)
    import consendxformer.phase5_descriptive_runner as old_runner
    monkeypatch.setattr(sources_api, "load_saved_prediction_inputs", old_runner.load_saved_prediction_inputs)
    def images(arrays):
        arrays["ct_crops"][:] = np.float32(.2)
        arrays["ct_crops"][:, 0, 0, 0] = np.arange(183, dtype=np.float32)/182
    replace_archive(evidence, "test_inputs.npz", images)
    kwargs = {"full_source_dirs": [all_sources[("ConsenDxFormer", s)].source_dir for s in (42, 43, 44)],
              "b1_source_dirs": [all_sources[("B1", s)].source_dir for s in (42, 43, 44)],
              "attention_source": evidence, "attention_manifest_sha256": sha256_file(evidence / "artifact_manifest.json"),
              "authorize_execution": True}
    monkeypatch.setattr(runner, "validate_evaluator_repository", lambda p: {"commit": "1d3f1157afe95415f68dcde5f8723588846cf061", "clean": True})
    monkeypatch.setattr(runner, "NAMESPACE", tmp_path / "insertion-only")
    return kwargs, all_sources, evidence


def install_toy_runtime(monkeypatch, calls, *, late_failure=False, mutate=None,
                        interrupt_at=None, interruption=None):
    @contextmanager
    def context(inputs, *, device, cache_dir):
        def predictor(key):
            offset = key[1] - 42 + (key[0] == "B1")
            def image_only(image):
                assert image.shape == (1, 1, 64, 64) and image.dtype == np.float32
                calls.append(key)
                if len(calls) == interrupt_at:
                    raise interruption
                index = int(np.rint(image[0, 0, 0, 0] * 182))
                chosen = (index + offset) % 3
                probabilities = np.full((1, 3), .125, np.float32)
                probabilities[0, chosen] = .75
                output = {"probs": probabilities, "logits": np.log(probabilities).astype(np.float32)}
                if late_failure and len(calls) == 1098: output["logits"][0, 0] += 1
                if mutate is not None and len(calls) == 1098: mutate()
                return output
            return image_only
        yield {key: predictor(key) for key in KEYS}, {"live_input_snapshot": {}, "fixture": "synthetic_image_only"}
    monkeypatch.setattr(runner, "frozen_predictors", context)


def load_kwargs(kwargs):
    return sources_api.load_perturbation_inputs(kwargs["full_source_dirs"], kwargs["b1_source_dirs"],
        kwargs["attention_source"], kwargs["attention_manifest_sha256"])


def test_new_protocol_and_proposal_unchanged_scope_only():
    policy, proposal = runner.load_protocol()
    assert policy == perturbation_protocol()
    assert policy["scope"] == "insertion_deletion_only" and policy["successful_run_budget"]["total"] == 417240
    assert len(policy["hypotheses"]) == 15 and not policy["full_supplemental_completion"]
    assert policy["synthetic_timing_evidence"] is None
    assert "global barrier" in proposal


def test_all_six_source_contracts_native_archive_and_no_ablation_dependency(tmp_path, monkeypatch):
    kwargs, _, evidence = setup_sources(tmp_path, monkeypatch)
    inputs = load_kwargs(kwargs)
    assert inputs.images.shape == (183, 1, 64, 64) and len(inputs.ids) == 183
    assert inputs.ids.patient_id.nunique() == 99 and set(inputs.sources) == set(KEYS)
    assert len(inputs.evidence_provenance["prior"]["training_identities"]) == 807
    assert str((evidence / "test_inputs.npz").resolve()) in inputs.snapshot
    assert not any("_off" in p for p in inputs.snapshot)


def test_explicit_manifest_cannot_change_between_pin_check_and_snapshot(tmp_path, monkeypatch):
    kwargs, _, evidence = setup_sources(tmp_path, monkeypatch)
    original = sources_api.load_saved_prediction_inputs
    def changed(*args):
        result = original(*args)
        path = evidence / "artifact_manifest.json"
        path.write_bytes(path.read_bytes() + b" ")  # JSON unchanged, pinned bytes different
        return result
    monkeypatch.setattr(sources_api, "load_saved_prediction_inputs", changed)
    with pytest.raises(ValueError, match="manifest changed during"):
        load_kwargs(kwargs)


@pytest.mark.parametrize("failure", [None, "live_input", "model_state"])
def test_frozen_runtime_preparation_image_only_adapter_and_exit_guards(tmp_path, monkeypatch, failure):
    kwargs, _, _ = setup_sources(tmp_path, monkeypatch)
    inputs = load_kwargs(kwargs)
    import torch
    calls, loaded = [], []
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.5))
        def forward(self, image):
            calls.append(True)
            logits = torch.stack([image.mean()*self.weight, image.mean()*0, image.mean()*0]).reshape(1, 3)
            return {"logits": logits, "probs": torch.softmax(logits, dim=1)}
    def frozen(path, *args, device="cpu"):
        key = next(k for k, s in inputs.sources.items() if s.source_dir == path)
        loaded.append(key)
        return SimpleNamespace(source=inputs.sources[key], model=Tiny().eval().requires_grad_(False),
            checkpoint_sha256=inputs.source_records[key]["best_checkpoint_sha256"], device=device)
    @contextmanager
    def pinned(*args): yield object(), {"fixture": "synthetic_pinned_api"}
    class Loader:
        input_snapshot = {}
        def __iter__(self):
            image = inputs.images.copy()
            if failure == "live_input": image[0, 0, 0, 0] += .1
            yield {**{k: inputs.ids[k].tolist() for k in inputs.ids}, "image": torch.from_numpy(image),
                   "soft_consensus_mask": torch.from_numpy(inputs.archive_inputs["soft_masks_input"].copy())}
    monkeypatch.setattr(sources_api, "frozen_bundle", lambda: (SimpleNamespace(phase4_config={}), object()))
    monkeypatch.setattr(sources_api, "revalidate_inputs", lambda *a: (None, SimpleNamespace(to_dict=lambda: {"synthetic": True}), inputs.ids))
    monkeypatch.setattr(sources_api, "guard_image_paths", lambda frame: None)
    monkeypatch.setattr(sources_api, "pinned_b1_api", pinned)
    monkeypatch.setattr(sources_api, "load_frozen_final_best_model", frozen)
    monkeypatch.setattr(sources_api, "load_frozen_b1", frozen)
    monkeypatch.setattr(sources_api, "build_frozen_attention_loader", lambda model: Loader())
    deterministic = torch.are_deterministic_algorithms_enabled()
    def run():
        with sources_api.frozen_predictors(inputs, device="cpu", cache_dir=tmp_path / "unused-cache") as (predictors, metadata):
            assert calls == [] and set(loaded) == set(KEYS)
            assert set(predictors) == set(KEYS) and metadata["autocast"] is False
            assert torch.are_deterministic_algorithms_enabled()
            predictors[KEYS[0]](inputs.images[:1])
            if failure == "model_state":
                with torch.no_grad(): predictors[KEYS[0]].model.weight.add_(1)
    if failure is None: run()
    else:
        with pytest.raises(ValueError, match="Live frozen inputs" if failure == "live_input" else "model state changed"):
            run()
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert len(calls) == (0 if failure == "live_input" else 1)


@pytest.mark.parametrize("change", ["explicit_manifest", "raw_bytes", "snapshot", "source_binding", "duplicate_seed",
    "unbound_provenance", "policy", "permission", "train_overlap", "prior_membership", "identity", "ct_dtype",
    "ct_range", "native_shape", "normalized_mask", "saved_argmax"])
def test_source_corruption_fails_before_any_model_preparation(tmp_path, monkeypatch, change):
    kwargs, _, evidence = setup_sources(tmp_path, monkeypatch)
    provenance_path = evidence / "provenance/provenance.json"
    provenance = runner.read_json(provenance_path)
    if change == "explicit_manifest": kwargs["attention_manifest_sha256"] = "0"*64
    elif change == "raw_bytes": (evidence / "location_prior.npz").write_bytes(b"SYNTHETIC corruption")
    elif change in {"snapshot", "source_binding", "duplicate_seed", "prior_membership", "train_overlap"}:
        if change == "snapshot": provenance["input_file_snapshot_sha256"] = "0"*64
        elif change == "source_binding": provenance["source_experiments"][0]["source_snapshot"]["checkpoints/best.pt"] = "0"*64
        elif change == "duplicate_seed": provenance["source_experiments"][1] = dict(provenance["source_experiments"][0])
        elif change == "prior_membership": provenance["prior"]["n_nodules"] = 806
        else: provenance["prior"]["training_identities"][0]["patient_id"] = "synthetic-p000"
        atomic_write_json(provenance_path, provenance, overwrite=True)
        _seal(evidence)
    elif change in {"policy", "permission"}:
        path = evidence / "evaluation.json"
        evaluation = runner.read_json(path)
        if change == "policy": evaluation["protocol"]["training_nodules"] = 806
        else: evaluation["comparison_interpretation_permitted"] = False
        atomic_write_json(path, evaluation, overwrite=True); _seal(evidence)
    elif change == "unbound_provenance":
        manifest = runner.read_json(evidence / "artifact_manifest.json")
        del manifest["files"]["provenance/provenance.json"]
        manifest["bundle_sha256"] = sha256_json(manifest["files"])
        atomic_write_json(evidence / "artifact_manifest.json", manifest, overwrite=True)
        marker = runner.read_json(evidence / "COMPLETED.json")
        marker["artifact_manifest_sha256"] = sha256_file(evidence / "artifact_manifest.json")
        atomic_write_json(evidence / "COMPLETED.json", marker, overwrite=True)
    else:
        def change_arrays(arrays):
            if change == "identity": arrays["patient_id"][0] = "wrong-patient"
            elif change == "ct_dtype": arrays["ct_crops"] = arrays["ct_crops"].astype(np.float64)
            elif change == "ct_range": arrays["ct_crops"][0, 0, 0, 0] = 2
            elif change == "normalized_mask": arrays["soft_masks_native_grid"] = arrays["target_distributions"].copy()
            elif change == "native_shape": arrays["b1_attention"] = np.ones((183, 1, 1, 1), np.float32)
            else: arrays["b1_probabilities"][0] = [.5, .25, .25]
        name = "seed_42_model_arrays.npz" if change in {"native_shape", "saved_argmax"} else "test_inputs.npz"
        replace_archive(evidence, name, change_arrays)
    if change not in {"explicit_manifest", "raw_bytes"}:
        kwargs["attention_manifest_sha256"] = sha256_file(evidence / "artifact_manifest.json")
    monkeypatch.setattr(runner, "frozen_predictors", lambda *a, **kw: pytest.fail("Model preparation before source validation"))
    with pytest.raises((ValueError, OSError)): runner.run_insertion_deletion(**kwargs)
    assert not runner.NAMESPACE.exists()


@pytest.mark.parametrize("failure", ["late_replay", "input_mutation"])
def test_global_1098_barrier_failures_preserve_failed_attempt_no_nonoriginal_calls(tmp_path, monkeypatch, failure):
    kwargs, sources, _ = setup_sources(tmp_path, monkeypatch)
    calls = []
    path = sources[("B1", 42)].source_dir / "checkpoints/best.pt"
    install_toy_runtime(monkeypatch, calls, late_failure=failure == "late_replay",
                        mutate=(lambda: path.write_bytes(b"synthetic concurrent mutation")) if failure == "input_mutation" else None)
    with pytest.raises((ValueError, RuntimeError)):
        runner.run_insertion_deletion(**kwargs)
    assert len(calls) == 1098
    output = next(runner.NAMESPACE.iterdir())
    record = runner.read_json(output / "FAILED.json")
    assert record["calls"]["attempted"] == {"unperturbed": 1098, "baseline": 0, "interior": 0}
    assert not record["subset_completed"] and not (output / "COMPLETED.json").exists()
    assert not (output / "replay_barrier.json").exists()
    if failure == "late_replay":
        with pytest.raises(FileExistsError): runner.run_insertion_deletion(**kwargs)
        assert len(calls) == 1098


def test_authorization_and_output_namespace_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "validate_evaluator_repository", lambda *a: pytest.fail("Unauthorized preflight"))
    with pytest.raises(ValueError, match="authorize-execution"):
        runner.run_insertion_deletion(full_source_dirs=[], b1_source_dirs=[], attention_source="absent", attention_manifest_sha256="0"*64)
    with pytest.raises(ValueError, match="authorize-execution"):
        runner.insertion_deletion_cli(["--full-source", "absent", "--b1-source", "absent", "--attention-source", "absent",
                                      "--attention-manifest-sha256", "0"*64])
    kwargs, _, _ = setup_sources(tmp_path, monkeypatch)
    inputs = load_kwargs(kwargs)
    with pytest.raises(ValueError, match="namespace"):
        runner.output_directory(tmp_path / "wrong", "evaluation", inputs)
    parent = runner.NAMESPACE / "completed"
    parent.mkdir(parents=True); (parent / "COMPLETED.json").write_text("{}")
    with pytest.raises(ValueError, match="completed bundle"):
        runner.output_directory(parent / "child", "evaluation", inputs)
    partial = runner.NAMESPACE / "partial"
    partial.mkdir()
    with pytest.raises(FileExistsError): runner.output_directory(runner.NAMESPACE, "partial", inputs)
    target, link = tmp_path / "junction-target", runner.NAMESPACE / "escaped-link"
    target.mkdir()
    if os.name == "nt":
        command = f"New-Item -ItemType Junction -Path '{link}' -Target '{target}' | Out-Null"
        subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True, capture_output=True)
    else: link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        runner.output_directory(link, "evaluation", inputs)


def test_endpoint_cache_export_failure_blocks_all_nonoriginal_forwards(tmp_path, monkeypatch):
    kwargs, _, _ = setup_sources(tmp_path, monkeypatch)
    calls = []
    install_toy_runtime(monkeypatch, calls)
    def fail(*args, **kw): raise OSError("synthetic cache export failure")
    monkeypatch.setattr(runner, "write_map_archive", fail)
    with pytest.raises(OSError, match="cache export failure"):
        runner.run_insertion_deletion(**kwargs)
    assert len(calls) == 1098
    output = next(runner.NAMESPACE.iterdir())
    failed = runner.read_json(output / "FAILED.json")
    assert not failed["calls"]["global_replay_barrier_passed"]
    assert failed["calls"]["attempted"]["baseline"] == failed["calls"]["attempted"]["interior"] == 0
    assert not (output / "COMPLETED.json").exists()


@pytest.mark.parametrize("error_type", [FileExistsError, KeyboardInterrupt])
def test_mkdir_race_never_writes_failure_log_into_another_attempt(tmp_path, monkeypatch, error_type):
    kwargs, _, _ = setup_sources(tmp_path, monkeypatch)
    calls = []
    install_toy_runtime(monkeypatch, calls)
    original = Path.mkdir
    def race(path, *args, **kw):
        if path.name.startswith("phase5-perturbation-"):
            original(path, *args, **kw)
            (path / "other-attempt.txt").write_text("preserve this existing attempt")
            (path / "COMPLETED.json").write_text("other attempt's marker")
            raise error_type("synthetic directory race")
        return original(path, *args, **kw)
    monkeypatch.setattr(Path, "mkdir", race)
    with pytest.raises(error_type, match="directory race"):
        runner.run_insertion_deletion(**kwargs)
    assert not calls
    output = next(runner.NAMESPACE.iterdir())
    assert sorted(p.name for p in output.iterdir()) == ["COMPLETED.json", "other-attempt.txt"]
    assert (output / "COMPLETED.json").read_text() == "other attempt's marker"


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
@pytest.mark.parametrize("stage", ["replay", "interior"])
def test_graceful_interruptions_persist_actual_forward_context(tmp_path, monkeypatch, error_type, stage):
    kwargs, _, _ = setup_sources(tmp_path, monkeypatch)
    snapshot = dict(load_kwargs(kwargs).snapshot)
    calls, interruption = [], error_type("synthetic catchable interruption")
    stop = 1098 if stage == "replay" else 1103
    install_toy_runtime(monkeypatch, calls, interrupt_at=stop, interruption=interruption)
    with pytest.raises(error_type) as caught:
        runner.run_insertion_deletion(**kwargs)
    assert caught.value is interruption and len(calls) == stop
    output = next(runner.NAMESPACE.iterdir())
    failed = runner.read_json(output / "FAILED.json")
    ledger = failed["calls"]
    expected = {"unperturbed": 1098, "baseline": 0 if stage == "replay" else 1,
                "interior": 0 if stage == "replay" else 4}
    accepted = dict(expected)
    accepted["unperturbed" if stage == "replay" else "interior"] -= 1
    assert ledger["attempted"] == expected and ledger["accepted"] == accepted
    assert ledger["total_attempted"] == stop and ledger["total_accepted"] == stop-1
    assert ledger["global_replay_barrier_passed"] is (stage == "interior")
    assert len(failed["replay_checks"]) == (1097 if stage == "replay" else 1098)
    assert all(check["consistent"] for check in failed["replay_checks"])
    context = ledger["last_attempt"]
    assert ledger["failure"] == {**context, "error_type": error_type.__name__, "message": str(interruption)}
    assert context["call_index"] == stop and len(context["input_sha256"]) == 64
    assert (context["model"], context["seed"], context["unit_id"], context["stage"]) == (
        ("B1", 44, "unit_1097", "unperturbed") if stage == "replay"
        else ("ConsenDxFormer", 42, "unit_0000", "interior"))
    if stage == "interior":
        assert (context["order"], context["direction"], context["step"]) == ("dsa", "insertion", 4)
        assert runner.read_json(output / "replay_barrier.json")["replays_passed"] == 1098
    else:
        assert not (output / "replay_barrier.json").exists()
    assert failed["error_type"] == error_type.__name__ and not failed["subset_completed"]
    assert not failed["automatic_resume_permitted"] and not (output / "COMPLETED.json").exists()
    assert (output / "protocol.json").is_file() and (output / "provenance/reviewed_proposal.md").is_file()
    before = {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError): runner.run_insertion_deletion(**kwargs)
    assert len(calls) == stop
    assert before == {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()}
    runner.assert_source_snapshot_unchanged(snapshot)


def test_failure_record_write_error_preserves_original_replay_interruption(tmp_path, monkeypatch):
    kwargs, _, _ = setup_sources(tmp_path, monkeypatch)
    calls, attempted_records, interruption = [], [], KeyboardInterrupt("original replay interruption")
    install_toy_runtime(monkeypatch, calls, interrupt_at=3, interruption=interruption)
    original_write = runner.atomic_write_json
    def failed_write(path, value, **options):
        if Path(path).name == "FAILED.json":
            attempted_records.append(value)
            raise OSError("synthetic failure-record write error")
        return original_write(path, value, **options)
    monkeypatch.setattr(runner, "atomic_write_json", failed_write)
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_insertion_deletion(**kwargs)
    assert caught.value is interruption and len(calls) == 3
    assert len(attempted_records) == 1
    failed = attempted_records[0]
    assert failed["calls"]["attempted"] == {"unperturbed": 3, "baseline": 0, "interior": 0}
    assert failed["calls"]["accepted"] == {"unperturbed": 2, "baseline": 0, "interior": 0}
    assert failed["calls"]["failure"]["error_type"] == "KeyboardInterrupt"
    assert len(failed["replay_checks"]) == 2
    output = next(runner.NAMESPACE.iterdir())
    assert not (output / "FAILED.json").exists() and not (output / "COMPLETED.json").exists()
    assert (output / "protocol.json").is_file()
    with pytest.raises(FileExistsError): runner.run_insertion_deletion(**kwargs)
    assert len(calls) == 3


@pytest.mark.parametrize("secondary_failure", [None, "record_write", "marker_remove"])
def test_final_validation_interruption_cleanup_is_best_effort(tmp_path, monkeypatch, secondary_failure):
    from test_phase5_perturbation import tiny_inputs, toy_output
    from consendxformer.phase5_perturbation_exports import expected_tables, validate_result
    from consendxformer.phase5_descriptive_runner import _write_csv
    kwargs, _, _ = setup_sources(tmp_path, monkeypatch)
    runtime_calls, calls = [], []
    install_toy_runtime(monkeypatch, runtime_calls)
    small = tiny_inputs()
    def image_only(image):
        calls.append(True)
        return toy_output(image)
    small["predictors"] = {key: image_only for key in KEYS}
    execute = runner.execute_curves
    validate_exports, validate_completed = runner.validate_result_exports, runner.validate_completed_perturbation
    # Finalization-only harness: real small-cohort curves/exports, not a replacement
    # production cohort. The unchanged full successful-run test covers all 183.
    monkeypatch.setattr(runner, "execute_curves", lambda *a, ledger, **kw: execute(**small, ledger=ledger))
    def small_exports(output, result):
        validate_result(result, expected_nodules=1, expected_patients=1)
        archives = {"replay_cache.npz": result.replay, "orders.npz": result.orders,
                    "curves.npz": result.curves, "baseline_cache.npz": result.baseline}
        records = [runner.write_map_archive(output / name, arrays) for name, arrays in archives.items()]
        for name, table in expected_tables(result).items():
            _write_csv(output / "tables" / name, table)
        atomic_write_json(output / "replay_barrier.json", {"synthetic_finalization_fixture": True})
        return records
    monkeypatch.setattr(runner, "write_result", small_exports)
    monkeypatch.setattr(runner, "validate_result_exports", lambda *a: validate_exports(
        *a, expected_nodules=1, expected_patients=1))
    interruption = KeyboardInterrupt("original final-validation interruption")
    observed = {}
    def interrupt_final_validation(output):
        assert (output / "COMPLETED.json").is_file()
        observed["ledger"] = runner.read_json(output / "call_ledger.json")
        observed["replays"] = runner.read_json(output / "replay_checks.json")
        observed["files"] = {p.relative_to(output).as_posix(): sha256_file(p)
                             for p in output.rglob("*") if p.is_file() and p.name != "COMPLETED.json"}
        observed["calls"] = len(calls)
        raise interruption
    monkeypatch.setattr(runner, "validate_completed_perturbation", interrupt_final_validation)
    write_json, unlink = runner.atomic_write_json, Path.unlink
    secondary_attempts = []
    def failed_write(path, value, **options):
        if secondary_failure == "record_write" and Path(path).name == "FAILED.json":
            secondary_attempts.append("record_write")
            raise OSError("synthetic failure-record write error")
        return write_json(path, value, **options)
    def failed_unlink(path, *args, **options):
        if secondary_failure == "marker_remove" and path.name == "COMPLETED.json":
            secondary_attempts.append("marker_remove")
            raise PermissionError("synthetic marker removal error")
        return unlink(path, *args, **options)
    monkeypatch.setattr(runner, "atomic_write_json", failed_write)
    monkeypatch.setattr(Path, "unlink", failed_unlink)
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_insertion_deletion(**kwargs)
    assert caught.value is interruption and runtime_calls == []
    assert len(calls) == observed["calls"] == 2280
    output = next(runner.NAMESPACE.iterdir())
    assert secondary_attempts == ([] if secondary_failure is None else [secondary_failure])
    assert (output / "COMPLETED.json").exists() is (secondary_failure == "marker_remove")
    if secondary_failure == "record_write":
        assert not (output / "FAILED.json").exists()
    else:
        failed = runner.read_json(output / "FAILED.json")
        assert failed["error_type"] == "KeyboardInterrupt" and not failed["subset_completed"]
        assert failed["calls"] == observed["ledger"]
        assert failed["replay_checks"] == observed["replays"] and len(observed["replays"]) == 6
        assert failed["calls"]["attempted"] == failed["calls"]["accepted"] == {
            "unperturbed": 6, "baseline": 6, "interior": 2268}
        # Even if the filesystem blocks marker removal, recorded failure is not completion.
        with pytest.raises(ValueError, match="failed attempt"):
            validate_completed(output)
    assert observed["files"] == {name: sha256_file(output / name) for name in observed["files"]}
    with pytest.raises(FileExistsError): runner.run_insertion_deletion(**kwargs)
    assert len(calls) == 2280 and runtime_calls == []


def test_full_synthetic_run_exact_417240_calls_exports_subset_and_no_old_analyses(tmp_path, monkeypatch):
    kwargs, _, _ = setup_sources(tmp_path, monkeypatch)
    inputs = load_kwargs(kwargs)
    snapshot = dict(inputs.snapshot)
    calls = []
    install_toy_runtime(monkeypatch, calls)
    import consendxformer.phase5_descriptive_runner as descriptive
    import consendxformer.phase4_bootstrap as bootstrap
    import consendxformer.phase4_attention_runtime as attention
    import consendxformer.phase4_evaluation_runner as old_evaluation
    import torch
    def forbidden(*args, **kw): pytest.fail("Old analysis or research model executed in synthetic subset test")
    for module, names in ((descriptive, ("run_descriptive", "pointing_tables", "accuracy_intervals")),
                          (bootstrap, ("bootstrap_model_metrics", "bootstrap_paired_differences")),
                          (attention, ("infer_matched_attention",)),
                          (old_evaluation, ("compute_saved_prediction_evaluation", "run_phase4_saved_evaluation"))):
        for name in names: monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    output = runner.run_insertion_deletion(**kwargs)
    assert len(calls) == 417240 and all(calls.count(k) == 183*380 for k in KEYS)
    marker = runner.validate_completed_perturbation(output)
    assert marker["scope"] == "insertion_deletion_only" and marker["subset_completed"]
    assert not marker["full_supplemental_completion"] and not marker["stage_a_complete"]
    assert not marker["formal_tests_computed"] and len(marker["hypotheses"]) == 15
    assert not any({"p_raw", "p_adj", "p_value", "Q"} & set(h) for h in marker["hypotheses"])
    assert all(h["status"] == "unresolved_not_computed" for h in marker["hypotheses"])
    barrier = runner.read_json(output / "replay_barrier.json")
    assert barrier["replays_passed"] == 1098 and barrier["baseline_calls_at_barrier"] == barrier["interior_calls_at_barrier"] == 0
    ledger = runner.read_json(output / "call_ledger.json")
    assert ledger["accepted"] == {"unperturbed": 1098, "baseline": 1098, "interior": 415044}
    assert len(pd.read_csv(output / "tables/perturbation_per_seed.csv")) == 36
    assert len(pd.read_csv(output / "tables/perturbation_paired_descriptive.csv")) == 24
    raw = pd.read_csv(output / "tables/perturbation_curves.csv", usecols=["step", "unit_id"])
    assert len(raw) == 428220 and raw.unit_id.nunique() == 1098
    runner.assert_source_snapshot_unchanged(snapshot)
    with pytest.raises(FileExistsError): runner.run_insertion_deletion(**kwargs)
    assert len(calls) == 417240
    # Read-back must reject full-completion claims and changed table values, without any forward.
    original = dict(marker)
    marker["full_supplemental_completion"] = True
    atomic_write_json(output / "COMPLETED.json", marker, overwrite=True)
    with pytest.raises(ValueError, match="only"):
        runner.validate_completed_perturbation(output)
    atomic_write_json(output / "COMPLETED.json", original, overwrite=True)
    (output / "tables/perturbation_per_seed.csv").write_text("synthetic corruption")
    with pytest.raises(ValueError, match="hash differs"):
        runner.validate_completed_perturbation(output)
    assert len(calls) == 417240
