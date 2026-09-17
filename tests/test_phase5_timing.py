"""Synthetic CPU tests. Clocks, architectures and CUDA stubs are explicitly labeled."""
from contextlib import contextmanager
from pathlib import Path
import copy
import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.phase3_artifacts import atomic_write_json
from consendxformer.phase4_evaluation_runner import Phase4EvaluationError
from consendxformer.phase5_perturbation import KEYS
from consendxformer.phase5_timing import (make_plan, timing_specification, TimingLedger, measure_schedule, extrapolate)
from consendxformer.phase5_timing import DURATION_FIELDS, validate_loop_envelope
import consendxformer.phase5_device as device_api
import consendxformer.phase5_timing_runner as timing
import consendxformer.phase5_timing_runtime as runtime


class FakeClock:
    name = "explicit_test_fake_clock"
    def __init__(self): self.value = 0
    def __call__(self):
        self.value += 1_000_000
        return self.value


def tiny_output(image):
    # A test function, not a replacement production model or timing claim.
    probabilities = np.array([[.3, .3, .4]], np.float32)  # Accepted raw mass error > 1e-8; never normalize.
    return {"probs": probabilities, "logits": np.log(probabilities).astype(np.float32)}


def fixture_models():
    return {"resident_model_count": 6, "research_artifacts_accessed": False, "probe_calls": 0,
        "explicit_test_architecture_standin": True,
        "loaded_models": [{"model": m, "seed": s, "synthetic_initialization": True, "checkpoint_loaded": False,
            "state_sha256": "1"*64, "structure_sha256": "2"*64, "state_bytes": 4, "parameter_count": 1} for m, s in KEYS]}


def install_timing_standins(monkeypatch, root, *, calls=None, error=None, stop=None):
    contract = {"explicit_test_architecture_standin": True, "configuration_sha256": sha256_file(timing.SPEC_PATH)}
    bootstrap = {"device": "cpu", "before_torch_import": True, "torch_threads_requested": None,
                 "checked_unix_seconds": 1.0, "physical_gpu": None, "inventory": [], "active_processes": [],
                 "visible_devices": "", "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                 "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
                 "explicit_test_bootstrap_stub": True}
    with monkeypatch.context() as scoped:
        scoped.setattr(device_api, "_BOOTSTRAP", bootstrap)
        environment = {**runtime.runtime_environment("cpu"), "explicit_test_device_stub": True}
    snapshot = {str(timing.SPEC_PATH.resolve()): sha256_file(timing.SPEC_PATH)}
    monkeypatch.setattr(timing, "architecture_contract", lambda: (copy.deepcopy(contract), dict(snapshot)))
    monkeypatch.setattr(timing, "runtime_environment", lambda d: dict(environment))
    monkeypatch.setattr(timing, "bootstrap_record", lambda: copy.deepcopy(bootstrap))
    monkeypatch.setattr(timing, "validate_evaluator_repository", lambda p: {"commit": "456c84dd45690a7af0a70ce2b75658d7782de85b", "clean": True})
    monkeypatch.setattr(timing, "NAMESPACE", root)
    monkeypatch.setattr(timing, "memory_snapshot", lambda d: {"host_peak_rss_bytes": 4096,
        "host_current_rss_bytes": 4096, "host_measurement": "explicit_test_memory_stub"})
    monkeypatch.setattr(timing, "synchronize", lambda d: None)  # CPU device stub, no CUDA use.
    @contextmanager
    def models(device, cache_dir):
        assert device == "cpu"
        def predictor(image):
            if calls is not None:
                calls.append(image.copy())
                if len(calls) == stop: raise error
            return tiny_output(image)
        yield {key: predictor for key in KEYS}, fixture_models()
    monkeypatch.setattr(timing, "synthetic_predictors", models)
    return contract, environment, snapshot


_BUNDLE_CACHE = {}


def install_research_timing_fixture(tmp_path, monkeypatch, research_runner):
    """One explicitly fake-architecture CPU bundle per isolated pytest root, never real evidence."""
    root = tmp_path.parent / "_explicit_test_timing_standins"
    with monkeypatch.context() as scoped:
        contract, environment, snapshot = install_timing_standins(scoped, root)
        if root not in _BUNDLE_CACHE:
            _BUNDLE_CACHE[root] = timing.run_synthetic_timing(authorize_synthetic_timing=True,
                                                            attempt_id="test-fixture")
        bundle = _BUNDLE_CACHE[root]
    # Some existing CPU tests intentionally change torch thread settings. This
    # cached fake runtime must replay its recorded identity, not the later host
    # state; production still compares the independently observed live runtime.
    identity = timing.read_json(bundle / "identity.json")
    contract, environment = identity["architecture"], identity["runtime"]
    monkeypatch.setattr(research_runner, "architecture_contract", lambda: (copy.deepcopy(contract), dict(snapshot)), raising=False)
    monkeypatch.setattr(research_runner, "runtime_environment", lambda d: dict(environment), raising=False)
    return {"timing_bundle": bundle, "timing_manifest_sha256": sha256_file(bundle / "artifact_manifest.json")}


def test_specification_sequence_and_exact_fake_clock_accounting():
    spec = timing.load_specification()
    assert spec == timing_specification() and spec["total_calls"] == 2496
    images, plan = make_plan()
    assert plan == make_plan()[1]
    assert [r["input"] for r in plan["sequence"][:6]] == ["zero", "ramp", "random"]*2
    assert plan["sequence"][389]["step"] == 64 and plan["sequence"][390]["step"] == 0
    expected_random = np.random.Generator(np.random.PCG64(20260917)).random((1, 64, 64), dtype=np.float32)
    np.testing.assert_array_equal(images["random"], expected_random)
    original = {k: v.copy() for k, v in images.items()}
    calls, syncs = [], []
    def predictor(image):
        calls.append(image.copy())
        return tiny_output(image)
    ledger = TimingLedger()
    result = measure_schedule({k: predictor for k in KEYS}, images, plan, clock=FakeClock(),
        synchronize=lambda: syncs.append(True), ledger=ledger, memory=lambda: {"explicit_test_memory_stub": True})
    assert len(calls) == 2496 and len(syncs) == 4992
    assert ledger.attempted == ledger.accepted == {"warmup": 192, "measured": 2304}
    assert len(result["blocks"]) == 18 and all(b["calls"] == 128 for b in result["blocks"])
    assert result["clock"] == "explicit_test_fake_clock"
    assert all(row["input_construction_nanoseconds"] == row["adapter_nanoseconds"] == 1_000_000
               and row["validation_hashing_nanoseconds"] == 2_000_000 for row in result["calls"])
    validate_loop_envelope(result)
    estimates = extrapolate(result["blocks"])
    assert estimates["estimated_adapter_seconds"] == pytest.approx(417.24)
    assert all(u["research_calls"] == 69540 for u in estimates["units"])
    assert not estimates["actual_research_runtime"] and not estimates["range_is_confidence_interval"]
    for i, image in enumerate(calls):
        position = plan["sequence"][i % 416]
        mask = np.zeros((8, 8), dtype=bool); mask.flat[:position["step"]] = True
        mask = np.repeat(np.repeat(mask, 8, axis=0), 8, axis=1)
        if position["direction"] == "deletion": mask = ~mask
        np.testing.assert_array_equal(image[0], original[position["input"]]*mask)
    for name, image in images.items(): np.testing.assert_array_equal(image, original[name])


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_schedule_interruption_no_retries_no_accepted_output(error_type):
    images, plan = make_plan(); calls = []; interruption = error_type("explicit synthetic interruption")
    def predictor(image):
        calls.append(True)
        if len(calls) == 2496: raise interruption
        return tiny_output(image)
    ledger = TimingLedger()
    with pytest.raises(error_type) as caught:
        measure_schedule({k: predictor for k in KEYS}, images, plan, clock=FakeClock(), synchronize=lambda: None,
                         ledger=ledger, memory=lambda: {})
    assert caught.value is interruption and len(calls) == 2496
    assert ledger.attempted == {"warmup": 192, "measured": 2304}
    assert ledger.accepted == {"warmup": 192, "measured": 2303}
    assert ledger.failure["stage"] == "adapter" and ledger.failure["model"] == "B1" and ledger.failure["seed"] == 44


def test_raw_policy_failure_and_nonmonotonic_clock():
    images, plan = make_plan(); ledger = TimingLedger()
    def invalid(image): return {"probs": np.zeros((1, 3), np.float32), "logits": np.zeros((1, 3), np.float32)}
    with pytest.raises(ValueError, match="mass"):
        measure_schedule({k: invalid for k in KEYS}, images, plan, clock=FakeClock(), synchronize=lambda: None, ledger=ledger, memory=lambda: {})
    assert ledger.attempted["warmup"] == 1 and ledger.accepted["warmup"] == 0
    class BackwardClock(FakeClock):
        def __call__(self): self.value -= 1; return self.value
    with pytest.raises(ValueError, match="clock"):
        measure_schedule({k: tiny_output for k in KEYS}, images, plan, clock=BackwardClock(), synchronize=lambda: None, ledger=TimingLedger(), memory=lambda: {})


@pytest.fixture
def timing_bundle(tmp_path, monkeypatch):
    # Real perf_counter with tiny NumPy test stubs, not actual architecture benchmarking.
    fake = SimpleNamespace()
    options = install_research_timing_fixture(tmp_path, monkeypatch, fake)
    return options["timing_bundle"], options["timing_manifest_sha256"], fake.architecture_contract()[0], fake.runtime_environment("cpu")


def test_timing_bundle_complete_export_and_no_overwrite(timing_bundle, tmp_path, monkeypatch):
    bundle, digest, contract, environment = timing_bundle
    evidence = timing.validate_timing_bundle(bundle, digest, expected_architecture=contract, expected_runtime=environment)
    assert len(evidence["measurements"]["blocks"]) == 18 and len(evidence["measurements"]["calls"]) == 2496
    assert not evidence["binding"]["research_execution_authorized"]
    copied = timing.copy_timing_evidence(bundle, tmp_path / "embedded", digest)
    assert copied["binding"] == evidence["binding"]
    with np.load(bundle / "synthetic_outputs.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["probabilities"], np.repeat(tiny_output(None)["probs"], 2496, axis=0))
    assert 1e-8 < evidence["measurements"]["calls"][0]["raw_mass_deviation"] < 1e-6
    assert not timing.read_json(bundle / "COMPLETED.json")["full_supplemental_completion"]
    assert len(timing.read_json(bundle / "COMPLETED.json")["hypotheses"]) == 15
    with pytest.raises(FileExistsError): timing.copy_timing_evidence(bundle, tmp_path / "embedded", digest)
    install_timing_standins(monkeypatch, bundle.parent)
    with pytest.raises(FileExistsError): timing.run_synthetic_timing(authorize_synthetic_timing=True, attempt_id="test-fixture")


@pytest.mark.parametrize("change", ["hash", "bytes", "count", "phase", "duration", "estimate", "fake_clock", "authorized", "failed", "architecture", "runtime", "plan",
                                   "memory", "archive", "resident", "last_attempt", "bootstrap"])
def test_corrupt_or_incompatible_evidence_fails_closed(timing_bundle, tmp_path, change):
    import shutil
    original, digest, contract, environment = timing_bundle
    bundle = tmp_path / "corrupted"; shutil.copytree(original, bundle)
    if change == "hash": digest = "0"*64
    elif change == "bytes": (bundle / "summary.md").write_text("synthetic corruption")
    elif change == "failed": (bundle / "FAILED.json").write_text("{}")
    elif change == "architecture": contract = {**contract, "wrong": True}
    elif change == "runtime": environment = {**environment, "device": "cuda"}
    else:
        name = {"authorized": "COMPLETED.json", "plan": "plan.json", "resident": "models.json",
                "bootstrap": "bootstrap.json"}.get(change, "measurements.json")
        record = timing.read_json(bundle / name)
        if change == "count": record["ledger"]["attempted"]["measured"] -= 1
        elif change == "phase": record["calls"][32]["phase"] = "warmup"
        elif change == "duration": record["calls"][0]["adapter_seconds"] = -1
        elif change == "estimate": record["estimates"]["estimated_adapter_seconds"] += 1
        elif change == "fake_clock": record["clock"] = "explicit_test_fake_clock"
        elif change == "authorized": record["research_execution_authorized"] = True
        elif change == "plan": record["sequence"][0]["step"] = 1
        elif change == "memory": record["memory_before"]["host_peak_rss_bytes"] = -1
        elif change == "archive": record["output_archive"]["sha256"] = "0"*64
        elif change == "resident": record["resident_model_count"] = 1
        elif change == "last_attempt": record["ledger"]["last_attempt"]["seed"] = 43
        elif change == "bootstrap": record["before_torch_import"] = False
        atomic_write_json(bundle / name, record, overwrite=True)
        if name != "COMPLETED.json":
            manifest = timing.read_json(bundle / "artifact_manifest.json")
            manifest["files"][name] = sha256_file(bundle / name); manifest["bundle_sha256"] = sha256_json(manifest["files"])
            atomic_write_json(bundle / "artifact_manifest.json", manifest, overwrite=True)
            digest = sha256_file(bundle / "artifact_manifest.json")
            marker = timing.read_json(bundle / "COMPLETED.json"); marker["artifact_manifest_sha256"] = digest
            atomic_write_json(bundle / "COMPLETED.json", marker, overwrite=True)
    with pytest.raises((ValueError, OSError)):
        timing.validate_timing_bundle(bundle, digest, expected_architecture=contract, expected_runtime=environment)


def test_research_timing_gate_precedes_any_research_source_access(timing_bundle, monkeypatch):
    import consendxformer.phase5_perturbation_runner as research
    bundle, digest, contract, environment = timing_bundle
    monkeypatch.setattr(research, "validate_evaluator_repository", lambda p: {})
    monkeypatch.setattr(research, "runtime_environment", lambda d: environment)
    monkeypatch.setattr(research, "architecture_contract", lambda: (contract, {}))
    monkeypatch.setattr(research, "load_perturbation_inputs", lambda *a: pytest.fail("Research source accessed before timing guard"))
    args = dict(full_source_dirs=[], b1_source_dirs=[], attention_source="forbidden", attention_manifest_sha256="0"*64, authorize_execution=True)
    with pytest.raises(ValueError, match="timing bundle"): research.run_insertion_deletion(**args)
    with pytest.raises(ValueError, match="hash"):
        research.run_insertion_deletion(**args, timing_bundle=bundle, timing_manifest_sha256="0"*64)
    monkeypatch.setattr(research, "runtime_environment", lambda d: {**environment, "incompatible": True})
    with pytest.raises(ValueError, match="compatibility"):
        research.run_insertion_deletion(**args, timing_bundle=bundle, timing_manifest_sha256=digest)
    monkeypatch.setattr(research, "runtime_environment", lambda d: environment)
    monkeypatch.setattr(research, "architecture_contract", lambda: ({**contract, "configuration_sha256": "0"*64}, {}))
    with pytest.raises(ValueError, match="incompatibility"):
        research.run_insertion_deletion(**args, timing_bundle=bundle, timing_manifest_sha256=digest)


@pytest.mark.parametrize("logging_failure", [False, True])
def test_timing_interruption_preserves_attempt_and_original_error(tmp_path, monkeypatch, logging_failure):
    calls, interruption = [], KeyboardInterrupt("synthetic timing interruption")
    install_timing_standins(monkeypatch, tmp_path / "timing", calls=calls, stop=33, error=interruption)
    original = timing.atomic_write_json
    def write(path, value, **kw):
        if logging_failure and Path(path).name == "FAILED.json": raise OSError("synthetic logging failure")
        return original(path, value, **kw)
    monkeypatch.setattr(timing, "atomic_write_json", write)
    with pytest.raises(KeyboardInterrupt) as caught:
        timing.run_synthetic_timing(authorize_synthetic_timing=True, attempt_id="interrupt")
    assert caught.value is interruption and len(calls) == 33
    output = next(timing.NAMESPACE.iterdir())
    assert not (output / "COMPLETED.json").exists()
    if not logging_failure:
        failed = timing.read_json(output / "FAILED.json")
        assert failed["ledger"]["attempted"] == {"warmup": 32, "measured": 1}
        assert failed["ledger"]["accepted"] == {"warmup": 32, "measured": 0}
    with pytest.raises(FileExistsError): timing.run_synthetic_timing(authorize_synthetic_timing=True, attempt_id="interrupt")
    assert len(calls) == 33


def test_no_authorization_no_preparation_and_namespace_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "architecture_contract", lambda: pytest.fail("Unauthorized preparation"))
    with pytest.raises(ValueError, match="authorize"): timing.run_synthetic_timing(attempt_id="forbidden")
    calls = []; install_timing_standins(monkeypatch, tmp_path / "allowed", calls=calls)
    with pytest.raises(ValueError, match="namespace"):
        timing.run_synthetic_timing(authorize_synthetic_timing=True, attempt_id="wrong", output_root=tmp_path / "elsewhere")
    assert calls == []


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_final_validation_interruption_removes_marker(tmp_path, monkeypatch, error_type):
    calls = []
    install_timing_standins(monkeypatch, tmp_path / "timing", calls=calls)
    original = timing.validate_timing_bundle
    interruption = error_type("explicit final validation interruption")
    def interrupted(path, *a, **kw):
        assert (path / "COMPLETED.json").exists()
        raise interruption
    monkeypatch.setattr(timing, "validate_timing_bundle", interrupted)
    with pytest.raises(error_type) as caught:
        timing.run_synthetic_timing(authorize_synthetic_timing=True, attempt_id="final-interrupt")
    assert caught.value is interruption and len(calls) == 2496
    output = next(timing.NAMESPACE.iterdir())
    assert not (output / "COMPLETED.json").exists()
    assert timing.read_json(output / "FAILED.json")["ledger"]["accepted"] == {"warmup": 192, "measured": 2304}
    with pytest.raises(ValueError, match="failed"):
        original(output, sha256_file(output / "artifact_manifest.json"))


def test_source_mutation_prevents_timing_completion(tmp_path, monkeypatch):
    contract, _, _ = install_timing_standins(monkeypatch, tmp_path / "timing")
    source = tmp_path / "explicit_synthetic_source.txt"; source.write_text("original test source")
    snapshot = {str(source.resolve()): sha256_file(source)}
    monkeypatch.setattr(timing, "architecture_contract", lambda: (contract, snapshot))
    original = timing.synthetic_predictors
    @contextmanager
    def mutate_after_calls(*args):
        with original(*args) as prepared:
            yield prepared
            source.write_text("test mutation")
    monkeypatch.setattr(timing, "synthetic_predictors", mutate_after_calls)
    with pytest.raises(Phase4EvaluationError, match="changed"):
        timing.run_synthetic_timing(authorize_synthetic_timing=True, attempt_id="source-mutation")
    output = next(timing.NAMESPACE.iterdir())
    assert (output / "FAILED.json").exists() and not (output / "COMPLETED.json").exists()


def test_raced_output_directory_is_not_owned_or_modified(tmp_path, monkeypatch):
    calls = []; install_timing_standins(monkeypatch, tmp_path / "timing", calls=calls)
    original = Path.mkdir; raced = []
    def mkdir(path, *args, **kwargs):
        if path.name.startswith("phase5-timing-"):
            original(path, parents=True, exist_ok=False)
            (path / "COMPLETED.json").write_text("another attempt's sentinel")
            raced.append(path)
            raise FileExistsError("explicit test ownership race")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(FileExistsError):
        timing.run_synthetic_timing(authorize_synthetic_timing=True, attempt_id="ownership")
    assert calls == [] and len(raced) == 1
    assert (raced[0] / "COMPLETED.json").read_text() == "another attempt's sentinel"
    assert not (raced[0] / "FAILED.json").exists()


@pytest.mark.parametrize("value", [np.zeros((1, 3), np.float32), np.array([[.3, .3, .5]], np.float32),
    np.array([[-.1, .6, .5]], np.float32), np.array([[1.1, 0, 0]], np.float32), np.array([[np.nan, 0, 1]], np.float32),
    np.array([[np.inf, 0, 1]], np.float32), np.ones((1, 2), np.float32), np.ones((1, 3), np.float64)])
def test_invalid_synthetic_output_has_one_attempt_no_retry(value):
    images, plan = make_plan(); ledger = TimingLedger()
    def invalid(image): return {"probs": value, "logits": np.zeros((1, 3), np.float32)}
    with pytest.raises(ValueError):
        measure_schedule({k: invalid for k in KEYS}, images, plan, clock=FakeClock(), synchronize=lambda: None,
                         ledger=ledger, memory=lambda: {})
    assert ledger.attempted == {"warmup": 1, "measured": 0} and ledger.accepted == {"warmup": 0, "measured": 0}


def test_synthetic_plan_cannot_change_during_calls():
    images, plan = make_plan(); ledger = TimingLedger()
    def mutate(image):
        plan["sequence"][1]["input_sha256"] = "0"*64
        return tiny_output(image)
    with pytest.raises(ValueError, match="sequence changed"):
        measure_schedule({k: mutate for k in KEYS}, images, plan, clock=FakeClock(), synchronize=lambda: None,
                         ledger=ledger, memory=lambda: {})
    assert ledger.attempted == {"warmup": 1, "measured": 0}


def _verify_actual_constructor_paths(tmp_path, monkeypatch):
    import torch
    import consendxformer.phase4_execution as execution
    import consendxformer.phase4_attention_runtime as attention
    calls = []
    complete, baseline = execution._model, runtime.construct_b1
    def full(config): calls.append(("ConsenDxFormer", config["dataset"]["image_size"])); return complete(config)
    def b1(cls, config): calls.append(("B1", config["dataset"]["image_size"])); return baseline(cls, config)
    def forbidden(*a, **kw): pytest.fail("Research artifact access or hidden probe during synthetic construction")
    monkeypatch.setattr(execution, "_model", full); monkeypatch.setattr(runtime, "construct_b1", b1)
    monkeypatch.setattr(torch, "load", forbidden); monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(attention, "load_frozen_b1", forbidden)
    import consendxformer.phase4_posthoc as posthoc
    monkeypatch.setattr(posthoc, "load_frozen_final_best_model", forbidden)
    import pandas as pd
    monkeypatch.setattr(pd, "read_csv", forbidden)
    with runtime.synthetic_predictors("cpu", tmp_path / "no-cache") as (predictors, metadata):
        assert len(predictors) == 6 and len(calls) == 6 and all(size == 64 for _, size in calls)
        assert metadata["resident_model_count"] == 6 and metadata["probe_calls"] == 0
        assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
        for key, predictor in predictors.items():
            assert predictor.model.__class__.__name__ == ("ConsenDxFormer" if key[0] == "ConsenDxFormer" else "LungDxFormer")
            assert not predictor.model.training and not any(p.requires_grad for p in predictor.model.parameters())


def test_actual_constructor_paths_prepare_six_models_without_research_access(tmp_path):
    # Earlier legacy tests intentionally import parts of lungdxformer directly.
    # Do not clear their modules or relax the pinned-source guard. The real CLI
    # always starts in a fresh interpreter; isolate this constructor-only test too.
    command = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "import test_phase5_timing as fixture; "
        "patcher = fixture.pytest.MonkeyPatch(); "
        "fixture._verify_actual_constructor_paths(Path(sys.argv[2]), patcher); "
        "patcher.undo(); print('SYNTHETIC_CONSTRUCTORS_OK_ZERO_FORWARDS')"
    )
    result = subprocess.run([sys.executable, "-B", "-c", command, str(Path(__file__).resolve().parent), str(tmp_path)],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SYNTHETIC_CONSTRUCTORS_OK_ZERO_FORWARDS" in result.stdout


def test_uuid_bootstrap_before_torch_and_mapping_with_explicit_device_stubs(monkeypatch):
    uuid = "GPU-12345678-1234-1234-1234-123456789abc"
    physical = {"index": 1, "uuid": uuid, "name": "explicit fake GPU", "pci_bus_id": "0000:01:00.0",
                "total_mib": 11264, "used_mib": 8, "driver_version": "test-driver"}
    monkeypatch.setattr(device_api, "_BOOTSTRAP", None)
    monkeypatch.delitem(sys.modules, "torch", raising=False)  # Explicit pre-import device stub; no CUDA calls.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8"); monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)  # Track restoration after the bootstrap writes it.
    monkeypatch.setattr(device_api, "gpu_inventory", lambda: ([physical], []))
    record = device_api.bootstrap_device("cuda", uuid, torch_threads=32)
    assert record["physical_gpu"]["index"] == 1 and os.environ["CUDA_VISIBLE_DEVICES"] == uuid
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
        get_device_properties=lambda i: SimpleNamespace(name=physical["name"], total_memory=11264*1024**2, major=7, minor=5)))
    mapping = device_api.verify_runtime_device(fake_torch, "cuda")
    assert mapping["logical_device"] == "cuda:0" and mapping["physical_gpu"]["uuid"] == uuid
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="isolation"): device_api.verify_runtime_device(fake_torch, "cuda")


@pytest.mark.parametrize("issue", ["late", "uuid", "workspace", "visible", "busy", "process"])
def test_cuda_preflight_fails_without_fallback(monkeypatch, issue):
    uuid = "GPU-12345678-1234-1234-1234-123456789abc"
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8"); monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(device_api, "gpu_inventory", lambda: ([{"uuid": uuid, "used_mib": 300 if issue == "busy" else 8}],
                                                              [{"uuid": uuid, "pid": 999}] if issue == "process" else []))
    if issue == "late": monkeypatch.setitem(sys.modules, "torch", object())
    if issue == "uuid": uuid = "GPU-short"
    if issue == "workspace": monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    if issue == "visible": monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises((ValueError, RuntimeError)): device_api.bootstrap_device("cuda", uuid)


def set_launch_mode(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)
    else:
        monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", value)


def capture_cpu_launch_stub(monkeypatch, value):
    """Explicit pre-import CPU/Torch-interface stub; no real CUDA/model execution."""
    monkeypatch.setattr(device_api, "_BOOTSTRAP", None)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    set_launch_mode(monkeypatch, value)
    captured = device_api.bootstrap_device("cpu")
    stub = SimpleNamespace(__version__="explicit_test_torch_stub", version=SimpleNamespace(cuda=None),
        __config__=SimpleNamespace(show=lambda: "explicit runtime stub, no forwards"),
        get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1,
        get_float32_matmul_precision=lambda: "highest",
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: None, allow_tf32=False),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))))
    monkeypatch.setitem(sys.modules, "torch", stub)
    return captured


@pytest.mark.parametrize("value", [None, "", "0", "1", "operator-supplied-text"])
def test_launch_mode_captured_raw_and_runtime_matched(monkeypatch, value):
    captured = capture_cpu_launch_stub(monkeypatch, value)
    observed = runtime.runtime_environment("cpu")
    assert captured["cuda_launch_blocking"] == observed["environment"]["CUDA_LAUNCH_BLOCKING"] == value
    assert os.environ.get("CUDA_LAUNCH_BLOCKING") == value  # No default or silent mutation.
    device_api.validate_device_evidence(captured, observed)


@pytest.mark.parametrize("before,after", [(None, "0"), ("0", "1"), ("1", None), ("", "0")])
def test_launch_mode_change_or_missing_capture_rejected_before_runtime(monkeypatch, before, after):
    capture_cpu_launch_stub(monkeypatch, before)
    set_launch_mode(monkeypatch, after)
    with pytest.raises(ValueError, match="CUDA_LAUNCH_BLOCKING changed"):
        runtime.runtime_environment("cpu")
    record = device_api.bootstrap_record(); record.pop("cuda_launch_blocking")
    monkeypatch.setattr(device_api, "_BOOTSTRAP", record)
    with pytest.raises(ValueError, match="Missing/invalid CUDA_LAUNCH_BLOCKING"):
        runtime.runtime_environment("cpu")


def test_launch_mode_changed_during_observation_is_rejected(monkeypatch):
    capture_cpu_launch_stub(monkeypatch, "0")
    def explicit_mutating_runtime_stub():
        monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
        return None
    sys.modules["torch"].backends.cudnn.version = explicit_mutating_runtime_stub
    with pytest.raises(ValueError, match="CUDA_LAUNCH_BLOCKING changed"):
        runtime.runtime_environment("cpu")


def reseal_timing_fixture(bundle):
    """Rehash deliberately modified temporary evidence, including changed runtime identities."""
    identity = timing.read_json(bundle / "identity.json")
    timing_id = "phase5-timing-" + sha256_json(identity)[:16]
    manifest = timing.read_json(bundle / "artifact_manifest.json")
    manifest.update(timing_id=timing_id, files={name: sha256_file(bundle / name) for name in sorted(timing.PAYLOADS)})
    manifest["bundle_sha256"] = sha256_json(manifest["files"])
    atomic_write_json(bundle / "artifact_manifest.json", manifest, overwrite=True)
    digest = sha256_file(bundle / "artifact_manifest.json")
    marker = timing.read_json(bundle / "COMPLETED.json")
    marker.update(timing_id=timing_id, artifact_manifest_sha256=digest)
    atomic_write_json(bundle / "COMPLETED.json", marker, overwrite=True)
    return digest


def cloned_timing_fixture(timing_bundle, tmp_path):
    import shutil
    result = tmp_path / "explicit_modified_toy_evidence"
    shutil.copytree(timing_bundle[0], result)
    return result


@pytest.mark.parametrize("value", [None, "", "0", "1"])
def test_matching_launch_evidence_copy_readback(timing_bundle, tmp_path, value):
    bundle = cloned_timing_fixture(timing_bundle, tmp_path)
    identity, bootstrap = (timing.read_json(bundle / name) for name in ("identity.json", "bootstrap.json"))
    identity["runtime"]["environment"]["CUDA_LAUNCH_BLOCKING"] = value
    bootstrap["cuda_launch_blocking"] = value
    atomic_write_json(bundle / "identity.json", identity, overwrite=True)
    atomic_write_json(bundle / "bootstrap.json", bootstrap, overwrite=True)
    digest = reseal_timing_fixture(bundle)
    checked = timing.validate_timing_bundle(bundle, digest, expected_runtime=identity["runtime"])
    copied = timing.copy_timing_evidence(bundle, tmp_path / "copy", digest)
    assert checked["binding"] == copied["binding"]
    assert copied["identity"]["runtime"]["environment"]["CUDA_LAUNCH_BLOCKING"] == value


@pytest.mark.parametrize("defect", ["missing_bootstrap", "missing_runtime", "both_missing", "binding", "nontext"])
def test_rehashed_missing_or_inconsistent_launch_evidence_rejected(timing_bundle, tmp_path, defect):
    bundle = cloned_timing_fixture(timing_bundle, tmp_path)
    identity, bootstrap = (timing.read_json(bundle / name) for name in ("identity.json", "bootstrap.json"))
    if defect in ("missing_bootstrap", "both_missing"): bootstrap.pop("cuda_launch_blocking")
    if defect in ("missing_runtime", "both_missing"): identity["runtime"]["environment"].pop("CUDA_LAUNCH_BLOCKING")
    if defect == "binding":
        bootstrap["cuda_launch_blocking"] = "0"
        identity["runtime"]["environment"]["CUDA_LAUNCH_BLOCKING"] = "1"
    if defect == "nontext":
        bootstrap["cuda_launch_blocking"] = 0
        identity["runtime"]["environment"]["CUDA_LAUNCH_BLOCKING"] = 0
    atomic_write_json(bundle / "identity.json", identity, overwrite=True)
    atomic_write_json(bundle / "bootstrap.json", bootstrap, overwrite=True)
    digest = reseal_timing_fixture(bundle)
    with pytest.raises(ValueError, match="CUDA_LAUNCH_BLOCKING"):
        timing.validate_timing_bundle(bundle, digest)
    with pytest.raises(ValueError, match="CUDA_LAUNCH_BLOCKING"):
        timing.copy_timing_evidence(bundle, tmp_path / "forbidden_copy", digest)
    assert not (tmp_path / "forbidden_copy").exists()


def test_launch_mismatch_reaches_neither_research_loaders_nor_forwards(timing_bundle, tmp_path, monkeypatch):
    import consendxformer.phase5_perturbation_runner as research
    import torch
    bundle = cloned_timing_fixture(timing_bundle, tmp_path)
    identity, bootstrap = (timing.read_json(bundle / name) for name in ("identity.json", "bootstrap.json"))
    identity["runtime"]["environment"]["CUDA_LAUNCH_BLOCKING"] = "0"
    bootstrap["cuda_launch_blocking"] = "0"
    atomic_write_json(bundle / "identity.json", identity, overwrite=True)
    atomic_write_json(bundle / "bootstrap.json", bootstrap, overwrite=True)
    digest = reseal_timing_fixture(bundle)
    observed = copy.deepcopy(identity["runtime"])
    observed["environment"]["CUDA_LAUNCH_BLOCKING"] = "1"  # Only this compatibility field differs.
    reached = []
    def forbidden(*a, **kw): reached.append(True); pytest.fail("Research access/forward before timing compatibility gate")
    monkeypatch.setattr(research, "validate_evaluator_repository", lambda p: {})
    monkeypatch.setattr(research, "architecture_contract", lambda: (identity["architecture"], {}))
    monkeypatch.setattr(research, "runtime_environment", lambda d: observed)  # Explicit observed-device stub.
    monkeypatch.setattr(research, "load_perturbation_inputs", forbidden)
    monkeypatch.setattr(research, "frozen_predictors", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    with pytest.raises(ValueError, match="runtime/device compatibility"):
        research.run_insertion_deletion(full_source_dirs=[], b1_source_dirs=[], attention_source="forbidden",
            attention_manifest_sha256="0"*64, authorize_execution=True, timing_bundle=bundle, timing_manifest_sha256=digest)
    assert reached == []


@pytest.mark.parametrize("case", ["zero", "undersized", "warmup_omitted", "overhead", "exact", "rounding", "one_ns", "missing_ns"])
def test_rehashed_loop_envelope_semantics(timing_bundle, tmp_path, case):
    bundle = cloned_timing_fixture(timing_bundle, tmp_path)
    measured = timing.read_json(bundle / "measurements.json")
    if case == "rounding":
        # Explicit hypothetical 13-ns intervals: fsum(seconds) is one ulp above
        # the canonical exact integer total. This is not measured runtime data.
        for row in measured["calls"]:
            for field in DURATION_FIELDS:
                row[field + "_nanoseconds"] = 13
                row[field + "_seconds"] = 13/1e9
        for block in measured["blocks"]:
            rows = [r for r in measured["calls"] if (r["model"], r["seed"], r["block"]) == (block["model"], block["seed"], block["block"])]
            for field in DURATION_FIELDS: block[field + "_seconds"] = math.fsum(r[field + "_seconds"] for r in rows)
        measured["estimates"] = extrapolate(measured["blocks"])
    contained_ns = sum(r[field + "_nanoseconds"] for r in measured["calls"] for field in DURATION_FIELDS)
    measured_only_ns = sum(r[field + "_nanoseconds"] for r in measured["calls"] if r["phase"] == "measured" for field in DURATION_FIELDS)
    loop_ns = {"zero": 0, "undersized": contained_ns//2, "warmup_omitted": measured_only_ns,
               "overhead": contained_ns + 1_000_000, "one_ns": contained_ns-1}.get(case, contained_ns)
    measured["loop_wall_nanoseconds"] = loop_ns
    measured["loop_wall_seconds"] = loop_ns/1e9
    if case == "warmup_omitted": assert measured_only_ns < contained_ns
    if case == "rounding":
        contained = math.fsum(r[field + "_seconds"] for r in measured["calls"] for field in DURATION_FIELDS)
        gap = contained - measured["loop_wall_seconds"]
        assert 0 < gap <= 4*math.ulp(contained)
    if case == "missing_ns": measured["calls"][0].pop("adapter_nanoseconds")
    atomic_write_json(bundle / "measurements.json", measured, overwrite=True)
    digest = reseal_timing_fixture(bundle)
    if case in ("overhead", "exact", "rounding"):
        checked = timing.validate_timing_bundle(bundle, digest)
        copied = timing.copy_timing_evidence(bundle, tmp_path / "valid_copy", digest)
        assert checked["binding"] == copied["binding"]
    else:
        with pytest.raises(ValueError, match="Loop envelope|nanosecond"):
            timing.validate_timing_bundle(bundle, digest)


def test_exact_ns_guard_rejects_deficit_hidden_by_float_resolution():
    # Deliberately huge hypothetical interval, not a real timing observation.
    rows = [{field + suffix: value for field in DURATION_FIELDS
             for suffix, value in (("_nanoseconds", 0), ("_seconds", 0.0))} for _ in range(2496)]
    rows[0].update(adapter_nanoseconds=2**54, adapter_seconds=2**54/1e9)
    measured = {"calls": rows, "loop_wall_nanoseconds": 2**54, "loop_wall_seconds": 2**54/1e9}
    validate_loop_envelope(measured)
    measured["loop_wall_nanoseconds"] -= 1
    assert measured["loop_wall_nanoseconds"]/1e9 == measured["loop_wall_seconds"]
    with pytest.raises(ValueError, match="Loop envelope"):
        validate_loop_envelope(measured)


@pytest.mark.parametrize("ulps", [4, 5])
def test_explicit_sum_rounding_stub_at_documented_ulp_boundary(monkeypatch, ulps):
    # An explicitly injected sum-rounding stub isolates the <=4-ULP comparison.
    # The 13-ns rehashed fixture above separately exercises actual math.fsum.
    rows = [{field + suffix: value for field in DURATION_FIELDS
             for suffix, value in (("_nanoseconds", 0), ("_seconds", 0.0))} for _ in range(2496)]
    rows[0].update(adapter_nanoseconds=1_000_000_000, adapter_seconds=1.0)
    measured = {"calls": rows, "loop_wall_nanoseconds": 1_000_000_000, "loop_wall_seconds": 1.0}
    monkeypatch.setattr(math, "fsum", lambda values: 1.0 + ulps*math.ulp(1.0))
    if ulps == 4:
        validate_loop_envelope(measured)
    else:
        with pytest.raises(ValueError, match="Loop envelope"):
            validate_loop_envelope(measured)
