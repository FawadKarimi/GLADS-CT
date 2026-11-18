"""CPU-only GPU identity tests: all management, driver and Torch devices are stubs."""
from pathlib import Path
import copy
import ctypes
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consendxformer import phase5_cuda_identity as native
from consendxformer import phase5_device as device
from consendxformer import phase5_timing_runner as timing
from consendxformer import phase5_timing_runtime as runtime
from consendxformer.phase3_artifacts import atomic_write_json
from test_phase5_timing import timing_bundle, cloned_timing_fixture, reseal_timing_fixture

UUID = "GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90"
PHYSICAL = {"index": 1, "uuid": UUID, "name": "NVIDIA GeForce RTX 2080 Ti",
            "pci_bus_id": "00000000:3B:00.0", "total_mib": 11264,
            "used_mib": 8, "driver_version": "535.230.02"}
TORCH_BYTES = 11546394624


def native_driver_standin(*, count=1, raw_uuid=None, pci=b"0000:3b:00.0\0", error=None):
    """Real ctypes callback ABI/buffers, fake CPU implementations, NOT a CUDA library."""
    calls = []
    driver = SimpleNamespace()
    raw_uuid = bytes.fromhex(UUID[4:].replace("-", "")) if raw_uuid is None else raw_uuid
    def init(flags):
        calls.append(("cuInit", flags)); return 0
    def get_count(out):
        calls.append(("cuDeviceGetCount",)); out[0] = count; return 0
    def get_device(out, ordinal):
        calls.append(("cuDeviceGet", ordinal)); out[0] = 17; return 0  # Handle != ordinal/index.
    def get_uuid(out, handle):
        calls.append(("cuDeviceGetUuid_v2", handle))
        ctypes.memmove(out, raw_uuid, len(raw_uuid)); return 0
    def get_pci(out, size, handle):
        calls.append(("cuDeviceGetPCIBusId", size, handle))
        ctypes.memmove(out, pci, min(size, len(pci))); return 0
    definitions = [
        ("cuInit", [ctypes.c_uint], init),
        ("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)], get_count),
        ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int], get_device),
        ("cuDeviceGetUuid_v2", [ctypes.POINTER(native.CUuuid), ctypes.c_int], get_uuid),
        ("cuDeviceGetPCIBusId", [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int], get_pci),
    ]
    for name, args, fn in definitions:
        if error == name:
            def failed(*values, _name=name):
                calls.append((_name, "failure")); return 34  # CUDA_ERROR_STUB_LIBRARY.
            fn = failed
        setattr(driver, name, ctypes.CFUNCTYPE(ctypes.c_int, *args)(fn))
    return driver, calls, definitions


def install_cuda_standins(monkeypatch, **driver_options):
    """Pre-import evidence and property-only Torch substitute; never imports/queries CUDA."""
    record = {"device": "cuda", "before_torch_import": True, "torch_threads_requested": 32,
              "checked_unix_seconds": 1.0, "physical_gpu": copy.deepcopy(PHYSICAL),
              "inventory": [copy.deepcopy(PHYSICAL)], "active_processes": [],
              "visible_devices": UUID, "cublas_workspace_config": ":4096:8",
              "cuda_launch_blocking": None, "explicit_test_device_stub": True}
    monkeypatch.setattr(device, "_BOOTSTRAP", record)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)
    driver, calls, signatures = native_driver_standin(**driver_options)
    monkeypatch.setattr(native, "_load_driver", lambda: driver)
    prop = SimpleNamespace(name=PHYSICAL["name"], total_memory=TORCH_BYTES, major=7, minor=5)
    stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                               get_device_properties=lambda i: prop))
    return record, stub, prop, driver, calls, signatures


def test_supplied_unequal_memory_native_boundary_and_offline(monkeypatch):
    record, torch_stub, _, driver, calls, signatures = install_cuda_standins(monkeypatch)
    mapping = device.verify_runtime_device(torch_stub, "cuda")
    assert calls == [("cuInit", 0), ("cuDeviceGetCount",), ("cuDeviceGet", 0),
                     ("cuDeviceGetUuid_v2", 17), ("cuDeviceGetPCIBusId", 32, 17)]
    assert ctypes.sizeof(native.CUuuid) == 16
    for name, args, _ in signatures:
        assert list(getattr(driver, name).argtypes) == args
        assert getattr(driver, name).restype is ctypes.c_int
    direct = mapping["direct_driver_identity"]
    assert direct["uuid"] == UUID and direct["uuid_bytes_hex"] == UUID[4:].replace("-", "")
    assert direct["pci_bus_id"] == "0000:3b:00.0" and direct["pci_components"] == [0, 59, 0, 0]
    assert mapping["physical_gpu"]["pci_bus_id"] == "00000000:3B:00.0"
    assert mapping["memory_diagnostics"] == {
        "schema": "phase5_gpu_memory_diagnostics_v1", "nvidia_smi_total_mib": 11264,
        "nvidia_smi_total_bytes": 11811160064, "torch_device_properties_total_bytes": TORCH_BYTES,
        "torch_minus_nvidia_smi_bytes": -264765440}
    observed = {**mapping, "torch_threads": 32, "environment": {
        "CUDA_VISIBLE_DEVICES": UUID, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_LAUNCH_BLOCKING": None}}
    monkeypatch.setattr(native, "_load_driver", forbidden_driver)
    device.validate_device_evidence(record, observed)


def forbidden_driver(*args, **kwargs):
    pytest.fail("CPU/offline validation must never query or load a GPU driver")


@pytest.mark.parametrize("pci", [b"0001:3b:00.0\0", b"0000:3c:00.0\0", b"0000:3b:01.0\0", b"0000:3b:00.1\0"])
def test_each_pci_component_mismatch_rejected(monkeypatch, pci):
    _, stub, _, _, _, _ = install_cuda_standins(monkeypatch, pci=pci)
    with pytest.raises(ValueError, match="PCI address differs"):
        device.verify_runtime_device(stub, "cuda")


@pytest.mark.parametrize("pci", [b"0000:3B:00.0\0", b"00000000:3b:00.0\0"])
def test_equivalent_full_pci_formats(monkeypatch, pci):
    _, stub, _, _, _, _ = install_cuda_standins(monkeypatch, pci=pci)
    assert device.verify_runtime_device(stub, "cuda")["direct_driver_identity"]["pci_components"] == [0, 59, 0, 0]


def test_wrong_uuid_rejected_even_with_equal_memory_and_name(monkeypatch):
    _, stub, prop, _, _, _ = install_cuda_standins(monkeypatch, raw_uuid=b"x"*16)
    prop.total_memory = PHYSICAL["total_mib"]*1024**2
    with pytest.raises(ValueError, match="UUID differs"):
        device.verify_runtime_device(stub, "cuda")


@pytest.mark.parametrize("pci", [b"3b:00.0\0", b"000:3b:00.0\0", b"0000:3b:00\0", b"0000:3b:20.0\0",
    b"0000:3b:00.8\0", b"0000:3b:00.0 \0", b"\xff\0", b"x"*32, b"\0"])
def test_malformed_or_unterminated_driver_pci_rejected(monkeypatch, pci):
    install_cuda_standins(monkeypatch, pci=pci)
    with pytest.raises(ValueError, match="PCI"):
        native.query_driver_identity()


def test_unwritten_uuid_and_auxiliary_name_rejected(monkeypatch):
    _, stub, _, _, _, _ = install_cuda_standins(monkeypatch, raw_uuid=b"")
    with pytest.raises(ValueError, match="Zero GPU UUID"):
        device.verify_runtime_device(stub, "cuda")
    _, stub, prop, _, _, _ = install_cuda_standins(monkeypatch)
    prop.name = "different name"
    with pytest.raises(ValueError, match="properties disagree"):
        device.verify_runtime_device(stub, "cuda")


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.parametrize("source", ["driver", "torch"])
def test_isolated_counts_fail_before_identity_or_model(monkeypatch, count, source):
    _, stub, _, _, calls, _ = install_cuda_standins(monkeypatch, count=count if source == "driver" else 1)
    if source == "torch": stub.cuda.device_count = lambda: count
    with pytest.raises(ValueError, match="Exactly one isolated"):
        device.verify_runtime_device(stub, "cuda")
    assert not any(row[0] == "cuDeviceGet" for row in calls)


@pytest.mark.parametrize("symbol", ["cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId"])
@pytest.mark.parametrize("defect", ["missing", "result"])
def test_every_native_error_or_missing_symbol_fails_no_retry(monkeypatch, symbol, defect):
    _, stub, _, driver, calls, definitions = install_cuda_standins(monkeypatch, error=symbol if defect == "result" else None)
    if defect == "missing": delattr(driver, symbol)
    with pytest.raises(RuntimeError, match=symbol):
        device.verify_runtime_device(stub, "cuda")
    if defect == "missing": assert calls == []
    else:
        assert calls[-1] == (symbol, "failure")
        assert len(calls) == [row[0] for row in definitions].index(symbol) + 1


def test_lazy_installed_linux_soname_and_unavailable_driver(monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "linux")
    seen = []
    def missing(name): seen.append(name); raise OSError("explicit CPU missing-library stub")
    monkeypatch.setattr(native.ctypes, "CDLL", missing)
    with pytest.raises(RuntimeError, match="libcuda.so.1 unavailable"):
        native._load_driver()
    assert seen == ["libcuda.so.1"]
    sentinel = object()
    monkeypatch.setattr(native.ctypes, "CDLL", lambda name: sentinel if name == "libcuda.so.1" else forbidden_driver())
    assert native._load_driver() is sentinel
    monkeypatch.setattr(native.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="requires Linux"):
        native._load_driver()


def test_cpu_never_loads_driver(monkeypatch):
    monkeypatch.setattr(native.ctypes, "CDLL", forbidden_driver)
    monkeypatch.setattr(native, "_load_driver", forbidden_driver)
    assert device.verify_runtime_device(object(), "cpu")["physical_gpu"] is None


@pytest.mark.parametrize("value", [0, -1, True, 11264.0])
@pytest.mark.parametrize("source", ["nvidia", "torch"])
def test_live_memory_totals_are_positive_integers(monkeypatch, value, source):
    record, stub, prop, _, _, _ = install_cuda_standins(monkeypatch)
    if source == "nvidia": record["physical_gpu"]["total_mib"] = value
    else: prop.total_memory = value
    with pytest.raises(ValueError, match="positive integers"):
        device.verify_runtime_device(stub, "cuda")


@pytest.mark.parametrize("total", [11811160064, 11811160064 + 8192])
def test_zero_and_positive_deltas_are_diagnostics_not_acceptance_limits(monkeypatch, total):
    _, stub, prop, _, _, _ = install_cuda_standins(monkeypatch)
    prop.total_memory = total
    observed = device.verify_runtime_device(stub, "cuda")
    assert observed["memory_diagnostics"]["torch_minus_nvidia_smi_bytes"] == total - 11811160064


def gpu_bundle(timing_bundle, tmp_path, monkeypatch):
    """Rehashed synthetic-only evidence with GPU metadata from native CPU callbacks."""
    bundle = cloned_timing_fixture(timing_bundle, tmp_path)
    with monkeypatch.context() as scoped:
        record, stub, _, _, _, _ = install_cuda_standins(scoped)
        mapping = device.verify_runtime_device(stub, "cuda")
    identity = timing.read_json(bundle / "identity.json")
    identity["runtime"].update(mapping, torch_threads=32)
    identity["runtime"]["environment"].update(CUDA_VISIBLE_DEVICES=UUID, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                                            CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_LAUNCH_BLOCKING=None)
    measured = timing.read_json(bundle / "measurements.json")
    for memory in [measured["memory_before"], measured["memory_after"], *(b["memory"] for b in measured["blocks"])]:
        memory.update(cuda_allocated_bytes=0, cuda_reserved_bytes=0, cuda_peak_allocated_bytes=0,
                      cuda_peak_reserved_bytes=0, cuda_free_bytes=TORCH_BYTES, cuda_total_bytes=TORCH_BYTES)
    atomic_write_json(bundle / "identity.json", identity, overwrite=True)
    atomic_write_json(bundle / "bootstrap.json", record, overwrite=True)
    atomic_write_json(bundle / "measurements.json", measured, overwrite=True)
    return bundle, reseal_timing_fixture(bundle), identity, record


def test_gpu_export_and_embedded_readback_without_driver(timing_bundle, tmp_path, monkeypatch):
    bundle, digest, identity, _ = gpu_bundle(timing_bundle, tmp_path, monkeypatch)
    monkeypatch.setattr(native, "_load_driver", forbidden_driver)
    monkeypatch.setattr(device, "query_driver_identity", forbidden_driver)
    monkeypatch.setattr(native.ctypes, "CDLL", forbidden_driver)
    checked = timing.validate_timing_bundle(bundle, digest, expected_runtime=identity["runtime"])
    copied = timing.copy_timing_evidence(bundle, tmp_path / "embedded", digest)
    assert checked["binding"] == copied["binding"]
    assert copied["identity"]["runtime"] == identity["runtime"]
    assert copied["identity"]["runtime"]["memory_diagnostics"]["torch_minus_nvidia_smi_bytes"] == -264765440


def test_old_timing_revision_cannot_certify_new_mapping(timing_bundle, tmp_path):
    bundle = cloned_timing_fixture(timing_bundle, tmp_path)
    identity = timing.read_json(bundle / "identity.json")
    spec = timing.read_json(bundle / "specification.json")
    for old in (spec, identity["specification"]):
        old.pop("evidence_revision")
        old["cuda"].pop("mapping_verification")
        old["cuda"].pop("memory_totals")
    atomic_write_json(bundle / "identity.json", identity, overwrite=True)
    atomic_write_json(bundle / "specification.json", spec, overwrite=True)
    digest = reseal_timing_fixture(bundle)
    with pytest.raises(ValueError, match="specification/sequence/identity"):
        timing.validate_timing_bundle(bundle, digest)


@pytest.mark.parametrize("defect", ["missing_direct", "missing_uuid", "wrong_uuid", "bad_uuid", "zero_uuid", "uuid_bytes",
    "missing_pci", "wrong_pci", "bad_pci", "components", "count", "ordinal", "schema", "api", "old_method",
    "missing_memory", "delta", "total_float", "total_bool", "total_zero", "torch_float", "torch_negative", "delta_float",
    "runtime_total_float", "inventory_total_float", "conversion"])
def test_rehashed_gpu_evidence_defects_rejected(timing_bundle, tmp_path, monkeypatch, defect):
    bundle, _, identity, record = gpu_bundle(timing_bundle, tmp_path, monkeypatch)
    observed = identity["runtime"]; direct = observed["direct_driver_identity"]
    if defect == "missing_direct": observed.pop("direct_driver_identity")
    elif defect == "missing_uuid": direct.pop("uuid")
    elif defect == "wrong_uuid": direct["uuid"] = "GPU-11111111-1111-1111-1111-111111111111"
    elif defect == "bad_uuid": direct["uuid"] = "GPU-short"
    elif defect == "zero_uuid": direct["uuid"] = "GPU-00000000-0000-0000-0000-000000000000"
    elif defect == "uuid_bytes": direct["uuid_bytes_hex"] = "ff"*16
    elif defect == "missing_pci": direct.pop("pci_bus_id")
    elif defect == "wrong_pci": direct["pci_bus_id"] = "0001:3b:00.0"; direct["pci_components"][0] = 1
    elif defect == "bad_pci": direct["pci_bus_id"] = "3b:00.0"
    elif defect == "components": direct["pci_components"][0] = False
    elif defect == "count": direct["device_count"] = 2
    elif defect == "ordinal": direct["logical_ordinal"] = 1
    elif defect == "schema": direct["schema"] = "obsolete"
    elif defect == "api": direct["api"] = "environment_only"
    elif defect == "old_method": observed["mapping_verification"] = "exact_nvidia_smi_uuid_before_torch_import_and_single_logical_device"
    elif defect == "missing_memory": observed.pop("memory_diagnostics")
    elif defect == "delta": observed["memory_diagnostics"]["torch_minus_nvidia_smi_bytes"] += 1
    elif defect == "delta_float": observed["memory_diagnostics"]["torch_minus_nvidia_smi_bytes"] = -264765440.0
    elif defect == "runtime_total_float": observed["physical_gpu"]["total_mib"] = 11264.0
    elif defect == "inventory_total_float": record["inventory"][0]["total_mib"] = 11264.0
    elif defect == "conversion": observed["memory_diagnostics"]["nvidia_smi_total_bytes"] += 1
    elif defect.startswith("total_"):
        value = {"total_float": 11264., "total_bool": True, "total_zero": 0}[defect]
        record["physical_gpu"]["total_mib"] = value; record["inventory"][0]["total_mib"] = value
        observed["physical_gpu"]["total_mib"] = value
    elif defect == "torch_float": observed["logical_total_bytes"] = float(TORCH_BYTES)
    elif defect == "torch_negative": observed["logical_total_bytes"] = -1
    atomic_write_json(bundle / "identity.json", identity, overwrite=True)
    atomic_write_json(bundle / "bootstrap.json", record, overwrite=True)
    digest = reseal_timing_fixture(bundle)
    monkeypatch.setattr(native, "_load_driver", forbidden_driver)
    with pytest.raises(ValueError): timing.validate_timing_bundle(bundle, digest)
    with pytest.raises(ValueError): timing.copy_timing_evidence(bundle, tmp_path / "rejected_copy", digest)
    assert not (tmp_path / "rejected_copy").exists()


def test_direct_identity_mismatch_precedes_research_loading(timing_bundle, tmp_path, monkeypatch):
    import consendxformer.phase5_perturbation_runner as research
    import torch
    bundle, digest, identity, _ = gpu_bundle(timing_bundle, tmp_path, monkeypatch)
    observed = copy.deepcopy(identity["runtime"])
    observed["direct_driver_identity"]["pci_bus_id"] = "0001:3b:00.0"
    def forbidden(*a, **kw): pytest.fail("Research/model access before identity compatibility gate")
    monkeypatch.setattr(research, "validate_evaluator_repository", lambda p: {})
    monkeypatch.setattr(research, "architecture_contract", lambda: (identity["architecture"], {}))
    monkeypatch.setattr(research, "runtime_environment", lambda d: observed)  # Labeled observed-runtime stub.
    monkeypatch.setattr(research, "load_perturbation_inputs", forbidden)
    monkeypatch.setattr(research, "frozen_predictors", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    with pytest.raises(ValueError, match="runtime/device compatibility"):
        research.run_insertion_deletion(full_source_dirs=[], b1_source_dirs=[], attention_source="forbidden",
            attention_manifest_sha256="0"*64, authorize_execution=True, timing_bundle=bundle,
            timing_manifest_sha256=digest, device="cuda")


@pytest.mark.parametrize("defect", ["library", "uuid", "pci"])
def test_live_failure_before_construction_sources_or_output(monkeypatch, tmp_path, defect):
    import consendxformer.phase5_perturbation_runner as research
    _, stub, _, _, _, _ = install_cuda_standins(monkeypatch,
        raw_uuid=b"x"*16 if defect == "uuid" else None,
        pci=b"0001:3b:00.0\0" if defect == "pci" else b"0000:3b:00.0\0")
    stub.set_num_threads = lambda n: None
    monkeypatch.setitem(sys.modules, "torch", stub)
    if defect == "library":
        def missing(): raise RuntimeError("explicit missing driver stub")
        monkeypatch.setattr(native, "_load_driver", missing)
    def forbidden(*a, **kw): pytest.fail("Access after failed live identity")
    monkeypatch.setattr(timing, "validate_evaluator_repository", lambda p: {})
    monkeypatch.setattr(timing, "architecture_contract", forbidden)
    monkeypatch.setattr(timing, "synthetic_predictors", forbidden)
    monkeypatch.setattr(timing, "NAMESPACE", tmp_path / "must_not_exist")
    with pytest.raises((ValueError, RuntimeError)):
        timing.run_synthetic_timing(authorize_synthetic_timing=True, attempt_id="stub", device="cuda")
    assert not timing.NAMESPACE.exists()
    monkeypatch.setattr(research, "validate_evaluator_repository", lambda p: {})
    monkeypatch.setattr(research, "load_perturbation_inputs", forbidden)
    monkeypatch.setattr(research, "frozen_predictors", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        research.run_insertion_deletion(full_source_dirs=[], b1_source_dirs=[], attention_source="forbidden",
            attention_manifest_sha256="0"*64, authorize_execution=True, timing_bundle="never-read",
            timing_manifest_sha256="0"*64, device="cuda")
