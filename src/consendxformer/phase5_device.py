"""Standard-library-only device gate. CLI entry points call it before importing torch."""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from copy import deepcopy

from .phase5_cuda_identity import (MAPPING_METHOD, query_driver_identity,
                                  validate_driver_identity, pci_components)

_BOOTSTRAP = None
UUID_PATTERN = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"


def gpu_inventory():
    """Read-only NVIDIA management query; never chooses a device."""
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.used,driver_version",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 7:
            raise ValueError("Incomplete NVIDIA device inventory")
        index, uuid, name, bus, total, used, driver = (v.strip() for v in row)
        if re.fullmatch(UUID_PATTERN, uuid) is None:
            raise ValueError("A full physical GPU UUID is required; MIG/abbreviations are not supported")
        record = dict(index=int(index), uuid=uuid, name=name, pci_bus_id=bus,
                      total_mib=int(total), used_mib=int(used), driver_version=driver)
        pci_components(bus)
        if not name or not driver or record["total_mib"] <= 0 or not 0 <= record["used_mib"] <= record["total_mib"]:
            raise ValueError("Invalid NVIDIA device inventory")
        rows.append(record)
    if not rows or len({r["uuid"] for r in rows}) != len(rows):
        raise ValueError("Missing/duplicate physical devices")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    active = []
    for row in csv.reader(processes.stdout.splitlines()):
        if len(row) != 2 or re.fullmatch(UUID_PATTERN, row[0].strip()) is None:
            raise ValueError("Cannot verify active GPU processes")
        active.append({"uuid": row[0].strip(), "pid": int(row[1].strip())})
    return rows, active


def bootstrap_device(device, gpu_uuid=None, *, torch_threads=None):
    """No framework has been imported yet; enforce one exact UUID, never fall back."""
    global _BOOTSTRAP
    if "torch" in sys.modules:
        raise RuntimeError("Device isolation must occur before importing PyTorch; use a fresh CLI process")
    if device not in ("cpu", "cuda"):
        raise ValueError("Explicit cpu or cuda device required")
    if torch_threads is not None and (type(torch_threads) is not int or torch_threads < 1):
        raise ValueError("Positive PyTorch thread count required")
    record = {"device": device, "before_torch_import": True, "torch_threads_requested": torch_threads,
              "checked_unix_seconds": time.time(), "physical_gpu": None, "inventory": [], "active_processes": [],
              "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING")}
    if device == "cuda":
        if not isinstance(gpu_uuid, str) or re.fullmatch(UUID_PATTERN, gpu_uuid) is None:
            raise ValueError("CUDA requires --gpu-uuid with a complete physical UUID")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before process startup")
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, gpu_uuid):
            raise ValueError("Existing CUDA_VISIBLE_DEVICES conflicts with the requested exact UUID")
        rows, processes = gpu_inventory()
        selected = [row for row in rows if row["uuid"] == gpu_uuid]
        if len(selected) != 1:
            raise ValueError("Requested GPU UUID is absent")
        # Operational safety gate, not a scientific threshold or availability guarantee.
        if selected[0]["used_mib"] > 256 or any(p["uuid"] == gpu_uuid for p in processes):
            raise ValueError("Requested physical GPU is not idle (256 MiB ceiling/no active compute processes)")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        record.update(physical_gpu=selected[0], inventory=rows, active_processes=processes)
    else:
        if gpu_uuid is not None:
            raise ValueError("A GPU UUID cannot be supplied for CPU execution")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    record["visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
    record["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    _BOOTSTRAP = deepcopy(record)
    return deepcopy(record)


def bootstrap_cli(argv, authorization_flag):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument(authorization_flag, action="store_true", dest="authorized")
    args, _ = parser.parse_known_args(argv)
    if "--help" in argv or "-h" in argv:
        return bootstrap_device("cpu")
    if not args.authorized:
        raise ValueError(f"Explicit {authorization_flag} required before runtime initialization")
    return bootstrap_device(args.device, args.gpu_uuid, torch_threads=args.torch_threads)


def verify_runtime_device(torch, device):
    """Verify the isolated logical mapping without forward/probe calls; works on torch 2.3."""
    if device == "cpu":
        return {"device": "cpu", "logical_device": "cpu", "physical_gpu": None}
    record = deepcopy(_BOOTSTRAP)
    if (device != "cuda" or not record or record["device"] != "cuda"
            or not record["before_torch_import"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != record["physical_gpu"]["uuid"]
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("Missing/changed pre-import UUID isolation")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one isolated CUDA device must be available; no fallback")
    gpu = record["physical_gpu"]
    direct = query_driver_identity()
    validate_driver_identity(direct, gpu)
    prop = torch.cuda.get_device_properties(0)
    if prop.name != gpu["name"]:
        raise ValueError("Logical device properties disagree with selected physical GPU")
    memory = memory_diagnostics(gpu["total_mib"], prop.total_memory)
    return {"device": "cuda", "logical_device": "cuda:0", "physical_gpu": {
        k: gpu[k] for k in ("uuid", "name", "pci_bus_id", "total_mib", "driver_version")},
        "capability": [prop.major, prop.minor], "logical_total_bytes": prop.total_memory,
        "logical_name": prop.name, "torch_device_count": 1, "direct_driver_identity": direct,
        "memory_diagnostics": memory, "mapping_verification": MAPPING_METHOD}


def memory_diagnostics(nvidia_total_mib, torch_total_bytes):
    if any(type(v) is not int or v <= 0 for v in (nvidia_total_mib, torch_total_bytes)):
        raise ValueError("GPU memory totals must be positive integers")
    return {"schema": "phase5_gpu_memory_diagnostics_v1",
            "nvidia_smi_total_mib": nvidia_total_mib,
            "nvidia_smi_total_bytes": nvidia_total_mib * 1024**2,
            "torch_device_properties_total_bytes": torch_total_bytes,
            "torch_minus_nvidia_smi_bytes": torch_total_bytes - nvidia_total_mib * 1024**2}


def bootstrap_record():
    return deepcopy(_BOOTSTRAP)


def _raw_launch_blocking(record, key):
    if not isinstance(record, dict) or key not in record or (record[key] is not None and not isinstance(record[key], str)):
        raise ValueError("Missing/invalid CUDA_LAUNCH_BLOCKING evidence; unset must be explicit null")
    return record[key]


def verify_launch_blocking_environment():
    """Preserve raw inherited text/None; never set a default or tune the operator's value."""
    record = bootstrap_record()
    captured = _raw_launch_blocking(record, "cuda_launch_blocking")
    if record.get("before_torch_import") is not True or os.environ.get("CUDA_LAUNCH_BLOCKING") != captured:
        raise ValueError("CUDA_LAUNCH_BLOCKING changed since pre-import capture")
    return captured


def validate_device_evidence(record, runtime):
    """Offline semantic check; never initializes CUDA or queries the current machine."""
    if (not isinstance(record, dict) or record.get("before_torch_import") is not True
            or record.get("device") != runtime.get("device")
            or type(record.get("checked_unix_seconds")) not in (int, float)
            or not math.isfinite(record["checked_unix_seconds"]) or record["checked_unix_seconds"] <= 0):
        raise ValueError("Invalid pre-import device evidence")
    threads = record.get("torch_threads_requested")
    if threads is not None and (type(threads) is not int or threads < 1 or threads != runtime["torch_threads"]):
        raise ValueError("Thread configuration differs from bootstrap")
    environment = runtime["environment"]
    if _raw_launch_blocking(record, "cuda_launch_blocking") != _raw_launch_blocking(environment, "CUDA_LAUNCH_BLOCKING"):
        raise ValueError("CUDA_LAUNCH_BLOCKING runtime/bootstrap binding differs")
    if (record["visible_devices"] != environment["CUDA_VISIBLE_DEVICES"]
            or record["cublas_workspace_config"] != environment["CUBLAS_WORKSPACE_CONFIG"]):
        raise ValueError("Device environment differs from pre-import evidence")
    if runtime["device"] == "cpu":
        if (runtime["logical_device"] != "cpu" or runtime["physical_gpu"] is not None
                or record["physical_gpu"] is not None or record["inventory"] or record["active_processes"]
                or record["visible_devices"] != ""):
            raise ValueError("Invalid CPU isolation evidence")
        return
    gpu = record["physical_gpu"]
    if (runtime["device"] != "cuda" or not isinstance(gpu, dict)
            or re.fullmatch(UUID_PATTERN, str(gpu.get("uuid"))) is None
            or runtime["logical_device"] != "cuda:0" or record["visible_devices"] != gpu["uuid"]
            or environment["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8" or environment["CUDA_DEVICE_ORDER"] != "PCI_BUS_ID"
            or runtime["physical_gpu"] != {k: gpu[k] for k in ("uuid", "name", "pci_bus_id", "total_mib", "driver_version")}
            or type(runtime["physical_gpu"].get("total_mib")) is not int
            or runtime.get("mapping_verification") != MAPPING_METHOD
            or runtime.get("logical_name") != gpu["name"]
            or type(runtime.get("torch_device_count")) is not int or runtime["torch_device_count"] != 1):
        raise ValueError("Invalid physical/logical GPU binding")
    validate_driver_identity(runtime.get("direct_driver_identity"), gpu)
    expected_memory = memory_diagnostics(gpu.get("total_mib"), runtime.get("logical_total_bytes"))
    memory = runtime.get("memory_diagnostics")
    if (not isinstance(memory, dict) or memory != expected_memory
            or any(type(memory.get(k)) is not int for k in expected_memory if k != "schema")):
        raise ValueError("GPU memory diagnostics differ from raw source totals/delta")
    inventory = record["inventory"]
    if (len({g["uuid"] for g in inventory}) != len(inventory)
            or any(type(g.get("total_mib")) is not int or g["total_mib"] <= 0 for g in inventory)
            or [g for g in inventory if g["uuid"] == gpu["uuid"]] != [gpu]
            or type(gpu["index"]) is not int or gpu["index"] < 0
            or type(gpu["used_mib"]) is not int or not 0 <= gpu["used_mib"] <= min(256, gpu["total_mib"])
            or any(p["uuid"] == gpu["uuid"] for p in record["active_processes"])):
        raise ValueError("GPU availability evidence differs from the startup gate")
