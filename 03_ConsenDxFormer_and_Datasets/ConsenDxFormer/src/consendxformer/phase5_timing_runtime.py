"""Synthetic constructors and runtime identity. No dataset/checkpoint/prediction loading."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
import pandas as pd

from .baseline_source import verify_baseline_source
from .hashing import sha256_json
from .phase4_config import load_phase4_config
from .phase4_attention_runtime import pinned_b1_api, construct_b1
from .phase4_evaluation_runner import PROJECT_ROOT, snapshot_source_files, assert_source_snapshot_unchanged
from .phase5_device import bootstrap_record, verify_runtime_device, verify_launch_blocking_environment
from .phase5_perturbation import KEYS
from .phase5_perturbation_sources import TorchImagePredictor, _model_state_hash

CONFIG_FILES = ("configs/base.yaml", "configs/baseline_common.yaml", "configs/b1.yaml",
                "configs/phase4_consensdxformer.yaml", "locks/baseline_sources.yaml",
                "configs/phase5_insertion_deletion_v1.json", "configs/phase5_synthetic_timing_v1.json",
                "docs/phase5_supplemental_evaluation_proposal.md")


def architecture_contract():
    """Read only code/configuration and the pinned external source, never frozen research inputs."""
    bundle = load_phase4_config(PROJECT_ROOT / "configs/phase4_consensdxformer.yaml")
    baseline = verify_baseline_source(PROJECT_ROOT / "external/LungDxFormer", PROJECT_ROOT / "locks/baseline_sources.yaml")
    files = [*(PROJECT_ROOT / name for name in CONFIG_FILES),
             *sorted((PROJECT_ROOT / "src/consendxformer").glob("*.py")),
             PROJECT_ROOT / "scripts/49_run_phase5_insertion_deletion.py",
             PROJECT_ROOT / "scripts/50_run_phase5_synthetic_timing.py"]
    normalized = {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}
    configs = {"ConsenDxFormer": bundle.phase4_config, "B1": bundle.b1_anchor_config}
    return {"schema": "phase5_timing_architecture_v1", "source_files_sha256_lf": normalized,
            "effective_configs_sha256": {k: sha256_json(v) for k, v in configs.items()},
            "constructors": {"ConsenDxFormer": "phase4_execution._model", "B1": "phase4_attention_runtime.construct_b1"},
            "architecture": {k: {"dataset": {f: v["dataset"][f] for f in ("image_size", "in_channels", "num_classes")},
                                  "model": v["model"]} for k, v in configs.items()},
            "baseline": {k: baseline.to_dict()[k] for k in ("repository_url", "head_commit", "source_tree_sha256")}}, snapshot_source_files(files)


def runtime_environment(device):
    launch_blocking = verify_launch_blocking_environment()
    import torch

    if device not in ("cpu", "cuda"):
        raise ValueError("Explicit CPU/CUDA execution is required")
    bootstrap = bootstrap_record()
    if bootstrap and bootstrap["torch_threads_requested"] is not None:
        torch.set_num_threads(bootstrap["torch_threads_requested"])
    mapping = verify_runtime_device(torch, device)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    result = {"schema": "phase5_timing_runtime_v1", "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "host": platform.node(), "system": platform.system(), "release": platform.release(),
        "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity, "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "environment": {**{k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
            "CUBLAS_WORKSPACE_CONFIG", "CUDA_DEVICE_ORDER", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_MODULE_LOADING")},
            "CUDA_LAUNCH_BLOCKING": launch_blocking},
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "batch_size": 1, "dtype": "float32", "autocast": False, "deterministic_algorithms": True,
        "deterministic_warn_only": False, "cudnn_benchmark": False, "cudnn_deterministic": True, **mapping}
    verify_launch_blocking_environment()
    return result


def host_memory():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in ("peak", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.K32GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        value = Counters(); value.cb = ctypes.sizeof(value)
        if not kernel.K32GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            raise OSError("Cannot measure process memory")
        return {"host_peak_rss_bytes": int(value.peak), "host_current_rss_bytes": int(value.rss), "host_measurement": "Windows process working set"}
    import resource
    scale = 1 if sys.platform == "darwin" else 1024
    return {"host_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*scale,
            "host_current_rss_bytes": None, "host_measurement": "process lifetime resource.ru_maxrss; not a resettable phase peak"}


def memory_snapshot(device):
    import torch
    result = host_memory()
    if device == "cuda":
        free, total = torch.cuda.mem_get_info(0)
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(0), cuda_reserved_bytes=torch.cuda.memory_reserved(0),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                      cuda_free_bytes=free, cuda_total_bytes=total)
    return result


def synchronize(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize(0)


@contextmanager
def synthetic_predictors(device, cache_dir):
    """Construct actual architectures once per seed, synthetic initializations only; zero probes."""
    import torch
    from .phase4_execution import _model

    bundle = load_phase4_config(PROJECT_ROOT / "configs/phase4_consensdxformer.yaml")
    old = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    py_state, np_state = random.getstate(), np.random.get_state()
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    try:
        with torch.random.fork_rng(devices=[0] if device == "cuda" else []), pinned_b1_api(Path(cache_dir)) as (b1_class, baseline):
            models, records = {}, []
            for name, seed in KEYS:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                model = _model(bundle.phase4_config) if name == "ConsenDxFormer" else construct_b1(b1_class, bundle.b1_anchor_config)
                model.to(device).eval().requires_grad_(False)
                models[(name, seed)] = model
                structure = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in model.state_dict().items()}
                records.append({"model": name, "seed": seed, "synthetic_initialization": True,
                    "checkpoint_loaded": False, "state_sha256": _model_state_hash(model), "structure_sha256": sha256_json(structure),
                    "state_bytes": sum(v.numel()*v.element_size() for v in model.state_dict().values()),
                    "parameter_count": sum(p.numel() for p in model.parameters())})
            # All six resident before any warm-up. No constructor probing/model forward.
            if device == "cuda": torch.cuda.reset_peak_memory_stats(0)
            yield {key: TorchImagePredictor(model, device) for key, model in models.items()}, {
                "loaded_models": records, "baseline": baseline, "resident_model_count": 6,
                "research_artifacts_accessed": False, "probe_calls": 0}
            for record in records:
                if _model_state_hash(models[(record["model"], record["seed"])]) != record["state_sha256"]:
                    raise ValueError("Synthetic model parameters/buffers changed")
    finally:
        random.setstate(py_state); np.random.set_state(np_state)
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[2:]
