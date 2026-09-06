#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.config import load_config
from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.reproducibility import set_global_seed


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/phase0.yaml")
    args = p.parse_args()

    cfg_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    source_lock_path = (ROOT / cfg["source_lock_file"]).resolve()

    seed_report = set_global_seed(
        int(cfg["reproducibility"]["seed"]),
        deterministic_algorithms=bool(cfg["reproducibility"].get("deterministic_algorithms", True)),
        cudnn_benchmark=bool(cfg["reproducibility"].get("cudnn_benchmark", False)),
    )

    gpu = {"available": False, "devices": []}
    try:
        import torch
        gpu["available"] = bool(torch.cuda.is_available())
        gpu["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        gpu["cudnn_version"] = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except Exception as exc:
        gpu["torch_probe_error"] = repr(exc)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 0,
        "project_root": str(ROOT),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "PyYAML": package_version("PyYAML"),
            "torch": package_version("torch"),
        },
        "gpu": gpu,
        "seed_control": seed_report,
        "project_git": {
            "head": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "status_porcelain": git_value(["status", "--porcelain"]),
        },
        "config_file": str(cfg_path),
        "config_sha256": sha256_json(cfg),
        "source_lock_file": str(source_lock_path),
        "source_lock_sha256": sha256_file(source_lock_path),
        "reproducibility_caveat": (
            "Seed control and deterministic-algorithm requests reduce nondeterminism but do not guarantee "
            "bitwise-identical results across different hardware, drivers, CUDA/cuDNN, or PyTorch versions."
        ),
    }

    out_dir = ROOT / cfg["paths"]["phase0_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "environment_lock.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("PHASE0_ENV_LOCK=PASS")
    print(f"environment_lock={out_path}")
    print(f"config_sha256={report['config_sha256']}")
    print(f"source_lock_sha256={report['source_lock_sha256']}")
    print(f"cuda_available={gpu['available']}")
    if gpu["devices"]:
        for d in gpu["devices"]:
            print(f"gpu[{d['index']}]={d['name']} cc={d['compute_capability']}")


if __name__ == "__main__":
    main()
