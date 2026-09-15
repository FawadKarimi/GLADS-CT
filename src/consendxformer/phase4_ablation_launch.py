"""Explicit GPU-UUID job assignments; no device discovery, scheduling or auto-retry."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys

from .hashing import sha256_json
from .phase3_artifacts import atomic_write_json
from .phase4_ablation_contracts import AblationRunSpec, NAMESPACE, execution_protocol

GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
REFERENCE_NAMES = {
    42: "phase4-final-s42-c1e650d77585ed9c",
    43: "phase4-final-s43-946220804796df93",
    44: "phase4-final-s44-8c74c3dd1e666bb5",
}


def validate_gpu(value):
    if not isinstance(value, str) or GPU_UUID.fullmatch(value) is None:
        raise ValueError("Assign one full physical GPU UUID (GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx); no implicit GPU 0")
    return value


@dataclass(frozen=True)
class AssignedJob:
    spec: AblationRunSpec
    gpu: str
    full_source: Path


def validate_jobs(jobs):
    if not jobs:
        raise ValueError("Explicit jobs are required")
    keys, gpus = set(), set()
    for job in jobs:
        job.spec.specification()
        validate_gpu(job.gpu)
        key = (job.spec.variant, job.spec.seed)
        if key in keys or job.gpu.lower() in gpus:
            raise ValueError("Duplicate run or GPU assignment; one independent job per explicit GPU")
        keys.add(key); gpus.add(job.gpu.lower())
    return tuple(jobs)


def job_command(job, script):
    return [sys.executable, str(script), "run", "--variant", job.spec.variant, "--seed", str(job.spec.seed),
            "--full-source", str(job.full_source), "--gpu", job.gpu, "--authorize-training"]


def launch_jobs(jobs, *, authorize_training=False):
    if authorize_training is not True:
        raise ValueError("Launching training needs separate explicit runtime authorization")
    jobs = validate_jobs(jobs)
    from .phase4_ablation_runner import ROOT, prepare_ablation, recheck_prepared, guard_new_namespace
    # Preflight the entire requested group before creating logs or spawning any job.
    prepared = [prepare_ablation(job.spec, job.full_source) for job in jobs]
    for item in prepared:
        recheck_prepared(item)
    record = [{"identity": p.identity, "gpu": job.gpu, "output_dir": str(p.output_dir)} for job, p in zip(jobs, prepared)]
    directory = ROOT / NAMESPACE / "launches" / f"group-{sha256_json(record)[:16]}"
    guard_new_namespace(directory)
    directory.mkdir(parents=True, exist_ok=False)
    atomic_write_json(directory / "plan.json", {"jobs": record, "runtime_training_explicitly_authorized": True,
                                               "protocol": execution_protocol(), "automatic_retry": False})
    processes, results = [], []
    interrupted = False
    try:
        for job, item in zip(jobs, prepared):
            recheck_prepared(item)
            name = f"{job.spec.variant}-s{job.spec.seed}"
            log = (directory / f"{name}.log").open("xb")
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": job.gpu, "PYTHONDONTWRITEBYTECODE": "1",
                           "PYTHONHASHSEED": str(job.spec.seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
            try:
                process = subprocess.Popen(job_command(job, ROOT / "scripts/45_run_phase4_ablations.py"), cwd=ROOT,
                    env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except BaseException:
                log.close()
                results.append({"job": name, "gpu": job.gpu, "exit_code": None, "status": "spawn_failed"})
                raise
            processes.append((name, job.gpu, process, log))
        for name, gpu, process, log in processes:
            code = process.wait()
            results.append({"job": name, "gpu": gpu, "exit_code": code, "status": "exited"})
            atomic_write_json(directory / f"{name}.exit.json", results[-1])
    except BaseException:
        interrupted = True
        for _, _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for name, gpu, process, log in processes:
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            log.close()
            if not any(row["job"] == name for row in results):
                row = {"job": name, "gpu": gpu, "exit_code": code, "status": "interrupted"}
                results.append(row)
                atomic_write_json(directory / f"{name}.exit.json", row)
        atomic_write_json(directory / "exit_codes.json", {"jobs": results, "interrupted": interrupted,
            "all_succeeded": len(results) == len(jobs) and all(row["exit_code"] == 0 for row in results) and not interrupted})
    return (0 if all(row["exit_code"] == 0 for row in results) else 1), directory


def ablation_cli(argv=None):
    import argparse
    import json
    from .phase4_ablations import ABLATION_NAMES, FINAL_SEEDS
    parser = argparse.ArgumentParser(description="Guarded component ablations; planning never trains")
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("plan", help="Read-only 12-run specification; no real inputs or GPUs needed")
    run = modes.add_parser("run")
    run.add_argument("--variant", choices=ABLATION_NAMES, required=True)
    run.add_argument("--seed", type=int, choices=FINAL_SEEDS, required=True)
    run.add_argument("--full-source")
    run.add_argument("--gpu", required=True, help="One explicitly assigned physical GPU UUID")
    run.add_argument("--authorize-training", action="store_true")
    launch = modes.add_parser("launch")
    launch.add_argument("--job", action="append", required=True, help="VARIANT:SEED:GPU-UUID; explicitly assign every job")
    launch.add_argument("--full-source-root", default="artifacts/phase4/experiments")
    launch.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "plan":
        order = ("attention_alignment", *[name for name in ABLATION_NAMES if name != "attention_alignment"])
        print(json.dumps({"read_only": True, "training_execution_authorized": False, "protocol": execution_protocol(),
                          "runs": [AblationRunSpec(name, seed).to_dict() for name in order for seed in FINAL_SEEDS]}, indent=2))
        return 0
    if not args.authorize_training:
        parser.error("explicit --authorize-training is required; no execution performed")
    if args.mode == "run":
        validate_gpu(args.gpu)
        # Set visibility before importing any Torch-dependent runner modules.
        if "torch" in sys.modules:
            raise ValueError("GPU run must start in a fresh process before Torch import")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        from .phase4_ablation_runner import ROOT, run_ablation
        source = args.full_source or ROOT / "artifacts/phase4/experiments" / REFERENCE_NAMES[args.seed]
        result = run_ablation(AblationRunSpec(args.variant, args.seed), source,
                              authorize_training=True, device="cuda", gpu_assignment=args.gpu)
        print(f"OUTPUT_DIR={result}")
        return 0
    jobs = []
    for assignment in args.job:
        parts = assignment.split(":")
        if len(parts) != 3:
            parser.error("each --job must be VARIANT:SEED:GPU-UUID")
        spec = AblationRunSpec(parts[0], int(parts[1])); spec.specification()
        jobs.append(AssignedJob(spec, parts[2], Path(args.full_source_root) / REFERENCE_NAMES[spec.seed]))
    code, directory = launch_jobs(jobs, authorize_training=True)
    print(f"LAUNCH_LOG_DIR={directory}")
    return code
