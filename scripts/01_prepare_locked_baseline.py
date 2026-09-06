#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.config import load_config


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p.stdout.strip()


def normalize_remote(url: str) -> str:
    return url.rstrip("/").removesuffix(".git")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0.yaml")
    ap.add_argument("--repo-path", default=None)
    ap.add_argument("--clone-if-missing", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    lock_path = ROOT / cfg["source_lock_file"]
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    expected_repo = lock["code"]["repository"]
    expected_sha = lock["code"]["commit_sha"]
    repo_path = Path(args.repo_path) if args.repo_path else ROOT / cfg["paths"]["baseline_repo_dir"]
    repo_path = repo_path.resolve()

    if not repo_path.exists():
        if not args.clone_if_missing:
            raise FileNotFoundError(
                f"Baseline repo not found at {repo_path}. Rerun with --clone-if-missing "
                "or clone the official repository manually."
            )
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", expected_repo, str(repo_path)])

    if not (repo_path / ".git").exists():
        raise RuntimeError(f"{repo_path} is not a Git repository")

    origin = run(["git", "remote", "get-url", "origin"], cwd=repo_path)
    if normalize_remote(origin) != normalize_remote(expected_repo):
        raise RuntimeError(f"Unexpected origin: {origin}; expected {expected_repo}")

    # Fetch the exact immutable object, then detach at it. We do not modify source files.
    run(["git", "fetch", "origin", expected_sha], cwd=repo_path)
    run(["git", "checkout", "--detach", expected_sha], cwd=repo_path)

    head = run(["git", "rev-parse", "HEAD"], cwd=repo_path)
    dirty = run(["git", "status", "--porcelain"], cwd=repo_path)
    if head != expected_sha:
        raise AssertionError(f"HEAD mismatch: {head} != {expected_sha}")
    if dirty:
        raise RuntimeError("Locked baseline working tree is dirty. Do not use it as B0 until clean.")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_repo_path": str(repo_path),
        "origin": origin,
        "head": head,
        "expected_head": expected_sha,
        "working_tree_clean": True,
        "paper_url": lock["paper"]["url"],
        "zenodo_doi": lock["archive"]["doi"],
        "zenodo_verification_status": lock["archive"]["verification_status"],
        "status": "PASS",
    }

    out_dir = ROOT / cfg["paths"]["phase0_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "source_lock_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("PHASE0_BASELINE_LOCK=PASS")
    print(f"baseline_head={head}")
    print(f"working_tree_clean=true")
    print(f"zenodo_doi={lock['archive']['doi']}")
    print(f"report={out}")


if __name__ == "__main__":
    main()
