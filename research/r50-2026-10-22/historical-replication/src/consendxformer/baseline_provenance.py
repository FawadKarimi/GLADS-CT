from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import subprocess


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_git_state(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()

    def git(*args: str) -> str | None:
        process = subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=False
        )
        return process.stdout.strip() if process.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "working_tree_clean": status == "" if status is not None else None,
    }


def resolve_device(requested: str) -> dict[str, Any]:
    report: dict[str, Any] = {"requested": requested, "resolved": "cpu"}
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        report["torch_version"] = str(torch.__version__)
        report["cuda_available"] = cuda_available
        # This intentionally mirrors the pinned source's actual selection rule.
        report["resolved"] = "cuda" if requested == "auto" and cuda_available else "cpu"
    except Exception as exc:  # infrastructure/dry runs do not require PyTorch
        report["probe_error"] = repr(exc)
    return report


def write_provenance(path: str | Path, record: dict[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
