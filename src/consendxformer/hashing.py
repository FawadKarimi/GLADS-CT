from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def patient_rank_hash(patient_id: str, seed: int, algorithm_version: str = "sha256_rank_v1") -> str:
    payload = f"{algorithm_version}|{seed}|{patient_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
