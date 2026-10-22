"""Safe deterministic output primitives for Phase 2 generated data."""

from __future__ import annotations

import json
import io
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .hashing import sha256_file, sha256_json


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def write_json_new(path: str | Path, value: Any) -> Path:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite JSON output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(value))
    return destination


def write_csv_new(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite CSV output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8", lineterminator="\n")
    return destination


def write_json_or_verify(path: str | Path, value: Any) -> Path:
    destination = Path(path).resolve()
    payload = canonical_json_bytes(value)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"Existing JSON differs during safe resume: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def write_provenance_or_verify(path: str | Path, value: dict[str, Any]) -> Path:
    """Preserve the initial run timestamp while verifying every other field on resume."""
    destination = Path(path).resolve()
    candidate = dict(value)
    if destination.exists():
        try:
            previous = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot validate existing provenance: {destination}") from exc
        if not isinstance(previous, dict) or "timestamp_utc" not in previous:
            raise RuntimeError(f"Existing provenance lacks timestamp_utc: {destination}")
        candidate["timestamp_utc"] = previous["timestamp_utc"]
    return write_json_or_verify(destination, candidate)


def write_csv_or_verify(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = Path(path).resolve()
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n")
    payload = stream.getvalue().encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"Existing CSV differs during safe resume: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def require_safe_phase2_output(path: str | Path, project_root: str | Path) -> Path:
    output = Path(path).resolve()
    root = Path(project_root).resolve()
    protected = (
        root / "artifacts" / "phase0" / "manifests",
        root / "external" / "LungDxFormer",
    )
    for protected_root in protected:
        if output == protected_root or _is_within(output, protected_root):
            raise ValueError(f"Phase 2 output must not be inside protected path {protected_root}")
    return output


def prepare_resumable_run(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    configuration_sha256: str,
    allow_resume: bool,
) -> tuple[Path, bool]:
    directory = require_safe_phase2_output(output_dir, project_root)
    config_path = directory / "run_configuration.json"
    completed_path = directory / "COMPLETED.json"
    if directory.exists():
        if completed_path.exists():
            raise FileExistsError(f"Completed Phase 2 output cannot be overwritten: {directory}")
        if not allow_resume:
            raise FileExistsError(f"Phase 2 output exists; pass explicit resume only if intended: {directory}")
        if not config_path.is_file():
            raise RuntimeError("Cannot resume output without run_configuration.json")
        try:
            previous = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Cannot validate existing run configuration") from exc
        if previous.get("configuration_sha256") != configuration_sha256:
            raise RuntimeError(
                "Refusing to resume Phase 2 output with a different configuration hash"
            )
        return directory, True
    directory.mkdir(parents=True, exist_ok=False)
    write_json_new(config_path, {"configuration_sha256": configuration_sha256})
    return directory, False


def deterministic_inventory_digest(
    root: str | Path, relative_files: Iterable[str | Path]
) -> tuple[dict[str, str], str]:
    directory = Path(root).resolve()
    checksums: dict[str, str] = {}
    for relative in sorted(Path(value).as_posix() for value in relative_files):
        target = (directory / relative).resolve()
        if not _is_within(target, directory) or not target.is_file():
            raise FileNotFoundError(f"Inventory member is missing or unsafe: {relative}")
        checksums[relative] = sha256_file(target)
    return checksums, sha256_json(checksums)


def write_checksum_ledger(
    output_dir: str | Path, relative_files: Iterable[str | Path]
) -> Path:
    directory = Path(output_dir).resolve()
    checksums, bundle = deterministic_inventory_digest(directory, relative_files)
    return write_json_or_verify(
        directory / "checksums.json",
        {
            "schema_version": 1,
            "files": checksums,
            "bundle_sha256": bundle,
            "hash_scope": "listed generated files; raw DICOM bytes are not hashed",
        },
    )


def mark_completed(output_dir: str | Path, checksum_ledger: str | Path) -> Path:
    directory = Path(output_dir).resolve()
    ledger = Path(checksum_ledger).resolve()
    return write_json_new(
        directory / "COMPLETED.json",
        {
            "schema_version": 1,
            "checksum_ledger": ledger.name,
            "checksum_ledger_sha256": sha256_file(ledger),
        },
    )
