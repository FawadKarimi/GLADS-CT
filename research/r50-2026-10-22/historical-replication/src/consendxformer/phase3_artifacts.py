"""Transactional artifact primitives for Phase 3 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

import pandas as pd

from .baseline_provenance import utc_timestamp
from .hashing import sha256_file, sha256_json


_EXPERIMENT_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{7,127}\Z")
_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"prepared"},
    "prepared": {"running", "synthetic_dry_run_completed", "failed"},
    "running": {"finalizing", "failed"},
    "finalizing": {"completed", "failed"},
    "synthetic_dry_run_completed": set(),
    "completed": set(),
    "failed": set(),
}


class Phase3ArtifactError(RuntimeError):
    """Raised when an experiment artifact would be unsafe or inconsistent."""


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _atomic_write(path: str | Path, payload: bytes, *, overwrite: bool) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite Phase 3 artifact: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                # A same-directory hard link is an atomic create-if-absent;
                # unlike a pre-check plus replace, it has no overwrite race.
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite Phase 3 artifact: {destination}"
                ) from exc
            temporary.unlink()
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_json(
    path: str | Path, value: Any, *, overwrite: bool = False
) -> Path:
    return _atomic_write(path, canonical_json_bytes(value), overwrite=overwrite)


def atomic_write_bytes(
    path: str | Path, value: bytes, *, overwrite: bool = False
) -> Path:
    return _atomic_write(path, bytes(value), overwrite=overwrite)


def atomic_write_text(
    path: str | Path, value: str, *, overwrite: bool = False
) -> Path:
    return _atomic_write(path, value.encode("utf-8"), overwrite=overwrite)


def atomic_write_csv(
    path: str | Path, frame: pd.DataFrame, *, overwrite: bool = False
) -> Path:
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, lineterminator="\n")
    return _atomic_write(path, stream.getvalue().encode("utf-8"), overwrite=overwrite)


@dataclass(frozen=True)
class ExperimentLayout:
    output_root: Path
    experiments_root: Path
    experiment_dir: Path
    config_dir: Path
    provenance_dir: Path
    checkpoints_dir: Path
    history_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    status_path: Path

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        experiment_id: str,
        *,
        protected_roots: Iterable[str | Path] = (),
    ) -> "ExperimentLayout":
        if _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None:
            raise ValueError(f"Unsafe Phase 3 experiment ID: {experiment_id!r}")
        root = Path(output_root).resolve()
        protected = tuple(Path(value).resolve() for value in protected_roots)
        for protected_root in protected:
            if root == protected_root or _is_within(root, protected_root):
                raise ValueError(
                    f"Phase 3 output must not be inside protected path {protected_root}"
                )
        experiments = root / "experiments"
        directory = experiments / experiment_id
        if directory.exists():
            marker = directory / "COMPLETED.json"
            dry_marker = directory / "DRY_RUN_COMPLETED.json"
            kind = "completed " if marker.exists() or dry_marker.exists() else "existing "
            raise FileExistsError(
                f"Refusing to overwrite {kind}Phase 3 experiment: {directory}"
            )
        paths = {
            "config_dir": directory / "config",
            "provenance_dir": directory / "provenance",
            "checkpoints_dir": directory / "checkpoints",
            "history_dir": directory / "history",
            "predictions_dir": directory / "predictions",
            "metrics_dir": directory / "metrics",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=False)
        return cls(
            output_root=root,
            experiments_root=experiments,
            experiment_dir=directory,
            status_path=directory / "status.json",
            **paths,
        )

    def relative(self, path: str | Path) -> str:
        target = Path(path).resolve()
        if not _is_within(target, self.experiment_dir):
            raise Phase3ArtifactError(
                f"Artifact is outside experiment directory: {target}"
            )
        return target.relative_to(self.experiment_dir).as_posix()


def transition_status(
    layout: ExperimentLayout,
    state: str,
    *,
    experiment_id: str,
    details: Mapping[str, Any] | None = None,
) -> Path:
    previous: dict[str, Any] | None = None
    current: str | None = None
    if layout.status_path.exists():
        try:
            loaded = json.loads(layout.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Phase3ArtifactError(
                f"Cannot read existing Phase 3 status: {layout.status_path}"
            ) from exc
        if not isinstance(loaded, dict):
            raise Phase3ArtifactError("Existing Phase 3 status is not a JSON object")
        previous = loaded
        current = str(previous.get("state"))
        if previous.get("experiment_id") != experiment_id:
            raise Phase3ArtifactError("Status belongs to a different experiment identity")
    if state not in _TRANSITIONS.get(current, set()):
        raise Phase3ArtifactError(f"Invalid Phase 3 status transition {current!r} -> {state!r}")
    transition = {"state": state, "timestamp_utc": utc_timestamp()}
    history = list(previous.get("transitions", ())) if previous else []
    history.append(transition)
    record: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "state": state,
        "transitions": history,
    }
    if details:
        record["details"] = dict(details)
    return atomic_write_json(layout.status_path, record, overwrite=previous is not None)


def write_artifact_manifest(
    layout: ExperimentLayout,
    artifact_paths: Iterable[str | Path],
    *,
    experiment_id: str,
    research_results_eligible: bool,
) -> Path:
    relative_paths = sorted({layout.relative(path) for path in artifact_paths})
    if not relative_paths:
        raise Phase3ArtifactError("Cannot complete an experiment without artifacts")
    files = {
        relative: sha256_file(layout.experiment_dir / relative)
        for relative in relative_paths
    }
    record = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "files": files,
        "bundle_sha256": sha256_json(files),
        "research_results_eligible": bool(research_results_eligible),
    }
    return atomic_write_json(layout.experiment_dir / "artifact_manifest.json", record)


def write_completion_marker(
    layout: ExperimentLayout,
    artifact_manifest: str | Path,
    *,
    experiment_id: str,
    synthetic_dry_run: bool,
) -> Path:
    manifest = Path(artifact_manifest).resolve()
    if manifest.parent != layout.experiment_dir or not manifest.is_file():
        raise Phase3ArtifactError("Completion requires the experiment artifact manifest")
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase3ArtifactError("Cannot validate the experiment artifact manifest") from exc
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise Phase3ArtifactError("Experiment artifact manifest has an invalid schema")
    if record.get("experiment_id") != experiment_id:
        raise Phase3ArtifactError("Artifact manifest belongs to another experiment")
    expected_eligibility = not synthetic_dry_run
    if record.get("research_results_eligible") is not expected_eligibility:
        raise Phase3ArtifactError(
            "Artifact manifest research eligibility disagrees with completion mode"
        )
    files = record.get("files")
    if not isinstance(files, dict) or not files:
        raise Phase3ArtifactError("Artifact manifest has no checksummed files")
    if sha256_json(files) != record.get("bundle_sha256"):
        raise Phase3ArtifactError("Artifact manifest bundle checksum is inconsistent")
    for relative, expected_digest in files.items():
        target = layout.experiment_dir / str(relative)
        if layout.relative(target) != relative or not target.is_file():
            raise Phase3ArtifactError(
                f"Artifact manifest contains an unsafe or missing member: {relative!r}"
            )
        if sha256_file(target) != expected_digest:
            raise Phase3ArtifactError(
                f"Artifact changed before completion: {relative!r}"
            )

    common_required = {
        "config/experiment_identity.json",
        "config/input_metadata.csv",
        "config/phase1_effective.yaml",
        "config/phase3_effective.yaml",
        "provenance/environment.json",
        "provenance/readiness.json",
        "provenance/run.json",
    }
    real_required = {
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "checkpoints/best.metadata.json",
        "checkpoints/last.metadata.json",
        "history/epoch_history.json",
        "predictions/val.csv",
        "predictions/test.csv",
        "metrics/val.json",
        "metrics/test.json",
    }
    missing = common_required - set(files)
    if not synthetic_dry_run:
        missing |= real_required - set(files)
    if missing:
        raise Phase3ArtifactError(
            f"Completion manifest lacks required artifacts: {sorted(missing)!r}"
        )
    if synthetic_dry_run and any(
        relative.startswith(("checkpoints/", "history/", "predictions/", "metrics/"))
        for relative in files
    ):
        raise Phase3ArtifactError(
            "Synthetic dry-run completion cannot contain result-shaped artifacts"
        )
    name = "DRY_RUN_COMPLETED.json" if synthetic_dry_run else "COMPLETED.json"
    return atomic_write_json(
        layout.experiment_dir / name,
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "artifact_manifest": manifest.name,
            "artifact_manifest_sha256": sha256_file(manifest),
            "synthetic_dry_run": bool(synthetic_dry_run),
            "research_results_eligible": not synthetic_dry_run,
        },
    )
