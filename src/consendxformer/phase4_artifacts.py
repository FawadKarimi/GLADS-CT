"""Transactional, mode-aware artifact lifecycle for Phase 4 single runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pandas as pd

from .baseline_provenance import utc_timestamp
from .hashing import sha256_file, sha256_json
from .phase3_artifacts import atomic_write_json


_EXPERIMENT_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{7,127}\Z")
_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"prepared"},
    "prepared": {"running", "failed"},
    "running": {"finalizing", "failed"},
    "finalizing": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
}


class Phase4ArtifactError(RuntimeError):
    """Raised when Phase 4 artifacts would be unsafe or incomplete."""


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class Phase4ExperimentLayout:
    output_root: Path
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
    ) -> "Phase4ExperimentLayout":
        if _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None:
            raise ValueError(f"Unsafe Phase 4 experiment ID: {experiment_id!r}")
        root = Path(output_root).resolve()
        for protected in (Path(value).resolve() for value in protected_roots):
            if root == protected or _within(root, protected):
                raise ValueError(f"Phase 4 output cannot be inside protected path {protected}")
        directory = root / experiment_id
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite Phase 4 experiment: {directory}")
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
            experiment_dir=directory,
            status_path=directory / "status.json",
            **paths,
        )

    def relative(self, path: str | Path) -> str:
        target = Path(path).resolve()
        if not _within(target, self.experiment_dir):
            raise Phase4ArtifactError(f"Artifact is outside experiment: {target}")
        return target.relative_to(self.experiment_dir).as_posix()


def transition_phase4_status(
    layout: Phase4ExperimentLayout,
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
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Phase4ArtifactError("Cannot read existing Phase 4 status") from exc
        if not isinstance(loaded, dict) or loaded.get("experiment_id") != experiment_id:
            raise Phase4ArtifactError("Phase 4 status identity is inconsistent")
        previous = loaded
        current = str(loaded.get("state"))
    if state not in _TRANSITIONS.get(current, set()):
        raise Phase4ArtifactError(f"Invalid Phase 4 status transition {current!r} -> {state!r}")
    history = list(previous.get("transitions", ())) if previous else []
    history.append({"state": state, "timestamp_utc": utc_timestamp()})
    record: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "state": state,
        "transitions": history,
    }
    if details is not None:
        record["details"] = dict(details)
    return atomic_write_json(layout.status_path, record, overwrite=previous is not None)


def write_phase4_artifact_manifest(
    layout: Phase4ExperimentLayout,
    artifacts: Iterable[str | Path],
    *,
    experiment_id: str,
    run_mode: str,
) -> Path:
    relative = sorted({layout.relative(path) for path in artifacts})
    if not relative:
        raise Phase4ArtifactError("Cannot complete Phase 4 without artifacts")
    files = {name: sha256_file(layout.experiment_dir / name) for name in relative}
    return atomic_write_json(
        layout.experiment_dir / "artifact_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "run_mode": run_mode,
            "files": files,
            "bundle_sha256": sha256_json(files),
            "research_results_eligible": True,
            "test_results_eligible": run_mode == "final",
        },
    )


def write_phase4_completion_marker(
    layout: Phase4ExperimentLayout,
    artifact_manifest: str | Path,
    *,
    experiment_id: str,
    run_mode: str,
) -> Path:
    manifest = Path(artifact_manifest).resolve()
    if manifest.parent != layout.experiment_dir or not manifest.is_file():
        raise Phase4ArtifactError("Completion requires the Phase 4 artifact manifest")
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4ArtifactError("Cannot read the Phase 4 artifact manifest") from exc
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != 1
        or record.get("experiment_id") != experiment_id
        or record.get("run_mode") != run_mode
    ):
        raise Phase4ArtifactError("Phase 4 artifact manifest identity is inconsistent")
    files = record.get("files")
    if not isinstance(files, dict) or sha256_json(files) != record.get("bundle_sha256"):
        raise Phase4ArtifactError("Phase 4 artifact manifest checksum is inconsistent")
    for relative, digest in files.items():
        target = layout.experiment_dir / str(relative)
        if layout.relative(target) != relative or not target.is_file():
            raise Phase4ArtifactError(f"Unsafe or missing Phase 4 artifact: {relative!r}")
        if sha256_file(target) != digest:
            raise Phase4ArtifactError(f"Phase 4 artifact changed before completion: {relative!r}")
    common = {
        "config/phase4_config.json",
        "config/b1_anchor_config.json",
        "config/effective_config.json",
        "config/run_identity.json",
        "config/input_lock.json",
        "config/input_supervision.csv",
        "provenance/readiness_before.json",
        "provenance/readiness_after.json",
        "provenance/phase0_manifest.json",
        "provenance/environment.json",
        "provenance/tuning_protocol.json",
        "provenance/run.json",
        "history/epoch_history.json",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "checkpoints/best.metadata.json",
        "checkpoints/last.metadata.json",
        "predictions/val.csv",
        "metrics/val.json",
    }
    required = set(common)
    if run_mode == "final":
        required |= {"predictions/test.csv", "metrics/test.json"}
    missing = sorted(required - set(files))
    if missing:
        raise Phase4ArtifactError(f"Completion manifest lacks artifacts: {missing}")
    if run_mode == "tuning" and any(
        name.startswith(("predictions/test.", "metrics/test.")) for name in files
    ):
        raise Phase4ArtifactError("Tuning completion cannot contain test artifacts")
    if run_mode == "tuning":
        try:
            supervision = pd.read_csv(
                layout.config_dir / "input_supervision.csv",
                dtype=str,
                keep_default_na=False,
            )
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise Phase4ArtifactError(
                "Cannot validate tuning input supervision snapshot"
            ) from exc
        if "split" not in supervision or bool((supervision["split"] == "test").any()):
            raise Phase4ArtifactError(
                "Tuning completion cannot snapshot test supervision rows"
            )
    marker = layout.experiment_dir / "COMPLETED.json"
    return atomic_write_json(
        marker,
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "run_mode": run_mode,
            "artifact_manifest": manifest.name,
            "artifact_manifest_sha256": sha256_file(manifest),
            "research_results_eligible": True,
            "test_results_eligible": run_mode == "final",
        },
    )


__all__ = [
    "Phase4ArtifactError",
    "Phase4ExperimentLayout",
    "transition_phase4_status",
    "write_phase4_artifact_manifest",
    "write_phase4_completion_marker",
]
