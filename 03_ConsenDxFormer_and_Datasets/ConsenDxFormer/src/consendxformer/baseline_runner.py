from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import argparse
import os
import subprocess
import sys

import yaml

from .baseline_config import (
    apply_safe_overrides,
    load_baseline_config,
    project_official_config,
)
from .baseline_metadata import adapt_metadata, write_adapted_metadata
from .baseline_provenance import (
    project_git_state,
    resolve_device,
    utc_timestamp,
    write_provenance,
)
from .baseline_source import (
    assert_source_snapshot_unchanged,
    snapshot_source_tree,
    verify_baseline_source,
)
from .hashing import sha256_file, sha256_json
from .manifest_guard import validate_frozen_manifest_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BaselineRunResult:
    mode: str
    returncode: int
    execution_status: str
    output_dir: Path
    provenance_path: Path
    official_config_path: Path
    prepared_metadata_path: Path
    command: tuple[str, ...]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_fresh_output(output_dir: Path, protected: Iterable[Path]) -> None:
    for protected_path in protected:
        if output_dir == protected_path or _is_within(output_dir, protected_path):
            raise ValueError(f"Output directory must not be inside protected path {protected_path}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output path already exists: {output_dir}. "
            "Use a fresh experiment directory; Phase 1 runners do not overwrite runs."
        )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def run_baseline(
    mode: str,
    *,
    config_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    baseline_dir: str | Path | None = None,
    manifest_dir: str | Path | None = None,
    image_root: str | Path | None = None,
    overrides: Iterable[str] = (),
    dry_run: bool = False,
) -> BaselineRunResult:
    """Prepare and optionally launch B0/B1 against the immutable official source."""
    expected_mode = mode.upper()
    config_file = _project_path(config_path)
    config = load_baseline_config(config_file, expected_mode=expected_mode)
    config = apply_safe_overrides(config, overrides)

    source_root = (
        Path(baseline_dir).resolve()
        if baseline_dir is not None
        else _project_path(config["baseline"]["repository_dir"])
    )
    frozen_root = (
        Path(manifest_dir).resolve()
        if manifest_dir is not None
        else _project_path(config["frozen_manifest"]["directory"])
    )
    lock_file = _project_path(config["baseline"]["source_lock_file"])
    input_metadata = Path(metadata_path).resolve()
    if not input_metadata.is_file():
        raise FileNotFoundError(f"Input metadata CSV does not exist: {input_metadata}")
    images = Path(image_root).resolve() if image_root is not None else input_metadata.parent
    run_dir = Path(output_dir).resolve()
    _require_fresh_output(run_dir, (source_root, frozen_root))

    # Guards deliberately run before creating a run directory or translating data.
    source_report = verify_baseline_source(source_root, lock_file)
    manifest_report = validate_frozen_manifest_bundle(frozen_root)
    before_source = snapshot_source_tree(source_root)

    canonical_metadata = adapt_metadata(
        input_metadata,
        manifest_report,
        columns=config["metadata"]["columns"],
        unknown_patient_policy=config["metadata"]["unknown_patient_policy"],
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    metadata_bundle = write_adapted_metadata(canonical_metadata, run_dir / "metadata")
    effective_config_path = run_dir / "effective_phase1_config.yaml"
    _write_yaml(effective_config_path, config)

    released_config_path = source_root / config["baseline"]["released_config"]
    training_entrypoint = source_root / config["baseline"]["training_entrypoint"]
    if not released_config_path.is_file():
        raise FileNotFoundError(f"Pinned released config is missing: {released_config_path}")
    if not training_entrypoint.is_file():
        raise FileNotFoundError(f"Pinned training entrypoint is missing: {training_entrypoint}")
    released_config = _load_yaml_mapping(released_config_path)
    official_config = project_official_config(
        config,
        released_config,
        metadata_csv=metadata_bundle.combined_csv,
        image_root=images,
        output_dir=run_dir,
    )
    official_config_path = run_dir / "lungdxformer_config.yaml"
    _write_yaml(official_config_path, official_config)

    for configured_output in official_config["paths"].values():
        Path(configured_output).mkdir(parents=True, exist_ok=True)
    # The released logger uses this hard-coded path relative to subprocess CWD.
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)

    command = (
        sys.executable,
        "-B",
        str(training_entrypoint.resolve()),
        "--config",
        str(official_config_path.resolve()),
    )
    provenance_path = run_dir / "provenance.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": utc_timestamp(),
        "experiment_type": expected_mode,
        "experiment_name": config["experiment"]["name"],
        "protocol": config["experiment"]["protocol"],
        "execution_status": "prepared",
        "baseline_process_returncode": None,
        "training_success_claimed": False,
        "project_git": project_git_state(PROJECT_ROOT),
        "baseline": source_report.to_dict(),
        "released_config_sha256": sha256_file(released_config_path),
        "configuration": {
            "input_path": str(config_file),
            "effective_path": str(effective_config_path),
            "effective_sha256": sha256_json(config),
            "official_external_path": str(official_config_path),
            "official_external_sha256": sha256_json(official_config),
        },
        "manifest": manifest_report.to_dict(),
        "manifest_checksum_reference": str(frozen_root / "manifest_checksums.json"),
        "seed": config["seed"],
        "device": resolve_device(config["device"]),
        "input_metadata_path": str(input_metadata),
        "image_root": str(images),
        "prepared_metadata": metadata_bundle.to_dict(),
        "output_path": str(run_dir),
        "command": list(command),
        "dry_run": bool(dry_run),
    }

    returncode = 0
    launch_error: Exception | None = None
    source_error: Exception | None = None
    if dry_run:
        record["execution_status"] = "dry_run_validated"
    else:
        stdout_path = run_dir / "baseline_stdout.log"
        stderr_path = run_dir / "baseline_stderr.log"
        child_env = os.environ.copy()
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        child_env["PYTHONHASHSEED"] = str(config["seed"])
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.run(
                    list(command),
                    cwd=run_dir,
                    env=child_env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                )
            returncode = int(process.returncode)
            record["baseline_process_returncode"] = returncode
            record["execution_status"] = (
                "process_returned" if returncode == 0 else "process_failed"
            )
        except Exception as exc:  # keep provenance for launch failures
            returncode = 1
            launch_error = exc
            record["execution_status"] = "process_launch_failed"
            record["launch_error"] = repr(exc)

    try:
        after_source = snapshot_source_tree(source_root)
        assert_source_snapshot_unchanged(before_source, after_source)
        verify_baseline_source(source_root, lock_file)
        record["baseline_unchanged_after_execution"] = True
    except Exception as exc:
        source_error = exc
        returncode = returncode or 1
        record["baseline_unchanged_after_execution"] = False
        record["source_guard_error"] = repr(exc)
        record["execution_status"] = "source_integrity_failed"

    record["finished_utc"] = utc_timestamp()
    write_provenance(provenance_path, record)
    if source_error is not None:
        raise RuntimeError("Baseline source integrity failed after execution") from source_error
    if launch_error is not None:
        raise RuntimeError("Could not launch the official baseline process") from launch_error

    return BaselineRunResult(
        mode=expected_mode,
        returncode=returncode,
        execution_status=str(record["execution_status"]),
        output_dir=run_dir,
        provenance_path=provenance_path,
        official_config_path=official_config_path,
        prepared_metadata_path=Path(metadata_bundle.combined_csv),
        command=command,
    )


def baseline_cli(expected_mode: str, default_config: str) -> int:
    parser = argparse.ArgumentParser(
        description=f"Run the locked LungDxFormer {expected_mode.upper()} baseline wrapper."
    )
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--metadata", required=True, help="Future Phase 2 or synthetic nodule metadata CSV")
    parser.add_argument("--output-dir", required=True, help="Fresh external experiment directory")
    parser.add_argument("--image-root", default=None, help="Root for relative image_path values (default: metadata directory)")
    parser.add_argument("--baseline-dir", default=None, help="Override location of the same locked checkout")
    parser.add_argument("--manifest-dir", default=None, help="Override location of a byte-identical frozen bundle")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Runtime-safe override; repeat for seed/device/workers/metadata column mapping",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate guards and metadata, emit config/provenance, but do not launch training",
    )
    args = parser.parse_args()
    result = run_baseline(
        expected_mode,
        config_path=args.config,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        baseline_dir=args.baseline_dir,
        manifest_dir=args.manifest_dir,
        image_root=args.image_root,
        overrides=args.overrides,
        dry_run=args.dry_run,
    )
    print(f"PHASE1_{expected_mode.upper()}={result.execution_status}")
    print(f"provenance={result.provenance_path}")
    print(f"official_config={result.official_config_path}")
    print(f"prepared_metadata={result.prepared_metadata_path}")
    return result.returncode
