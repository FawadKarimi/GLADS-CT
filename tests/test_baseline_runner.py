from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer import baseline_runner
from consendxformer.baseline_config import load_baseline_config
from consendxformer.baseline_source import SourceGuardReport, snapshot_source_tree
from consendxformer.hashing import sha256_json
from consendxformer.manifest_guard import validate_frozen_manifest_bundle


CONFIG_PATH = ROOT / "configs" / "b0.yaml"
B1_CONFIG_PATH = ROOT / "configs" / "b1.yaml"
MANIFEST_DIR = ROOT / "artifacts" / "phase0" / "manifests"
SMOKE_METADATA = ROOT / "tests" / "fixtures" / "phase1_smoke_metadata.csv"


@pytest.fixture(scope="module")
def manifest_report():
    return validate_frozen_manifest_bundle(MANIFEST_DIR)


def _make_fake_baseline(
    tmp_path: Path, *, exit_code: int = 0, mutate_source: bool = False
) -> Path:
    """Create only the released config/entrypoint surface used by the wrapper."""
    source_root = tmp_path / "fake-baseline"
    project_root = source_root / "LungDxFormer"
    config_dir = project_root / "configs"
    source_dir = project_root / "src"
    config_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)

    phase1 = load_baseline_config(CONFIG_PATH, expected_mode="B0")
    released = {
        name: deepcopy(phase1[name])
        for name in (
            "seed",
            "device",
            "dataset",
            "augmentation",
            "model",
            "training",
            "evaluation",
        )
    }
    released["paths"] = {
        "checkpoint_dir": "outputs/checkpoints",
        "metrics_dir": "outputs/metrics",
        "plots_dir": "outputs/plots",
        "predictions_dir": "outputs/predictions",
    }
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(released, sort_keys=False), encoding="utf-8"
    )

    child_lines = [
                "import json",
                "import os",
                "from pathlib import Path",
                "import sys",
                "config_index = sys.argv.index('--config') + 1",
                "context = {",
                "    'cwd': str(Path.cwd().resolve()),",
                "    'config': str(Path(sys.argv[config_index]).resolve()),",
                "    'config_exists': Path(sys.argv[config_index]).is_file(),",
                "    'dont_write_bytecode': sys.dont_write_bytecode,",
                "    'pythondontwritebytecode': os.environ.get('PYTHONDONTWRITEBYTECODE'),",
                "    'pythonhashseed': os.environ.get('PYTHONHASHSEED'),",
                "}",
                "Path('child_context.json').write_text(json.dumps(context), encoding='utf-8')",
    ]
    if mutate_source:
        child_lines.append(
            "Path(__file__).with_name('wrapper-mutated-source.txt').write_text('bad', encoding='utf-8')"
        )
    child_lines.extend((f"raise SystemExit({exit_code})", ""))
    (source_dir / "train.py").write_text(
        "\n".join(child_lines),
        encoding="utf-8",
    )
    return source_root


def _install_guards(
    monkeypatch: pytest.MonkeyPatch,
    source_root: Path,
    manifest_report,
    *,
    events: list[str] | None = None,
) -> list[str]:
    observed = [] if events is None else events

    def verify_source(repository_path, lock_file):
        assert Path(repository_path).resolve() == source_root.resolve()
        observed.append("source")
        return SourceGuardReport(
            repository_path=str(source_root.resolve()),
            lock_file=str(Path(lock_file).resolve()),
            repository_url="https://example.invalid/LungDxFormer.git",
            expected_commit="0e5bcadcadfce5ea99df8484d23705d7bd4d778a",
            head_commit="0e5bcadcadfce5ea99df8484d23705d7bd4d778a",
            working_tree_clean=True,
            source_tree_sha256=sha256_json(snapshot_source_tree(source_root)),
        )

    def verify_manifest(manifest_path):
        assert Path(manifest_path).resolve() == MANIFEST_DIR.resolve()
        observed.append("manifest")
        return manifest_report

    monkeypatch.setattr(baseline_runner, "verify_baseline_source", verify_source)
    monkeypatch.setattr(
        baseline_runner, "validate_frozen_manifest_bundle", verify_manifest
    )
    monkeypatch.setattr(
        baseline_runner,
        "project_git_state",
        lambda _root: {
            "commit": "095bd166cd5ac4750197101dbd6a89b4422d6500",
            "branch": "phase1-baselines",
            "working_tree_clean": True,
        },
    )
    return observed


def _run(
    source_root: Path,
    output_dir: Path,
    *,
    dry_run: bool,
    overrides: tuple[str, ...] = (),
    mode: str = "B0",
):
    config_path = CONFIG_PATH if mode == "B0" else B1_CONFIG_PATH
    return baseline_runner.run_baseline(
        mode,
        config_path=config_path,
        metadata_path=SMOKE_METADATA,
        output_dir=output_dir,
        baseline_dir=source_root,
        manifest_dir=MANIFEST_DIR,
        image_root=ROOT / "synthetic-images-that-need-not-exist",
        overrides=overrides,
        dry_run=dry_run,
    )


def _read_provenance(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_dry_run_smoke_is_cpu_safe_and_tolerates_missing_torch(
    tmp_path, monkeypatch, manifest_report
):
    source_root = _make_fake_baseline(tmp_path)
    events = _install_guards(monkeypatch, source_root, manifest_report)
    monkeypatch.setitem(sys.modules, "torch", None)

    run_dir = tmp_path / "dry-run"
    result = _run(
        source_root, run_dir, dry_run=True, overrides=("device=cpu", "seed=73")
    )
    provenance = _read_provenance(result.provenance_path)

    assert result.returncode == 0
    assert result.execution_status == "dry_run_validated"
    assert provenance["device"]["requested"] == "cpu"
    assert provenance["device"]["resolved"] == "cpu"
    assert "probe_error" in provenance["device"]
    assert provenance["baseline_process_returncode"] is None
    assert provenance["training_success_claimed"] is False
    assert not (run_dir / "child_context.json").exists()
    assert events == ["source", "manifest", "source"]


@pytest.mark.parametrize("failing_guard", ["source", "manifest"])
def test_failed_guards_abort_before_launch_or_output_creation(
    tmp_path, monkeypatch, manifest_report, failing_guard
):
    source_root = _make_fake_baseline(tmp_path)
    events: list[str] = []

    def source_guard(*_args, **_kwargs):
        events.append("source")
        if failing_guard == "source":
            raise RuntimeError("source rejected")
        return SourceGuardReport(
            repository_path=str(source_root),
            lock_file="lock.yaml",
            repository_url="https://example.invalid/LungDxFormer.git",
            expected_commit="locked",
            head_commit="locked",
            working_tree_clean=True,
            source_tree_sha256="0" * 64,
        )

    def manifest_guard(*_args, **_kwargs):
        events.append("manifest")
        raise RuntimeError("manifest rejected")

    monkeypatch.setattr(baseline_runner, "verify_baseline_source", source_guard)
    monkeypatch.setattr(
        baseline_runner, "validate_frozen_manifest_bundle", manifest_guard
    )
    run_dir = tmp_path / "must-not-be-created"

    expected = "source rejected" if failing_guard == "source" else "manifest rejected"
    with pytest.raises(RuntimeError, match=expected):
        _run(source_root, run_dir, dry_run=False)

    assert events == (["source"] if failing_guard == "source" else ["source", "manifest"])
    assert not run_dir.exists()
    assert not (source_root / "child_context.json").exists()


def test_runner_writes_external_official_config_and_frozen_split_metadata(
    tmp_path, monkeypatch, manifest_report
):
    source_root = _make_fake_baseline(tmp_path)
    _install_guards(monkeypatch, source_root, manifest_report)
    before = snapshot_source_tree(source_root)
    run_dir = tmp_path / "prepared-run"

    result = _run(source_root, run_dir, dry_run=True)

    official = yaml.safe_load(result.official_config_path.read_text(encoding="utf-8"))
    prepared = pd.read_csv(result.prepared_metadata_path)
    assert result.official_config_path.parent == run_dir
    assert result.prepared_metadata_path.parent == run_dir / "metadata"
    assert official["dataset"]["metadata_csv"] == str(
        result.prepared_metadata_path.resolve()
    )
    assert official["dataset"]["image_root"] == str(
        (ROOT / "synthetic-images-that-need-not-exist").resolve()
    )
    assert all(
        Path(path).resolve().is_relative_to(run_dir.resolve())
        for path in official["paths"].values()
    )
    assert prepared[["patient_id", "split"]].values.tolist() == [
        ["LIDC-IDRI-0002", "train"],
        ["LIDC-IDRI-0009", "val"],
        ["LIDC-IDRI-0001", "test"],
    ]
    for split in ("train", "val", "test"):
        split_frame = pd.read_csv(run_dir / "metadata" / f"{split}_metadata.csv")
        assert split_frame["split"].tolist() == [split]
    assert snapshot_source_tree(source_root) == before


def test_b1_runner_uses_same_source_with_paper_aligned_external_protocol(
    tmp_path, monkeypatch, manifest_report
):
    source_root = _make_fake_baseline(tmp_path)
    events = _install_guards(monkeypatch, source_root, manifest_report)
    before = snapshot_source_tree(source_root)
    run_dir = tmp_path / "b1-prepared-run"

    result = _run(source_root, run_dir, dry_run=True, mode="B1")
    official = yaml.safe_load(result.official_config_path.read_text(encoding="utf-8"))
    provenance = _read_provenance(result.provenance_path)

    assert result.mode == "B1"
    assert provenance["experiment_type"] == "B1"
    assert provenance["protocol"] == "paper-aligned-released-architecture"
    assert official["dataset"]["image_size"] == 64
    assert official["dataset"]["batch_size"] == 32
    assert official["training"]["epochs"] == 100
    assert official["training"]["lr"] == pytest.approx(1e-4)
    assert official["training"]["early_stopping_patience"] == 10
    assert official["model"]["dropout"] == pytest.approx(0.5)
    assert official["model"]["cnn_channels"] == [32, 64, 128]
    assert snapshot_source_tree(source_root) == before
    assert events == ["source", "manifest", "source"]


def test_configuration_hashes_are_deterministic_for_fixed_seed_and_paths(
    tmp_path, monkeypatch, manifest_report
):
    source_root = _make_fake_baseline(tmp_path)
    _install_guards(monkeypatch, source_root, manifest_report)
    run_dir = tmp_path / "repeatable-run"

    first = _run(source_root, run_dir, dry_run=True, overrides=("seed=91",))
    first_record = _read_provenance(first.provenance_path)
    first_effective = yaml.safe_load(
        (run_dir / "effective_phase1_config.yaml").read_text(encoding="utf-8")
    )
    first_official = yaml.safe_load(
        (run_dir / "lungdxformer_config.yaml").read_text(encoding="utf-8")
    )
    assert first_record["configuration"]["effective_sha256"] == sha256_json(
        first_effective
    )
    assert first_record["configuration"]["official_external_sha256"] == sha256_json(
        first_official
    )

    shutil.rmtree(run_dir)
    second = _run(source_root, run_dir, dry_run=True, overrides=("seed=91",))
    second_record = _read_provenance(second.provenance_path)
    assert second_record["configuration"] == first_record["configuration"]


def test_nonzero_child_exit_propagates_and_launch_is_source_safe(
    tmp_path, monkeypatch, manifest_report
):
    source_root = _make_fake_baseline(tmp_path, exit_code=23)
    events = _install_guards(monkeypatch, source_root, manifest_report)
    before = snapshot_source_tree(source_root)
    run_dir = tmp_path / "failed-process"

    result = _run(
        source_root, run_dir, dry_run=False, overrides=("device=cpu", "seed=123")
    )
    provenance = _read_provenance(result.provenance_path)
    child = json.loads((run_dir / "child_context.json").read_text(encoding="utf-8"))

    assert result.returncode == 23
    assert result.execution_status == "process_failed"
    assert provenance["baseline_process_returncode"] == 23
    assert provenance["execution_status"] == "process_failed"
    assert provenance["training_success_claimed"] is False
    assert result.command[0] == sys.executable
    assert result.command[1] == "-B"
    assert child["cwd"] == str(run_dir.resolve())
    assert not Path(child["cwd"]).is_relative_to(source_root.resolve())
    assert child["config"] == str(result.official_config_path.resolve())
    assert child["config_exists"] is True
    assert child["dont_write_bytecode"] is True
    assert child["pythondontwritebytecode"] == "1"
    assert child["pythonhashseed"] == "123"
    assert not list(source_root.rglob("__pycache__"))
    assert snapshot_source_tree(source_root) == before
    assert events == ["source", "manifest", "source"]


def test_child_source_mutation_hard_fails_and_is_recorded(
    tmp_path, monkeypatch, manifest_report
):
    source_root = _make_fake_baseline(tmp_path, mutate_source=True)
    _install_guards(monkeypatch, source_root, manifest_report)
    run_dir = tmp_path / "source-mutating-process"

    with pytest.raises(RuntimeError, match="source integrity failed"):
        _run(source_root, run_dir, dry_run=False)

    provenance = _read_provenance(run_dir / "provenance.json")
    assert provenance["execution_status"] == "source_integrity_failed"
    assert provenance["baseline_unchanged_after_execution"] is False
    assert provenance["training_success_claimed"] is False


def test_cli_returns_the_child_failure_code(
    tmp_path, monkeypatch, manifest_report
):
    source_root = _make_fake_baseline(tmp_path, exit_code=19)
    _install_guards(monkeypatch, source_root, manifest_report)
    run_dir = tmp_path / "cli-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "10_run_b0.py",
            "--config",
            str(CONFIG_PATH),
            "--metadata",
            str(SMOKE_METADATA),
            "--output-dir",
            str(run_dir),
            "--baseline-dir",
            str(source_root),
            "--manifest-dir",
            str(MANIFEST_DIR),
            "--set",
            "device=cpu",
        ],
    )

    assert baseline_runner.baseline_cli("B0", str(CONFIG_PATH)) == 19
    assert _read_provenance(run_dir / "provenance.json")[
        "baseline_process_returncode"
    ] == 19


@pytest.mark.parametrize("existing_kind", ["empty_directory", "nonempty_directory", "file"])
def test_existing_output_paths_are_rejected_without_overwrite_or_guard_work(
    tmp_path, monkeypatch, manifest_report, existing_kind
):
    source_root = _make_fake_baseline(tmp_path)
    run_path = tmp_path / "existing-run"
    if existing_kind == "file":
        run_path.write_text("keep me\n", encoding="utf-8")
    else:
        run_path.mkdir()
        if existing_kind == "nonempty_directory":
            (run_path / "sentinel.txt").write_text("keep me\n", encoding="utf-8")

    def unexpected_guard(*_args, **_kwargs):
        raise AssertionError("freshness must be checked before expensive guards")

    monkeypatch.setattr(baseline_runner, "verify_baseline_source", unexpected_guard)
    monkeypatch.setattr(
        baseline_runner, "validate_frozen_manifest_bundle", unexpected_guard
    )
    before = run_path.read_bytes() if run_path.is_file() else snapshot_source_tree(run_path)

    with pytest.raises(FileExistsError, match="do not overwrite"):
        _run(source_root, run_path, dry_run=True)

    after = run_path.read_bytes() if run_path.is_file() else snapshot_source_tree(run_path)
    assert after == before


@pytest.mark.parametrize("protected_kind", ["baseline", "manifest"])
def test_output_inside_baseline_or_manifest_is_rejected_before_writes(
    tmp_path, monkeypatch, manifest_report, protected_kind
):
    source_root = _make_fake_baseline(tmp_path)

    def unexpected_guard(*_args, **_kwargs):
        raise AssertionError("protected output path must fail before guards")

    monkeypatch.setattr(baseline_runner, "verify_baseline_source", unexpected_guard)
    monkeypatch.setattr(
        baseline_runner, "validate_frozen_manifest_bundle", unexpected_guard
    )
    protected_root = source_root if protected_kind == "baseline" else MANIFEST_DIR
    protected_output = protected_root / "would-write-into-protected-tree"

    with pytest.raises(ValueError, match="protected path"):
        _run(source_root, protected_output, dry_run=True)
    assert not protected_output.exists()


def test_provenance_has_required_identity_and_never_equates_exit_zero_with_training_success(
    tmp_path, monkeypatch, manifest_report
):
    source_root = _make_fake_baseline(tmp_path, exit_code=0)
    _install_guards(monkeypatch, source_root, manifest_report)
    run_dir = tmp_path / "returned-zero"

    result = _run(source_root, run_dir, dry_run=False, overrides=("seed=211",))
    provenance = _read_provenance(result.provenance_path)

    assert result.returncode == 0
    assert result.execution_status == "process_returned"
    assert provenance["schema_version"] == 1
    assert provenance["experiment_type"] == "B0"
    assert provenance["project_git"]["commit"]
    assert provenance["baseline"]["head_commit"] == (
        "0e5bcadcadfce5ea99df8484d23705d7bd4d778a"
    )
    assert len(provenance["configuration"]["effective_sha256"]) == 64
    assert provenance["manifest"]["bundle_sha256"] == manifest_report.bundle_sha256
    assert provenance["manifest_checksum_reference"] == str(
        MANIFEST_DIR / "manifest_checksums.json"
    )
    assert provenance["seed"] == 211
    assert provenance["created_utc"]
    assert provenance["finished_utc"]
    assert provenance["device"]["requested"] == "auto"
    assert provenance["input_metadata_path"] == str(SMOKE_METADATA.resolve())
    assert provenance["output_path"] == str(run_dir.resolve())
    assert provenance["execution_status"] == "process_returned"
    assert provenance["training_success_claimed"] is False
    assert "success" not in provenance["execution_status"]
