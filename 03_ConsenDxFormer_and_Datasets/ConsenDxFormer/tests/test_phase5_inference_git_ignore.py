"""Real temporary Git repository; no mocked status or research execution."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import consendxformer.phase4_config as config
import consendxformer.phase4_evaluation_runner as repository


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    # Isolate host ignore files, hooks, signing and identity. The only commit
    # created by this test is in this disposable synthetic repository.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-b", "phase4-consendxformer")
    hooks = tmp_path / "empty-hooks"
    hooks.mkdir()
    git(root, "config", "core.hooksPath", str(hooks))
    git(root, "config", "core.autocrlf", "false")
    git(root, "config", "user.name", "Synthetic Git Regression")
    git(root, "config", "user.email", "synthetic@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    for name in (".gitignore", "configs/phase4_evaluation.json",
                 "configs/phase4_consensdxformer.yaml", "configs/b1.yaml",
                 "configs/baseline_common.yaml", "configs/base.yaml",
                 "src/consendxformer/phase5_inference_runner.py"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    assert (root / ".gitignore").read_bytes() == (ROOT / ".gitignore").read_bytes()
    git(root, "add", "--", ".gitignore", "configs", "src")
    git(root, "commit", "-m", "Synthetic repository guard fixture")
    # Relocate only repository/ancestry trust anchors. Git status, ancestry,
    # committed config comparisons and the actual config loader all run.
    monkeypatch.setattr(repository, "PROJECT_ROOT", root)
    monkeypatch.setattr(repository, "SOURCE_FINAL_MODEL_COMMIT", git(root, "rev-parse", "HEAD"))
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    assert repository.validate_evaluator_repository(root)["clean"] is True
    return root


def generated_outputs(root):
    directory = root / "artifacts/phase5/paired_correctness/phase5-inference-preflight-synthetic"
    for name in ("evaluation.json", "provenance.json", "tables/hypotheses.csv",
                 "artifact_manifest.json", "PREFLIGHT.json", "FAILED.json"):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic ignore-rule fixture only\n", encoding="utf-8")
    return directory


def test_generated_paired_correctness_does_not_dirty_real_git(checkout):
    before = repository.validate_evaluator_repository(checkout)
    directory = generated_outputs(checkout)
    for path in directory.rglob("*"):
        if path.is_file():
            relative = path.relative_to(checkout).as_posix()
            match = git(checkout, "check-ignore", "-v", "--", relative)
            assert ":artifacts/phase5/paired_correctness/\t" in match
    assert git(checkout, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert repository.validate_evaluator_repository(checkout) == before


@pytest.mark.parametrize("change", ["tracked_source", "unrelated_untracked", "adjacent_namespace"])
def test_actual_clean_guard_still_detects_non_output_changes(checkout, change):
    generated_outputs(checkout)
    if change == "tracked_source":
        relative = "src/consendxformer/phase5_inference_runner.py"
        path = checkout / relative
        path.write_bytes(path.read_bytes() + b"\n# synthetic tracked edit\n")
    else:
        relative = ("unrelated.txt" if change == "unrelated_untracked" else
                    "artifacts/phase5/unrelated_analysis/untracked.txt")
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic unrelated file\n", encoding="utf-8")
    status = git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    assert relative in status
    assert "paired_correctness" not in status
    with pytest.raises(repository.Phase4EvaluationError, match="repository must be clean"):
        repository.validate_evaluator_repository(checkout)
