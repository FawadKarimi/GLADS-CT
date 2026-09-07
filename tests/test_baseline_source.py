from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.baseline_source import (
    _verify_baseline_source_against_trust,
    snapshot_source_tree,
    verify_baseline_source,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _make_tiny_locked_repo(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repo = tmp_path / "baseline"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "phase1-tests@example.invalid")
    _git(repo, "config", "user.name", "Phase 1 Tests")
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-m", "locked source")
    sha = _git(repo, "rev-parse", "HEAD")
    remote = "https://example.invalid/official/baseline.git"
    _git(repo, "remote", "add", "origin", remote)

    lock = tmp_path / "baseline_sources.yaml"
    lock.write_text(
        yaml.safe_dump(
            {
                "baseline_name": "LungDxFormer",
                "code": {"repository": remote, "commit_sha": sha},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return repo, lock, sha, remote


def test_source_guard_accepts_exact_clean_commit(tmp_path):
    repo, lock, sha, remote = _make_tiny_locked_repo(tmp_path)
    report = _verify_baseline_source_against_trust(
        repo, lock, trusted_sha=sha, trusted_repository=remote
    )
    assert report.head_commit == sha
    assert report.working_tree_clean is True
    assert len(report.source_tree_sha256) == 64


def test_source_guard_rejects_wrong_commit(tmp_path):
    repo, lock, locked_sha, remote = _make_tiny_locked_repo(tmp_path)
    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-m", "unlocked change")

    with pytest.raises(RuntimeError, match="HEAD mismatch"):
        _verify_baseline_source_against_trust(
            repo, lock, trusted_sha=locked_sha, trusted_repository=remote
        )


def test_source_guard_rejects_dirty_checkout(tmp_path):
    repo, lock, sha, remote = _make_tiny_locked_repo(tmp_path)
    (repo / "source.py").write_text("VALUE = 99\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="working tree is dirty"):
        _verify_baseline_source_against_trust(
            repo, lock, trusted_sha=sha, trusted_repository=remote
        )


def test_source_guard_rejects_assume_unchanged_content_edit(tmp_path):
    repo, lock, sha, remote = _make_tiny_locked_repo(tmp_path)
    _git(repo, "update-index", "--assume-unchanged", "source.py")
    (repo / "source.py").write_text("VALUE = 9\n", encoding="utf-8")

    # This is the bypass being guarded against: Git reports a clean worktree.
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    with pytest.raises(RuntimeError, match="differs from the pinned commit"):
        _verify_baseline_source_against_trust(
            repo, lock, trusted_sha=sha, trusted_repository=remote
        )


def test_public_source_guard_trust_anchors_are_not_overridable():
    with pytest.raises(TypeError, match="trusted_sha"):
        verify_baseline_source(
            "unused", "unused", trusted_sha="0" * 40  # type: ignore[call-arg]
        )


def test_source_snapshot_includes_ignored_and_untracked_files(tmp_path):
    repo, _, _, _ = _make_tiny_locked_repo(tmp_path)
    before = snapshot_source_tree(repo)
    (repo / "runtime.cache").write_text("new\n", encoding="utf-8")
    after = snapshot_source_tree(repo)
    assert "runtime.cache" not in before
    assert "runtime.cache" in after
