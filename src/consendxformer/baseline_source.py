from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any
import subprocess

import yaml

from .hashing import sha256_file, sha256_json

PINNED_BASELINE_SHA = "0e5bcadcadfce5ea99df8484d23705d7bd4d778a"
OFFICIAL_BASELINE_REPOSITORY = "https://github.com/Srinivasa459/LungDxFormer.git"


@dataclass(frozen=True)
class SourceGuardReport:
    repository_path: str
    lock_file: str
    repository_url: str
    expected_commit: str
    head_commit: str
    working_tree_clean: bool
    source_tree_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"Git command failed in {repo}: git {' '.join(args)}\n{detail}")
    return process.stdout.strip()


def _git_bytes(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    process = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=stdin,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git command failed in {repo}: git {' '.join(args)}\n{detail}")
    return process.stdout


def _normalize_remote(value: str) -> str:
    return value.strip().replace("\\", "/").rstrip("/").removesuffix(".git").lower()


def load_baseline_lock(lock_file: str | Path) -> dict[str, Any]:
    path = Path(lock_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Baseline source lock does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("code"), dict):
        raise ValueError(f"Malformed baseline source lock: {path}")
    if not data["code"].get("commit_sha") or not data["code"].get("repository"):
        raise ValueError(f"Baseline source lock lacks code.commit_sha/repository: {path}")
    return data


def snapshot_source_tree(repository_path: str | Path) -> dict[str, str]:
    """Hash every baseline checkout file except Git's own administrative data."""
    repo = Path(repository_path).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"Baseline directory does not exist: {repo}")
    snapshot: dict[str, str] = {}
    for path in sorted(repo.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(repo)
        if relative.parts and relative.parts[0] == ".git":
            continue
        snapshot[relative.as_posix()] = sha256_file(path)
    return snapshot


def assert_source_snapshot_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    if before == after:
        return
    before_names = set(before)
    after_names = set(after)
    added = sorted(after_names - before_names)
    removed = sorted(before_names - after_names)
    changed = sorted(name for name in before_names & after_names if before[name] != after[name])
    raise RuntimeError(
        "The baseline checkout changed during wrapper execution; external/LungDxFormer must remain immutable. "
        f"added={added[:10]}, removed={removed[:10]}, changed={changed[:10]}"
    )


def _pinned_tree_blobs(repo: Path, revision: str) -> list[tuple[str, str, bytes]]:
    """Return ``(mode, object id, raw path)`` entries from a pinned Git tree."""
    payload = _git_bytes(repo, "ls-tree", "-r", "-z", "--full-tree", revision)
    entries: list[tuple[str, str, bytes]] = []
    for record in payload.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = header.decode("ascii").split()
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("Could not parse the pinned baseline Git tree") from exc
        if object_type != "blob":
            path = os.fsdecode(raw_path)
            raise RuntimeError(
                f"Unsupported tracked object in pinned baseline tree: "
                f"path={path!r}, mode={mode}, type={object_type}"
            )
        entries.append((mode, object_id, raw_path))
    return entries


def _verify_tracked_worktree_matches_commit(repo: Path, revision: str) -> None:
    """Compare worktree bytes with ``revision`` without trusting index stat flags.

    ``git status`` deliberately honors ``assume-unchanged`` and ``skip-worktree``.
    Hashing the checked-out files and comparing those object IDs with the pinned
    commit prevents either index flag from hiding a source edit. ``hash-object``
    also applies Git's clean conversion, so an ordinary CRLF checkout remains
    equivalent to the commit's LF blobs on Windows.
    """
    entries = _pinned_tree_blobs(repo, revision)
    mismatched: list[str] = []
    regular_entries: list[tuple[str, bytes]] = []

    for mode, expected_object_id, raw_path in entries:
        relative = os.fsdecode(raw_path)
        worktree_path = repo / Path(relative)
        if not worktree_path.exists() and not worktree_path.is_symlink():
            mismatched.append(f"{relative} (missing)")
            continue
        if mode in {"100644", "100755"}:
            if worktree_path.is_symlink() or not worktree_path.is_file():
                mismatched.append(f"{relative} (wrong file type)")
                continue
            regular_entries.append((expected_object_id, raw_path))
            continue
        if mode == "120000":
            if worktree_path.is_symlink():
                link_target = os.fsencode(os.readlink(worktree_path))
                actual_object_id = _git_bytes(
                    repo, "hash-object", "--stdin", stdin=link_target
                ).decode("ascii").strip()
            elif worktree_path.is_file():
                # Git may materialize a symlink as a plain file on Windows when
                # core.symlinks is disabled. The file then stores the link text.
                actual_object_id = _git(
                    repo, "hash-object", "--no-filters", "--", relative
                )
            else:
                mismatched.append(f"{relative} (wrong file type)")
                continue
            if actual_object_id != expected_object_id:
                mismatched.append(relative)
            continue
        raise RuntimeError(
            f"Unsupported tracked file mode in pinned baseline tree: "
            f"path={relative!r}, mode={mode}"
        )

    # Batch hashing avoids one Git process per source file. Git receives the raw
    # tree paths, hashes the actual worktree files, and emits one object ID per
    # path in the same order. A newline in a Git path cannot be represented by
    # --stdin-paths, so handle that rare case individually.
    batch_entries = [entry for entry in regular_entries if b"\n" not in entry[1]]
    individual_entries = [entry for entry in regular_entries if b"\n" in entry[1]]
    if batch_entries:
        path_input = b"\n".join(raw_path for _, raw_path in batch_entries) + b"\n"
        try:
            output = _git_bytes(repo, "hash-object", "--stdin-paths", stdin=path_input)
        except RuntimeError as exc:
            raise RuntimeError(
                "Tracked baseline source cannot be hashed against the pinned commit"
            ) from exc
        actual_ids = output.decode("ascii").splitlines()
        if len(actual_ids) != len(batch_entries):
            raise RuntimeError(
                "Git returned an unexpected number of hashes while verifying baseline source"
            )
        for (expected_object_id, raw_path), actual_object_id in zip(
            batch_entries, actual_ids
        ):
            if actual_object_id != expected_object_id:
                mismatched.append(os.fsdecode(raw_path))

    for expected_object_id, raw_path in individual_entries:
        relative = os.fsdecode(raw_path)
        actual_object_id = _git(
            repo, "hash-object", f"--path={relative}", "--", relative
        )
        if actual_object_id != expected_object_id:
            mismatched.append(relative)

    if mismatched:
        raise RuntimeError(
            "Tracked baseline source differs from the pinned commit; refusing execution: "
            f"{mismatched[:20]}"
        )


def _verify_baseline_source_against_trust(
    repository_path: str | Path,
    lock_file: str | Path,
    *,
    trusted_sha: str,
    trusted_repository: str,
) -> SourceGuardReport:
    """Hard-fail unless the exact locked official checkout is present and clean.

    This private function supplies a narrow seam for tiny local test repositories.
    Production callers use :func:`verify_baseline_source`, whose trust anchors
    cannot be overridden.
    """
    repo = Path(repository_path).resolve()
    lock_path = Path(lock_file).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(
            f"Locked baseline checkout is missing: {repo}. Prepare it with "
            "scripts/01_prepare_locked_baseline.py --clone-if-missing before running B0/B1."
        )
    if not (repo / ".git").exists():
        raise RuntimeError(f"Baseline path is not a Git checkout: {repo}")

    lock = load_baseline_lock(lock_path)
    if lock.get("baseline_name") != "LungDxFormer":
        raise RuntimeError(
            "Source lock baseline_name disagrees with the expected official baseline: "
            f"{lock.get('baseline_name')!r}"
        )
    locked_sha = str(lock["code"]["commit_sha"]).strip()
    locked_repository = str(lock["code"]["repository"]).strip()
    if locked_sha != trusted_sha:
        raise RuntimeError(
            f"Source lock commit disagrees with the trusted Phase 0 pin: {locked_sha} != {trusted_sha}"
        )
    if _normalize_remote(locked_repository) != _normalize_remote(trusted_repository):
        raise RuntimeError(
            "Source lock repository disagrees with the trusted official repository: "
            f"{locked_repository!r} != {trusted_repository!r}"
        )

    inside = _git(repo, "rev-parse", "--is-inside-work-tree")
    if inside.lower() != "true":
        raise RuntimeError(f"Baseline path is not inside a Git work tree: {repo}")
    head = _git(repo, "rev-parse", "HEAD")
    if head != trusted_sha:
        raise RuntimeError(f"Baseline HEAD mismatch: expected {trusted_sha}, got {head}")

    origin = _git(repo, "remote", "get-url", "origin")
    if _normalize_remote(origin) != _normalize_remote(trusted_repository):
        raise RuntimeError(f"Baseline origin mismatch: expected {trusted_repository}, got {origin}")

    dirty = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        preview = "\n".join(dirty.splitlines()[:20])
        raise RuntimeError(f"Locked baseline working tree is dirty; refusing execution:\n{preview}")
    _verify_tracked_worktree_matches_commit(repo, trusted_sha)

    snapshot = snapshot_source_tree(repo)
    return SourceGuardReport(
        repository_path=str(repo),
        lock_file=str(lock_path),
        repository_url=origin,
        expected_commit=trusted_sha,
        head_commit=head,
        working_tree_clean=True,
        source_tree_sha256=sha256_json(snapshot),
    )


def verify_baseline_source(
    repository_path: str | Path,
    lock_file: str | Path,
) -> SourceGuardReport:
    """Verify the production baseline against the immutable Phase 0 trust anchors."""
    return _verify_baseline_source_against_trust(
        repository_path,
        lock_file,
        trusted_sha=PINNED_BASELINE_SHA,
        trusted_repository=OFFICIAL_BASELINE_REPOSITORY,
    )
