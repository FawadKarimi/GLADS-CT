"""Immutable real-data acceptance lock for Phase 4 execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_LOCK_KEYS = {
    "schema_version",
    "cohort_view",
    "expected_resolution",
    "sample_count",
    "split_counts",
    "reader_count_counts",
    "manifest_bundle_sha256",
    "supervision_sha256",
    "ordered_supervision_sha256",
}

EXPECTED_PHASE4_INPUT_LOCK: dict[str, Any] = {
    "schema_version": 1,
    "cohort_view": "paper_aligned_multi_reader_2_to_4",
    "expected_resolution": 64,
    "sample_count": 1133,
    "split_counts": {"train": 807, "val": 143, "test": 183},
    "reader_count_counts": {"2": 307, "3": 305, "4": 521},
    "manifest_bundle_sha256": (
        "0c983f0ebbc6dd70623b570c8a657aa14ccc550ac590b577648b3163bc8a01e3"
    ),
    "supervision_sha256": (
        "cbe4e402825665428ee890887963437d2bc480f8616bd7ec7999c667759ccc25"
    ),
    "ordered_supervision_sha256": (
        "2bae364b180f75cf2332714cc486a517540fcff3b9e94c48d83f59d0d9d6789a"
    ),
}


class Phase4InputLockError(RuntimeError):
    """Raised when real Phase 4 inputs differ from the approved lock."""


@dataclass(frozen=True)
class Phase4InputLock:
    schema_version: int
    cohort_view: str
    expected_resolution: int
    sample_count: int
    split_counts: Mapping[str, int]
    reader_count_counts: Mapping[str, int]
    manifest_bundle_sha256: str
    supervision_sha256: str
    ordered_supervision_sha256: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["split_counts"] = dict(self.split_counts)
        value["reader_count_counts"] = dict(self.reader_count_counts)
        return value


def _fail(message: str) -> None:
    raise Phase4InputLockError(message)


def _validate_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("Phase 4 input lock must be a JSON object")
    payload = dict(value)
    if set(payload) != _LOCK_KEYS:
        missing = sorted(_LOCK_KEYS - set(payload))
        extra = sorted(set(payload) - _LOCK_KEYS)
        _fail(f"Phase 4 input lock fields differ: missing={missing}, extra={extra}")
    if payload != EXPECTED_PHASE4_INPUT_LOCK:
        changed = sorted(
            key
            for key in _LOCK_KEYS
            if payload.get(key) != EXPECTED_PHASE4_INPUT_LOCK.get(key)
        )
        _fail(f"Phase 4 input lock differs from the approved values: {changed}")
    for role in (
        "manifest_bundle_sha256",
        "supervision_sha256",
        "ordered_supervision_sha256",
    ):
        if _SHA256_RE.fullmatch(str(payload[role])) is None:
            _fail(f"Phase 4 input lock {role} is not a lowercase SHA-256 digest")
    if sum(payload["split_counts"].values()) != payload["sample_count"]:
        _fail("Phase 4 input lock split counts do not sum to sample_count")
    if sum(payload["reader_count_counts"].values()) != payload["sample_count"]:
        _fail("Phase 4 input lock reader counts do not sum to sample_count")
    return payload


def load_phase4_input_lock(path: str | Path) -> Phase4InputLock:
    lock_path = Path(path).resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(f"Phase 4 input lock does not exist: {lock_path}")
    try:
        loaded = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4InputLockError(
            f"Cannot read Phase 4 input lock: {lock_path}"
        ) from exc
    payload = _validate_payload(loaded)
    return Phase4InputLock(**payload)


def validate_phase4_input_lock(
    lock: Phase4InputLock,
    readiness: Mapping[str, Any] | Any,
) -> None:
    """Require the current readiness evidence to equal every locked input field."""

    if not isinstance(lock, Phase4InputLock):
        raise TypeError("lock must be a Phase4InputLock")
    if isinstance(readiness, Mapping):
        report = dict(readiness)
    elif callable(getattr(readiness, "to_dict", None)):
        report = dict(readiness.to_dict())
    else:
        raise TypeError("readiness must be a mapping or expose to_dict()")

    expected = lock.to_dict()
    roles = (
        "cohort_view",
        "expected_resolution",
        "sample_count",
        "split_counts",
        "reader_count_counts",
        "manifest_bundle_sha256",
        "supervision_sha256",
        "ordered_supervision_sha256",
    )
    for role in roles:
        actual = report.get(role)
        if role in {"split_counts", "reader_count_counts"} and isinstance(
            actual, Mapping
        ):
            actual = {str(key): int(value) for key, value in actual.items()}
            locked = {str(key): int(value) for key, value in expected[role].items()}
        else:
            locked = expected[role]
        if actual != locked:
            _fail(
                f"Phase 4 readiness field {role!r} differs from the input lock: "
                f"actual={actual!r}, expected={locked!r}"
            )


__all__ = [
    "EXPECTED_PHASE4_INPUT_LOCK",
    "Phase4InputLock",
    "Phase4InputLockError",
    "load_phase4_input_lock",
    "validate_phase4_input_lock",
]
