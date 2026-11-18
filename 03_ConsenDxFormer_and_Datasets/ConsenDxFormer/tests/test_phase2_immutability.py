from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.manifest_guard import FROZEN_BUNDLE_SHA256, validate_frozen_manifest_bundle
from consendxformer.phase2_io import require_safe_phase2_output


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def test_phase0_manifests_and_phase1_baseline_configs_remain_unchanged():
    manifest = validate_frozen_manifest_bundle(ROOT / "artifacts" / "phase0" / "manifests")
    assert manifest.bundle_sha256 == FROZEN_BUNDLE_SHA256
    for path, expected_blob in {
        "configs/b0.yaml": "99484625b45b782466a926a5fd258044b0aa6492",
        "configs/b1.yaml": "6d09fb309561a1c1aa9ce41d4bfdc36ffba10a6a",
    }.items():
        assert _git("hash-object", path) == expected_blob
        assert _git("rev-parse", f"2c5527c1e9afce6b2dd09e186913452ed10340f2:{path}") == expected_blob


@pytest.mark.parametrize(
    "protected",
    [
        ROOT / "artifacts" / "phase0" / "manifests" / "bad-output",
        ROOT / "external" / "LungDxFormer" / "bad-output",
    ],
)
def test_phase2_output_cannot_write_into_frozen_or_baseline_trees(protected):
    with pytest.raises(ValueError, match="protected path"):
        require_safe_phase2_output(protected, ROOT)
    assert not protected.exists()


def test_large_raw_and_generated_data_patterns_are_gitignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/", "*.dcm", "*.npy", "*.npz", "artifacts/phase2/runtime/"):
        assert pattern in ignore


def test_phase2_does_not_implement_model_training_or_random_split():
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src" / "consendxformer").glob("lidc_*.py"))
    )
    assert "torch" not in sources
    assert "random_split" not in sources
    assert "Radiologist_1" not in sources
