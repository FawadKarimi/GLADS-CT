from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase2_config import (
    ClusteringConfig,
    HUWindow,
    PYLIDC_VERSION,
    load_phase2_config,
    phase2_config_sha256,
)
from consendxformer.pylidc_provider import ProviderDependencyError, PylidcProvider


def test_phase2_config_loads_with_explicit_locked_algorithms_and_no_hu_claim():
    config = load_phase2_config(ROOT / "configs" / "phase2.yaml")
    assert config["project"]["phase"] == 2
    assert config["clustering"] == {
        "implementation": "pylidc.Scan.cluster_annotations",
        "metric": "min",
        "tol": None,
        "factor": 0.9,
        "min_tol": 0.1,
    }
    assert config["consensus"]["ret_masks"] is True
    assert config["hu"] == {
        "provenance": "explicit_required",
        "window_min": None,
        "window_max": None,
    }
    assert config["rendering"]["supported_resolutions"] == [64, 96]
    assert len(phase2_config_sha256(config)) == 64


def test_hu_window_is_explicit_and_validated():
    assert HUWindow(-1000, 400, "user_supplied").to_dict()["window_min"] == -1000.0
    with pytest.raises(ValueError, match="strictly less"):
        HUWindow(0, 0)
    with pytest.raises(ValueError, match="provenance"):
        HUWindow(-1, 1, "paper_reported")


def test_pylidc_is_lazy_optional_and_exactly_pinned(monkeypatch, tmp_path):
    provider = PylidcProvider(tmp_path)
    assert provider.name == "official_pylidc"

    original = importlib.metadata.version

    def missing(name):
        if name == "pylidc":
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(ProviderDependencyError, match=r"pylidc==0\.2\.3"):
        tuple(provider.iter_scans())
    assert PYLIDC_VERSION == "0.2.3"
    assert ClusteringConfig().to_dict() == {
        "metric": "min",
        "tol": None,
        "factor": 0.9,
        "min_tol": 0.1,
    }
