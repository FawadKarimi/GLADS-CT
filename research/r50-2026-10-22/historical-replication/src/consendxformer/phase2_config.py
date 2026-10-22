"""Validation for the Phase 2 data/annotation protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .hashing import sha256_json
from .manifest_guard import FROZEN_BUNDLE_SHA256


SOURCE_MODES = ("strict_full", "partial_development")
PYLIDC_VERSION = "0.2.3"
REPRESENTATIVE_SLICE_RULE = "maximum_soft_consensus_area_lowest_index_v1"
CROP_RULE = "square_union_bbox_context_v1"


@dataclass(frozen=True)
class ClusteringConfig:
    metric: str = "min"
    tol: float | None = None
    factor: float = 0.9
    min_tol: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "tol": self.tol,
            "factor": self.factor,
            "min_tol": self.min_tol,
        }


@dataclass(frozen=True)
class ConsensusConfig:
    clevel: float = 0.5
    pad: int | float | list[list[int]] | None = None
    ret_masks: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"clevel": self.clevel, "pad": self.pad, "ret_masks": self.ret_masks}


@dataclass(frozen=True)
class CropConfig:
    context_fraction: float = 0.25
    min_side_pixels: int = 16
    pad_hu: float = -1000.0
    rule: str = CROP_RULE

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "context_fraction": self.context_fraction,
            "min_side_pixels": self.min_side_pixels,
            "pad_mode": "constant",
            "pad_hu": self.pad_hu,
        }


@dataclass(frozen=True)
class HUWindow:
    minimum: float
    maximum: float
    provenance: str = "project_defined"

    def __post_init__(self) -> None:
        if not self.minimum < self.maximum:
            raise ValueError("HU window minimum must be strictly less than maximum")
        if self.provenance not in {"project_defined", "user_supplied"}:
            raise ValueError("HU provenance must be project_defined or user_supplied")

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_min": float(self.minimum),
            "window_max": float(self.maximum),
            "provenance": self.provenance,
        }


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Phase 2 config section {name!r} must be a mapping")
    return value


def validate_phase2_config(config: dict[str, Any]) -> None:
    if _section(config, "project").get("phase") != 2:
        raise ValueError("project.phase must be 2")
    source = _section(config, "source")
    if source.get("provider") != "pylidc":
        raise ValueError("source.provider must remain the official pylidc adapter")
    if source.get("mode") not in SOURCE_MODES:
        raise ValueError(f"source.mode must be one of {SOURCE_MODES}")
    if source.get("expected_unique_patients") != 1010:
        raise ValueError("source.expected_unique_patients must remain 1010")
    if source.get("expected_unique_ct_series") != 1018:
        raise ValueError("source.expected_unique_ct_series must remain 1018")

    frozen = _section(config, "frozen_manifest")
    if frozen.get("directory") != "artifacts/phase0/manifests":
        raise ValueError("frozen_manifest.directory must point to the Phase 0 bundle")
    if frozen.get("bundle_sha256") != FROZEN_BUNDLE_SHA256:
        raise ValueError("frozen_manifest.bundle_sha256 does not match the source-code trust anchor")

    clustering = _section(config, "clustering")
    expected_clustering = {
        "implementation": "pylidc.Scan.cluster_annotations",
        "metric": "min",
        "tol": None,
        "factor": 0.9,
        "min_tol": 0.1,
    }
    if clustering != expected_clustering:
        raise ValueError(
            "clustering must explicitly preserve the documented pylidc parameters: "
            f"{expected_clustering!r}"
        )

    consensus = _section(config, "consensus")
    if consensus.get("implementation") != "pylidc.utils.consensus":
        raise ValueError("consensus implementation must be pylidc.utils.consensus")
    if consensus.get("ret_masks") is not True:
        raise ValueError("consensus.ret_masks must be true to preserve individual masks")
    clevel = consensus.get("clevel")
    if not isinstance(clevel, (int, float)) or isinstance(clevel, bool) or not 0 < clevel <= 1:
        raise ValueError("consensus.clevel must be in (0, 1]")

    representative = _section(config, "representative_slice")
    if representative.get("rule") != REPRESENTATIVE_SLICE_RULE:
        raise ValueError("representative slice rule is immutable within Phase 2")

    crop = _section(config, "crop")
    if crop.get("rule") != CROP_RULE or crop.get("pad_mode") != "constant":
        raise ValueError("crop rule/padding must use the deterministic Phase 2 protocol")
    context = crop.get("context_fraction")
    if not isinstance(context, (int, float)) or isinstance(context, bool) or context < 0:
        raise ValueError("crop.context_fraction must be non-negative")
    minimum_side = crop.get("min_side_pixels")
    if not isinstance(minimum_side, int) or isinstance(minimum_side, bool) or minimum_side <= 0:
        raise ValueError("crop.min_side_pixels must be a positive integer")

    hu = _section(config, "hu")
    if hu.get("provenance") != "explicit_required":
        raise ValueError("HU endpoints must remain explicitly required for real rendering")
    if hu.get("window_min") is not None or hu.get("window_max") is not None:
        if hu.get("window_min") is None or hu.get("window_max") is None:
            raise ValueError("Both HU endpoints must be supplied together")
        HUWindow(float(hu["window_min"]), float(hu["window_max"]), "project_defined")

    rendering = _section(config, "rendering")
    if rendering.get("supported_resolutions") != [64, 96]:
        raise ValueError("rendering.supported_resolutions must remain [64, 96]")
    if rendering.get("image_dtype") != "float32":
        raise ValueError("rendered images must be float32")
    if rendering.get("individual_mask_dtype") != "uint8":
        raise ValueError("individual masks must be uint8")
    if rendering.get("soft_mask_dtype") != "float32":
        raise ValueError("soft masks must be float32")

    output = _section(config, "output")
    root = str(output.get("root", "")).replace("\\", "/")
    if not root.startswith("data/processed/"):
        raise ValueError("Phase 2 output must remain under ignored data/processed/")


def load_phase2_config(path: str | Path) -> dict[str, Any]:
    config = load_config(path)
    validate_phase2_config(config)
    return config


def phase2_config_sha256(config: dict[str, Any]) -> str:
    validate_phase2_config(config)
    return sha256_json(config)


def clustering_config(config: dict[str, Any]) -> ClusteringConfig:
    validate_phase2_config(config)
    values = config["clustering"]
    return ClusteringConfig(
        metric=values["metric"],
        tol=values["tol"],
        factor=float(values["factor"]),
        min_tol=float(values["min_tol"]),
    )


def consensus_config(config: dict[str, Any]) -> ConsensusConfig:
    validate_phase2_config(config)
    values = config["consensus"]
    return ConsensusConfig(
        clevel=float(values["clevel"]),
        pad=values["pad"],
        ret_masks=True,
    )


def crop_config(config: dict[str, Any]) -> CropConfig:
    validate_phase2_config(config)
    values = config["crop"]
    return CropConfig(
        context_fraction=float(values["context_fraction"]),
        min_side_pixels=int(values["min_side_pixels"]),
        pad_hu=float(values["pad_hu"]),
    )
