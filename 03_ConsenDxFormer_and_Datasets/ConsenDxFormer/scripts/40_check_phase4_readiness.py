#!/usr/bin/env python3
"""Check guarded Phase 4 real-data readiness without constructing a model."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.manifest_guard import validate_frozen_manifest_bundle  # noqa: E402
from consendxformer.phase4_config import load_phase4_config  # noqa: E402
from consendxformer.phase4_readiness import validate_phase4_readiness  # noqa: E402


def main() -> int:
    try:
        bundle = load_phase4_config(ROOT / "configs" / "phase4_consensdxformer.yaml")
        config = bundle.phase4_config
        data_config = config["data"]
        manifest_path = Path(data_config["frozen_split_dir"])
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        manifest = validate_frozen_manifest_bundle(manifest_path.resolve())
        report, _ = validate_phase4_readiness(
            data_config,
            manifest=manifest,
            expected_resolution=int(config["dataset"]["image_size"]),
            project_root=ROOT,
            synthetic_dry_run=False,
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "ready_for_requested_mode": False,
            "real_research_ready": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(
            json.dumps(failure, indent=2, sort_keys=True, allow_nan=False),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
