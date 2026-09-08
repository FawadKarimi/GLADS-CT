#!/usr/bin/env python3
"""Check raw LIDC/pylidc readiness without extracting nodules."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase2_io import require_safe_phase2_output, write_json_new  # noqa: E402
from consendxformer.phase2_runtime import prepare_real_runtime  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an explicit raw LIDC source")
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase2.yaml"))
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--mode", required=True, choices=("strict_full", "partial_development"))
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    config, manifest, provider, _scans, report = prepare_real_runtime(
        config_path=args.config,
        raw_root=args.raw_root,
        source_mode=args.mode,
        permit_incomplete_report=True,
    )
    output = require_safe_phase2_output(args.output_json, ROOT)
    payload = report.to_dict()
    payload.update(
        {
            "configuration_mode": config["source"]["mode"],
            "manifest_bundle_sha256": manifest.bundle_sha256,
            "provider_versions": dict(provider.versions()),
            "full_dicom_file_cryptographic_hashing_performed": False,
        }
    )
    write_json_new(output, payload)
    print(f"LIDC_SOURCE_MODE={report.source_mode}")
    print(f"FINAL_DATASET_ELIGIBLE={str(report.final_dataset_eligible).lower()}")
    print(f"report={output}")
    return 0 if args.mode != "strict_full" or report.final_dataset_eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
