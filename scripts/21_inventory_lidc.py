#!/usr/bin/env python3
"""Inventory real LIDC annotations without writing image or mask patches."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.lidc_pipeline import run_inventory  # noqa: E402
from consendxformer.phase2_runtime import prepare_real_runtime  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory physical LIDC nodule clusters")
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase2.yaml"))
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--mode", required=True, choices=("strict_full", "partial_development"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config, manifest, provider, scans, report = prepare_real_runtime(
        config_path=args.config,
        raw_root=args.raw_root,
        source_mode=args.mode,
    )
    summary = run_inventory(
        provider=provider,
        scans=scans,
        manifest=manifest,
        source_report=report,
        config=config,
        output_dir=args.output_dir,
        project_root=ROOT,
    )
    print(f"PHYSICAL_NODULE_CLUSTERS={summary['physical_nodule_cluster_count']}")
    print(f"output={Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
