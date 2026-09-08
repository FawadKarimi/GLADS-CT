#!/usr/bin/env python3
"""Render explicit-HU 64x64 or 96x96 model-ready 2D ROIs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.lidc_render_pipeline import run_rendering  # noqa: E402
from consendxformer.phase2_config import HUWindow  # noqa: E402
from consendxformer.phase2_runtime import prepare_real_runtime  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render canonical nodules as 2D ROIs")
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase2.yaml"))
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--mode", required=True, choices=("strict_full", "partial_development"))
    parser.add_argument("--canonical-ledger", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", required=True, type=int, choices=(64, 96))
    parser.add_argument("--hu-min", required=True, type=float)
    parser.add_argument("--hu-max", required=True, type=float)
    parser.add_argument(
        "--hu-provenance",
        default="user_supplied",
        choices=("user_supplied", "project_defined"),
    )
    args = parser.parse_args(argv)
    config, manifest, provider, scans, report = prepare_real_runtime(
        config_path=args.config,
        raw_root=args.raw_root,
        source_mode=args.mode,
    )
    rendered = run_rendering(
        provider=provider,
        scans=scans,
        manifest=manifest,
        source_report=report,
        config=config,
        canonical_ledger_path=args.canonical_ledger,
        output_dir=args.output_dir,
        project_root=ROOT,
        hu_window=HUWindow(args.hu_min, args.hu_max, args.hu_provenance),
        resolution=args.resolution,
    )
    print(f"RENDERED_ROIS={len(rendered)}")
    print(f"output={Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
