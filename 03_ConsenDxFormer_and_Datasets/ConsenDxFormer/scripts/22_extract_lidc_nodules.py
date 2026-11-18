#!/usr/bin/env python3
"""Extract the canonical multi-reader nodule ledger and 2D mask artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.lidc_pipeline import run_extraction  # noqa: E402
from consendxformer.phase2_runtime import prepare_real_runtime  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract canonical LIDC nodule records")
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase2.yaml"))
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--mode", required=True, choices=("strict_full", "partial_development"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config, manifest, provider, scans, report = prepare_real_runtime(
        config_path=args.config,
        raw_root=args.raw_root,
        source_mode=args.mode,
    )
    output = args.output_dir or str(ROOT / config["output"]["root"])
    ledger = run_extraction(
        provider=provider,
        scans=scans,
        manifest=manifest,
        source_report=report,
        config=config,
        output_dir=output,
        project_root=ROOT,
        allow_resume=args.resume,
    )
    print(f"CANONICAL_NODULE_ROWS={len(ledger)}")
    print(f"output={Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
