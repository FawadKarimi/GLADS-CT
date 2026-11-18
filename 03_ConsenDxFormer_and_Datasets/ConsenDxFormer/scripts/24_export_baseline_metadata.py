#!/usr/bin/env python3
"""Export one explicitly named Phase 2 cohort view for Phase 1 B0/B1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.baseline_metadata_export import COHORT_VIEWS, export_baseline_metadata  # noqa: E402
from consendxformer.manifest_guard import validate_frozen_manifest_bundle  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export an explicit baseline cohort view")
    parser.add_argument("--rendered-ledger", required=True)
    parser.add_argument("--rendered-root", required=True)
    parser.add_argument("--view", required=True, choices=COHORT_VIEWS)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args(argv)
    manifest = validate_frozen_manifest_bundle(ROOT / "artifacts" / "phase0" / "manifests")
    result = export_baseline_metadata(
        args.rendered_ledger,
        manifest,
        cohort_view=args.view,
        rendered_root=args.rendered_root,
        output_csv=args.output_csv,
    )
    print(f"BASELINE_METADATA_ROWS={len(result)}")
    print(f"cohort_view={args.view}")
    print(f"output={Path(args.output_csv).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
