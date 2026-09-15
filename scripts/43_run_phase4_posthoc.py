#!/usr/bin/env python3
"""Explicit frozen-checkpoint attention and optional boundary inference."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_posthoc_runner import phase4_posthoc_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(phase4_posthoc_cli())
