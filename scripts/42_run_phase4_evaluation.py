#!/usr/bin/env python3
"""Run the guarded Phase 4 saved-prediction evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_evaluation_runner import phase4_evaluation_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(phase4_evaluation_cli())
