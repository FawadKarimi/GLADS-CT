#!/usr/bin/env python3
"""Launch one guarded Phase 3 B0 or B1 experiment."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase3_runner import phase3_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(phase3_cli())
