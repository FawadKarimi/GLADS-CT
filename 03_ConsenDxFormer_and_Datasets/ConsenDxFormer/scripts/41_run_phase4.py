#!/usr/bin/env python3
"""Run one guarded Phase 4 ConsenDxFormer experiment."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_runner import phase4_cli


if __name__ == "__main__":
    raise SystemExit(phase4_cli())
