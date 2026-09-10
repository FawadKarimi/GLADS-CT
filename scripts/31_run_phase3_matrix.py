#!/usr/bin/env python3
"""Launch an explicit guarded Phase 3 B0/B1 seed matrix."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase3_orchestrator import phase3_matrix_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(phase3_matrix_cli())
