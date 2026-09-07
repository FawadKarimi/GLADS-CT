#!/usr/bin/env python3
"""Guarded evaluation of a supplied B0/B1 prediction CSV."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.baseline_evaluator import baseline_evaluation_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(baseline_evaluation_cli())
