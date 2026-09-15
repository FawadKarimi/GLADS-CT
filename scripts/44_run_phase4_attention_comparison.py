#!/usr/bin/env python3
"""Explicit exploratory no-retraining supplementary attention evidence."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.phase4_attention_comparison import attention_comparison_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(attention_comparison_cli())
