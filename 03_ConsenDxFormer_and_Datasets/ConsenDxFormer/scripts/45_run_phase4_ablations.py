#!/usr/bin/env python3
"""Read-only plan or explicitly authorized, separately identified ablation jobs."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consendxformer.phase4_ablation_launch import ablation_cli

if __name__ == "__main__":
    raise SystemExit(ablation_cli())
