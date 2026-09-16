#!/usr/bin/env python3
"""Separate opt-in insertion/deletion subset; no other analysis is invoked."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consendxformer.phase5_perturbation_runner import insertion_deletion_cli

if __name__ == "__main__":
    raise SystemExit(insertion_deletion_cli())
