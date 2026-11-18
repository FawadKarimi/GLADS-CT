#!/usr/bin/env python3
"""Separate opt-in insertion/deletion subset; no other analysis is invoked."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    from consendxformer.phase5_device import bootstrap_cli
    bootstrap_cli(sys.argv[1:], "--authorize-execution")
    from consendxformer.phase5_perturbation_runner import insertion_deletion_cli
    raise SystemExit(insertion_deletion_cli())
