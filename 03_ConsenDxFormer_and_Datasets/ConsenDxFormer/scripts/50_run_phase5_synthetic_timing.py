#!/usr/bin/env python3
"""Separate opt-in synthetic timing. Device isolation precedes all framework imports."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    from consendxformer.phase5_device import bootstrap_cli
    bootstrap_cli(sys.argv[1:], "--authorize-synthetic-timing")
    from consendxformer.phase5_timing_runner import timing_cli
    raise SystemExit(timing_cli())
