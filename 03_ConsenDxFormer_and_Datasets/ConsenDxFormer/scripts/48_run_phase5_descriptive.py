"""Explicitly opted-in saved-artifact pointing and accuracy only; no inference."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from consendxformer.phase5_descriptive_runner import descriptive_cli

if __name__ == "__main__":
    raise SystemExit(descriptive_cli())
