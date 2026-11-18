"""Separate opt-in R3 archived-record analysis; future execution authorization required."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consendxformer.r3_entropy import cli

if __name__ == "__main__":
    raise SystemExit(cli())
