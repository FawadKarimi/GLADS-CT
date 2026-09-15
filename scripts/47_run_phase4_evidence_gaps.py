"""Explicit analysis-only evidence export. Run from the repository root."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from consendxformer.phase4_evidence_gaps import evidence_gap_cli

if __name__ == "__main__":
    raise SystemExit(evidence_gap_cli())
