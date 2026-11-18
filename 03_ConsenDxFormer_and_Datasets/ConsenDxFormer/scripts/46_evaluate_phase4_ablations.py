#!/usr/bin/env python3
"""Explicit completed-ablation analysis; never trains or alters the frozen prior."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-source", action="append", required=True)
    parser.add_argument("--supplementary-source", required=True)
    parser.add_argument("--authorize-analysis", action="store_true")
    args = parser.parse_args()
    from consendxformer.phase4_ablation_analysis import analyze_ablations
    print(analyze_ablations(args.ablation_source, args.supplementary_source,
                           authorize_analysis=args.authorize_analysis, device="cpu"))
