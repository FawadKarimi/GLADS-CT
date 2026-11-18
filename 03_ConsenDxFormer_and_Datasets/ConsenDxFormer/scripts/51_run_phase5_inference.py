"""Separate saved-prediction preflight / explicitly released inference subset."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consendxformer.phase5_inference_runner import inference_cli

if __name__ == "__main__":
    raise SystemExit(inference_cli())
