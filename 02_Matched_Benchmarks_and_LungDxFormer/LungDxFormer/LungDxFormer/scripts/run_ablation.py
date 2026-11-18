from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
from lungdxformer.evaluation.ablation import generate_ablation_settings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    settings = generate_ablation_settings()
    for i, s in enumerate(settings, start=1):
        overrides = [
            f"model.use_transformer={str(s['use_transformer']).lower()}",
            f"model.use_positional_encoding={str(s['use_positional_encoding']).lower()}",
            f"model.use_spatial_attention={str(s['use_spatial_attention']).lower()}",
        ]
        print(f"Running ablation {i}: {s}")
        subprocess.run(["python", "src/train.py", "--config", args.config, "--override", *overrides], check=True)

if __name__ == "__main__":
    main()
