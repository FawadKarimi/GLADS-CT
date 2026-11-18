from pathlib import Path
import subprocess
import sys

def test_smoke():
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "processed" / "synthetic_demo"
    subprocess.run([sys.executable, "scripts/generate_synthetic_dataset.py", "--output_dir", str(out), "--num_samples", "30"], cwd=root, check=True)
    subprocess.run([sys.executable, "src/train.py", "--config", "configs/config.yaml",
                    "--override", f"dataset.metadata_csv={out/'metadata.csv'}", f"dataset.image_root={out}"], cwd=root, check=True)
