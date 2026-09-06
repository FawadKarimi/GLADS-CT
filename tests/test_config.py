from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consendxformer.config import load_config


def test_phase0_config_extends_base():
    cfg = load_config(ROOT / "configs" / "phase0.yaml")
    assert cfg["project"]["phase"] == 0
    assert cfg["reproducibility"]["seed"] == 2026
    assert cfg["dataset_inventory"]["tcia_reference"]["unique_patients"] == 1010
    assert cfg["dataset_inventory"]["tcia_reference"]["unique_ct_series"] == 1018
