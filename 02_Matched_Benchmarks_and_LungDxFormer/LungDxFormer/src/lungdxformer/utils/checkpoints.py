from pathlib import Path
import torch

def save_checkpoint(state: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)
