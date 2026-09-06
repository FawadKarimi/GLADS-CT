from __future__ import annotations

from typing import Any
import os
import random
import sys


def set_global_seed(seed: int, deterministic_algorithms: bool = True, cudnn_benchmark: bool = False) -> dict[str, Any]:
    """Set seeds for Python, NumPy, and PyTorch when available.

    This improves reproducibility but does not guarantee bitwise-identical results
    across different GPUs, drivers, CUDA/cuDNN versions, or framework versions.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    report: dict[str, Any] = {
        "seed": seed,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "numpy_seeded": False,
        "torch_seeded": False,
        "deterministic_algorithms_requested": deterministic_algorithms,
        "cudnn_benchmark_requested": cudnn_benchmark,
    }

    try:
        import numpy as np
        np.random.seed(seed)
        report["numpy_seeded"] = True
    except Exception as exc:  # pragma: no cover
        report["numpy_error"] = repr(exc)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
            torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(bool(deterministic_algorithms), warn_only=True)
        report["torch_seeded"] = True
        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        report["torch_error"] = repr(exc)

    report["python_version"] = sys.version
    return report
