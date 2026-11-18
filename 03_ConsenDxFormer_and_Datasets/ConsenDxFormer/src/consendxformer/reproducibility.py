"""Central reproducibility controls and environment audit helpers."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import random
import sys
from typing import Any, Iterable


MAX_SEED = 2**32 - 1
DEFAULT_RELEVANT_PACKAGES = (
    "numpy",
    "pandas",
    "PyYAML",
    "torch",
    "torchvision",
    "scikit-learn",
    "opencv-python",
    "Pillow",
    "pylidc",
    "pydicom",
)


class ReproducibilityError(RuntimeError):
    """Raised when requested real-training reproducibility cannot be established."""


def validate_seed(seed: Any) -> int:
    """Return a NumPy/PyTorch/DataLoader-compatible unsigned 32-bit seed."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in the range 0..{MAX_SEED}")
    return seed


def _import_torch() -> Any:
    return importlib.import_module("torch")


def set_global_seed(
    seed: int,
    deterministic_algorithms: bool = True,
    cudnn_benchmark: bool = False,
    *,
    warn_only: bool = True,
    require_torch: bool = False,
) -> dict[str, Any]:
    """Seed Python, NumPy and (when installed) PyTorch.

    ``require_torch=False`` keeps infrastructure-only synthetic dry-runs usable
    in the project's lightweight test environment. Real training must pass
    ``require_torch=True`` so an unavailable or unusable PyTorch runtime fails
    loudly. These controls improve repeatability but cannot promise bitwise
    equality across GPUs, drivers, CUDA/cuDNN, or framework versions.
    """

    seed = validate_seed(seed)
    for role, value in (
        ("deterministic_algorithms", deterministic_algorithms),
        ("cudnn_benchmark", cudnn_benchmark),
        ("warn_only", warn_only),
        ("require_torch", require_torch),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{role} must be a boolean")

    # Setting this inside an interpreter cannot retroactively change that
    # interpreter's hash secret. It is exported for any later child process,
    # and the current-interpreter limitation is recorded below.
    os.environ["PYTHONHASHSEED"] = str(seed)
    cublas_workspace = None
    if deterministic_algorithms:
        # PyTorch requires one of these workspace settings for deterministic
        # CUDA BLAS operations. Preserve an explicitly configured valid value.
        configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if configured not in {":16:8", ":4096:8"}:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        cublas_workspace = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    random.seed(seed)

    report: dict[str, Any] = {
        "seed": seed,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "python_random_seeded": True,
        "pythonhashseed_effective_for_current_interpreter_startup": False,
        "numpy_seeded": False,
        "torch_seeded": False,
        "torch_cpu_seeded": False,
        "torch_cuda_seeded": False,
        "deterministic_algorithms_requested": deterministic_algorithms,
        "deterministic_algorithms_warn_only_requested": warn_only,
        "cudnn_benchmark_requested": cudnn_benchmark,
        "cublas_workspace_config": cublas_workspace,
        "bitwise_reproducibility_across_environments_claimed": False,
    }

    try:
        import numpy as np

        np.random.seed(seed)
        report["numpy_seeded"] = True
        report["numpy_version"] = np.__version__
    except Exception as exc:  # NumPy is a core Phase 3 dependency.
        report["numpy_error"] = repr(exc)
        raise ReproducibilityError(
            "NumPy reproducibility controls could not be initialized"
        ) from exc

    try:
        torch = _import_torch()
        torch.manual_seed(seed)
        report["torch_cpu_seeded"] = True
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            torch.cuda.manual_seed_all(seed)
            report["torch_cuda_seeded"] = True

        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn is not None:
            cudnn.benchmark = bool(cudnn_benchmark)
            cudnn.deterministic = bool(deterministic_algorithms)

        deterministic_api = getattr(torch, "use_deterministic_algorithms", None)
        warn_only_supported = False
        if callable(deterministic_api):
            try:
                deterministic_api(bool(deterministic_algorithms), warn_only=bool(warn_only))
                warn_only_supported = True
            except TypeError:  # Compatibility with older optional PyTorch installs.
                deterministic_api(bool(deterministic_algorithms))

        enabled_probe = getattr(torch, "are_deterministic_algorithms_enabled", None)
        warning_probe = getattr(
            torch, "is_deterministic_algorithms_warn_only_enabled", None
        )
        report.update(
            {
                "torch_seeded": True,
                "torch_version": str(getattr(torch, "__version__", "unknown")),
                "cuda_available": cuda_available,
                "cuda_runtime_version": (
                    None
                    if getattr(torch, "version", None) is None
                    else getattr(torch.version, "cuda", None)
                ),
                "deterministic_algorithms_enabled": (
                    bool(enabled_probe())
                    if callable(enabled_probe)
                    else bool(deterministic_algorithms)
                ),
                "deterministic_algorithms_warn_only_enabled": (
                    bool(warning_probe())
                    if callable(warning_probe)
                    else (bool(warn_only) if warn_only_supported else None)
                ),
                "cudnn_deterministic": (
                    bool(cudnn.deterministic) if cudnn is not None else None
                ),
                "cudnn_benchmark": (
                    bool(cudnn.benchmark) if cudnn is not None else None
                ),
            }
        )
    except Exception as exc:
        report["torch_error"] = repr(exc)
        if require_torch:
            raise ReproducibilityError(
                "PyTorch is required for a real Phase 3 experiment, but its "
                "reproducibility controls could not be initialized"
            ) from exc

    report["python_version"] = sys.version
    return report


def dataloader_worker_seed(worker_id: int) -> int:
    """Seed Python/NumPy in one PyTorch DataLoader worker and return its seed."""

    if not isinstance(worker_id, int) or isinstance(worker_id, bool) or worker_id < 0:
        raise ValueError("worker_id must be a non-negative integer")
    try:
        torch = _import_torch()
    except Exception as exc:
        raise ReproducibilityError(
            "PyTorch is required to initialize a DataLoader worker"
        ) from exc
    worker_seed = int(torch.initial_seed()) % (MAX_SEED + 1)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except Exception as exc:  # pragma: no cover - NumPy is a core dependency.
        raise ReproducibilityError("NumPy worker seeding failed") from exc
    return worker_seed


def seed_dataloader_worker(worker_id: int) -> None:
    """PyTorch-compatible ``worker_init_fn`` for deterministic augmentation."""

    dataloader_worker_seed(worker_id)


def make_dataloader_generator(seed: int, *, require_torch: bool = True) -> Any | None:
    """Create a lazily imported, deterministically seeded ``torch.Generator``."""

    seed = validate_seed(seed)
    if not isinstance(require_torch, bool):
        raise TypeError("require_torch must be a boolean")
    try:
        torch = _import_torch()
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator
    except Exception as exc:
        if require_torch:
            raise ReproducibilityError(
                "PyTorch is required to create the seeded DataLoader generator"
            ) from exc
        return None


def _distribution_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        name = str(package)
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        except Exception as exc:  # Keep provenance creation diagnostic and total.
            versions[name] = f"ERROR: {type(exc).__name__}: {exc}"
    return versions


def environment_audit(
    *,
    require_torch: bool = False,
    relevant_packages: Iterable[str] = DEFAULT_RELEVANT_PACKAGES,
) -> dict[str, Any]:
    """Collect machine-readable runtime provenance without requiring a GPU."""

    if not isinstance(require_torch, bool):
        raise TypeError("require_torch must be a boolean")
    package_names = tuple(dict.fromkeys(str(value) for value in relevant_packages))
    record: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "numpy": {"version": None},
        "packages": _distribution_versions(package_names),
        "torch": {
            "available": False,
            "version": None,
            "cuda_available": False,
            "cuda_runtime_version": None,
            "cudnn_version": None,
            "gpu_names": [],
        },
        "bitwise_reproducibility_across_environments_claimed": False,
    }
    try:
        import numpy as np

        record["numpy"]["version"] = np.__version__
    except Exception as exc:  # pragma: no cover - NumPy is a core dependency.
        record["numpy"]["error"] = repr(exc)

    try:
        torch = _import_torch()
        cuda_available = bool(torch.cuda.is_available())
        gpu_names: list[str] = []
        if cuda_available:
            for index in range(int(torch.cuda.device_count())):
                gpu_names.append(str(torch.cuda.get_device_name(index)))
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        cudnn_version = None
        if cudnn is not None and callable(getattr(cudnn, "version", None)):
            cudnn_version = cudnn.version()
        torch_version_namespace = getattr(torch, "version", None)
        record["torch"] = {
            "available": True,
            "version": str(getattr(torch, "__version__", "unknown")),
            "cuda_available": cuda_available,
            "cuda_runtime_version": (
                getattr(torch_version_namespace, "cuda", None)
                if torch_version_namespace is not None
                else None
            ),
            "cudnn_version": cudnn_version,
            "gpu_names": gpu_names,
        }
    except Exception as exc:
        record["torch"]["import_error"] = repr(exc)
        if require_torch:
            raise ReproducibilityError(
                "PyTorch environment audit is required for a real Phase 3 experiment"
            ) from exc
    return record


# Concise aliases for DataLoader construction and provenance call sites.
seed_worker = seed_dataloader_worker
make_torch_generator = make_dataloader_generator
capture_environment = environment_audit
