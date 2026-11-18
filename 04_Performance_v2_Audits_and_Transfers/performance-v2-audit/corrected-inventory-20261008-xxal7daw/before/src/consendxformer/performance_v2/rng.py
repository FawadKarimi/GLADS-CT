"""CPU-only initialization scopes and private data-order/augmentation streams."""
from contextlib import contextmanager
import hashlib
import random
import threading

STREAMS = (
    "encoder_initialization", "diagnostic_head_initialization",
    "auxiliary_head_initialization", "shuffle", "augmentation",
    "shared_fusion_initialization",
)
_CONSTRUCTION_LOCK = threading.RLock()


def stream_seed(seed: int, stream: str) -> int:
    if type(seed) is not int or seed not in (42, 43, 44) or stream not in STREAMS:
        raise ValueError("unreviewed seed or RNG stream")
    raw = f"performance-v2-rng-v1|{seed}|{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def private_random(seed: int, stream: str) -> random.Random:
    """No mutation of Python's global RNG; caller owns this stream's lifetime."""
    return random.Random(stream_seed(seed, stream))


def cpu_generator(seed: int, stream: str):
    import torch
    return torch.Generator(device="cpu").manual_seed(stream_seed(seed, stream))


@contextmanager
def initialization_scope(seed: int, stream: str):
    """Restore CPU RNG after construction; never seed or inspect CUDA devices.

    Construction is serialized among these scopes. Other application threads must
    not consume the global torch CPU generator during construction.
    """
    import torch
    value = stream_seed(seed, stream)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("construction requires explicit float32 default")
    with _CONSTRUCTION_LOCK, torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(value)
        yield
