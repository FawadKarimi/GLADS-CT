"""Owned data RNG streams and trace plans, not an image transform or loader."""
import copy
import math
import random

from .rng import STREAMS, private_random, stream_seed
from .reuse import canonical, digest


class TrainingStreams:
    def __init__(self, seed):
        self.seed = seed
        self.shuffle = private_random(seed, "shuffle")
        self.augmentation = private_random(seed, "augmentation")
        self.last_epoch = 0

    def epoch_plan(self, ids, *, epoch):
        ids = tuple(ids)
        if (type(epoch) is not int or epoch != self.last_epoch + 1 or not 1 <= epoch <= 100
                or not ids or len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids)):
            raise ValueError("sequential epoch and unique stable identities required")
        ordered = list(ids)
        self.shuffle.shuffle(ordered)
        traces = []
        for identifier in ordered:
            h, v = self.augmentation.random() < .5, self.augmentation.random() < .5
            angle = self.augmentation.uniform(-10, 10)
            tx, ty = self.augmentation.uniform(-.08, .08), self.augmentation.uniform(-.08, .08)
            traces.append({"id": identifier, "horizontal_flip": h, "vertical_flip": v,
                           "clockwise_angle": angle, "translate_fraction": [tx, ty],
                           "translate_pixels": [int(round(tx * 224)), int(round(ty * 224))]})
        self.last_epoch = epoch
        return {"epoch": epoch, "order": ordered, "augmentation": traces,
                "batches": [ordered[i:i + 16] for i in range(0, len(ordered), 16)],
                "batch_size": 16, "drop_last": False, "workers": 0}

    def snapshot(self):
        return {"schema": "performance_v2_data_streams_v1", "seed": self.seed,
                "last_epoch": self.last_epoch, "shuffle": self.shuffle.getstate(),
                "augmentation": self.augmentation.getstate(),
                "stream_seeds": {s: stream_seed(self.seed, s) for s in STREAMS}}

    def restore(self, state):
        if (type(state) is not dict or set(state) != set(self.snapshot())
                or state["schema"] != "performance_v2_data_streams_v1"
                or type(state["seed"]) is not int or state["seed"] != self.seed
                or state["stream_seeds"] != self.snapshot()["stream_seeds"]
                or type(state["last_epoch"]) is not int or not 0 <= state["last_epoch"] <= 100):
            raise ValueError("RNG stream identity mismatch")
        staged = []
        for key in ("shuffle", "augmentation"):
            generator = random.Random(0)
            try:
                generator.setstate(state[key])
            except (ValueError, TypeError) as error:
                raise ValueError("invalid RNG state") from error
            staged.append(generator)
        self.shuffle, self.augmentation = staged
        self.last_epoch = state["last_epoch"]


def capture_cpu_rng(streams):
    """Explicit CPU-only resume interface; no CUDA queries or implicit seeding."""
    import numpy as np
    import torch
    n = np.random.get_state()
    return {"schema": "performance_v2_cpu_rng_v1", "torch_cpu": torch.get_rng_state().clone(),
            "python": random.getstate(),
            "numpy": (n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])),
            "data_streams": copy.deepcopy(streams.snapshot()), "cuda": "not_used_cpu_only"}


def validate_cpu_rng(state, *, seed):
    """Validate on private generators before touching any process-global RNG."""
    import numpy as np
    import torch
    if (type(state) is not dict or set(state) != {"schema", "torch_cpu", "python", "numpy", "data_streams", "cuda"}
            or state["schema"] != "performance_v2_cpu_rng_v1" or state["cuda"] != "not_used_cpu_only"):
        raise ValueError("exact CPU RNG bundle required; GPU resume is unimplemented")
    candidate = TrainingStreams(seed)
    candidate.restore(state["data_streams"])
    cpu = state["torch_cpu"]
    if not isinstance(cpu, torch.Tensor) or cpu.device.type != "cpu" or cpu.dtype != torch.uint8:
        raise ValueError("CPU Torch RNG bytes required")
    torch.Generator(device="cpu").set_state(cpu)
    random.Random(0).setstate(state["python"])
    n = state["numpy"]
    if (type(n) is not tuple or len(n) != 5 or n[0] != "MT19937"
            or type(n[1]) is not list or len(n[1]) != 624
            or any(type(x) is not int or not 0 <= x < 2**32 for x in n[1])
            or type(n[2]) is not int or not 0 <= n[2] <= 624
            or type(n[3]) is not int or n[3] not in (0, 1)
            or type(n[4]) not in (int, float) or not math.isfinite(n[4])):
        raise ValueError("NumPy RNG identity mismatch")
    converted = (n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4])
    np.random.RandomState(0).set_state(converted)
    return candidate, converted


def restore_cpu_rng(state, streams):
    import numpy as np
    import torch
    candidate, converted = validate_cpu_rng(state, seed=streams.seed)
    # All formats validated before any global mutation.
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["python"])
    np.random.set_state(converted)
    streams.restore(candidate.snapshot())


def trace_sha256(plan):
    return digest(canonical(plan))
