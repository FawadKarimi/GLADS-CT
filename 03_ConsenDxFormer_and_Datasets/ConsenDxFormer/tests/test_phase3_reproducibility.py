from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import consendxformer.reproducibility as reproducibility
from consendxformer.reproducibility import (
    dataloader_worker_seed,
    environment_audit,
    make_dataloader_generator,
    set_global_seed,
    validate_seed,
)


def test_global_seed_repeats_python_and_numpy_sequences():
    first_report = set_global_seed(12345, require_torch=False)
    first = ([random.random() for _ in range(4)], np.random.random(4))
    second_report = set_global_seed(12345, require_torch=False)
    second = ([random.random() for _ in range(4)], np.random.random(4))

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert first_report["python_random_seeded"] is True
    assert second_report["numpy_seeded"] is True
    assert first_report["bitwise_reproducibility_across_environments_claimed"] is False


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises((TypeError, ValueError)):
        validate_seed(seed)


def test_dataloader_worker_seeds_python_and_numpy(monkeypatch):
    class FakeTorch:
        @staticmethod
        def initial_seed():
            return 2**32 + 77

    monkeypatch.setattr(reproducibility, "_import_torch", lambda: FakeTorch())
    worker_seed = dataloader_worker_seed(0)
    first = (random.random(), float(np.random.random()))
    dataloader_worker_seed(0)
    second = (random.random(), float(np.random.random()))

    assert worker_seed == 77
    assert first == second


def test_seeded_dataloader_generator_uses_explicit_seed(monkeypatch):
    class Generator:
        def __init__(self):
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class FakeTorch:
        pass

    FakeTorch.Generator = Generator

    monkeypatch.setattr(reproducibility, "_import_torch", lambda: FakeTorch())
    generator = make_dataloader_generator(90210)
    assert generator.seed == 90210


def test_environment_audit_is_explicit_without_requiring_gpu():
    record = environment_audit(require_torch=False)

    assert record["python"]["version"]
    assert record["numpy"]["version"] == np.__version__
    assert isinstance(record["torch"]["gpu_names"], list)
    assert record["bitwise_reproducibility_across_environments_claimed"] is False
    assert "torch" in record["packages"]
