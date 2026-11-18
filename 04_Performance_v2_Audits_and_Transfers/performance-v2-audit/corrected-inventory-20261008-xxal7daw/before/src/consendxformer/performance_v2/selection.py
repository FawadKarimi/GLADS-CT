"""Pure validation-only selection. No checkpoint file loading or metric I/O."""
from dataclasses import dataclass
import math

OBJECTIVE_ORDER = ("HARD", "LEGACY", "SHARED-010", "SEPARATE-010")
INITIALIZATION_ORDER = ("RANDOM-A", "IMAGENET-A", "RADIMAGENET-A")


@dataclass(frozen=True)
class ValidationEpoch:
    epoch: int
    macro_f1: float
    diagnostic_ce: float
    split: str = "validation"

    def __post_init__(self):
        if self.split != "validation":
            raise ValueError("only validation records may select checkpoints")
        if type(self.epoch) is not int or not 1 <= self.epoch <= 100:
            raise ValueError("invalid epoch")
        if (isinstance(self.macro_f1, bool) or not math.isfinite(self.macro_f1)
                or not 0 <= self.macro_f1 <= 1 or isinstance(self.diagnostic_ce, bool)
                or not math.isfinite(self.diagnostic_ce) or self.diagnostic_ce < 0):
            raise ValueError("invalid diagnostic validation metrics")


def select_checkpoint(history):
    records = list(history)
    if not records or any(type(r) is not ValidationEpoch for r in records):
        raise ValueError("typed validation history required")
    if len({r.epoch for r in records}) != len(records):
        raise ValueError("duplicate epoch")
    maximum = max(r.macro_f1 for r in records)
    candidates = [r for r in records if maximum - r.macro_f1 <= 1e-12]
    return min(candidates, key=lambda r: (r.diagnostic_ce, r.epoch))


class ValidationPatience:
    """Raw min-delta counter, independent of tolerance-ranked checkpoint choice."""
    def __init__(self):
        self.last_epoch = 0
        self.best = -math.inf
        self.without_improvement = 0
        self.stopped = False

    def observe(self, record):
        if (type(record) is not ValidationEpoch or self.stopped
                or record.epoch != self.last_epoch + 1):
            raise ValueError("sequential validation epochs required before stopping")
        self.last_epoch = record.epoch
        if record.macro_f1 > self.best + 1e-4:
            self.best = record.macro_f1
            self.without_improvement = 0
        else:
            self.without_improvement += 1
        self.stopped = (record.epoch == 100 or
                        (record.epoch >= 20 and self.without_improvement >= 15))
        return self.stopped


@dataclass(frozen=True)
class SeedValidation:
    seed: int
    checkpoint: ValidationEpoch
    complete: bool

    def __post_init__(self):
        if (type(self.seed) is not int or self.seed not in (42, 43, 44)
                or type(self.checkpoint) is not ValidationEpoch or self.complete is not True):
            raise ValueError("complete validation-selected seed result required")


def _rank(scores, order):
    if not scores or not set(scores) <= set(order):
        raise ValueError("unknown/empty candidate set")
    maximum = max(v[0] for v in scores.values())
    eligible = [name for name, (f1, _) in scores.items() if maximum - f1 <= 1e-4]
    winner = min(eligible, key=lambda name: (scores[name][1], order.index(name)))
    return {"selected": winner, "exact_maximum": maximum,
            "selected_macro_f1": scores[winner][0], "scores": dict(scores)}


def select_screen(candidates, *, kind, eligible_initializations=None):
    if kind not in ("initialization", "objective"):
        raise ValueError("unknown screening phase")
    if kind == "initialization":
        if (not isinstance(eligible_initializations, (list, tuple))
                or not eligible_initializations
                or len(eligible_initializations) != len(set(eligible_initializations))
                or not set(eligible_initializations) <= set(INITIALIZATION_ORDER)):
            raise ValueError("declare all provenance-eligible initialization arms")
        required = set(eligible_initializations)
    else:
        if eligible_initializations is not None:
            raise ValueError("objective screens cannot override the frozen arm set")
        required = set(OBJECTIVE_ORDER)
    if set(candidates) != required:
        raise ValueError("all declared eligible screens required; no survivor selection")
    scores = {}
    for name, record in candidates.items():
        if type(record) is not SeedValidation or record.seed != 42:
            raise ValueError("screen requires complete seed42 records")
        scores[name] = (record.checkpoint.macro_f1, record.checkpoint.diagnostic_ce)
    return _rank(scores, INITIALIZATION_ORDER if kind == "initialization" else OBJECTIVE_ORDER)


def select_repeated_family(families):
    if len(families) != 2:
        raise ValueError("exactly nominated candidate and control required")
    scores = {}
    for name, records in families.items():
        records = list(records)
        if (len(records) != 3 or any(type(r) is not SeedValidation for r in records)
                or {r.seed for r in records} != {42, 43, 44}):
            raise ValueError("both families require all three complete seeds")
        scores[name] = (math.fsum(r.checkpoint.macro_f1 for r in records) / 3,
                        math.fsum(r.checkpoint.diagnostic_ce for r in records) / 3)
    return _rank(scores, OBJECTIVE_ORDER)
