"""Candidate C arithmetic only. No research loader, runner, I/O or model imports.

Implements the accepted conditional independent heterogeneous-patient mean null.
The caller must supply already validated paired counts. This module cannot
establish source validity, patient independence, or authorization for research.
All scores are rational; transcendental tails are enclosed with directed
Decimal arithmetic. Unresolved numerical decisions raise, never change alpha.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    Decimal, Context, DecimalException, ROUND_FLOOR, ROUND_CEILING,
    ROUND_HALF_EVEN, localcontext, Underflow, Subnormal, Clamped,
)
from fractions import Fraction
from typing import Mapping, Sequence, Optional

SEEDS = (42, 43, 44)
HYPOTHESIS_IDS = tuple(
    f"C_{comparator}_{seed}"
    for comparator in ("B1", "attention_off", "disagreement_off", "ordinal_off", "fusion_off")
    for seed in SEEDS
)
FAMILY_ALPHA = Fraction(1, 20)
METHOD_ID = "hoeffding_equal_patient_two_sided_v1"
NULL_ID = "conditional_independent_heterogeneous_patient_mean_zero_v1"


class InputIntegrityError(ValueError):
    """Malformed counts, identities, or family; no deletion/fallback allowed."""


class NumericalFailure(ArithmeticError):
    """Tail or decision could not be enclosed at permitted precision."""


@dataclass(frozen=True)
class PatientCounts:
    patient_id: str
    a: int
    b: int
    c: int
    d: int

    def __post_init__(self) -> None:
        if not isinstance(self.patient_id, str) or not self.patient_id or self.patient_id.strip() != self.patient_id:
            raise InputIntegrityError("patient_id must be nonblank canonical text")
        if any(type(v) is not int or v < 0 for v in (self.a, self.b, self.c, self.d)):
            raise InputIntegrityError("Count cells must be nonnegative Python integers, not bool/float")
        if self.n <= 0:
            raise InputIntegrityError("Every patient must retain at least one nodule")

    @property
    def n(self) -> int:
        return self.a + self.b + self.c + self.d

    @property
    def u(self) -> Fraction:
        return Fraction(self.b - self.c, self.n)


@dataclass(frozen=True)
class TailBounds:
    exponent: Fraction
    patient_count: int
    precision: int
    log_lower: Decimal
    log_upper: Decimal
    p_lower: Decimal
    p_upper: Decimal


@dataclass(frozen=True)
class MeanResult:
    counts: tuple[PatientCounts, ...]
    score_sum: Fraction
    patient_mean: Fraction
    nodule_mean: Fraction
    nodule_count: int
    nonzero_patient_count: int
    discordant_patient_count: int
    cancellation_patient_count: int
    tail: TailBounds


def _context(precision: int, rounding: str = ROUND_HALF_EVEN) -> Context:
    if type(precision) is not int or not 8 <= precision <= 4096:
        raise InputIntegrityError("precision must be an integer in [8,4096]")
    ctx = Context(prec=precision, rounding=rounding, Emin=-999999, Emax=999999)
    for signal in (Underflow, Subnormal, Clamped):
        ctx.traps[signal] = True
    return ctx


def fraction_decimal(value: Fraction, precision: int, rounding: str) -> Decimal:
    ctx = _context(precision, rounding)
    return ctx.divide(Decimal(value.numerator), Decimal(value.denominator))


def bounded_tail(score_sum: Fraction, patient_count: int, *, precision: int = 80) -> TailBounds:
    """Enclose min(1,2 exp(-S^2/(2P))); P includes zero/cancelled scores.

    Decimal ln/exp are correctly rounded to nearest. Adjacent representable
    numbers enclose them; all subsequent operations use directed rounding.
    The returned upper endpoint is a conservative numerical representation,
    not an estimated/tighter statistical bound. No floating-point input accepted.
    """
    if type(patient_count) is not int or patient_count <= 0:
        raise InputIntegrityError("patient_count must be a positive integer")
    if not isinstance(score_sum, Fraction) or abs(score_sum) > patient_count:
        raise InputIntegrityError("score_sum must be an exact Fraction in [-P,P]")
    q = score_sum * score_sum / (2 * patient_count)
    try:
        mid = _context(precision)
        low = _context(precision, ROUND_FLOOR)
        high = _context(precision, ROUND_CEILING)
        ln2 = mid.ln(Decimal(2))
        qlo = fraction_decimal(q, precision, ROUND_FLOOR)
        qhi = fraction_decimal(q, precision, ROUND_CEILING)
        loglo = min(Decimal(0), low.subtract(mid.next_minus(ln2), qhi))
        loghi = min(Decimal(0), high.subtract(mid.next_plus(ln2), qlo))
        plo = Decimal(1) if loglo == 0 else max(Decimal(0), mid.next_minus(mid.exp(loglo)))
        phi = Decimal(1) if loghi == 0 else min(Decimal(1), mid.next_plus(mid.exp(loghi)))
        if not (Decimal(0) < plo <= phi <= Decimal(1)) or not loglo <= loghi <= 0:
            raise NumericalFailure("Invalid or underflowed probability enclosure")
        return TailBounds(q, patient_count, precision, loglo, loghi, plo, phi)
    except DecimalException as exc:
        raise NumericalFailure("Decimal enclosure failed") from exc


def bounded_patient_mean(counts: Sequence[PatientCounts], *, precision: int = 80) -> MeanResult:
    """Count-based core; generic P supports synthetic unit fixtures.

    Future research integration must enforce frozen P=99, N=183 and the exact
    source/pairing identities BEFORE calling this core. It is not provided here.
    """
    rows = tuple(counts)
    if not rows or any(not isinstance(row, PatientCounts) for row in rows):
        raise InputIntegrityError("A nonempty sequence of PatientCounts is required")
    ids = [row.patient_id for row in rows]
    if len(set(ids)) != len(ids):
        raise InputIntegrityError("Duplicate patient_id")
    rows = tuple(sorted(rows, key=lambda row: row.patient_id))
    s = sum((row.u for row in rows), Fraction(0))
    n = sum(row.n for row in rows)
    return MeanResult(
        rows, s, s / len(rows), Fraction(sum(row.b - row.c for row in rows), n), n,
        sum(row.b != row.c for row in rows), sum(row.b + row.c > 0 for row in rows),
        sum(row.b == row.c and row.b + row.c > 0 for row in rows),
        bounded_tail(s, len(rows), precision=precision),
    )


def decide_tail(score_sum: Fraction, patient_count: int, alpha: Fraction, *,
                precisions: tuple[int, ...] = (80, 160, 320)) -> tuple[bool, TailBounds]:
    """Decide p<=alpha only when the entire numerical enclosure resolves it."""
    if not isinstance(alpha, Fraction) or not 0 < alpha < 1 or not precisions:
        raise InputIntegrityError("Exact alpha in (0,1) and nonempty precision schedule required")
    for precision in precisions:
        tail = bounded_tail(score_sum, patient_count, precision=precision)
        if Fraction(tail.p_upper) <= alpha:
            return True, tail
        if Fraction(tail.p_lower) > alpha:
            return False, tail
    raise NumericalFailure("Decision straddles alpha; no relaxed threshold or partial family")


def holm_exact(pvalues: Mapping[str, Optional[Fraction]]) -> dict[str, dict]:
    """Exact rational Holm bookkeeping on all 15 slots, ties by ID.

    None is a prespecified per-slot statistical undefined placeholder ONLY.
    It never represents a source error or method-level block. Such failures
    must be raised by the caller and must prevent family publication.
    """
    if set(pvalues) != set(HYPOTHESIS_IDS) or len(pvalues) != 15:
        raise InputIntegrityError("Exactly the frozen 15 hypothesis IDs are required")
    for value in pvalues.values():
        if value is not None and (not isinstance(value, Fraction) or not 0 <= value <= 1):
            raise InputIntegrityError("p-values must be exact Fractions in [0,1] or None")
    work = {key: Fraction(1) if value is None else value for key, value in pvalues.items()}
    order = sorted(HYPOTHESIS_IDS, key=lambda key: (work[key], key))
    running = Fraction(0)
    output = {}
    for rank, key in enumerate(order, 1):
        running = max(running, (16 - rank) * work[key])
        adjusted = min(Fraction(1), running)
        undefined = pvalues[key] is None
        output[key] = {
            "rank": rank, "raw": pvalues[key],
            "adjusted": None if undefined else adjusted,
            "reject": False if undefined else adjusted <= FAMILY_ALPHA,
            "bookkeeping_placeholder": undefined,
        }
    return {key: output[key] for key in HYPOTHESIS_IDS}


def bounded_family(counts_by_id: Mapping[str, Sequence[PatientCounts]], *,
                   precisions: tuple[int, ...] = (80, 160, 320)) -> dict[str, dict]:
    """Enclosed Holm decisions for a complete valid count-level synthetic family.

    This is NOT a research entry point: source loaders, authorizations and
    original-source provenance are intentionally not supplied in this tranche.
    Monotonicity of Holm gives an enclosure by adjusting both endpoint vectors.
    """
    if set(counts_by_id) != set(HYPOTHESIS_IDS) or len(counts_by_id) != 15:
        raise InputIntegrityError("Missing/extra hypothesis slots")
    if not precisions:
        raise InputIntegrityError("Empty precision schedule")
    for precision in precisions:
        results = {key: bounded_patient_mean(counts_by_id[key], precision=precision) for key in HYPOTHESIS_IDS}
        signatures = {tuple((row.patient_id, row.n) for row in result.counts) for result in results.values()}
        if len(signatures) != 1:
            raise InputIntegrityError("The family must share all patient identities and nodule counts")
        if any(res.tail.log_lower < 0 and res.tail.log_upper == 0 for res in results.values()):
            # Exact ordering against the p=1 plateau is numerically unresolved.
            continue
        lower = holm_exact({key: Fraction(res.tail.p_lower) for key, res in results.items()})
        upper = holm_exact({key: Fraction(res.tail.p_upper) for key, res in results.items()})
        output = {}
        unresolved = False
        # Formula order is exact: all saturated p=1 values tie; otherwise q descending.
        formula_order = sorted(HYPOTHESIS_IDS, key=lambda key: (
            1 if results[key].tail.p_lower == 1 else 0,
            Fraction(0) if results[key].tail.p_lower == 1 else -results[key].tail.exponent,
            key,
        ))
        for key in HYPOTHESIS_IDS:
            lo, hi = lower[key]["adjusted"], upper[key]["adjusted"]
            if hi <= FAMILY_ALPHA:
                reject = True
            elif lo > FAMILY_ALPHA:
                reject = False
            else:
                unresolved = True
                break
            output[key] = {
                "result": results[key], "rank": formula_order.index(key) + 1,
                "adjusted_lower": lo, "adjusted_upper": hi, "reject": reject,
            }
        if not unresolved:
            return output
    raise NumericalFailure("At least one Holm decision unresolved; entire family blocked")


@dataclass(frozen=True)
class NodulePrediction:
    patient_id: str
    nodule_id: str
    series_instance_uid: str
    split: str
    true_class: str
    predicted_class: str
    seed: int

    def __post_init__(self) -> None:
        for value in (self.patient_id, self.nodule_id, self.series_instance_uid):
            if not isinstance(value, str) or not value or value != value.strip():
                raise InputIntegrityError("Noncanonical prediction identity")
        if self.split != 'test' or type(self.seed) is not int or self.seed not in SEEDS:
            raise InputIntegrityError("Wrong split/seed")
        names = ('benign', 'indeterminate', 'malignant')
        if self.true_class not in names or self.predicted_class not in names:
            raise InputIntegrityError("Unknown class")

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (self.patient_id, self.nodule_id, self.series_instance_uid, self.split, self.true_class)


def pair_correctness(complete: Sequence[NodulePrediction], comparator: Sequence[NodulePrediction], *,
                     seed: int, expected_identities: Sequence[tuple[str, str, str, str, str]]) -> tuple[PatientCounts, ...]:
    """Pure exact pairing AFTER future historical source-loader validation.

    Expected identities must come from that trusted source contract, not a
    convenient inner join. This function alone does NOT authenticate sources.
    """
    if type(seed) is not int or seed not in SEEDS:
        raise InputIntegrityError("Unknown seed")
    expected = tuple(expected_identities)
    if any(not isinstance(x, tuple) or len(x) != 5
           or any(not isinstance(v, str) or not v or v != v.strip() for v in x)
           or x[3] != "test" or x[4] not in ("benign", "indeterminate", "malignant")
           for x in expected):
        raise InputIntegrityError("Malformed expected identities")
    if (not expected or len(set(expected)) != len(expected)
            or len({x[1] for x in expected}) != len(expected)):
        raise InputIntegrityError("Empty or duplicate expected identities")
    maps = []
    for group in (complete, comparator):
        rows = tuple(group)
        if any(not isinstance(row, NodulePrediction) or row.seed != seed for row in rows):
            raise InputIntegrityError("Mixed seeds or invalid records")
        mapping = {row.nodule_id: row for row in rows}
        if len(mapping) != len(rows) or {row.identity for row in rows} != set(expected):
            raise InputIntegrityError("Missing/extra/duplicate/unpaired identities or labels")
        maps.append(mapping)
    cells = {}
    for identity in expected:
        patient, nodule, _, _, label = identity
        correct_a = maps[0][nodule].predicted_class == label
        correct_b = maps[1][nodule].predicted_class == label
        index = 0 if correct_a and correct_b else 1 if correct_a else 2 if correct_b else 3
        cells.setdefault(patient, [0, 0, 0, 0])[index] += 1
    return tuple(PatientCounts(patient, *cells[patient]) for patient in sorted(cells))
