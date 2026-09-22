"""Independent synthetic-only oracle: mpmath, not production Decimal helpers."""
from fractions import Fraction
import mpmath as mp


def mp_fraction(value):
    return mp.mpf(value.numerator) / mp.mpf(value.denominator)


def candidate_p(s, p):
    with mp.workdps(130):
        x = mp_fraction(s)
        return +min(mp.mpf(1), 2 * mp.exp(-x * x / (2 * p)))


def patient_oracle(cells):
    """Rebuild counts/scores independently without production PatientCounts."""
    scores = [Fraction(b - c, a + b + c + d) for a, b, c, d in cells]
    s = sum(scores, Fraction(0))
    return s, s / len(cells), candidate_p(s, len(cells))


def legacy_p(s, second):
    if second == 0:
        return None
    with mp.workdps(100):
        return +mp.erfc(mp.sqrt(mp_fraction(s * s / second) / 2))


def holm_oracle(pvalues):
    """Independent rank-free closed form: max over all p_j <= p_i.

    On tied p-values the largest eligible multiplier is the first tied rank.
    Uses exact fractions and retains undefined slots as internal ones.
    """
    values = {k: Fraction(1) if v is None else v for k, v in pvalues.items()}
    out = {}
    for key, p in values.items():
        candidates = []
        for q in values.values():
            if q <= p:
                # rank of the first q is 1 + number strictly smaller than q.
                multiplier = len(values) - sum(t < q for t in values.values())
                candidates.append(multiplier * q)
        out[key] = None if pvalues[key] is None else min(Fraction(1), max(candidates))
    return out
