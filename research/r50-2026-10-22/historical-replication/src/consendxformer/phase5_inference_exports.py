"""Deterministic records and independent arithmetic read-back, without models."""
from fractions import Fraction

import mpmath as mp

from .phase5_bounded_mean import HYPOTHESIS_IDS, pair_correctness, bounded_family
from .phase5_inference_protocol import COMPARATORS
from .phase5_inference_sources import IDENTITY, validate_records


def contrast(key):
    name, seed = key[2:].rsplit("_", 1)
    return COMPARATORS[name], int(seed)


def compute_family(records, expected):
    groups = validate_records(records, expected)
    ids = [tuple(row[k] for k in IDENTITY) for row in expected]
    counts = {}
    for key in HYPOTHESIS_IDS:
        model, seed = contrast(key)
        counts[key] = pair_correctness(groups[("ConsenDxFormer", seed)], groups[(model, seed)],
                                      seed=seed, expected_identities=ids)
    family = bounded_family(counts, precisions=(80, 160, 320))
    hypotheses, patients = [], []
    for key in HYPOTHESIS_IDS:
        item = family[key]
        result, tail = item["result"], item["result"].tail
        model, seed = contrast(key)
        hypotheses.append({"hypothesis_id": key, "comparator": model, "seed": seed,
            "status": "computed", "P": 99, "N": 183, "score_sum": str(result.score_sum),
            "patient_mean": str(result.patient_mean), "nodule_mean_descriptive": str(result.nodule_mean),
            "nonzero_patients": result.nonzero_patient_count, "discordant_patients": result.discordant_patient_count,
            "cancelled_patients": result.cancellation_patient_count, "exponent": str(tail.exponent),
            "precision": tail.precision, "log_lower": str(tail.log_lower), "log_upper": str(tail.log_upper),
            "raw_lower": str(tail.p_lower), "raw_upper": str(tail.p_upper), "rank": item["rank"],
            "adjusted_lower": str(item["adjusted_lower"]), "adjusted_upper": str(item["adjusted_upper"]),
            "reject": item["reject"]})
        patients.extend({"hypothesis_id": key, "patient_id": row.patient_id, "a": row.a, "b": row.b,
                         "c": row.c, "d": row.d, "n": row.n, "u": str(row.u)} for row in result.counts)
    return {"hypotheses": hypotheses, "patients": patients}


def validate_arithmetic(records, expected, result):
    """Rebuild integer cells and rank-free Holm independently of core functions.

    mpmath uses precision exceeding the emitted Decimal enclosure by 60 digits.
    This is deterministic read-back, not a new inference method or calibration.
    """
    groups = validate_records(records, expected)
    if set(result) != {"hypotheses", "patients"} or len(result["hypotheses"]) != 15:
        raise ValueError("Incomplete family export")
    hypotheses = result["hypotheses"]
    if [r["hypothesis_id"] for r in hypotheses] != list(HYPOTHESIS_IDS):
        raise ValueError("Hypothesis IDs/order differ")
    if any(type(r["precision"]) is not int or r["precision"] not in (80, 160, 320) for r in hypotheses):
        raise ValueError("Unreviewed numerical precision")
    expected_patients, sums, nodule_sums, diagnostics = [], {}, {}, {}
    for key in HYPOTHESIS_IDS:
        model, seed = contrast(key)
        a = {r.nodule_id: r for r in groups[("ConsenDxFormer", seed)]}
        b = {r.nodule_id: r for r in groups[(model, seed)]}
        cells = {}
        for row in expected:
            ca = a[row["nodule_id"]].predicted_class == row["true_class"]
            cb = b[row["nodule_id"]].predicted_class == row["true_class"]
            cell = cells.setdefault(row["patient_id"], [0, 0, 0, 0])
            cell[{(True, True): 0, (True, False): 1, (False, True): 2, (False, False): 3}[(ca, cb)]] += 1
        if len(cells) != 99 or sum(map(sum, cells.values())) != 183:
            raise ValueError("Read-back cohort differs")
        sums[key] = sum((Fraction(c[1] - c[2], sum(c)) for c in cells.values()), Fraction(0))
        nodule_sums[key] = Fraction(sum(c[1] - c[2] for c in cells.values()), 183)
        diagnostics[key] = (sum(c[1] != c[2] for c in cells.values()),
                            sum(c[1] + c[2] > 0 for c in cells.values()),
                            sum(c[1] == c[2] and c[1] > 0 for c in cells.values()))
        expected_patients.extend({"hypothesis_id": key, "patient_id": p, "a": c[0], "b": c[1],
            "c": c[2], "d": c[3], "n": sum(c), "u": str(Fraction(c[1] - c[2], sum(c)))}
            for p, c in sorted(cells.items()))
    if result["patients"] != expected_patients:
        raise ValueError("Patient cell/score export differs from original paired predictions")
    def number(value):
        f = Fraction(value)
        return mp.mpf(f.numerator) / f.denominator
    with mp.workdps(max(r["precision"] for r in hypotheses) + 60):
        ps = {k: min(mp.mpf(1), 2 * mp.exp(-number(s) ** 2 / 198)) for k, s in sums.items()}
        order = sorted(HYPOTHESIS_IDS, key=lambda k: (ps[k], k))
        # Rank-free definition is separate from the core's sorted cumulative max.
        adjusted = {k: min(mp.mpf(1), max((15 - sum(p < q for p in ps.values())) * q
                           for q in ps.values() if q <= ps[k])) for k in HYPOTHESIS_IDS}
        for row in hypotheses:
            key = row["hypothesis_id"]
            model, seed = contrast(key)
            if any(type(row[k]) is not int for k in ("seed", "P", "N", "rank", "nonzero_patients",
                                                      "discordant_patients", "cancelled_patients")):
                raise ValueError("Integer export fields must not be coerced")
            if (not 0 < Fraction(row["raw_lower"]) <= Fraction(row["raw_upper"]) <= 1
                    or not 0 < Fraction(row["adjusted_lower"]) <= Fraction(row["adjusted_upper"]) <= 1
                    or not Fraction(row["log_lower"]) <= Fraction(row["log_upper"]) <= 0):
                raise ValueError("Invalid numerical enclosure range")
            if (row["comparator"] != model or row["seed"] != seed or row["status"] != "computed"
                    or row["P"] != 99 or row["N"] != 183
                    or Fraction(row["score_sum"]) != sums[key]
                    or Fraction(row["patient_mean"]) != sums[key] / 99
                    or Fraction(row["nodule_mean_descriptive"]) != nodule_sums[key]
                    or Fraction(row["exponent"]) != sums[key] ** 2 / 198
                    or (row["nonzero_patients"], row["discordant_patients"], row["cancelled_patients"]) != diagnostics[key]
                    or row["rank"] != order.index(key) + 1):
                raise ValueError("Family point estimate/identity/rank mismatch")
            for low, high, value in (("raw_lower", "raw_upper", ps[key]),
                                     ("adjusted_lower", "adjusted_upper", adjusted[key]),
                                     ("log_lower", "log_upper", mp.log(ps[key]))):
                if not number(row[low]) <= value <= number(row[high]):
                    raise ValueError("Independent oracle outside exported enclosure")
            if type(row["reject"]) is not bool or row["reject"] != bool(adjusted[key] <= mp.mpf(1) / 20):
                raise ValueError("Independent Holm decision mismatch")
            lo, hi = Fraction(row["adjusted_lower"]), Fraction(row["adjusted_upper"])
            if not (hi <= Fraction(1, 20) if row["reject"] else lo > Fraction(1, 20)):
                raise ValueError("Unresolved exported decision enclosure")
