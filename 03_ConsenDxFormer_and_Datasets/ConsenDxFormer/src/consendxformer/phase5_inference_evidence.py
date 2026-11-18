"""Read-only prior-evidence binding. Never runs a campaign or an R process."""
from __future__ import annotations

import csv
from decimal import Decimal, DecimalException
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile

from .hashing import sha256_file
from .phase5_bounded_mean import HYPOTHESIS_IDS
from .phase5_inference_sources import strict_path

REVIEW_ZIP_SHA256 = "6c6c6c0bb308813fdaea2217c131cd34465cbb46e1fae1c408d62ac48c2e1e47"
PACKAGE_MANIFEST_SHA256 = "8c5fd787f35815693d3857df1fef524921779140e3011f45e09f7a269c46dfc5"
CORE_SHA256 = "9fd7927f8497ac54b4db8f0629de3aa1663252f16a3c55a2d79deca196c1c684"
DOCUMENT_HASHES = {
    "phase5_supplemental_evaluation_proposal.md": "4d3f7500071a5c352b2a918c0d26ffabc90b4e11af5982cdfdef7880e1054620",
    "phase5_inference_resolution_proposal.md": "30a15d75f3bb09a6c44680c652100354713a492e2f0a5bfd2df3eed76699a315",
    "phase5_inference_amendment_decision.md": "e3e6232cab56261a126122aa34ad9a6ef2ddde4c22036321349bc509f02dacda",
    "phase5_inference_implementation_authorization.md": "f79711c9e6ca69d297b30e6ac87768930973d5959378ac6c2abe71159a968710",
}
RUN = "validation/phase5_candidate_c/runs/run02/"
R_INPUT = "validation/phase5_candidate_c/runs/r_reference_inputs/"
R_FILES = ("STATUS.txt", "reference_identity.txt", "legacy_results.csv", "holm_results.csv", "binomial_results.csv")
MEMBERS = (RUN + "VALIDATION_STATUS.json", RUN + "artifact_manifest.json",
           "evidence/unit-tests-final.txt", "evidence/independent-readback-run02.json",
           *(R_INPUT + n for n in ("fixture_manifest.json", "legacy_counts.csv", "legacy_expected.csv",
                                  "holm_cases.csv", "binomial_cases.csv")))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def lf_digest(data):
    return digest(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def document_snapshot(root):
    paths = {str(strict_path(Path(root) / "docs" / name)): sha256_file(Path(root) / "docs" / name)
             for name in DOCUMENT_HASHES}
    for name, expected in DOCUMENT_HASHES.items():
        if lf_digest((Path(root) / "docs" / name).read_bytes()) != expected:
            raise ValueError("Governing document changed: " + name)
    core = strict_path(Path(root) / "src/consendxformer/phase5_bounded_mean.py")
    if lf_digest(core.read_bytes()) != CORE_SHA256:
        raise ValueError("Reviewed arithmetic core changed")
    paths[str(core)] = sha256_file(core)
    return paths


def _csv(text, columns):
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != columns:
        raise ValueError("Reference CSV columns mismatch")
    rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed reference CSV")
    return rows


def _close(actual, expected):
    if expected == "":
        return actual == "NA"
    try:
        a, e = float(actual), float(Fraction(expected))
        return math.isfinite(a) and abs(a - e) <= 5e-13 * max(1, abs(e))
    except (ValueError, ZeroDivisionError, OverflowError):
        return False


def _same_count(actual, expected):
    """Compare decimal/scientific count text exactly, never with a tolerance.

    Decimal construction preserves all digits, including beyond float64's
    integer range. Reject nonnumeric spellings, negative values, nonfinite
    values and fractional values before comparing the two integer values.
    """
    values = []
    try:
        for text in (actual, expected):
            if not isinstance(text, str) or not re.fullmatch(
                r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text
            ):
                return False
            value = Decimal(text)
            if not value.is_finite() or value < 0 or value != value.to_integral_value():
                return False
            values.append(value)
        return values[0] == values[1]
    except (DecimalException, ValueError, OverflowError):
        return False


def validate_evidence_payload(payload):
    """Offline checks of copied small evidence; no live files or R re-execution."""
    package_text = payload["package_manifest"]
    if (payload["review_zip_sha256"] != REVIEW_ZIP_SHA256
            or digest(package_text.encode()) != PACKAGE_MANIFEST_SHA256):
        raise ValueError("Reviewed package identity mismatch")
    package = json.loads(package_text)
    members = payload["members"]
    if set(members) != set(MEMBERS):
        raise ValueError("Missing/extra copied evidence")
    for name, text in members.items():
        if digest(text.encode()) != package[name]:
            raise ValueError("Prior evidence member corrupted")
    manifest = json.loads(members[RUN + "artifact_manifest.json"])
    status = json.loads(members[RUN + "VALIDATION_STATUS.json"])
    if any(package[RUN + n] != h for n, h in manifest.items()):
        raise ValueError("Campaign/package manifest disagreement")
    required = {"scope": "synthetic_core_validation_only", "candidate_finite_suite_status": "PASS",
                "candidate_adverse_or_inconclusive_cells": 0, "frozen_settings": 52,
                "additional_exact_fixtures": 6, "simulated_replicates_total": 600000,
                "tail_cells": 2610, "candidate_cells": 870, "research_ready": False,
                "model_forwards": 0, "formal_research_tests_computed": False}
    if any(status.get(k) != v for k, v in required.items()):
        raise ValueError("Prior synthetic validation status differs")
    if status["hypothesis_status"] != {k: "unresolved_not_computed" for k in HYPOTHESIS_IDS}:
        raise ValueError("Historical hypothesis status changed")
    if status["sources"]["src/consendxformer/phase5_bounded_mean.py"] != CORE_SHA256:
        raise ValueError("Evidence does not validate this core")
    compatibility = payload["source_compatibility"]
    if set(compatibility) != set(status["sources"]):
        raise ValueError("Missing prior/current source compatibility ledger")
    for name, expected in status["sources"].items():
        item = compatibility[name]
        current = item["current_lf_sha256"]
        if (item["prior_sha256"] != expected or not re.fullmatch(r"[0-9a-f]{64}", current)
                or item["same"] is not (current == expected)
                or (current != expected and name != "tests/test_phase5_bounded_mean.py")):
            raise ValueError("Prior evidence source compatibility mismatch")
    hashes_text = payload["r_hashes_text"]
    if digest(hashes_text.encode()) != payload["r_hashes_sha256"]:
        raise ValueError("R output hash receipt mismatch")
    hashes, outputs = json.loads(hashes_text), payload["r_outputs"]
    if set(hashes) != set(R_FILES) or set(outputs) != set(R_FILES):
        raise ValueError("Exactly five actual R output hashes/files required")
    for name in R_FILES:
        if not re.fullmatch(r"[0-9a-f]{64}", hashes[name]) or digest(outputs[name].encode()) != hashes[name]:
            raise ValueError("Actual R output hash mismatch")
    lines = outputs["STATUS.txt"].splitlines()
    if lines[:3] != ["R_REFERENCE_COMPARISONS=PASS", "SYNTHETIC_ONLY=True", "RESEARCH_READY=False"]:
        raise ValueError("R reference did not pass or falsely claims research readiness")
    identity = outputs["reference_identity.txt"]
    for token in ("R version 3.6.3", "clust.bin.pair_0.1.2",
                  "e464ac7e5d094bffd25b7c4cbba67820b60f1cc1", "f6523d982dca310c1a76954c9965e36aaa2f9cf1",
                  "R stats p.adjust function:", "R stats binom.test function:"):
        if token not in identity:
            raise ValueError("R runtime/reference identity mismatch")
    legacy = _csv(outputs["legacy_results.csv"], ["case_id", "Q", "p"])
    expected = _csv(members[R_INPUT + "legacy_expected.csv"], ["case_id", "Q", "p"])
    if len(legacy) != 26 or len(expected) != 26 or any(
        a["case_id"] != b["case_id"] or not all(_close(a[k], b[k]) for k in ("Q", "p"))
        for a, b in zip(legacy, expected)
    ):
        raise ValueError("R legacy results differ from pinned synthetic fixtures")
    holm = _csv(outputs["holm_results.csv"], ["case_id", "id", "adjusted", "reject"])
    expected = _csv(members[R_INPUT + "holm_cases.csv"], ["case_id", "id", "raw", "adjusted", "reject"])
    if len(holm) != 90 or len(expected) != 90 or any(
        (a["case_id"], a["id"]) != (b["case_id"], b["id"])
        or not _close(a["adjusted"], b["adjusted"])
        or a["reject"] != ("TRUE" if b["reject"] == "1" else "FALSE") for a, b in zip(holm, expected)
    ):
        raise ValueError("R Holm results differ from pinned synthetic fixtures")
    binomial = _csv(outputs["binomial_results.csv"], ["rejections", "lower", "upper"])
    expected = _csv(members[R_INPUT + "binomial_cases.csv"], ["rejections", "B", "confidence", "lower", "upper"])
    if len(binomial) != 9 or len(expected) != 9 or any(
        not _same_count(a["rejections"], b["rejections"]) or not all(_close(a[k], b[k]) for k in ("lower", "upper"))
        for a, b in zip(binomial, expected)
    ):
        raise ValueError("R binomial results differ from pinned synthetic fixtures")
    return {"prior_campaign": "hash_verified_not_rerun", "r_reference": "outputs_verified_against_pinned_fixtures",
            "r_reexecuted": False, "research_ready": False, "independent_integrated_review": "separate_gate"}


def load_evidence(zip_path, r_directory, hashes_path, hashes_sha256, root):
    """Hash the supplied archive, all payloads and actual R outputs before sources."""
    zip_path, r_directory, hashes_path = map(strict_path, (zip_path, r_directory, hashes_path))
    if sha256_file(zip_path) != REVIEW_ZIP_SHA256 or sha256_file(hashes_path) != hashes_sha256:
        raise ValueError("Supplied validation archive/R hash receipt identity mismatch")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate ZIP members")
        prefix = "phase5_candidate_c_review/"
        for item in archive.infolist():
            p = PurePosixPath(item.filename)
            if (not item.filename.startswith(prefix) or "\\" in item.filename or ":" in item.filename
                    or any(part in ("", ".", "..") for part in item.filename.split("/"))
                    or p.is_absolute() or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe ZIP member")
        text = archive.read(prefix + "PACKAGE_SHA256.json").decode("utf-8")
        if digest(text.encode()) != PACKAGE_MANIFEST_SHA256:
            raise ValueError("Package manifest mismatch")
        package = json.loads(text)
        if set(names) != {prefix + n for n in package} | {prefix + "PACKAGE_SHA256.json"}:
            raise ValueError("Package membership mismatch")
        for name, expected in package.items():
            if digest(archive.read(prefix + name)) != expected:
                raise ValueError("Package payload corruption")
        members = {name: archive.read(prefix + name).decode("utf-8") for name in MEMBERS}
    status = json.loads(members[RUN + "VALIDATION_STATUS.json"])
    # Only the standalone test fixture is adapted. All arithmetic/campaign/oracle
    # sources stay byte-equivalent (LF convention) to prior validated evidence.
    compatibility = {}
    for name, expected in status["sources"].items():
        actual = lf_digest((Path(root) / name).read_bytes())
        if actual != expected and name != "tests/test_phase5_bounded_mean.py":
            raise ValueError("Prior evidence source no longer applicable: " + name)
        compatibility[name] = {"prior_sha256": expected, "current_lf_sha256": actual,
                               "same": actual == expected}
    if {p.name for p in r_directory.iterdir()} != set(R_FILES):
        raise ValueError("R directory must contain exactly the five reference outputs")
    outputs = {n: strict_path(r_directory / n).read_bytes().decode("utf-8") for n in R_FILES}
    payload = {"review_zip_sha256": REVIEW_ZIP_SHA256, "package_manifest": text, "members": members,
               "r_outputs": outputs, "r_hashes_text": hashes_path.read_bytes().decode("utf-8"),
               "r_hashes_sha256": hashes_sha256, "source_compatibility": compatibility}
    validate_evidence_payload(payload)
    paths = (zip_path, hashes_path, *(r_directory / n for n in R_FILES),
             *(strict_path(Path(root) / name) for name in status["sources"]))
    snapshot = {str(p): sha256_file(p) for p in paths}
    return payload, snapshot
