"""Invented records only; leaf-loader/Git/evidence trust-anchor stubs are explicit.

No source paths from the study, actual R outputs, models or calibration runs.
The historical full/B1 aggregate loader and new integration guards run unchanged.
"""
from copy import deepcopy
import csv
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.phase3_artifacts import atomic_write_json
from consendxformer.phase5_bounded_mean import HYPOTHESIS_IDS, NumericalFailure
from consendxformer.phase5_descriptive import CLASS_ORDER, MODELS, SEEDS
import consendxformer.phase4_evaluation_runner as historical
import consendxformer.phase5_inference_sources as sources
import consendxformer.phase5_inference_evidence as evidence
import consendxformer.phase5_inference_protocol as protocol
import consendxformer.phase5_inference_exports as exports
import consendxformer.phase5_inference_runner as runner
from test_phase4_evaluation_runner import _descriptor


def forbidden(*args, **kwargs):
    raise AssertionError("Forbidden model/old-analysis/statistics path")


@pytest.fixture(autouse=True)
def no_models_or_reference_processes(monkeypatch):
    import subprocess
    import torch
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def fake_evidence(tmp_path, monkeypatch):
    """Small invented archive + fake R transcript, NOT actual validation evidence.

    Test-only SHA trust anchors replace the external reviewed ZIP pins. All
    content, receipt, membership and numerical-reference checks still execute.
    """
    prior_sources = {"src/consendxformer/phase5_bounded_mean.py": evidence.CORE_SHA256,
                     "tests/test_phase5_bounded_mean.py": "0" * 64}
    status = {"scope": "synthetic_core_validation_only", "candidate_finite_suite_status": "PASS",
        "candidate_adverse_or_inconclusive_cells": 0, "frozen_settings": 52,
        "additional_exact_fixtures": 6, "simulated_replicates_total": 600000,
        "tail_cells": 2610, "candidate_cells": 870, "research_ready": False,
        "model_forwards": 0, "formal_research_tests_computed": False,
        "hypothesis_status": {k: "unresolved_not_computed" for k in HYPOTHESIS_IDS},
        "sources": prior_sources, "fixture_warning": "INVENTED STATUS, NO CAMPAIGN RAN"}
    members = {n: "invented fixture only\n" for n in evidence.MEMBERS}
    members[evidence.RUN + "VALIDATION_STATUS.json"] = json.dumps(status)
    members[evidence.RUN + "artifact_manifest.json"] = json.dumps({"VALIDATION_STATUS.json":
        evidence.digest(members[evidence.RUN + "VALIDATION_STATUS.json"].encode())})
    legacy = [{"case_id": f"fake_{i}", "Q": "" if i == 0 else "0", "p": "" if i == 0 else "1"}
              for i in range(26)]
    holm = [{"case_id": f"fake_{i}", "id": k, "raw": "1", "adjusted": "1", "reject": "0"}
            for i in range(6) for k in HYPOTHESIS_IDS]
    binomial = [{"rejections": str(i), "B": "100000", "confidence": "1/2", "lower": "0", "upper": "1"}
                for i in range(9)]
    members[evidence.R_INPUT + "legacy_expected.csv"] = csv_text(legacy, ["case_id", "Q", "p"])
    members[evidence.R_INPUT + "holm_cases.csv"] = csv_text(holm, ["case_id", "id", "raw", "adjusted", "reject"])
    members[evidence.R_INPUT + "binomial_cases.csv"] = csv_text(binomial, ["rejections", "B", "confidence", "lower", "upper"])
    package_text = json.dumps({k: evidence.digest(v.encode()) for k, v in members.items()}, sort_keys=True)
    archive_path = tmp_path / "FAKE-review.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for k, v in members.items(): archive.writestr("phase5_candidate_c_review/" + k, v)
        archive.writestr("phase5_candidate_c_review/PACKAGE_SHA256.json", package_text)
    monkeypatch.setattr(evidence, "REVIEW_ZIP_SHA256", sha256_file(archive_path))
    monkeypatch.setattr(protocol, "REVIEW_ZIP_SHA256", sha256_file(archive_path))
    monkeypatch.setattr(evidence, "PACKAGE_MANIFEST_SHA256", evidence.digest(package_text.encode()))
    rdir = tmp_path / "FAKE-R-outputs"
    rdir.mkdir()
    outputs = {
        "STATUS.txt": "R_REFERENCE_COMPARISONS=PASS\nSYNTHETIC_ONLY=True\nRESEARCH_READY=False\nFAKE TEST TRANSCRIPT\n",
        "reference_identity.txt": "FAKE TEST IDENTITY, NO R PROCESS\nR version 3.6.3\nclust.bin.pair_0.1.2\n"
            "e464ac7e5d094bffd25b7c4cbba67820b60f1cc1\nf6523d982dca310c1a76954c9965e36aaa2f9cf1\n"
            "R stats p.adjust function:\nR stats binom.test function:\n",
        "legacy_results.csv": csv_text([{k: v or "NA" for k, v in row.items()} for row in legacy], ["case_id", "Q", "p"]),
        "holm_results.csv": csv_text([{k: ("FALSE" if k == "reject" else r[k]) for k in ("case_id", "id", "adjusted", "reject")}
                                      for r in holm], ["case_id", "id", "adjusted", "reject"]),
        "binomial_results.csv": csv_text([{k: r[k] for k in ("rejections", "lower", "upper")} for r in binomial],
                                           ["rejections", "lower", "upper"]),
    }
    for name, text in outputs.items(): (rdir / name).write_bytes(text.encode())
    receipt = tmp_path / "FAKE-R-hashes.json"
    receipt.write_text(json.dumps({n: sha256_file(rdir / n) for n in evidence.R_FILES}), encoding="utf-8")
    return dict(core_evidence_zip=archive_path, r_reference_dir=rdir, r_reference_hashes=receipt,
                r_reference_hashes_sha256=sha256_file(receipt))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    patient = [i // 2 if i < 168 else i - 84 for i in range(183)]
    ids = pd.DataFrame({"patient_id": [f"fake-p{i:03}" for i in patient],
        "nodule_id": [f"fake-n{i:03}" for i in range(183)],
        "series_instance_uid": [f"fake-s{i:03}" for i in patient], "split": ["test"] * 183,
        "true_class": [CLASS_ORDER[i % 3] for i in range(183)]})
    supervision = ids.rename(columns={"true_class": "label"})
    objects = {}
    for model in MODELS:
        for seed in SEEDS:
            path = tmp_path / "invented-sources" / f"{model}-{seed}"
            (path / "checkpoints").mkdir(parents=True)
            (path / "predictions").mkdir()
            (path / "checkpoints/best.pt").write_bytes(b"NOT A TORCH CHECKPOINT, INVENTED SENTINEL")
            frame = ids.copy()
            chosen = np.array([(i + MODELS.index(model) + seed - 42) % 3 for i in range(183)])
            probs = np.full((183, 3), .125, dtype=np.float32)
            probs[np.arange(183), chosen] = .75
            frame["predicted_class"] = [CLASS_ORDER[j] for j in chosen]
            for j, name in enumerate(CLASS_ORDER):
                frame[f"prob_{name}"] = probs[:, j]
                frame[f"logit_{name}"] = np.log(probs[:, j])
            frame.to_csv(path / "predictions/test.csv", index=False)
            descriptor = _descriptor(seed, model=model)
            descriptor["phase3_input_fingerprint"] = {"fixture": "synthetic_shared"}
            obj = SimpleNamespace(source_dir=path, descriptor=descriptor, predictions=frame,
                supervision=supervision.copy(deep=True), source_snapshot={str(p): sha256_file(p)
                for p in path.rglob("*") if p.is_file()})
            if model not in MODELS[:2]:
                obj.identity = {"variant": model.removesuffix("_off"), "seed": seed,
                    "full_model_reference": sources.reference_record(objects[("ConsenDxFormer", seed)])}
                obj.predictions = {"test": frame}
            objects[(model, seed)] = obj
    # Only completed leaf source loaders are stubbed. The actual full/B1 loader,
    # its source-descriptor/fingerprint/cohort guards and new barrier all execute.
    calls = []
    def leaf(path, role):
        calls.append((role, Path(path)))
        obj = next(o for o in objects.values() if o.source_dir == Path(path))
        assert (obj.descriptor["model"] == role) if role != "OFF" else obj.descriptor["model"].endswith("_off")
        return obj
    monkeypatch.setattr(historical, "load_phase4_final_prediction_source", lambda p: leaf(p, "ConsenDxFormer"))
    monkeypatch.setattr(historical, "load_b1_prediction_source", lambda p: leaf(p, "B1"))
    monkeypatch.setattr(sources, "load_ablation_source", lambda p: leaf(p, "OFF"))
    monkeypatch.setattr(runner, "validate_evaluator_repository", lambda root: {"commit": "b" * 40, "fixture_git_stub": True})
    monkeypatch.setattr(runner, "NAMESPACE", tmp_path / "only-new-outputs")
    kwargs = fake_evidence(tmp_path, monkeypatch)
    policy = tmp_path / "FAKE-trust-anchor-protocol.json"
    atomic_write_json(policy, protocol.protocol())
    monkeypatch.setattr(runner, "PROTOCOL_PATH", policy)
    kwargs.update(full_source_dirs=[objects[(MODELS[0], s)].source_dir for s in SEEDS],
        b1_source_dirs=[objects[(MODELS[1], s)].source_dir for s in SEEDS],
        ablation_source_dirs=[objects[(m, s)].source_dir for m in MODELS[2:] for s in SEEDS],
        authorize_preflight=True)
    return SimpleNamespace(kwargs=kwargs, objects=objects, calls=calls, tmp=tmp_path)


def run_kwargs(setup, compute=False):
    kwargs = dict(setup.kwargs)
    if compute:
        kwargs.update(preflight_only=False, authorize_preflight=False, authorize_research_statistics=True)
    return kwargs


def test_preflight_zero_statistics_and_zero_forwards(setup, monkeypatch):
    monkeypatch.setattr(runner, "compute_family", forbidden)
    monkeypatch.setattr(exports, "pair_correctness", forbidden)
    import torch
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)  # fake sentinel files are not checkpoints
    import consendxformer.phase4_bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "bootstrap_model_metrics", forbidden)
    before = {str(p): sha256_file(p) for p in (setup.tmp / "invented-sources").rglob("*") if p.is_file()}
    output = runner.run_inference(**setup.kwargs)
    marker = runner.validate_completion(output)
    assert marker["preflight_completed"] and not marker["subset_completed"]
    assert not marker["formal_tests_computed"] and not marker["research_readiness_certified"]
    assert not (output / "COMPLETED.json").exists()
    assert len(setup.calls) == 18
    assert [r for r, _ in setup.calls].count("OFF") == 12
    assert all(sha256_file(Path(p)) == h for p, h in before.items())
    assert all(r["status"] == "unresolved_not_computed" for r in runner.read_json(output / "inference.json")["hypotheses"])


def test_synthetic_family_deterministic_export_and_independent_readback(setup, monkeypatch):
    import torch
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    first = runner.run_inference(**run_kwargs(setup, True))
    result = runner.read_json(first / "inference.json")
    assert len(result["hypotheses"]) == 15 and len(result["patients"]) == 1485
    assert {r["P"] for r in result["hypotheses"]} == {99}
    assert {r["N"] for r in result["hypotheses"]} == {183}
    assert any(r["reject"] for r in result["hypotheses"])
    assert any(not r["reject"] for r in result["hypotheses"])
    kwargs = run_kwargs(setup, True)
    kwargs["output_root"] = runner.NAMESPACE / "second-container"
    second = runner.run_inference(**kwargs)
    assert first.name == second.name
    assert {p.relative_to(first).as_posix(): p.read_bytes() for p in first.rglob("*") if p.is_file()} == {
        p.relative_to(second).as_posix(): p.read_bytes() for p in second.rglob("*") if p.is_file()}
    monkeypatch.setattr(exports, "bounded_family", forbidden)
    monkeypatch.setattr(exports, "pair_correctness", forbidden)
    marker = runner.validate_completion(first)
    assert marker["subset_completed"] and not marker["full_supplemental_completion"] and not marker["stage_a_complete"]
    assert not marker["research_readiness_certified"]
    with pytest.raises(FileExistsError): runner.run_inference(**run_kwargs(setup, True))


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "seed", "variant", "label", "identity", "split",
                                       "reference", "fingerprint", "supervision", "checkpoint", "prediction", "hash", "late_loader"])
def test_bad_source_fails_whole_barrier_before_statistics(setup, monkeypatch, problem):
    obj = setup.objects[(MODELS[-1], 44)]
    kwargs = run_kwargs(setup, True)
    if problem == "missing": kwargs["ablation_source_dirs"] = kwargs["ablation_source_dirs"][:-1]
    elif problem == "extra": kwargs["b1_source_dirs"] = kwargs["b1_source_dirs"] * 2
    elif problem == "duplicate": kwargs["ablation_source_dirs"][-1] = kwargs["ablation_source_dirs"][0]
    elif problem == "seed": obj.identity["seed"] = 45
    elif problem == "variant": obj.identity["variant"] = "not_reviewed"
    elif problem == "label": obj.predictions["test"].loc[0, "true_class"] = "malignant"
    elif problem == "identity": obj.predictions["test"].loc[0, "nodule_id"] = "unknown"
    elif problem == "split": obj.predictions["test"].loc[0, "split"] = "train"
    elif problem == "reference": obj.identity["full_model_reference"] = sources.reference_record(setup.objects[(MODELS[0], 42)])
    elif problem == "fingerprint": setup.objects[("B1", 44)].descriptor["phase3_input_fingerprint"] = {"changed": True}
    elif problem == "supervision": obj.supervision.loc[0, "label"] = "malignant"
    elif problem == "checkpoint": (obj.source_dir / "checkpoints/best.pt").write_bytes(b"changed")
    elif problem == "prediction": (obj.source_dir / "predictions/test.csv").write_bytes(b"changed")
    elif problem == "hash": obj.source_snapshot[str(obj.source_dir / "checkpoints/best.pt")] = "0" * 64
    else:
        old = sources.load_ablation_source
        def fail_last(p):
            if p == obj.source_dir: raise ValueError("last source rejected by historical contract")
            return old(p)
        monkeypatch.setattr(sources, "load_ablation_source", fail_last)
    monkeypatch.setattr(runner, "compute_family", forbidden)
    expected_error = historical.Phase4EvaluationError if problem == "fingerprint" else (ValueError, FileNotFoundError)
    with pytest.raises(expected_error): runner.run_inference(**kwargs)
    assert not runner.NAMESPACE.exists()


@pytest.mark.parametrize("problem", ["no_auth", "both_modes", "research_without_auth"])
def test_authorization_before_any_access(setup, monkeypatch, problem):
    kwargs = dict(setup.kwargs)
    if problem == "no_auth": kwargs["authorize_preflight"] = False
    elif problem == "both_modes": kwargs["authorize_research_statistics"] = True
    else: kwargs["preflight_only"] = False
    monkeypatch.setattr(runner, "load_evidence", forbidden)
    monkeypatch.setattr(runner, "load_inference_inputs", forbidden)
    with pytest.raises(ValueError, match="authorization"): runner.run_inference(**kwargs)


@pytest.mark.parametrize("problem", ["missing_R", "receipt_hash", "R_bytes", "extra_R", "zip", "reference_numbers", "false_ready", "R_identity"])
def test_evidence_failure_precedes_source_access(setup, monkeypatch, problem):
    kwargs = dict(setup.kwargs)
    rdir = kwargs["r_reference_dir"]
    if problem == "missing_R": (rdir / "STATUS.txt").unlink()
    elif problem == "receipt_hash": kwargs["r_reference_hashes_sha256"] = "f" * 64
    elif problem == "R_bytes": (rdir / "legacy_results.csv").write_bytes(b"corrupt")
    elif problem == "extra_R": (rdir / "unexpected").write_bytes(b"extra")
    elif problem == "zip": kwargs["core_evidence_zip"].write_bytes(b"corrupt")
    else:
        name = {"reference_numbers": "legacy_results.csv", "false_ready": "STATUS.txt", "R_identity": "reference_identity.txt"}[problem]
        p = rdir / name
        text = p.read_text()
        text = text.replace("fake_1,0,1", "fake_1,7,1") if problem == "reference_numbers" else (
            text.replace("RESEARCH_READY=False", "RESEARCH_READY=True") if problem == "false_ready" else text.replace("3.6.3", "4.0.0"))
        p.write_bytes(text.encode())
        receipt = kwargs["r_reference_hashes"]
        receipt.write_text(json.dumps({n: sha256_file(rdir / n) for n in evidence.R_FILES}))
        kwargs["r_reference_hashes_sha256"] = sha256_file(receipt)
    monkeypatch.setattr(runner, "load_inference_inputs", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)): runner.run_inference(**kwargs)
    assert not runner.NAMESPACE.exists()


@pytest.mark.parametrize("problem", ["numerical", "mutation", "interrupt"])
def test_failure_and_interruption_never_publish_partial_family(setup, monkeypatch, problem):
    if problem == "numerical":
        def fail(*a, **k): raise NumericalFailure("synthetic forced straddle")
        monkeypatch.setattr(exports, "bounded_family", fail)
        error = NumericalFailure
    elif problem == "mutation":
        old = runner.compute_family
        def mutate(*a, **k):
            result = old(*a, **k)
            path = setup.objects[(MODELS[-1], 44)].source_dir / "checkpoints/best.pt"
            path.write_bytes(b"synthetic in-flight corruption")
            return result
        monkeypatch.setattr(runner, "compute_family", mutate)
        error = historical.Phase4EvaluationError
    else:
        def interrupt(*a, **k): raise KeyboardInterrupt("synthetic late read-back interruption")
        monkeypatch.setattr(runner, "validate_completion", interrupt)
        error = KeyboardInterrupt
    with pytest.raises(error): runner.run_inference(**run_kwargs(setup, True))
    output, = runner.NAMESPACE.iterdir()
    assert (output / "FAILED.json").is_file()
    assert not (output / "COMPLETED.json").exists() and not (output / "PREFLIGHT.json").exists()
    if problem == "numerical": assert not (output / "inference.json").exists()


def reseal(output):
    manifest = runner.read_json(output / "artifact_manifest.json")
    manifest["files"] = {n: sha256_file(output / n) for n in manifest["files"]}
    manifest["bundle_sha256"] = sha256_json(manifest["files"])
    atomic_write_json(output / "artifact_manifest.json", manifest, overwrite=True)
    name = "PREFLIGHT.json" if manifest["preflight_only"] else "COMPLETED.json"
    marker = runner.read_json(output / name)
    marker["artifact_manifest_sha256"] = sha256_file(output / "artifact_manifest.json")
    atomic_write_json(output / name, marker, overwrite=True)


@pytest.mark.parametrize("problem", ["cell", "score", "raw", "adjusted", "reject", "csv", "scope", "extra", "failed", "preflight_statistics", "boolean_type"])
def test_offline_readback_rejects_resealed_corruption(setup, problem):
    output = runner.run_inference(**run_kwargs(setup, problem != "preflight_statistics"))
    result = runner.read_json(output / "inference.json")
    if problem == "cell": result["patients"][0]["b"] += 1
    elif problem == "score": result["hypotheses"][0]["score_sum"] = "98"
    elif problem == "raw": result["hypotheses"][0]["raw_upper"] = "2"
    elif problem == "adjusted": result["hypotheses"][0]["adjusted_upper"] = "0"
    elif problem == "reject": result["hypotheses"][0]["reject"] = not result["hypotheses"][0]["reject"]
    elif problem == "preflight_statistics": result["hypotheses"][0]["raw"] = "1"
    elif problem == "csv": (output / "tables/hypotheses.csv").write_text("corrupt\n")
    elif problem == "scope":
        p = output / "evaluation.json"
        value = runner.read_json(p); value["stage_a_complete"] = True
        atomic_write_json(p, value, overwrite=True)
    elif problem == "boolean_type":
        p = output / "provenance.json"
        value = runner.read_json(p); value["model_forwards"] = False
        atomic_write_json(p, value, overwrite=True)
    elif problem == "extra": (output / "extra.txt").write_text("unmanifested")
    else: (output / "FAILED.json").write_text("{}")
    atomic_write_json(output / "inference.json", result, overwrite=True)
    reseal(output)
    with pytest.raises(ValueError): runner.validate_completion(output)


def test_pairing_order_zero_and_cancellation_patients_retained(setup):
    loaded = sources.load_inference_inputs(setup.kwargs["full_source_dirs"], setup.kwargs["b1_source_dirs"], setup.kwargs["ablation_source_dirs"])
    records = sources.prediction_records(loaded.frames)
    # Make all nodules concordant, then one patient's two nodules cancel.
    full = {g["seed"]: g for g in records if g["model"] == "ConsenDxFormer"}
    for group in records:
        group["rows"] = deepcopy(full[group["seed"]]["rows"])
    result = exports.compute_family(records, loaded.expected_identities)
    assert all(r["raw_lower"] == "1" and r["P"] == 99 and r["score_sum"] == "0" for r in result["hypotheses"])
    a = next(g for g in records if g["model"] == MODELS[0] and g["seed"] == 42)
    b = next(g for g in records if g["model"] == "B1" and g["seed"] == 42)
    for i, (ca, cb) in enumerate(((True, False), (False, True))):
        for group, correct in ((a, ca), (b, cb)):
            row = group["rows"][i]; label = row["true_class"]
            row["predicted_class"] = label if correct else CLASS_ORDER[(CLASS_ORDER.index(label) + 1) % 3]
    result = exports.compute_family(records, loaded.expected_identities)
    row = result["hypotheses"][0]
    assert row["P"] == 99 and row["cancelled_patients"] == 1 and row["score_sum"] == "0"
    for group in records: group["rows"].reverse()
    assert exports.compute_family(records[::-1], loaded.expected_identities[::-1]) == result
    exports.validate_arithmetic(records, loaded.expected_identities, result)


@pytest.mark.parametrize("problem", ["nodule_count", "patient_count", "missing_group", "duplicate_group", "unknown_label", "bool_seed"])
def test_fixed_cohort_and_record_contract(setup, problem):
    loaded = sources.load_inference_inputs(setup.kwargs["full_source_dirs"], setup.kwargs["b1_source_dirs"], setup.kwargs["ablation_source_dirs"])
    records, expected = sources.prediction_records(loaded.frames), deepcopy(loaded.expected_identities)
    if problem == "nodule_count": expected.pop()
    elif problem == "patient_count": expected[-1]["patient_id"] = expected[0]["patient_id"]
    elif problem == "missing_group": records.pop()
    elif problem == "duplicate_group": records[-1] = records[0]
    elif problem == "unknown_label": records[0]["rows"][0]["predicted_class"] = "unknown"
    else: records[0]["seed"] = True
    with pytest.raises(ValueError): sources.validate_records(records, expected)


def test_output_protection_symlink_overlap_and_partial(setup, monkeypatch):
    with pytest.raises(ValueError): runner._output_path(setup.tmp, "run", (), {})
    protected = setup.objects[(MODELS[0], 42)].source_dir
    monkeypatch.setattr(runner, "NAMESPACE", protected)
    with pytest.raises(ValueError): runner._output_path(protected, "run", (protected,), {})
    real = setup.tmp / "real"; real.mkdir()
    link = setup.tmp / "link"; link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError): sources.strict_path(link / "nested")
    monkeypatch.setattr(runner, "NAMESPACE", real)
    (real / "partial").mkdir()
    sentinel = real / "partial/sentinel"; sentinel.write_bytes(b"preserve")
    with pytest.raises(FileExistsError): runner._output_path(real, "partial", (), {})
    assert sentinel.read_bytes() == b"preserve"


def test_protocol_and_document_hashes_fail_closed(tmp_path):
    policy = protocol.protocol(); policy["patients"] = 98
    path = tmp_path / "wrong.json"; atomic_write_json(path, policy)
    with pytest.raises(ValueError): protocol.load_protocol(path)
    snapshot = evidence.document_snapshot(ROOT)
    assert len(snapshot) == 5
    assert evidence.lf_digest((ROOT / "src/consendxformer/phase5_bounded_mean.py").read_bytes()) == evidence.CORE_SHA256


def test_source_compatibility_ledger_is_required(setup):
    payload, snapshot = evidence.load_evidence(setup.kwargs["core_evidence_zip"], setup.kwargs["r_reference_dir"],
        setup.kwargs["r_reference_hashes"], setup.kwargs["r_reference_hashes_sha256"], ROOT)
    assert str(ROOT / "tests/test_phase5_bounded_mean.py") in snapshot
    payload["source_compatibility"]["src/consendxformer/phase5_bounded_mean.py"]["current_lf_sha256"] = "0" * 64
    with pytest.raises(ValueError): evidence.validate_evidence_payload(payload)


def test_snapshot_rejects_same_byte_symlink_replacement(tmp_path):
    original = tmp_path / "original"
    target = tmp_path / "target"
    original.write_bytes(b"identical")
    target.write_bytes(b"identical")
    snapshot = {str(original): sha256_file(original)}
    original.unlink()
    original.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        sources.assert_inference_snapshot_unchanged(snapshot)


@pytest.mark.parametrize("field", ["reference", "prediction_sha256", "source_snapshot"])
def test_offline_source_binding_checks_nested_ledgers(setup, field):
    loaded = sources.load_inference_inputs(setup.kwargs["full_source_dirs"], setup.kwargs["b1_source_dirs"], setup.kwargs["ablation_source_dirs"])
    identity = {"sources": deepcopy(loaded.records), "source_snapshot_sha256": sha256_json(loaded.snapshot)}
    record = next(r for r in identity["sources"] if r["model"] == MODELS[-1])
    if field == "reference": record["identity"]["full_model_reference"]["best_checkpoint_sha256"] = "0" * 64
    elif field == "source_snapshot": record[field]["predictions/test.csv"] = "0" * 64
    else: record[field] = "0" * 64
    with pytest.raises(ValueError): runner._source_binding(identity, {"source_snapshot": loaded.snapshot})


def test_late_interrupt_cleanup_survives_failed_failure_writer(setup, monkeypatch):
    old = runner.atomic_write_json
    def write(path, *args, **kwargs):
        if Path(path).name == "FAILED.json": raise OSError("synthetic failure-write error")
        return old(path, *args, **kwargs)
    def interrupt(*args): raise KeyboardInterrupt("original synthetic interrupt")
    monkeypatch.setattr(runner, "atomic_write_json", write)
    monkeypatch.setattr(runner, "validate_completion", interrupt)
    with pytest.raises(KeyboardInterrupt, match="original synthetic interrupt"):
        runner.run_inference(**setup.kwargs)
    output, = runner.NAMESPACE.iterdir()
    assert not (output / "PREFLIGHT.json").exists() and not (output / "COMPLETED.json").exists()


@pytest.mark.parametrize("actual,expected", [
    ("1e+05", "100000"), ("100000.000", "1E5"), ("+1.0e+5", "0100000"),
    ("0e+05", "0"), ("-0e5", "0"), ("10e-1", "1"),
    ("9007199254740993", "9.007199254740993e15"),
])
def test_binomial_count_exact_integer_representations(actual, expected):
    assert evidence._same_count(actual, expected)
    assert evidence._same_count(expected, actual)


@pytest.mark.parametrize("bad", [
    "", " ", " 100000", "100000\n", "NaN", "sNaN", "Inf", "Infinity", "-Infinity",
    "-1", "-1e5", "1e-5", "99999.5", "100000.00000000000000000001",
    "1.0000000000000000000001e5", "1_00000", "100000/1", "1,00000", "0x186a0",
    "1e", "1e++5", "True", "１０００００", None, True, 100000, 100000.0,
])
def test_binomial_count_rejects_malformed_fractional_negative_and_nonfinite(bad):
    assert not evidence._same_count(bad, "100000")
    assert not evidence._same_count("100000", bad)


@pytest.mark.parametrize("actual,expected", [
    ("100001", "100000"), ("1e4", "100000"), ("0", "100000"),
    ("9007199254740993", "9007199254740992"),
])
def test_binomial_count_rejects_incorrect_integer_without_float_rounding(actual, expected):
    assert not evidence._same_count(actual, expected)


@pytest.mark.parametrize("count,valid", [
    ("1e+05", True), ("100000", True), ("100001", False), ("99999.5", False),
    ("100000.00000000000000000001", False), ("-100000", False), ("NaN", False), ("bad", False),
])
def test_binomial_count_in_complete_evidence_validation(setup, monkeypatch, count, valid):
    payload, _ = evidence.load_evidence(setup.kwargs["core_evidence_zip"], setup.kwargs["r_reference_dir"],
        setup.kwargs["r_reference_hashes"], setup.kwargs["r_reference_hashes_sha256"], ROOT)
    # A NEW in-memory invented reference case, not an edit to a pinned fixture
    # or any actual R file. Resign test-only anchors so numeric validation runs.
    fixture = evidence.R_INPUT + "binomial_cases.csv"
    rows = list(csv.DictReader(io.StringIO(payload["members"][fixture])))
    rows[-1]["rejections"] = "100000"
    payload["members"][fixture] = csv_text(rows, list(rows[0]))
    package = json.loads(payload["package_manifest"])
    package[fixture] = evidence.digest(payload["members"][fixture].encode())
    payload["package_manifest"] = json.dumps(package, sort_keys=True)
    monkeypatch.setattr(evidence, "PACKAGE_MANIFEST_SHA256", evidence.digest(payload["package_manifest"].encode()))
    rows = list(csv.DictReader(io.StringIO(payload["r_outputs"]["binomial_results.csv"])))
    rows[-1]["rejections"] = count
    payload["r_outputs"]["binomial_results.csv"] = csv_text(rows, list(rows[0]))
    payload["r_hashes_text"] = json.dumps({n: evidence.digest(payload["r_outputs"][n].encode()) for n in evidence.R_FILES})
    payload["r_hashes_sha256"] = evidence.digest(payload["r_hashes_text"].encode())
    if valid:
        assert evidence.validate_evidence_payload(payload)["r_reference"] == "outputs_verified_against_pinned_fixtures"
    else:
        with pytest.raises(ValueError, match="R binomial results differ"):
            evidence.validate_evidence_payload(payload)
