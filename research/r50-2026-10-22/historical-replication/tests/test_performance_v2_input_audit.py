"""Invented local fixtures only. Production archive/count pins are never bypassed."""
import copy
import importlib.util
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/55_audit_performance_v2_inputs.py"
spec = importlib.util.spec_from_file_location("input_audit",SCRIPT)
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def fixture(tmp_path):
    root = tmp_path/"rendered"; root.mkdir()
    sup, expected, rendered, canonical, patients, series, hashes = [],[],[],[],[],[],{}
    for i,split in enumerate(("train","val")):
        nid = f"invented-{i}"
        rel = f"resolution_64/{nid}.npy"
        row = dict(patient_id=f"p{i}", nodule_id=nid, series_instance_uid=f"s{i}",
                   split=split,label="benign",n_valid_malignancy_ratings="2",
                   nodule_sha256=str(i)*64,resolution="64",image_path=str(root/rel))
        sup.append(row); expected.append({k:row[k] for k in audit.FIELDS})
        rendered.append(dict(row,image_path=rel,paper_label="benign"))
        canonical.append(dict(row,paper_label="benign",paper_primary_eligible="True",source_validity_status="valid"))
        patients.append(dict(patient_id=f"p{i}",split=split))
        series.append(dict(patient_id=f"p{i}",series_instance_uid=f"s{i}",split=split))
        hashes[rel] = "a"*64
    # The test image deliberately does not exist. Its metadata must not lead to an open.
    rendered.append(dict(rendered[0],nodule_id="test-nodule",split="test",image_path="resolution_64/TEST.npy"))
    return [sup, expected, rendered, canonical, patients, series, root, hashes]


def test_exact_plan_ignores_test_image(tmp_path):
    args = fixture(tmp_path)
    result = audit.build_plan(*args)
    assert len(result)==2 and {r["split"] for r in result} == {"train","val"}
    assert all("TEST" not in r["path"] for r in result)


@pytest.mark.parametrize("case", ["missing", "duplicate", "label", "patient", "series", "split", "invalid",
    "boundary", "resolution", "readers", "path", "traversal", "unbound", "test_alias", "unknown_split"])
def test_plan_corruption_fails_closed(tmp_path,case):
    args = fixture(tmp_path); sup,expected,rend,canon,patients,series,root,hashes = args
    if case == "missing": sup.pop()
    elif case == "duplicate": sup.append(copy.deepcopy(sup[0]))
    elif case == "label": sup[0]["label"]="malignant"
    elif case == "patient": patients[0]["split"]="test"
    elif case == "series": series[0]["patient_id"]="wrong"
    elif case == "split": sup[0]["split"]="test"
    elif case == "invalid": canon[0]["source_validity_status"]="excluded_malformed_geometry"
    elif case == "boundary": canon[0]["paper_primary_eligible"]="False"
    elif case == "resolution": rend[0]["resolution"]="96"
    elif case == "readers": canon[0]["n_valid_malignancy_ratings"]="1"
    elif case == "path": sup[0]["image_path"]=str(tmp_path/"outside.npy")
    elif case == "traversal": rend[0]["image_path"]="../escape.npy"
    elif case == "unbound": hashes.clear()
    elif case == "test_alias": rend[-1]["image_path"]=rend[0]["image_path"]
    elif case == "unknown_split": sup.append(dict(sup[0],nodule_id="extra",split="typo"))
    with pytest.raises((ValueError,KeyError)): audit.build_plan(*args)


def test_patient_overlap_rejected_before_path_access(tmp_path):
    args=fixture(tmp_path)
    args[0][1]["patient_id"] = args[0][0]["patient_id"]
    with pytest.raises(ValueError,match="overlap"): audit.build_plan(*args)


def npy(array):
    stream=io.BytesIO(); np.save(stream,array); return stream.getvalue()


@pytest.mark.parametrize("value,dtype,divide,maximum", [(0,np.uint8,False,0),(1,np.float32,False,1),
    (255,np.uint8,True,1),(128,np.float32,True,128/255)])
def test_real_numpy_opencv_decoder(value,dtype,divide,maximum):
    facts=audit.decode(np,cv2,npy(np.full((64,64),value,dtype=dtype)))
    assert facts["division_by_255"] is divide
    assert facts["scaled_max"] == pytest.approx(maximum,abs=1e-7)
    assert facts["decoded_dtype"] == str(np.dtype(dtype))
    assert facts["scaled_dtype"] == "float32" and facts["all_finite"]


@pytest.mark.parametrize("array", [np.full((64,64),np.nan),np.full((64,64),np.inf),
    np.full((64,64),-1),np.full((64,64),256),np.zeros((32,32)),np.zeros((64,64,3)),
    np.zeros((64,64),dtype=complex),np.full((64,64),"object",dtype=object)])
def test_bad_decodes_rejected(array):
    with pytest.raises(ValueError): audit.decode(np,cv2,npy(array))


def test_corrupt_npy_and_no_pickle():
    with pytest.raises(ValueError): audit.decode(np,cv2,b"not an image")


def test_hash_barrier_and_source_mutation(tmp_path):
    first=tmp_path/"first.npy"; second=tmp_path/"second.npy"
    first.write_bytes(npy(np.zeros((64,64),np.uint8)))
    plan=[dict(split="train",path=str(first),image_sha256=audit.sha(first.read_bytes())),
          dict(split="val",path=str(second),image_sha256="a"*64)]
    reader=audit.Reader()
    with pytest.raises(ValueError,match="missing"): audit.verify_image_hashes(reader,plan)
    # The global barrier has no decoder and cannot advance to decoding on this failure.
    first.write_bytes(b"changed")
    with pytest.raises(ValueError,match="hash mismatch"): reader.recheck()


def test_no_test_read_even_if_path_exists(tmp_path):
    path=tmp_path/"test.npy"; path.write_bytes(b"sentinel")
    reader=audit.Reader()
    with pytest.raises(ValueError,match="noneligible"):
        audit.verify_image_hashes(reader,[dict(split="test",path=str(path),image_sha256=audit.sha(b"sentinel"))])
    assert not reader.records


def test_no_overwrite_or_repo_output(tmp_path):
    repo=tmp_path/"repo"; repo.mkdir()
    out=tmp_path/"out"; out.mkdir(); (out/"report.json").write_text("sentinel")
    with pytest.raises(FileExistsError): audit.run(repo,tmp_path/"missing.zip",out)
    assert (out/"report.json").read_text()=="sentinel"
    with pytest.raises(ValueError,match="outside"): audit.run(repo,tmp_path/"missing.zip",repo/"out")


def test_fatal_guard_keeps_report_no_completion(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    repo=tmp_path/"repo"; repo.mkdir()
    assert audit.run(repo,tmp_path/"missing.zip",tmp_path/"out")==1
    report=json.loads((tmp_path/"out/report.json").read_text())
    assert report["status"]=="failed" and "hide CUDA" in report["failure_traceback"]
    assert not (tmp_path/"out/COMPLETED.json").exists()


def test_unsafe_map_and_hash_mismatch(tmp_path):
    with pytest.raises(ValueError,match="unsafe"): audit.files_map(dict(files={"../escape":"a"*64},bundle_sha256="b"*64))
    with pytest.raises(ValueError,match="digest"): audit.files_map(dict(files={"ok":"a"*64},bundle_sha256="b"*64))
    path=tmp_path/"source"; path.write_bytes(b"payload")
    with pytest.raises(ValueError,match="hash mismatch"): audit.Reader().read(path,"a"*64)


def test_phase0_lf_contract_preserves_raw_immutability(tmp_path):
    path=tmp_path/"split.csv"; path.write_bytes(b"a,b\r\n")
    reader=audit.Reader()
    assert audit.frozen_text(reader,path,audit.sha(b"a,b\n")) == b"a,b\r\n"
    path.write_bytes(b"a,b\n")
    with pytest.raises(ValueError,match="hash mismatch"): reader.recheck()
