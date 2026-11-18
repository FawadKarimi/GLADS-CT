"""Standalone read-only frozen TRAIN/VAL image audit. No torch, models, test images or GPU API."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
import traceback
import warnings
import zipfile

ARCHIVE_SHA = "e5663b68523b345a3b270108e6958e1f49498e3bec14ef6959e9f4d53394e36b"
RUN = "phase4-final-s42-c1e650d77585ed9c"
SPLIT_LEDGER_SHA = "b139beb0b329dd1b30baf138e161360c23821c2c4a54fc80cda3f95c328df6b9"
DECODER_SHA = "f47a12c5b999bbd657eb5f0ccfa1814655e3ba4627c9e3afcb11615552c0b595"
TRANSFORM_REPORT_SHA = "3e79f956d99d39e79cd81dd218397f66f8bc4faa71ce07ccb62220ed3a69c894"
CLASSES = ("benign", "indeterminate", "malignant")
FIELDS = ("patient_id", "nodule_id", "series_instance_uid", "split", "label", "n_valid_malignancy_ratings")
DATA = "data/processed/consendxformer_phase2"
RENDER = DATA + "/rendered_64"
MAX_BYTES = 32*1024*1024


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(data): return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def parse(data):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, "duplicate JSON key")
            result[k] = v
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def safe(path):
    path = Path(os.path.abspath(path))
    for part in [*reversed(path.parents), path]:
        if part.exists() or part.is_symlink():
            info = part.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
                    "symlink/reparse path")
    return path


def member(root, name):
    p = PurePosixPath(name)
    require(str(p) == name and not p.is_absolute() and ".." not in p.parts
            and "\\" not in name and ":" not in name, "unsafe relative path")
    path = safe(root / name)
    require(path.is_relative_to(root), "path escape")
    return path


class Reader:
    def __init__(self): self.records = {}
    def read(self, path, expected=None):
        path = safe(path)
        require(path.is_file() and path.stat().st_size <= MAX_BYTES, "missing/oversized source: " + str(path))
        raw = path.read_bytes()
        require(len(raw) <= MAX_BYTES, "oversized source")
        digest = sha(raw)
        require(expected is None or digest == expected, "source hash mismatch: " + str(path))
        if str(path) in self.records:
            require(self.records[str(path)]["sha256"] == digest, "source changed")
        self.records[str(path)] = {"sha256": digest, "size_bytes": len(raw)}
        return raw
    def recheck(self):
        for path, row in list(self.records.items()): self.read(Path(path), row["sha256"])


def rows(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "duplicate/missing CSV headers")
    out = []
    for row in reader:
        require(None not in row and all(v is not None for v in row.values()), "malformed CSV")
        out.append(row)
    return out


def frozen_text(reader, path, expected):
    raw = reader.read(path)
    require(sha(raw.replace(b"\r\n", b"\n")) == expected, "Phase 0 LF-normalized hash mismatch")
    return raw


def index(records, key):
    result = {}
    for row in records:
        require(row[key] and row[key] not in result, "duplicate/empty " + key)
        result[row[key]] = row
    return result


def files_map(value):
    mapping = value["files"]
    require(isinstance(mapping, dict) and mapping, "empty file map")
    for name, digest in mapping.items():
        p = PurePosixPath(name)
        require(str(p) == name and not p.is_absolute() and ".." not in p.parts and "\\" not in name
                and ":" not in name, "unsafe file map")
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "bad hash")
    require(sha(canonical(mapping)) == value["bundle_sha256"], "file map digest")
    return mapping


def phase2(reader, root, expected_sha, expected_bundle):
    ledger = parse(reader.read(root/"checksums.json", expected_sha))
    completion = parse(reader.read(root/"COMPLETED.json"))
    require(completion == {"schema_version": 1, "checksum_ledger": "checksums.json",
                           "checksum_ledger_sha256": expected_sha}, "Phase 2 completion binding")
    require(ledger["schema_version"] == 1 and ledger["bundle_sha256"] == expected_bundle
            and ledger["hash_scope"] == "listed generated files; raw DICOM bytes are not hashed", "Phase 2 ledger contract")
    return files_map(ledger)  # never visit every listed member (contains TEST/masks)


def build_plan(supervision, expected, rendered, canonical_rows, patient_rows, series_rows, render_root, image_hashes):
    """Pure selection/path checks. Main supplies only byte-bound frozen inputs."""
    expected = index(expected, "nodule_id")
    require(all(r["split"] in ("train", "val", "test") for r in supervision), "unexpected source split")
    selected = index([r for r in supervision if r["split"] in ("train", "val")], "nodule_id")
    require(selected.keys() == expected.keys(), "cohort membership mismatch")
    rendered = index(rendered, "nodule_id")
    canonical_rows = index(canonical_rows, "nodule_id")
    patients, series = index(patient_rows, "patient_id"), index(series_rows, "series_instance_uid")
    train = {r["patient_id"] for r in selected.values() if r["split"] == "train"}
    val = {r["patient_id"] for r in selected.values() if r["split"] == "val"}
    require(not train & val, "train/val patient overlap")
    # Test-path strings are inspected solely to reject aliasing; never stat/open them.
    test_paths = {r["image_path"] for r in rendered.values() if r["split"] == "test"}
    plan, paths = [], set()
    for nid, row in selected.items():
        require(all(row[k] == expected[nid][k] for k in FIELDS), "identity/label/reader mismatch")
        require(row["label"] in CLASSES and row["n_valid_malignancy_ratings"] in ("2", "3", "4"), "eligibility mismatch")
        render, canon = rendered[nid], canonical_rows[nid]
        for key in ("patient_id", "nodule_id", "series_instance_uid", "split", "nodule_sha256"):
            require(row[key] == render[key] == canon[key], "render/canonical identity mismatch")
        require(row["resolution"] == render["resolution"] == "64", "resolution mismatch")
        require(canon["source_validity_status"] in ("valid", "valid_with_missing_ratings"), "source invalid")
        require(canon["paper_primary_eligible"].lower() == "true" and
                canon["paper_label"] == render["paper_label"] == row["label"] and
                canon["n_valid_malignancy_ratings"] == row["n_valid_malignancy_ratings"], "canonical eligibility")
        require(patients[row["patient_id"]]["split"] == row["split"] and
                series[row["series_instance_uid"]]["split"] == row["split"] and
                series[row["series_instance_uid"]]["patient_id"] == row["patient_id"], "frozen split mismatch")
        rel = render["image_path"]
        require(rel not in test_paths and rel not in paths, "duplicate/test image alias")
        require(rel == f"resolution_64/{nid}.npy", "unexpected frozen rendered image layout")
        path = member(render_root, rel)
        require(Path(row["image_path"]).is_absolute() and safe(row["image_path"]) == path, "supervision path mismatch")
        require(rel in image_hashes, "unbound image")
        paths.add(rel)
        plan.append({**{k: row[k] for k in FIELDS}, "nodule_sha256": row["nodule_sha256"],
                     "path": str(path), "relative_image_path": rel, "image_sha256": image_hashes[rel]})
    return sorted(plan, key=lambda r: (r["split"], r["nodule_id"]))


def decode(np, cv2, raw):
    image = np.load(io.BytesIO(raw), allow_pickle=False)
    require(isinstance(image, np.ndarray) and image.shape == (64, 64), "unexpected image shape")
    require(image.dtype.kind in "uif" and np.isfinite(image).all(), "invalid dtype/nonfinite image")
    require(image.min() >= 0 and image.max() <= 255, "image range outside [0,255]")
    facts = {"decoded_shape": list(image.shape), "decoded_dtype": str(image.dtype),
             "decoded_min": float(image.min()), "decoded_max": float(image.max()), "all_finite": True}
    # Same NPY branch, cast, resize and conditional scaling as frozen _read_image.
    image = cv2.resize(np.asarray(image, dtype=np.float32), (64,64), interpolation=cv2.INTER_LINEAR)
    divided = bool(image.max() > 1)
    if divided: image = image / 255.0
    image = image.astype(np.float32)
    require(image.shape == (64,64) and np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1,
            "invalid scaled image")
    return {**facts, "division_by_255": divided, "scaled_dtype": str(image.dtype),
            "scaled_min": float(image.min()), "scaled_max": float(image.max()),
            "scaled_array_sha256": sha(image.tobytes(order="C"))}


def verify_image_hashes(reader, plan):
    for row in plan:
        require(row["split"] in ("train", "val"), "noneligible image read rejected")
        reader.read(Path(row["path"]), row["image_sha256"])


def binary_digest(path):
    path = safe(path)
    require(path.is_file() and path.stat().st_size <= 512*1024*1024, "invalid implementation binary")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def git_state(repo):
    return {key: subprocess.check_output(["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(repo), *args], text=True, timeout=30).strip()
            for key,args in (("head", ["rev-parse","HEAD"]), ("branch",["branch","--show-current"]),
                             ("status",["status","--porcelain=v1","--untracked-files=all"]))}


def save(path, value):
    with path.with_suffix(".tmp").open("wb") as stream:
        stream.write(canonical(value)+b"\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(path.with_suffix(".tmp"), path)


def run(repo, archive, output):
    repo, output, archive = safe(repo), safe(output), safe(archive)
    require(not output.is_relative_to(repo) and not repo.is_relative_to(output), "output must be outside repository")
    output.mkdir(exist_ok=False)
    reader = Reader()
    report = dict(schema="performance_v2_train_val_input_audit_v1", status="running",
                  pilot_execution_authorized=False, model_forwards=0, training_performed=False,
                  gpu_api_calls=0, test_images_accessed=0, mixed_metadata_bytes_hashed=True,
                  full_original_bundles_verified=False, images_decoded=0, warnings=[],
                  script_sha256=sha(Path(__file__).read_bytes()), python=platform.python_version(),
                  transform_report_sha256=TRANSFORM_REPORT_SHA, transform_report_reexecuted=False)
    save(output/"report.json", report)
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "hide CUDA before process startup")
        before = git_state(repo); report["git_before"] = before
        require(before["branch"] == "phase4-consendxformer" and not before["status"], "unexpected/dirty source checkout")
        zip_bytes = reader.read(archive, ARCHIVE_SHA)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            names = z.namelist(); require(len(names) == len(set(names)) == 14, "archive inventory")
            prefix = "audit-20260927T100333767013Z-539541171cb6/"
            manifest_raw = z.read(prefix+"artifact_manifest.json")
            manifest = parse(manifest_raw)
            require(parse(z.read(prefix+"AUDIT_COMPLETED.json"))["artifact_manifest_sha256"] == sha(manifest_raw), "archive completion")
            for name, digest in files_map(manifest).items():
                require(sha(z.read(prefix+name)) == digest, "archive payload mismatch")
            source = parse(z.read(prefix+"source_manifest.json"))
            expected = [r for r in rows(z.read(prefix+"train_val_source_rows.csv")) if r["experiment_id"] == RUN]
        require(Counter(r["split"] for r in expected) == {"train":807,"val":143}, "frozen count mismatch")
        for split, n, counts in (("train",437,[348,224,235]),("val",84,[65,38,40])):
            subset = [r for r in expected if r["split"] == split]
            require(len({r["patient_id"] for r in subset}) == n and
                    [sum(r["label"] == c for r in subset) for c in CLASSES] == counts, "frozen patient/class count")
        identity = source["recorded_identities"][RUN]
        fingerprint = identity["phase3_input_fingerprint"]
        runroot = repo/"artifacts/phase4/experiments"/RUN
        records = {r["relative"]:r for r in source["files"] if "/"+RUN+"/" in r["path"]}
        def bound(name): return reader.read(member(runroot,name), records[name]["sha256"])
        live_manifest = parse(bound("artifact_manifest.json")); mapping = files_map(live_manifest)
        completion = parse(bound("COMPLETED.json"))
        require(completion["artifact_manifest_sha256"] == records["artifact_manifest.json"]["sha256"], "run completion")
        require(parse(bound("config/run_identity.json")) == identity, "archived identity mismatch")
        for name in ("config/input_supervision.csv", "config/run_identity.json", "config/input_lock.json"):
            require(mapping[name] == records[name]["sha256"], "run source binding mismatch")
        require(sha(canonical(parse(bound("config/input_lock.json")))) == identity["input_lock_sha256"], "input lock mismatch")
        supervision = rows(bound("config/input_supervision.csv"))
        splitroot = repo/"artifacts/phase0/manifests"
        split = parse(frozen_text(reader,splitroot/"manifest_checksums.json", SPLIT_LEDGER_SHA))
        require(split["bundle_sha256"] == identity["manifest_bundle_sha256"], "split bundle mismatch")
        splitbytes = {name:frozen_text(reader,member(splitroot,name),digest) for name,digest in split.items() if name != "bundle_sha256"}
        canonical_root, render_root = repo/DATA, repo/RENDER
        canonmap = phase2(reader, canonical_root, fingerprint["canonical_checksum_ledger_sha256"], fingerprint["canonical_bundle_sha256"])
        rendermap = phase2(reader, render_root, fingerprint["rendered_checksum_ledger_sha256"], fingerprint["rendered_bundle_sha256"])
        canonical_rows = rows(reader.read(canonical_root/"canonical_nodule_ledger.csv",canonmap["canonical_nodule_ledger.csv"]))
        rendered = rows(reader.read(render_root/"rendered_rois.csv",rendermap["rendered_rois.csv"]))
        plan = build_plan(supervision, expected, rendered, canonical_rows,
                          rows(splitbytes["patient_split.csv"]), rows(splitbytes["series_split.csv"]), render_root, rendermap)
        report["cohort_ledger_sha256"] = sha(canonical(plan))
        # Global pre-decode barrier: exact membership, paths and all selected image hashes.
        verify_image_hashes(reader, plan)
        report["predecode_image_hashes_verified"] = len(plan)
        source_text = reader.read(repo/"src/consendxformer/phase4_dataset.py").decode().replace("\r\n","\n")
        node = next(n for n in ast.parse(source_text).body if isinstance(n,ast.FunctionDef) and n.name == "_read_image")
        decoder_hash = sha(ast.get_source_segment(source_text,node).encode())
        require(decoder_hash == DECODER_SHA, "frozen decoder implementation differs")
        report["frozen_decoder_source_sha256"] = decoder_hash
        report["decoder_policy"] = "frozen NPY branch: allow_pickle=False; float32; cv2.resize INTER_LINEAR to64; divide by255 iff resized max>1; no repair"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import numpy as np
            import cv2
            report["versions"] = {"numpy":np.__version__, "cv2":cv2.__version__,
                                  "opencv-python":importlib.metadata.version("opencv-python")}
            require(report["versions"] == {"numpy":"2.2.6","cv2":"5.0.0","opencv-python":"5.0.0.93"}, "decoder environment drift")
            report["transform_package_metadata_only"] = {name:importlib.metadata.version(name) for name in ("torch","torchvision")}
            require(report["transform_package_metadata_only"] == {"torch":"2.3.0+cu121","torchvision":"0.18.0+cu121"}, "transform environment metadata drift")
            report["implementation_files"] = {}
            for module in (np, cv2):
                path = Path(module.__file__)
                report["implementation_files"][str(path)] = sha(reader.read(path))
            report["implementation_binaries"] = {}
            for package, prefix in (("numpy","_multiarray_umath"),("opencv-python","cv2")):
                dist = importlib.metadata.distribution(package)
                selected = [f for f in (dist.files or []) if Path(str(f)).name.startswith(prefix)
                            and str(f).endswith((".so",".pyd"))]
                require(len(selected) == 1, "ambiguous/missing decoder binary")
                path = safe(dist.locate_file(selected[0]))
                report["implementation_binaries"][str(path)] = binary_digest(path)
            report["opencv_build_information_sha256"] = sha(cv2.getBuildInformation().encode())
            require(not caught, "dependency warnings")
            facts = []
            for row in plan:
                report["active_nodule_id"] = row["nodule_id"]
                save(output/"report.json",report)
                raw = reader.read(Path(row["path"]),row["image_sha256"])
                facts.append({**row, **decode(np,cv2,raw)})
                require(not caught, "decode warnings")
                report["images_decoded"] = len(facts)
            report["warnings"] = [str(w.message) for w in caught]
        reader.recheck()
        for path, digest in report["implementation_binaries"].items():
            require(binary_digest(Path(path)) == digest, "decoder binary changed")
        report["git_after"] = git_state(repo)
        require(report["git_after"] == before, "source checkout changed")
        report["split_summary"] = {split:{"nodules":len([r for r in facts if r["split"] == split]),
            "patients":len({r["patient_id"] for r in facts if r["split"] == split}),
            "division_by_255":sum(r["division_by_255"] for r in facts if r["split"] == split),
            "decoded_dtypes":dict(Counter(r["decoded_dtype"] for r in facts if r["split"] == split))}
            for split in ("train","val")}
        report.update(status="passed",active_nodule_id=None)
        save(output/"input_ledger.json",facts)
        save(output/"source_hashes.json",reader.records)
        save(output/"report.json",report)
        outputs = {name:sha((output/name).read_bytes()) for name in ("input_ledger.json","source_hashes.json","report.json")}
        save(output/"artifact_manifest.json",dict(files=outputs,bundle_sha256=sha(canonical(outputs))))
        save(output/"COMPLETED.json",dict(artifact_manifest_sha256=sha((output/"artifact_manifest.json").read_bytes()),
                                         scope="train_val_inputs_only",pilot_execution_authorized=False))
        return 0
    except BaseException as exc:
        report.update(status="failed",failure_traceback=traceback.format_exc())
        if "caught" in locals(): report["warnings"] = [str(w.message) for w in caught]
        save(output/"report.json",report)
        save(output/"source_hashes.json",reader.records)
        if "facts" in locals(): save(output/"input_ledger.json",facts)
        return 130 if isinstance(exc,KeyboardInterrupt) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.repo,args.audit_zip,args.output))
