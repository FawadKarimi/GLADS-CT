#!/usr/bin/env python3
"""Authorized 6-run GLADS-CT TRAIN1297 + Multi-Slice Extension training and evaluation runner.

Executes the 6 paired runs serially on GPU 1 (UUID GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90):
  1. HARD-42   (seed=42, objective='HARD')
  2. LEGACY-42 (seed=42, objective='LEGACY')
  3. HARD-43   (seed=43, objective='HARD')
  4. LEGACY-43 (seed=43, objective='LEGACY')
  5. HARD-44   (seed=44, objective='HARD')
  6. LEGACY-44 (seed=44, objective='LEGACY')

Enforces all scientific invariants from GLADS_CT_Project_Brief.md, HANDOVER.md,
TRAINING-RECIPE.INACTIVE.json, and SIX-RUN-RESOURCE-PLAN.INACTIVE.json:
  - Patient separation across TRAIN (529), VAL (84), TEST (99)
  - Zero TEST pixel access; checkpoint selection and early stopping on frozen VAL (N=143) only
  - Whole-nodule inverse-frequency class weights from 1,297 TRAIN nodules:
    [1297/1746, 1297/1326, 1297/819]
  - Original verified RadImageNet backbone initialization (318 mapped entries, 53 BN counters)
  - Identical backbone + non-backbone initial state across HARD and LEGACY within each seed
  - Identical pre-frozen view schedules and augmentation traces across HARD and LEGACY within each seed
  - Single-reader nodules (490) have disagreement u=0.0; multi-reader nodules (807) preserve reader distribution
  - Sparse checkpoint retention: retains best and final/latest checkpoint bodies per run (<=2 bodies)
"""
import os
import sys

# Pin environment variables before importing numpy/torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
import gzip
import json
import math
import resource
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import torch

# Add authenticated reference module paths
CANDIDATE_SRC = "/home/chaidir/.performance-v2-replication-20261013/candidate/src"
REFERENCE_SRC = "/home/chaidir/.performance-v2-replication-20261013/reference"
for p in (CANDIDATE_SRC, REFERENCE_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from consendxformer.performance_v2.bindings import CLASS_ORDER
from consendxformer.performance_v2.checkpoint import decode_state, encode_state, state_sha256
from consendxformer.performance_v2.input_adapter import decode_grayscale, transform
from consendxformer.performance_v2.metrics import metric_definition_sha256
from consendxformer.performance_v2.model import R50Hybrid
from consendxformer.performance_v2.objectives import (
    class_weights_from_training_counts,
    objective_loss,
)
from consendxformer.performance_v2.optimization import (
    make_adamw,
    set_epoch_learning_rates,
)
from consendxformer.performance_v2.reuse import canonical, digest
from consendxformer.performance_v2.selection import (
    ValidationEpoch,
    ValidationPatience,
    select_checkpoint,
)
from consendxformer.performance_v2.streams import TrainingStreams, capture_cpu_rng
from consendxformer.performance_v2.rad_model_import import import_verified_cpu
from consendxformer.performance_v2.cuda_identity import exact_uuid

# Pinned cryptographic constants and paths
EXPECTED_GPU_UUID = "GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90"
RESIDENT_PIDS = (3340885, 3477966)

CROP_DIR = Path("/home/chaidir/.performance-v2-train1297-multislice-crops-20261019-001")
VIEW_MANIFEST_MAT_PATH = CROP_DIR / "VIEW-MANIFEST.MATERIALIZED.json"
VIEW_MANIFEST_MAT_SHA = "073c7ac48ee2f0fd96754f4c8647d677e83b6d3d1e0e69c42f590980293a2cad"
CROP_MANIFEST_PATH = CROP_DIR / "CROP-MANIFEST.json"
CROP_MANIFEST_SHA = "d77509a311083c8d7345266be6cce90dc3c399d76eb7e202aa9ef0e333ab673e"

SCHEDULE_DIR = Path("/home/chaidir/.performance-v2-train1297-schedule-freeze-20261019-008/outputs")
SCHEDULE_SHAS = {
    42: {
        "gz_sha256": "728ee3242a0abb368c190689d91aa5dfc1da81cf232e4294be1dd54616c0b559",
        "jsonl_sha256": "b436277071d6d2e169a06ba95727411aa72506b59df1bd894701a663537f64e0",
    },
    43: {
        "gz_sha256": "04abd1b4ad7f98f132217d16ea76e52356650b89a8cbb1b01c62bd8d9cf01acf",
        "jsonl_sha256": "f7c88ffa3936ce18bd846f9affff2f373f6aa6b8883bcec62b4faaf44fb55622",
    },
    44: {
        "gz_sha256": "57242c1f4b4ee387579ac9dace47e1b5f3bfd3fd97259b609b6f96ead3d40f37",
        "jsonl_sha256": "3388adebd4ea2f71cc8a9894c516229acc126b67f3ce1e2e99dbff091b581418",
    },
}

DELIVERY_DIR = Path("/home/chaidir/.performance-v2-train1297-final-delivery-20261019-009")
VAL_FROZEN_PATH = DELIVERY_DIR / "VAL-RECORDS.FROZEN.json"
VAL_FROZEN_SHA = "766ccd07c75a0ff2b0833867203cc57b146f9c3ac9ae52f07500d605298ad8a9"
TEST_NO_PIXELS_PATH = DELIVERY_DIR / "TEST-IDENTITIES.FROZEN-NO-PIXELS.json"

EXISTING_REP_ROOT = Path(
    "/experiment-storage/chaidir/fawad-dsai/ConsenDxFormer/data/processed/consendxformer_phase2/rendered_64"
)
EXISTING_PHASE2_BASE = EXISTING_REP_ROOT.parent

RAD_DIR = Path("/home/chaidir/.performance-v2-radimagenet-receiver-20261013/artifacts")
RAD_EXPORT_PATH = RAD_DIR / "backbone.typed-json"
RAD_RECEIPT_PATH = RAD_DIR / "export-receipt.json"
RAD_RECEIPT_SHA = "3817d15712462a6a500b7efc8aa8720e201c029e5110e38eeddb523c82e69918"
RAD_PLAN_SHA = "7809f8ff53150fde8e4273a21c0c4559d1ae6f665f34b22245b6398bc94b3f5c"
RAD_RAW_WEIGHT_SHA = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"

HISTORICAL_INIT_FROZEN = Path(
    "/home/chaidir/.performance-v2-supervision-20261015/inputs/INITIALIZATION-VALIDATION-TEST.FROZEN.json"
)
HISTORICAL_SUP_FROZEN = Path(
    "/home/chaidir/.performance-v2-supervision-20261015/analysis/SUPERVISION-VALIDATION.FROZEN.json"
)

OUTPUT_ROOT = Path("/experiment-storage/chaidir/fawad-dsai/glads-ct-train1297-sixrun-20261019")
HOME_RECEIPT_DIR = Path("/home/chaidir/.performance-v2-train1297-sixrun-20261019")

RUN_ORDER = (
    ("HARD-42", 42, "HARD"),
    ("LEGACY-42", 42, "LEGACY"),
    ("HARD-43", 43, "HARD"),
    ("LEGACY-43", 43, "LEGACY"),
    ("HARD-44", 44, "HARD"),
    ("LEGACY-44", 44, "LEGACY"),
)

EXPECTED_TRAIN_COUNTS = {0: 582, 1: 442, 2: 273}
EXPECTED_VAL_COUNTS = {0: 65, 1: 38, 2: 40}


def log_progress(status_dict):
    """Write live status JSON atomically for external progress inspection."""
    status_dict["updated_at_unix"] = time.time()
    tmp = OUTPUT_ROOT / "LIVE-STATUS.json.tmp"
    dst = OUTPUT_ROOT / "LIVE-STATUS.json"
    tmp.write_bytes(json.dumps(status_dict, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"))
    os.replace(tmp, dst)


def write_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)
    return digest(raw)


def resolve_representative_crop_path(rel_or_abs_path: str) -> Path:
    """Resolve representative crop path against EXISTING_REP_ROOT or EXISTING_PHASE2_BASE."""
    p = Path(rel_or_abs_path)
    if p.is_absolute() and p.is_file():
        return p
    cand1 = EXISTING_REP_ROOT / rel_or_abs_path
    if cand1.is_file():
        return cand1
    cand2 = EXISTING_PHASE2_BASE / rel_or_abs_path
    if cand2.is_file():
        return cand2
    raise FileNotFoundError(f"Representative crop not found for {rel_or_abs_path}")


def compute_split_diagnostics(ids, logits_cpu: torch.Tensor, labels_cpu: torch.Tensor, weights_cpu: torch.Tensor, *, split: str):
    """Compute exact 3-class diagnostics accumulated over the complete split (matching ref/metrics.py)."""
    ids = tuple(ids)
    n = len(ids)
    if (
        not ids
        or len(set(ids)) != n
        or not isinstance(logits_cpu, torch.Tensor)
        or logits_cpu.device.type != "cpu"
        or logits_cpu.dtype != torch.float32
        or tuple(logits_cpu.shape) != (n, 3)
        or not torch.isfinite(logits_cpu).all()
        or not isinstance(labels_cpu, torch.Tensor)
        or labels_cpu.device.type != "cpu"
        or labels_cpu.dtype != torch.int64
        or tuple(labels_cpu.shape) != (n,)
        or not ((labels_cpu >= 0) & (labels_cpu < 3)).all()
    ):
        raise ValueError(f"Invalid split diagnostics inputs for split={split}")

    with torch.no_grad():
        predictions = logits_cpu.argmax(dim=1).tolist()
        ce = -logits_cpu.log_softmax(dim=1).gather(1, labels_cpu[:, None]).flatten()
        if not torch.isfinite(ce).all():
            raise ValueError("Non-finite diagnostic CE")
        w_sample = weights_cpu[labels_cpu]
        weighted_ce = float(math.fsum((w_sample * ce).tolist()) / math.fsum(w_sample.tolist()))

    confusion = [[0] * 3 for _ in range(3)]
    for y_val, p_val in zip(labels_cpu.tolist(), predictions):
        confusion[int(y_val)][int(p_val)] += 1

    per_class = {}
    for c_idx, c_name in enumerate(CLASS_ORDER):
        tp = confusion[c_idx][c_idx]
        support = sum(confusion[c_idx])
        predicted = sum(row[c_idx] for row in confusion)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        entry = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
        per_class[c_name] = entry
        per_class[str(c_idx)] = entry

    acc = sum(confusion[c][c] for c in range(3)) / n
    m_f1 = math.fsum(per_class[c_name]["f1"] for c_name in CLASS_ORDER) / 3.0
    diag_ce = math.fsum(ce.tolist()) / n

    return {
        "split": split,
        "n": int(n),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "macro_f1": float(m_f1),
        "accuracy": float(acc),
        "micro_f1": float(acc),
        "diagnostic_ce": float(diag_ce),
        "weighted_diagnostic_ce": float(weighted_ce),
        "prediction_identity_sha256": digest(canonical(list(ids))),
        "metric_definition_sha256": metric_definition_sha256(),
    }


def apply_resource_limits():
    """Apply bounded process resource limits compatible with PyTorch CUDA context & 350 MiB checkpoints."""
    try:
        os.nice(10)
    except OSError:
        pass
    soft_nofile, hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_nofile = min(2048, hard_nofile) if hard_nofile != resource.RLIM_INFINITY else 2048
    if soft_nofile < target_nofile:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_nofile, hard_nofile))
    _, hard_fsize = resource.getrlimit(resource.RLIMIT_FSIZE)
    fsize_cap = 536870912  # 512 MiB
    if hard_fsize == resource.RLIM_INFINITY or hard_fsize >= fsize_cap:
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_cap, hard_fsize))


def configure_determinism():
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly 1 visible CUDA device, found {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    gpu_uuid = exact_uuid({"index": 0, "uuid": EXPECTED_GPU_UUID})
    if gpu_uuid != EXPECTED_GPU_UUID:
        raise RuntimeError(f"GPU UUID mismatch: got {gpu_uuid}, expected {EXPECTED_GPU_UUID}")
    for pid in RESIDENT_PIDS:
        if not Path(f"/proc/{pid}").exists():
            raise RuntimeError(f"Resident process PID {pid} not found in /proc")
    return {"device_name": props.name, "gpu_uuid": gpu_uuid, "total_memory_bytes": int(props.total_memory)}


def load_and_verify_inputs(status):
    """Verify all manifests, schedules, RadImageNet weights, and pre-decode all 4,273 TRAIN + 143 VAL crops."""
    status["phase"] = "verifying_manifests_and_schedules"
    log_progress(status)

    vm_bytes = VIEW_MANIFEST_MAT_PATH.read_bytes()
    if digest(vm_bytes) != VIEW_MANIFEST_MAT_SHA:
        raise ValueError("VIEW-MANIFEST.MATERIALIZED.json SHA-256 mismatch")
    cm_bytes = CROP_MANIFEST_PATH.read_bytes()
    if digest(cm_bytes) != CROP_MANIFEST_SHA:
        raise ValueError("CROP-MANIFEST.json SHA-256 mismatch")
    val_bytes = VAL_FROZEN_PATH.read_bytes()
    if digest(val_bytes) != VAL_FROZEN_SHA:
        raise ValueError("VAL-RECORDS.FROZEN.json SHA-256 mismatch")

    vm = json.loads(vm_bytes)
    val_records = json.loads(val_bytes)
    if isinstance(val_records, dict) and "records" in val_records:
        val_records = val_records["records"]
    test_records = json.loads(TEST_NO_PIXELS_PATH.read_bytes())
    if isinstance(test_records, dict) and "records" in test_records:
        test_records = test_records["records"]

    train_records = vm["records"]
    if len(train_records) != 1297 or len(val_records) != 143 or len(test_records) != 183:
        raise ValueError(f"Split counts mismatch: {len(train_records)}, {len(val_records)}, {len(test_records)}")

    train_patients = {r["patient_id"] for r in train_records}
    val_patients = {r["patient_id"] for r in val_records}
    test_patients = {r["patient_id"] for r in test_records}
    if len(train_patients) != 529 or len(val_patients) != 84 or len(test_patients) != 99:
        raise ValueError("Patient count mismatch across splits")
    if (train_patients & val_patients) or (train_patients & test_patients) or (val_patients & test_patients):
        raise ValueError("Patient leakage detected across splits")

    # Verify whole-nodule class counts and single-reader disagreement rule
    train_class_counts = {0: 0, 1: 0, 2: 0}
    single_reader_count = 0
    multi_reader_count = 0
    nodule_meta = {}
    view_path_and_sha = {}

    for r in train_records:
        nid = r["nodule_id"]
        label = int(r["label"])
        train_class_counts[label] += 1
        vr = int(r["valid_readers"])
        u = float(r["disagreement"])
        q = [float(x) for x in r["reader_target"]]
        if vr == 1:
            single_reader_count += 1
            expected_q = [1.0 if i == label else 0.0 for i in range(3)]
            if abs(u) > 1e-12 or any(abs(q[i] - expected_q[i]) > 1e-12 for i in range(3)):
                raise ValueError(f"Single-reader invariant violated for {nid}")
        elif vr >= 2:
            multi_reader_count += 1
        else:
            raise ValueError(f"Invalid valid_readers={vr} for {nid}")
        nodule_meta[nid] = {
            "nodule_id": nid,
            "patient_id": r["patient_id"],
            "label": label,
            "valid_readers": vr,
            "disagreement": u,
            "reader_target": q,
        }
        for v in r["views"]:
            vid = v["view_id"]
            if v["existing_representative"]:
                p = resolve_representative_crop_path(v["image_path"])
            else:
                p = Path(v["image_path"])
            view_path_and_sha[vid] = (p, v["image_sha256"], nid)

    if train_class_counts != EXPECTED_TRAIN_COUNTS:
        raise ValueError(f"Train class counts mismatch: {train_class_counts}")
    if single_reader_count != 490 or multi_reader_count != 807 or len(view_path_and_sha) != 4273:
        raise ValueError(f"Reader/view count mismatch: {single_reader_count}, {multi_reader_count}, {len(view_path_and_sha)}")

    val_class_counts = {0: 0, 1: 0, 2: 0}
    for r in val_records:
        val_class_counts[int(r["label"])] += 1
    if val_class_counts != EXPECTED_VAL_COUNTS:
        raise ValueError(f"Val class counts mismatch: {val_class_counts}")

    # Verify schedules for seeds 42, 43, 44
    schedules = {}
    for seed, shas in SCHEDULE_SHAS.items():
        s_path = SCHEDULE_DIR / f"MATCHED-VIEWS-seed-{seed}.jsonl.gz"
        gz_bytes = s_path.read_bytes()
        if digest(gz_bytes) != shas["gz_sha256"]:
            raise ValueError(f"Schedule gz SHA mismatch for seed {seed}")
        raw_jsonl = gzip.decompress(gz_bytes)
        if digest(raw_jsonl) != shas["jsonl_sha256"]:
            raise ValueError(f"Schedule uncompressed JSONL SHA mismatch for seed {seed}")
        lines = [json.loads(line) for line in raw_jsonl.decode("utf-8").splitlines() if line.strip()]
        if len(lines) != 100:
            raise ValueError(f"Expected 100 epochs in schedule seed {seed}, got {len(lines)}")
        for idx, ep in enumerate(lines, start=1):
            if ep["epoch"] != idx or ep["seed"] != seed or len(ep["selected"]) != 1297 or len(ep["batches"]) != 82:
                raise ValueError(f"Schedule epoch structure invalid at seed={seed} epoch={idx}")
        schedules[seed] = lines

    # Pre-verify and decode all 4,273 TRAIN view crops into CPU (1, 64, 64) float32 tensors
    status["phase"] = "verifying_and_loading_4273_train_crops"
    log_progress(status)
    train_view_tensors = {}
    for vid, (p, expected_sha, _nid) in sorted(view_path_and_sha.items()):
        raw = p.read_bytes()
        if len(raw) != 16512 or digest(raw) != expected_sha:
            raise ValueError(f"Train view crop hash/size mismatch for {vid} at {p}")
        train_view_tensors[vid] = decode_grayscale(raw)

    # Pre-verify and transform all 143 VAL crops into CPU (3, 224, 224) float32 tensors
    status["phase"] = "verifying_and_loading_143_val_crops"
    log_progress(status)
    val_ordered = sorted(val_records, key=lambda r: r["nodule_id"])
    val_tensors = []
    val_labels = []
    val_ids = []
    for r in val_ordered:
        nid = r["nodule_id"]
        p = resolve_representative_crop_path(f"resolution_64/{nid}.npy")
        raw = p.read_bytes()
        if len(raw) != 16512 or digest(raw) != r["image_sha256"]:
            raise ValueError(f"Val crop hash/size mismatch for {nid}")
        val_tensors.append(transform(decode_grayscale(raw), trace=None))
        val_labels.append(int(r["label"]))
        val_ids.append(nid)

    # Load and verify RadImageNet export bytes & receipt
    status["phase"] = "verifying_radimagenet_backbone_artifact"
    log_progress(status)
    rad_receipt_bytes = RAD_RECEIPT_PATH.read_bytes()
    if digest(rad_receipt_bytes) != RAD_RECEIPT_SHA:
        raise ValueError("RadImageNet export-receipt.json SHA-256 mismatch")
    rad_receipt = json.loads(rad_receipt_bytes)
    if rad_receipt.get("artifact", {}).get("sha256") != RAD_RAW_WEIGHT_SHA:
        raise ValueError("RadImageNet raw_weight_sha256 mismatch in receipt")
    rad_raw = RAD_EXPORT_PATH.read_bytes()

    return {
        "nodule_meta": nodule_meta,
        "train_view_tensors": train_view_tensors,
        "val_tensors": val_tensors,
        "val_labels": val_labels,
        "val_ids": val_ids,
        "schedules": schedules,
        "rad_raw": rad_raw,
        "rad_receipt_bytes": rad_receipt_bytes,
        "train_class_counts": train_class_counts,
        "val_class_counts": val_class_counts,
    }


def build_initialized_model(seed: int, objective: str, rad_raw: bytes, rad_receipt_bytes: bytes):
    """Construct R50Hybrid on CPU and import verified RadImageNet backbone weights."""
    streams = TrainingStreams(seed)
    model = R50Hybrid(seed=seed, objective=objective, initialization="RANDOM")
    report = import_verified_cpu(
        model,
        rad_raw,
        rad_receipt_bytes,
        trusted_receipt_sha=RAD_RECEIPT_SHA,
        trusted_plan_sha=RAD_PLAN_SHA,
        streams=streams,
    )
    sd = model.state_dict()
    backbone_sha = state_sha256({k: v for k, v in sd.items() if k.startswith("backbone.")})
    common_sha = state_sha256({k: v for k, v in sd.items() if not k.startswith("reader_head.")})
    full_sha = state_sha256(dict(sd))
    return model, streams, {
        "backbone_state_sha256": backbone_sha,
        "common_encoder_and_diagnostic_head_initial_state_sha256": common_sha,
        "full_initial_state_sha256": full_sha,
        "verified_count": int(report["verified_count"]),
        "supplied_count": int(report["supplied_count"]),
    }


def verify_seed_pair_initial_parity(rad_raw: bytes, rad_receipt_bytes: bytes):
    """Verify before training that HARD and LEGACY have bit-identical initial states per seed."""
    parity = {}
    for seed in (42, 43, 44):
        _, _, h_info = build_initialized_model(seed, "HARD", rad_raw, rad_receipt_bytes)
        _, _, l_info = build_initialized_model(seed, "LEGACY", rad_raw, rad_receipt_bytes)
        if h_info["backbone_state_sha256"] != l_info["backbone_state_sha256"]:
            raise ValueError(f"Backbone initial state mismatch for seed {seed}")
        if (
            h_info["common_encoder_and_diagnostic_head_initial_state_sha256"]
            != l_info["common_encoder_and_diagnostic_head_initial_state_sha256"]
        ):
            raise ValueError(f"Common non-backbone initial state mismatch for seed {seed}")
        if h_info["full_initial_state_sha256"] != l_info["full_initial_state_sha256"]:
            raise ValueError(f"Full initial state mismatch for seed {seed}")
        parity[seed] = h_info
    return parity


def prune_unretained_checkpoints(run_dir: Path, retained_epochs):
    retained_names = {f"epoch-{int(ep):03d}.state.json" for ep in retained_epochs}
    removed = []
    for p in sorted(run_dir.glob("epoch-*.state.json")):
        if p.name not in retained_names:
            p.unlink()
            removed.append(p.name)
    return sorted(retained_names), removed


def execute_single_run(
    run_id: str,
    seed: int,
    objective: str,
    inputs: dict,
    init_parity_info: dict,
    status: dict,
):
    run_dir = OUTPUT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / "receipt.json"
    summary_path = run_dir / "summary.json"

    # Skip if already completed and verified
    if receipt_path.exists() and summary_path.exists():
        existing_receipt = json.loads(receipt_path.read_bytes())
        if existing_receipt.get("status") == "complete" and existing_receipt.get("run_id") == run_id:
            return json.loads(summary_path.read_bytes())

    t_run_start = time.time()
    device = torch.device("cuda:0")
    model, streams, init_info = build_initialized_model(
        seed, objective, inputs["rad_raw"], inputs["rad_receipt_bytes"]
    )
    if init_info["full_initial_state_sha256"] != init_parity_info["full_initial_state_sha256"]:
        raise ValueError(f"Initial state drift detected for {run_id}")

    model.to(device)
    optimizer, opt_ledger = make_adamw(model)
    group_counts = {g["name"]: len(g["params"]) for g in optimizer.param_groups}

    counts_seq = [int(inputs["train_class_counts"][c]) for c in (0, 1, 2)]
    w_cpu = class_weights_from_training_counts(counts_seq)
    w_list = [float(x) for x in w_cpu.tolist()]
    w_dev = w_cpu.to(device)

    nodule_meta = inputs["nodule_meta"]
    train_view_tensors = inputs["train_view_tensors"]
    val_tensors = inputs["val_tensors"]
    val_labels = inputs["val_labels"]
    val_ids = inputs["val_ids"]
    schedule_lines = inputs["schedules"][seed]

    patience = ValidationPatience()
    history = []
    epoch_records = []
    best_epoch_saved = None
    best_state_sha = None
    best_state_bytes_len = None
    best_val_snapshot = None
    best_train_snapshot = None

    for ep_line in schedule_lines:
        ep_t0 = time.time()
        epoch = int(ep_line["epoch"])
        selected_map = {item["nodule_id"]: item["view_id"] for item in ep_line["selected"]}
        aug_map = {item["id"]: item for item in ep_line["augmentation"]}
        batches = ep_line["batches"]

        set_epoch_learning_rates(optimizer, epoch)
        lrs = {g["name"]: float(g["lr"]) for g in optimizer.param_groups}
        model.train()

        train_ids_ep = []
        train_logits_ep = []
        train_labels_ep = []
        batch_components = []
        grad_norms = []
        exposed_views = []

        for batch_nids in batches:
            xs = []
            ys = []
            qs = []
            us = []
            for nid in batch_nids:
                vid = selected_map[nid]
                trace = aug_map[nid]
                meta = nodule_meta[nid]
                xs.append(transform(train_view_tensors[vid], trace=trace))
                ys.append(meta["label"])
                qs.append(meta["reader_target"])
                us.append(meta["disagreement"])
                train_ids_ep.append(nid)
                exposed_views.append(vid)

            bx = torch.stack(xs, dim=0).to(device, non_blocking=True)
            by_cpu = torch.tensor(ys, dtype=torch.int64)
            by = by_cpu.to(device, non_blocking=True)
            bq = torch.tensor(qs, dtype=torch.float32, device=device)
            bu = torch.tensor(us, dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            out = model(bx)
            loss_parts = objective_loss(
                objective,
                out["logits"],
                by,
                w_dev,
                reader_targets=bq,
                disagreement=bu,
                reader_logits=out["reader_logits"],
            )
            loss_parts["total"].backward()
            gnorm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                .detach()
                .cpu()
                .item()
            )
            optimizer.step()

            train_logits_ep.append(out["logits"].detach().cpu())
            train_labels_ep.append(by_cpu)
            comp = {k: float(v.detach().cpu().item()) for k, v in loss_parts.items()}
            comp["n"] = int(len(batch_nids))
            comp["gradient_norm_before_clip"] = gnorm
            batch_components.append(comp)
            grad_norms.append(gnorm)

        train_metrics = compute_split_diagnostics(
            train_ids_ep,
            torch.cat(train_logits_ep, dim=0),
            torch.cat(train_labels_ep, dim=0),
            w_cpu,
            split="train",
        )
        total_n = sum(x["n"] for x in batch_components)
        hard_epoch_loss = sum(x["hard_numerator"] for x in batch_components) / sum(
            x["hard_denominator"] for x in batch_components
        )
        reader_epoch_loss = sum(x["reader"] * x["n"] for x in batch_components) / total_n
        legacy_epoch_loss = (
            sum(x["legacy_numerator"] for x in batch_components)
            / sum(x["legacy_denominator"] for x in batch_components)
            if objective == "LEGACY"
            else 0.0
        )
        epoch_objective_loss = legacy_epoch_loss if objective == "LEGACY" else hard_epoch_loss

        # Validation on frozen 143 representative nodules (batch size 16)
        model.eval()
        val_logits_ep = []
        with torch.no_grad():
            for i in range(0, len(val_tensors), 16):
                vx = torch.stack(val_tensors[i : i + 16], dim=0).to(device, non_blocking=True)
                vout = model(vx)
                val_logits_ep.append(vout["logits"].detach().cpu())
        val_metrics = compute_split_diagnostics(
            val_ids,
            torch.cat(val_logits_ep, dim=0),
            torch.tensor(val_labels, dtype=torch.int64),
            w_cpu,
            split="val",
        )

        val_rec = ValidationEpoch(
            epoch=epoch,
            macro_f1=float(val_metrics["macro_f1"]),
            diagnostic_ce=float(val_metrics["diagnostic_ce"]),
        )
        is_stop = bool(patience.observe(val_rec))
        history.append(val_rec)
        best_rec = select_checkpoint(history)
        best = {
            "epoch": int(best_rec.epoch),
            "macro_f1": float(best_rec.macro_f1),
            "diagnostic_ce": float(best_rec.diagnostic_ce),
        }
        p_status = {
            "last_epoch": int(patience.last_epoch),
            "best": float(patience.best) if math.isfinite(patience.best) else None,
            "without_improvement": int(patience.without_improvement),
            "stopped": bool(patience.stopped),
            "stop": is_stop,
        }
        is_new_best = best["epoch"] == epoch

        # Serialize full checkpoint state when this epoch is the new best OR final stopping epoch
        ckpt_sha = None
        ckpt_size = None
        if is_new_best or is_stop:
            cpu_model_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            opt_sd = copy.deepcopy(optimizer.state_dict())
            for st in opt_sd.get("state", {}).values():
                for k, val in list(st.items()):
                    if isinstance(val, torch.Tensor):
                        st[k] = val.detach().cpu().clone()
            history_plain = [
                {
                    "epoch": int(r.epoch),
                    "macro_f1": float(r.macro_f1),
                    "diagnostic_ce": float(r.diagnostic_ce),
                }
                for r in history
            ]
            payload = {
                "schema": "performance_v2_checkpoint_v1",
                "run_id": run_id,
                "seed": seed,
                "objective": objective,
                "initialization": "RADIMAGENET",
                "epoch": epoch,
                "model": cpu_model_sd,
                "optimizer": opt_sd,
                "rng": capture_cpu_rng(streams),
                "patience": copy.deepcopy(p_status),
                "history": history_plain,
                "selection": copy.deepcopy(best),
            }
            raw_state = encode_state(payload)
            ckpt_sha = digest(raw_state)
            ckpt_size = len(raw_state)
            state_path = run_dir / f"epoch-{epoch:03d}.state.json"
            tmp_state = state_path.with_suffix(".state.json.tmp")
            tmp_state.write_bytes(raw_state)
            os.replace(tmp_state, state_path)

            if is_new_best:
                best_epoch_saved = epoch
                best_state_sha = ckpt_sha
                best_state_bytes_len = ckpt_size
                best_val_snapshot = copy.deepcopy(val_metrics)
                best_train_snapshot = copy.deepcopy(train_metrics)

            retained_set = {best["epoch"], epoch} if is_stop else {best["epoch"]}
            prune_unretained_checkpoints(run_dir, retained_set)

        ep_elapsed = time.time() - ep_t0
        metrics_payload = {
            "run_id": run_id,
            "seed": seed,
            "objective": objective,
            "epoch": epoch,
            "learning_rates": lrs,
            "train": {
                "epoch_objective_loss": float(epoch_objective_loss),
                "loss_totals": {
                    "hard": float(hard_epoch_loss),
                    "legacy": float(legacy_epoch_loss),
                    "reader": float(reader_epoch_loss),
                },
                "mean_batch_objective_loss": float(
                    sum(x["total"] for x in batch_components) / len(batch_components)
                ),
                "max_grad_norm": float(max(grad_norms)),
                "mean_grad_norm": float(sum(grad_norms) / len(grad_norms)),
                "macro_f1": train_metrics["macro_f1"],
                "accuracy": train_metrics["accuracy"],
                "diagnostic_ce": train_metrics["diagnostic_ce"],
                "weighted_diagnostic_ce": train_metrics["weighted_diagnostic_ce"],
                "per_class": train_metrics["per_class"],
                "confusion_matrix": train_metrics["confusion_matrix"],
                "exposed_views_sha256": digest(canonical(exposed_views)),
            },
            "validation": val_metrics,
            "patience": p_status,
            "selected_so_far": best,
            "checkpoint_written": bool(is_new_best or is_stop),
            "checkpoint_sha256": ckpt_sha,
            "elapsed_seconds": float(ep_elapsed),
        }
        write_json_atomic(run_dir / f"epoch-{epoch:03d}.metrics.json", metrics_payload)
        write_json_atomic(
            run_dir / f"epoch-{epoch:03d}.runner.json",
            {
                "run_id": run_id,
                "epoch": epoch,
                "selected_epoch": best["epoch"],
                "stop": is_stop,
                "checkpoint_sha256": ckpt_sha if ckpt_sha else best_state_sha,
                "elapsed_seconds": float(ep_elapsed),
            },
        )
        epoch_records.append(metrics_payload)

        status["phase"] = f"training_{run_id}"
        status["current_run"] = {
            "run_id": run_id,
            "seed": seed,
            "objective": objective,
            "epoch": epoch,
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "val_diagnostic_ce": round(val_metrics["diagnostic_ce"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "best_epoch": best["epoch"],
            "best_val_macro_f1": round(best["macro_f1"], 6),
            "without_improvement": p_status["without_improvement"],
            "epoch_seconds": round(ep_elapsed, 2),
        }
        log_progress(status)

        if is_stop:
            break

    final_epoch = epoch_records[-1]["epoch"]
    retained_files, _ = prune_unretained_checkpoints(run_dir, {best_epoch_saved, final_epoch})
    retained_digests = {}
    for fname in retained_files:
        p = run_dir / fname
        retained_digests[fname] = {
            "sha256": digest(p.read_bytes()),
            "size_bytes": int(p.stat().st_size),
        }

    selection_summary = {
        "run_id": run_id,
        "seed": seed,
        "objective": objective,
        "initialization": "RADIMAGENET",
        "selected_epoch": int(best_epoch_saved),
        "stopped_epoch": int(final_epoch),
        "stopped_early": bool(final_epoch < 100),
        "best_checkpoint_file": f"epoch-{best_epoch_saved:03d}.state.json",
        "best_checkpoint_sha256": best_state_sha,
        "best_checkpoint_size_bytes": int(best_state_bytes_len),
        "latest_checkpoint_file": f"epoch-{final_epoch:03d}.state.json",
        "retained_checkpoints": retained_digests,
        "retained_checkpoint_count": int(len(retained_digests)),
        "validation_at_selected": best_val_snapshot,
        "train_at_selected": best_train_snapshot,
        "validation_at_final": epoch_records[-1]["validation"],
        "initial_state": init_info,
        "optimizer_group_counts": group_counts,
        "optimizer_parameter_total": int(len(opt_ledger)),
        "class_weights_nodule_level": w_list,
        "schedule_gz_sha256": SCHEDULE_SHAS[seed]["gz_sha256"],
        "schedule_jsonl_sha256": SCHEDULE_SHAS[seed]["jsonl_sha256"],
        "total_elapsed_seconds": float(time.time() - t_run_start),
    }
    write_json_atomic(run_dir / "selection.json", selection_summary)
    write_json_atomic(summary_path, selection_summary)
    write_json_atomic(
        receipt_path,
        {
            "status": "complete",
            "run_id": run_id,
            "seed": seed,
            "objective": objective,
            "selected_epoch": int(best_epoch_saved),
            "stopped_epoch": int(final_epoch),
            "best_checkpoint_sha256": best_state_sha,
            "retained_checkpoints": retained_digests,
            "summary_sha256": digest(summary_path.read_bytes()),
        },
    )

    # Mirror lightweight metadata (summary, selection, receipt, epoch metrics) to HOME_RECEIPT_DIR
    home_run_dir = HOME_RECEIPT_DIR / "runs" / run_id
    home_run_dir.mkdir(parents=True, exist_ok=True)
    for fn in ("summary.json", "selection.json", "receipt.json"):
        shutil.copy2(run_dir / fn, home_run_dir / fn)
    for mp in sorted(run_dir.glob("epoch-*.metrics.json")):
        shutil.copy2(mp, home_run_dir / mp.name)

    return selection_summary


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"values": [float(v) for v in vals], "mean": float(m), "sample_std": float(s)}


def build_final_comparison_report(run_summaries: dict, init_parity: dict, gpu_info: dict):
    """Aggregate all 6 runs, compute per-objective mean/SD, paired differences, and historical comparison."""
    by_obj = {"HARD": [], "LEGACY": []}
    for run_id, _seed, obj in RUN_ORDER:
        by_obj[obj].append(run_summaries[run_id])

    def summarize_arm(runs_list):
        macro_f1s = [r["validation_at_selected"]["macro_f1"] for r in runs_list]
        ces = [r["validation_at_selected"]["diagnostic_ce"] for r in runs_list]
        accs = [r["validation_at_selected"]["accuracy"] for r in runs_list]
        w_ces = [r["validation_at_selected"]["weighted_diagnostic_ce"] for r in runs_list]
        ep_sel = [r["selected_epoch"] for r in runs_list]
        ep_stop = [r["stopped_epoch"] for r in runs_list]
        per_class_agg = {}
        for c_name in CLASS_ORDER:
            per_class_agg[c_name] = {
                "precision": mean_std([r["validation_at_selected"]["per_class"][c_name]["precision"] for r in runs_list]),
                "recall": mean_std([r["validation_at_selected"]["per_class"][c_name]["recall"] for r in runs_list]),
                "f1": mean_std([r["validation_at_selected"]["per_class"][c_name]["f1"] for r in runs_list]),
                "support": int(runs_list[0]["validation_at_selected"]["per_class"][c_name]["support"]),
            }
        return {
            "macro_f1": mean_std(macro_f1s),
            "diagnostic_ce": mean_std(ces),
            "weighted_diagnostic_ce": mean_std(w_ces),
            "accuracy": mean_std(accs),
            "selected_epoch": mean_std(ep_sel),
            "stopped_epoch": mean_std(ep_stop),
            "per_class": per_class_agg,
        }

    arm_summary = {
        "HARD": summarize_arm(by_obj["HARD"]),
        "LEGACY": summarize_arm(by_obj["LEGACY"]),
    }

    # Paired differences (LEGACY - HARD) per seed
    paired_by_seed = {}
    for seed in (42, 43, 44):
        h = run_summaries[f"HARD-{seed}"]["validation_at_selected"]
        l = run_summaries[f"LEGACY-{seed}"]["validation_at_selected"]
        paired_by_seed[str(seed)] = {
            "seed": seed,
            "delta_macro_f1": float(l["macro_f1"] - h["macro_f1"]),
            "delta_diagnostic_ce": float(l["diagnostic_ce"] - h["diagnostic_ce"]),
            "delta_weighted_diagnostic_ce": float(l["weighted_diagnostic_ce"] - h["weighted_diagnostic_ce"]),
            "delta_accuracy": float(l["accuracy"] - h["accuracy"]),
            "delta_per_class_f1": {
                c_name: float(l["per_class"][c_name]["f1"] - h["per_class"][c_name]["f1"])
                for c_name in CLASS_ORDER
            },
            "delta_per_class_recall": {
                c_name: float(l["per_class"][c_name]["recall"] - h["per_class"][c_name]["recall"])
                for c_name in CLASS_ORDER
            },
            "delta_per_class_precision": {
                c_name: float(l["per_class"][c_name]["precision"] - h["per_class"][c_name]["precision"])
                for c_name in CLASS_ORDER
            },
        }

    paired_summary = {
        "delta_macro_f1": mean_std([paired_by_seed[str(s)]["delta_macro_f1"] for s in (42, 43, 44)]),
        "delta_diagnostic_ce": mean_std([paired_by_seed[str(s)]["delta_diagnostic_ce"] for s in (42, 43, 44)]),
        "delta_accuracy": mean_std([paired_by_seed[str(s)]["delta_accuracy"] for s in (42, 43, 44)]),
        "delta_benign_f1": mean_std([paired_by_seed[str(s)]["delta_per_class_f1"]["benign"] for s in (42, 43, 44)]),
        "delta_indeterminate_f1": mean_std(
            [paired_by_seed[str(s)]["delta_per_class_f1"]["indeterminate"] for s in (42, 43, 44)]
        ),
        "delta_malignant_f1": mean_std(
            [paired_by_seed[str(s)]["delta_per_class_f1"]["malignant"] for s in (42, 43, 44)]
        ),
    }

    # Load frozen 807-nodule historical supervision results for comparison
    historical_807 = None
    if HISTORICAL_SUP_FROZEN.exists():
        historical_807 = json.loads(HISTORICAL_SUP_FROZEN.read_bytes())

    return {
        "schema": "GLADS_CT_TRAIN1297_MULTISLICE_SIX_RUN_FINAL_REPORT_v1",
        "project": "GLADS-CT",
        "extension": "TRAIN1297 + Multi-Slice (1,297 nodules, 4,273 views, 1 view/nodule/epoch)",
        "evaluation_partition": "Frozen Validation (N=143 representative nodules, 84 patients; TEST=183 untouched)",
        "hardware": gpu_info,
        "initial_state_parity_by_seed": {str(k): v for k, v in init_parity.items()},
        "runs": run_summaries,
        "arm_summary": arm_summary,
        "paired_differences_legacy_minus_hard": {
            "by_seed": paired_by_seed,
            "summary": paired_summary,
        },
        "historical_807_reference": historical_807,
    }


def run_self_test():
    """End-to-end synthetic smoke test verifying all APIs, signatures, and checkpoint serialization."""
    torch.set_default_dtype(torch.float32)
    streams = TrainingStreams(42)
    model = R50Hybrid(seed=42, objective="LEGACY", initialization="RANDOM")
    optimizer, ledger = make_adamw(model)
    set_epoch_learning_rates(optimizer, 1)
    lrs = {g["name"]: float(g["lr"]) for g in optimizer.param_groups}
    assert len(lrs) == 4 and len(ledger) > 0

    w_cpu = class_weights_from_training_counts([582, 442, 273])
    assert tuple(w_cpu.shape) == (3,)

    # Test decode_grayscale + transform
    buf = np.zeros((64, 64), dtype=np.float32)
    import io
    s = io.BytesIO()
    np.save(s, buf, allow_pickle=False)
    dec = decode_grayscale(s.getvalue())
    tr_val = transform(dec, trace=None)
    plan = streams.epoch_plan(["n1", "n2"], epoch=1)
    tr_train = transform(dec, trace=plan["augmentation"][0])
    assert tuple(tr_val.shape) == (3, 224, 224) and tuple(tr_train.shape) == (3, 224, 224)

    # Test forward + objective_loss + backward + step
    bx = torch.stack([tr_train, tr_val], dim=0)
    by = torch.tensor([0, 2], dtype=torch.int64)
    bq = torch.tensor([[1.0, 0.0, 0.0], [0.25, 0.25, 0.50]], dtype=torch.float32)
    bu = torch.tensor([0.0, 0.5], dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    out = model(bx)
    for obj in ("HARD", "LEGACY"):
        lp = objective_loss(
            obj,
            out["logits"],
            by,
            w_cpu,
            reader_targets=bq,
            disagreement=bu,
            reader_logits=out["reader_logits"],
        )
        assert torch.isfinite(lp["total"]).all()
    lp["total"].backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True).item())
    assert math.isfinite(gnorm)
    optimizer.step()

    # Test diagnostics + selection + checkpoint encoding/decoding
    diag = compute_split_diagnostics(["n1", "n2"], out["logits"].detach(), by, w_cpu, split="val")
    vrec = ValidationEpoch(epoch=1, macro_f1=float(diag["macro_f1"]), diagnostic_ce=float(diag["diagnostic_ce"]))
    pat = ValidationPatience()
    stop = bool(pat.observe(vrec))
    best_rec = select_checkpoint([vrec])
    payload = {
        "schema": "performance_v2_checkpoint_v1",
        "run_id": "SELF-TEST",
        "seed": 42,
        "objective": "LEGACY",
        "initialization": "RADIMAGENET",
        "epoch": 1,
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": capture_cpu_rng(streams),
        "patience": {
            "last_epoch": int(pat.last_epoch),
            "best": float(pat.best) if math.isfinite(pat.best) else None,
            "without_improvement": int(pat.without_improvement),
            "stopped": bool(pat.stopped),
            "stop": stop,
        },
        "history": [{"epoch": int(vrec.epoch), "macro_f1": float(vrec.macro_f1), "diagnostic_ce": float(vrec.diagnostic_ce)}],
        "selection": {"epoch": int(best_rec.epoch), "macro_f1": float(best_rec.macro_f1), "diagnostic_ce": float(best_rec.diagnostic_ce)},
    }
    raw_ckpt = encode_state(payload)
    rt = decode_state(raw_ckpt, expected_sha256=digest(raw_ckpt))
    assert rt["epoch"] == 1 and rt["selection"]["epoch"] == 1
    print(json.dumps({"self_test": "PASS", "checkpoint_bytes": len(raw_ckpt)}))


def main():
    if "--self-test" in sys.argv:
        run_self_test()
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    HOME_RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    status = {"status": "starting", "started_at_unix": time.time(), "completed_runs": []}
    log_progress(status)

    apply_resource_limits()
    gpu_info = configure_determinism()
    status["gpu_info"] = gpu_info

    inputs = load_and_verify_inputs(status)
    status["phase"] = "verifying_seed_pair_initial_parity"
    log_progress(status)
    init_parity = verify_seed_pair_initial_parity(inputs["rad_raw"], inputs["rad_receipt_bytes"])
    status["init_parity"] = {str(k): v for k, v in init_parity.items()}
    log_progress(status)

    run_summaries = {}
    for run_id, seed, objective in RUN_ORDER:
        summary = execute_single_run(
            run_id=run_id,
            seed=seed,
            objective=objective,
            inputs=inputs,
            init_parity_info=init_parity[seed],
            status=status,
        )
        run_summaries[run_id] = summary
        status["completed_runs"].append(
            {
                "run_id": run_id,
                "selected_epoch": summary["selected_epoch"],
                "stopped_epoch": summary["stopped_epoch"],
                "val_macro_f1": summary["validation_at_selected"]["macro_f1"],
                "val_diagnostic_ce": summary["validation_at_selected"]["diagnostic_ce"],
                "val_accuracy": summary["validation_at_selected"]["accuracy"],
            }
        )
        log_progress(status)

    final_report = build_final_comparison_report(run_summaries, init_parity, gpu_info)
    report_sha = write_json_atomic(OUTPUT_ROOT / "SIX-RUN-FINAL-REPORT.json", final_report)
    write_json_atomic(HOME_RECEIPT_DIR / "SIX-RUN-FINAL-REPORT.json", final_report)

    status["status"] = "complete"
    status["phase"] = "complete"
    status["final_report_sha256"] = report_sha
    log_progress(status)
    print(json.dumps({"status": "PASS", "final_report_sha256": report_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
