"""Frozen metadata interfaces. Caller supplies bytes; no paths or data readers."""
from dataclasses import dataclass
import math

from .reuse import _hash, _json, canonical, digest

CLASS_ORDER = ("benign", "indeterminate", "malignant")
PRODUCTION_COUNTS = {"train": (807, 437, (348, 224, 235)),
                     "val": (143, 84, (65, 38, 40))}
RECORD_FIELDS = {"patient_id", "series_instance_uid", "nodule_id", "split",
                 "label", "reader_target", "disagreement", "valid_readers",
                 "image_sha256", "supervision_sha256"}


@dataclass(frozen=True)
class InputRecord:
    patient_id: str
    series_instance_uid: str
    nodule_id: str
    split: str
    label: int
    reader_target: tuple
    disagreement: float
    valid_readers: int
    image_sha256: str
    supervision_sha256: str

    @classmethod
    def parse(cls, row):
        if type(row) is not dict or set(row) != RECORD_FIELDS:
            raise ValueError("exact frozen record fields required")
        for key in ("patient_id", "series_instance_uid", "nodule_id"):
            if not isinstance(row[key], str) or not row[key] or len(row[key]) > 256:
                raise ValueError("nonempty bounded identity required")
        if (row["split"] not in ("train", "val") or type(row["label"]) is not int
                or row["label"] not in (0, 1, 2) or type(row["valid_readers"]) is not int
                or row["valid_readers"] not in (2, 3, 4)):
            raise ValueError("frozen split/label/eligibility mismatch; no TEST adapter")
        q, u = row["reader_target"], row["disagreement"]
        finite_unit = lambda x: type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1
        if (type(q) is not list or len(q) != 3 or not all(finite_unit(x) for x in q)
                or abs(math.fsum(q) - 1) > 1e-6 or not finite_unit(u)):
            raise ValueError("invalid frozen supervision; no renormalization")
        if not all(_hash(row[k]) for k in ("image_sha256", "supervision_sha256")):
            raise ValueError("exact source digests required")
        return cls(**dict(row, reader_target=tuple(q)))

    def as_dict(self):
        return dict(self.__dict__, reader_target=list(self.reader_target))


class FrozenInputs:
    """Hash-checked new interface lock, distinct from the archived cohort ledger.

    expected_sha256 must come from independent review, not from this payload.
    Source/supervision hashes are declarations until a future approved adapter
    verifies original bytes. verify_image_bytes verifies only supplied bytes.
    """
    def __init__(self, raw, *, expected_sha256):
        if type(raw) is not bytes or not _hash(expected_sha256) or digest(raw) != expected_sha256:
            raise ValueError("input lock digest mismatch")
        obj = _json(raw)
        if (type(obj) is not dict or set(obj) != {"schema", "scope", "class_order",
                "archived_cohort_sha256", "transform_sha256", "records"}
                or obj["schema"] != "performance_v2_input_interface_v1"
                or obj["scope"] not in ("synthetic", "production")
                or obj["class_order"] != list(CLASS_ORDER)
                or not _hash(obj["archived_cohort_sha256"])
                or not _hash(obj["transform_sha256"])
                or type(obj["records"]) is not list or not obj["records"]):
            raise ValueError("invalid input lock")
        records = tuple(InputRecord.parse(x) for x in obj["records"])
        ids = [r.nodule_id for r in records]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate prediction identity")
        train = {r.patient_id for r in records if r.split == "train"}
        val = {r.patient_id for r in records if r.split == "val"}
        if not train or not val or train & val:
            raise ValueError("both patient-disjoint splits required")
        series = {}
        for row in records:
            identity = (row.patient_id, row.split)
            if series.setdefault(row.series_instance_uid, identity) != identity:
                raise ValueError("series identity crosses patient or split")
        if obj["scope"] == "production":
            for split, expected in PRODUCTION_COUNTS.items():
                part = [r for r in records if r.split == split]
                observed = (len(part), len({r.patient_id for r in part}),
                            tuple(sum(r.label == c for r in part) for c in range(3)))
                if observed != expected:
                    raise ValueError("frozen production membership/counts mismatch")
            if obj["archived_cohort_sha256"] != "74ade0365dc6b7ea5593b4736692956b863b4245e122829e2921b0937663b5b1":
                raise ValueError("archived cohort identity mismatch")
        self._records = records
        self._index = {r.nodule_id: r for r in records}
        self.sha256 = expected_sha256
        self.scope = obj["scope"]
        self.transform_sha256 = obj["transform_sha256"]

    def records(self, split):
        if split not in ("train", "val"):
            raise ValueError("only frozen TRAIN/VALIDATION interfaces exist")
        return tuple(r for r in self._records if r.split == split)

    def ids(self, split):
        return tuple(r.nodule_id for r in self.records(split))

    def record(self, nodule_id):
        if nodule_id not in self._index:
            raise ValueError("unknown input identity")
        return self._index[nodule_id]

    def verify_image_bytes(self, nodule_id, raw):
        if type(raw) is not bytes or digest(raw) != self.record(nodule_id).image_sha256:
            raise ValueError("supplied source-image bytes mismatch")
        return digest(raw)

    def supervision(self, ids, *, split):
        import torch
        ids = tuple(ids)
        if not ids or len(ids) != len(set(ids)) or any(self.record(i).split != split for i in ids):
            raise ValueError("unique batch identities in declared split required")
        self.records(split)
        rows = [self.record(i) for i in ids]
        return {"labels": torch.tensor([r.label for r in rows], dtype=torch.int64),
                "reader_targets": torch.tensor([r.reader_target for r in rows], dtype=torch.float32),
                "disagreement": torch.tensor([r.disagreement for r in rows], dtype=torch.float32)}

    def validate_batch(self, ids, *, split, labels, reader_targets, disagreement, image_hashes):
        import torch
        expected = self.supervision(ids, split=split)
        for key, value in (("labels", labels), ("reader_targets", reader_targets),
                           ("disagreement", disagreement)):
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.requires_grad or value.dtype != expected[key].dtype
                    or not torch.equal(value, expected[key])):
                raise ValueError("frozen CPU supervision drift: " + key)
        if image_hashes != {i: self.record(i).image_sha256 for i in ids}:
            raise ValueError("fresh supplied image digest bindings required")
        return digest(canonical({"lock": self.sha256, "split": split, "ids": list(ids)}))
