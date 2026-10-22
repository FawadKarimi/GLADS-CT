"""Versioned, separately opted-in saved-prediction inference contract."""
import json
from pathlib import Path

from .hashing import sha256_json
from .phase5_bounded_mean import HYPOTHESIS_IDS, METHOD_ID, NULL_ID
from .phase5_inference_evidence import DOCUMENT_HASHES, CORE_SHA256, REVIEW_ZIP_SHA256

SCOPE = "paired_correctness_only"
COMPARATORS = {
    "B1": "B1", "attention_off": "attention_alignment_off",
    "disagreement_off": "disagreement_weighting_off", "ordinal_off": "ordinal_auxiliary_off",
    "fusion_off": "hierarchical_fusion_off",
}


def protocol():
    return {"schema": "phase5_inference_v1", "scope": SCOPE, "method": METHOD_ID, "null": NULL_ID,
            "documents_lf_sha256": DOCUMENT_HASHES, "core_lf_sha256": CORE_SHA256,
            "review_zip_sha256": REVIEW_ZIP_SHA256, "seeds": [42, 43, 44],
            "sources": {"complete": 3, "B1": 3, "OFF": 12}, "nodules": 183, "patients": 99,
            "logical_score_bounds": [-1, 1], "weighting": "equal_patient", "two_sided": True,
            "hypothesis_ids": list(HYPOTHESIS_IDS), "family_alpha": "1/20", "adjustment": "Holm_15",
            "precisions": [80, 160, 320], "numerical_failure": "block_entire_family",
            "all_zero": "defined_p_one_after_valid_computation", "model_forwards": 0,
            "analysis_status": "exploratory_post_hoc", "primary_endpoint": "macro_F1",
            "existing_nodule_weighted_intervals": "unchanged_descriptive_unadjusted",
            "full_supplemental_completion": False, "stage_a_complete": False,
            "requires_actual_r_outputs": True, "r_runtime": "3.6.3", "r_package": "clust.bin.pair_0.1.2"}


def load_protocol(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if sha256_json(value) != sha256_json(protocol()):
        raise ValueError("Inference integration protocol differs from reviewed v1")
    return value
