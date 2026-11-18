from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch = pytest.importorskip("torch")

from consendxformer.hashing import sha256_file
from consendxformer.phase4_attention_runtime import FrozenB1, infer_matched_attention
import consendxformer.phase4_attention_runtime as runtime
from test_phase4_posthoc import TinyPosthocModel, _load, _attention_identity_batch, _sealed_attention_loader


class ImageOnlyModel(TinyPosthocModel):
    def forward(self, image):
        assert isinstance(image, torch.Tensor)
        assert not torch.is_grad_enabled()
        assert not self.training
        return super().forward(image)


def _pair(tmp_path, monkeypatch):
    cdx = _load(tmp_path, monkeypatch)
    model = ImageOnlyModel()
    model.load_state_dict(cdx.model.state_dict())
    model.eval().requires_grad_(False)
    cdx = replace(cdx, model=model)
    image = torch.ones((2, 1, 2, 2))
    with torch.no_grad():
        outputs = model(image)
    predictions = cdx.source.predictions.copy()
    predictions["true_class"] = cdx.source.supervision["label"].tolist()
    for i, name in enumerate(("benign", "indeterminate", "malignant")):
        predictions[f"logit_{name}"] = outputs["logits"][:, i].numpy()
        predictions[f"prob_{name}"] = outputs["probs"][:, i].numpy()
    cdx = replace(cdx, source=replace(cdx.source, predictions=predictions))
    b1_model = ImageOnlyModel()
    b1_model.load_state_dict(model.state_dict())
    b1_model.eval().requires_grad_(False)
    b1 = FrozenB1(b1_model, replace(cdx.source, model="B1", predictions=predictions.iloc[::-1].copy()), "a" * 64, {})
    batch = {"image": image, "soft_consensus_mask": torch.ones_like(image),
             "true_class": cdx.source.supervision["label"].tolist(), **_attention_identity_batch(cdx)}
    loader = _sealed_attention_loader(cdx, [batch])
    return cdx, b1, loader


def test_matched_no_grad_image_only_inference_replays_reordered_saved_rows(tmp_path, monkeypatch):
    cdx, b1, loader = _pair(tmp_path, monkeypatch)
    result = infer_matched_attention(cdx, b1, loader)
    assert result["actual_attention_shapes"] == {"b1": [2, 1, 2, 2], "cdx": [2, 1, 2, 2]}
    assert result["replay"]["b1"]["consistent"] and result["replay"]["cdx"]["consistent"]
    np.testing.assert_array_equal(result["arrays"]["b1_attention"], result["arrays"]["cdx_attention"])
    assert all(p.grad is None for p in cdx.model.parameters())


def test_prediction_discrepancy_is_reported_without_rewriting_source(tmp_path, monkeypatch):
    cdx, b1, loader = _pair(tmp_path, monkeypatch)
    changed = b1.source.predictions.copy()
    changed["logit_benign"] += 1.
    b1 = replace(b1, source=replace(b1.source, predictions=changed))
    result = infer_matched_attention(cdx, b1, loader)
    assert not result["replay"]["b1"]["consistent"]
    assert result["replay"]["b1"]["logits"]["differing_values"] == 2
    pd.testing.assert_frame_equal(changed, b1.source.predictions)
    for path, digest in b1.source.source_snapshot.items():
        assert sha256_file(path) == digest


def test_inference_rejects_unfrozen_models_wrong_seed_and_changed_inputs(tmp_path, monkeypatch):
    cdx, b1, loader = _pair(tmp_path, monkeypatch)
    b1.model.train()
    with pytest.raises(ValueError, match="frozen"):
        infer_matched_attention(cdx, b1, loader)
    b1.model.eval()
    wrong = replace(b1, source=replace(b1.source, descriptor={**b1.source.descriptor, "seed": 44}))
    with pytest.raises(ValueError, match="seeds differ"):
        infer_matched_attention(cdx, wrong, loader)
    path = Path(next(iter(loader.input_snapshot)))
    path.write_bytes(b"unit-test tampering")
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        infer_matched_attention(cdx, b1, loader)


def _actual_pinned_api_roundtrip(directory: Path):
    """Runs in a fresh subprocess: actual pinned class, synthetic untrained weights."""
    from consendxformer.baseline_source import PINNED_BASELINE_SHA, snapshot_source_tree
    from consendxformer.phase3_config import load_phase3_config
    from consendxformer.phase3_identity import build_experiment_identity
    from consendxformer.phase3_checkpoints import build_best_checkpoint
    from consendxformer.phase4_evaluation_runner import SavedPredictionSource
    from consendxformer.phase4_config import load_phase4_config
    from consendxformer.phase4_execution import _model
    from consendxformer.phase4_dataset import _read_image
    import yaml

    before = snapshot_source_tree(ROOT / "external/LungDxFormer")
    bundle = load_phase3_config(ROOT / "configs/phase3_b1.yaml", seed=42, device="cpu")
    config = {**bundle.effective_phase1_config, "phase3_reproducibility": bundle.phase3_config["determinism"]}
    identity = build_experiment_identity(
        baseline_variant="B1", seed=42, baseline_source_commit=PINNED_BASELINE_SHA,
        project_commit="1" * 40, combined_config_sha256=bundle.combined_config_sha256,
        metadata_sha256="2" * 64, cohort_sha256="3" * 64,
        ordered_sample_sha256="4" * 64, frozen_split_bundle_sha256="5" * 64,
    ).to_dict()
    for name in ("config", "checkpoints", "history"):
        (directory / name).mkdir()
    (directory / "config/experiment_identity.json").write_text(json.dumps(identity))
    (directory / "config/phase1_effective.yaml").write_text(yaml.safe_dump(bundle.effective_phase1_config))
    (directory / "config/phase3_effective.yaml").write_text(yaml.safe_dump(bundle.phase3_config))
    (directory / "history/epoch_history.json").write_text(json.dumps([{"epoch": 1, "train": {"loss": 1.}, "validation": {"loss": .5}}]))
    with runtime.pinned_b1_api(directory / "disabled_cache") as (model_class, source_report):
        model = runtime.construct_b1(model_class, config)
        record = build_best_checkpoint(
            epoch=1, model_state=model.state_dict(), optimizer_state={}, scheduler_state=None,
            best_validation_loss=.5, seed=42, baseline_variant="B1",
            config_sha256=identity["combined_config_sha256"], experiment_id=identity["experiment_id"],
        )
        record.update({key: identity[key] for key in ("baseline_source_commit", "project_commit", "metadata_sha256", "cohort_sha256", "ordered_sample_sha256")})
        record.update(frozen_split_sha256=identity["frozen_split_bundle_sha256"], class_order=["benign", "indeterminate", "malignant"],
                      effective_baseline_config=config, validation_metrics={"loss": .5})
        torch.save(record, directory / "checkpoints/best.pt")
        (directory / "checkpoints/last.pt").write_bytes(b"must never load last")
        snapshot = {str(path.resolve()): sha256_file(path) for path in directory.rglob("*") if path.is_file()}
        source = SavedPredictionSource("B1", directory, {"seed": 42}, pd.DataFrame(), None, {}, snapshot)
        original_loader = runtime.load_b1_prediction_source
        runtime.load_b1_prediction_source = lambda path: source
        try:
            loaded = runtime.load_frozen_b1(directory, model_class, device="cpu")
            assert not loaded.model.training and all(not p.requires_grad for p in loaded.model.parameters())
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(loaded.model.state_dict()[key], tensor)
            cdx = _model(load_phase4_config(ROOT / "configs/phase4_consensdxformer.yaml").phase4_config)
            cdx.eval().requires_grad_(False)
            with torch.no_grad():
                image = torch.ones((1, 1, 64, 64))
                a, b = loaded.model(image), cdx(image)
            assert a["attention_map"].shape == b["attention_map"].shape == (1, 1, 8, 8)
            torch.testing.assert_close(a["attention_map"].sum(), torch.tensor(1.))
            # Verify the reused grayscale reader against the actual B1 reader.
            from lungdxformer.data.dataset import _read_image as baseline_read
            crop = directory / "synthetic_ct.npy"
            np.save(crop, np.arange(64 * 64, dtype=np.float32).reshape(64, 64))
            np.testing.assert_array_equal(_read_image(crop, 64), baseline_read(str(crop), 64))
            record["effective_baseline_config"] = {**config, "unexpected": True}
            torch.save(record, directory / "checkpoints/best.pt")
            with pytest.raises(ValueError, match="effective_baseline_config"):
                runtime.load_frozen_b1(directory, model_class, device="cpu")
        finally:
            runtime.load_b1_prediction_source = original_loader
    assert before == snapshot_source_tree(ROOT / "external/LungDxFormer")
    print("PINNED_API_ROUNDTRIP_OK")


def test_actual_pinned_model_checkpoint_and_preprocessing_api_in_fresh_process(tmp_path):
    if not (ROOT / "external/LungDxFormer/.git").exists():
        pytest.skip("Optional pinned external checkout is not installed; never downloads it")
    code = "import sys; from pathlib import Path; sys.path.insert(0, 'tests'); from test_phase4_attention_runtime import _actual_pinned_api_roundtrip; _actual_pinned_api_roundtrip(Path(sys.argv[1]))"
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PINNED_API_ROUNDTRIP_OK" in result.stdout
