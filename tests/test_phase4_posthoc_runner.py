"""CPU-only integration of real tiny checkpoint inference and artifact sealing."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch = pytest.importorskip("torch")

from consendxformer.hashing import sha256_file, sha256_json
from consendxformer.phase4_evaluation_runner import Phase4EvaluationError
import consendxformer.phase4_posthoc as inference
import consendxformer.phase4_posthoc_runner as runner
from test_phase4_posthoc import (
    TinyPosthocModel, _fixture, _attention_identity_batch, _sealed_attention_loader,
)


def _install(tmp_path, monkeypatch):
    sources = [_fixture(tmp_path / str(seed), seed=seed)[0] for seed in (42, 43, 44)]
    # All three tiny checkpoints use the same test images, as in production.
    sources = [replace(source, supervision=sources[0].supervision.copy()) for source in sources]
    by_path = {source.source_dir: source for source in sources}
    monkeypatch.setattr(runner, "validate_evaluator_repository", lambda root: {
        "branch": "phase4-consendxformer", "commit": "c" * 40, "clean": True,
    })
    # Synthetic fixtures cannot satisfy real cohort pins. Only the source
    # trust boundary is replaced; checkpoint loading/inference/metrics run.
    monkeypatch.setattr(runner, "load_phase4_final_prediction_source", lambda path: by_path[Path(path)])
    monkeypatch.setattr(inference, "load_phase4_final_prediction_source", lambda path: by_path[Path(path)])
    monkeypatch.setattr(runner, "validate_final_source_descriptors", lambda rows: rows)
    monkeypatch.setattr(runner, "load_frozen_final_best_model", lambda path, device: inference.load_frozen_final_best_model(
        path, device=device, model_factory=lambda config: TinyPosthocModel(),
    ))

    def loader(frozen):
        return _sealed_attention_loader(frozen, [{
            "image": torch.ones((2, 1, 2, 2)),
            "soft_consensus_mask": torch.ones((2, 1, 2, 2)),
            **_attention_identity_batch(frozen),
        }])

    # Resolve all masks against the shared first source (no file rewrites).
    def shared_loader(frozen):
        shared = replace(frozen, source=replace(frozen.source, source_dir=sources[0].source_dir))
        return loader(shared)

    monkeypatch.setattr(runner, "build_frozen_attention_loader", shared_loader)
    return sources


def test_frozen_inference_writes_deterministic_sealed_bundle(tmp_path, monkeypatch):
    sources = _install(tmp_path, monkeypatch)
    paths = [source.source_dir for source in sources]
    kwargs = dict(phase4_source_dirs=paths, output_root=tmp_path / "evaluation")
    before = torch.are_deterministic_algorithms_enabled()
    result = runner.run_phase4_posthoc_evaluation(**kwargs)
    assert torch.are_deterministic_algorithms_enabled() == before
    data = json.loads(result.machine_results.read_text())
    assert data["training_performed"] is False
    assert data["boundary_stress"]["status"] == "not_requested"
    assert data["attention_alignment"]["aggregate"]["dice"]["mean_of_seed_means"] == 1.0
    manifest = json.loads(result.artifact_manifest.read_text())
    assert manifest["bundle_sha256"] == sha256_json(manifest["files"])
    for relative, digest in manifest["files"].items():
        assert sha256_file(result.output_dir / relative) == digest
    assert json.loads(result.completion_marker.read_text())["artifact_manifest_sha256"] == sha256_file(result.artifact_manifest)
    for source in sources:
        for path, digest in source.source_snapshot.items():
            assert sha256_file(path) == digest
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run_phase4_posthoc_evaluation(**kwargs)
    second = runner.run_phase4_posthoc_evaluation(**{**kwargs, "output_root": tmp_path / "second"})
    assert second.evaluation_id == result.evaluation_id
    assert second.machine_results.read_bytes() == result.machine_results.read_bytes()
    assert second.artifact_manifest.read_bytes() == result.artifact_manifest.read_bytes()


def test_inference_source_mutation_never_completes(tmp_path, monkeypatch):
    sources = _install(tmp_path, monkeypatch)
    original = runner.evaluate_attention_alignment_from_loader

    def mutate(frozen, loader):
        result = original(frozen, loader)
        frozen.checkpoint_path.write_bytes(b"synthetic tampering")
        return result

    monkeypatch.setattr(runner, "evaluate_attention_alignment_from_loader", mutate)
    output = tmp_path / "evaluation"
    with pytest.raises(Phase4EvaluationError, match="changed during evaluation"):
        runner.run_phase4_posthoc_evaluation(
            phase4_source_dirs=[source.source_dir for source in sources], output_root=output,
        )
    assert not list(output.rglob("COMPLETED.json"))


def test_missing_seed_rejected_before_inference(tmp_path):
    with pytest.raises(Phase4EvaluationError, match="exactly three"):
        runner.run_phase4_posthoc_evaluation(phase4_source_dirs=[], output_root=tmp_path)


def test_boundary_inference_artifacts_are_explicit_and_separate(tmp_path, monkeypatch):
    from test_phase4_boundary_runtime import _fixture_source

    sources = _install(tmp_path, monkeypatch)
    boundary, primary, _ = _fixture_source(tmp_path / "boundary", monkeypatch)
    monkeypatch.setattr(runner, "load_repository_phase4_boundary_source", lambda **kwargs: boundary)

    def boundary_predict(frozen, verified):
        # Bind the separate tiny boundary fixture to each tiny frozen model;
        # real cohort validation and boundary inference still execute.
        bound = replace(frozen, source=replace(
            frozen.source, supervision=primary,
            descriptor={**frozen.source.descriptor, **boundary.provenance["primary_cohort_binding"]},
        ))
        return inference.predict_validated_boundary_cohort(bound, verified)

    monkeypatch.setattr(runner, "predict_validated_boundary_cohort", boundary_predict)
    result = runner.run_phase4_posthoc_evaluation(
        phase4_source_dirs=[source.source_dir for source in sources],
        output_root=tmp_path / "evaluation", include_boundary=True,
    )
    data = json.loads(result.machine_results.read_text())
    stress = data["boundary_stress"]
    assert stress["status"] == "evaluated_separately"
    assert len(stress["per_seed"]) == 3
    assert stress["primary_metrics_computed"] is False
    assert all("true_class" not in row for seed in stress["per_seed"] for row in seed["rows"])
    assert (result.output_dir / "tables" / "boundary_predictions_separate.csv").is_file()
    assert result.completion_marker.is_file()
