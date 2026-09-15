"""Launcher tests use fake processes only; no GPU discovery or jobs are started."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import consendxformer.phase4_ablation_launch as launch
import consendxformer.phase4_ablation_runner as runner
from consendxformer.phase4_ablation_contracts import AblationRunSpec


def jobs(tmp_path):
    return [launch.AssignedJob(AblationRunSpec("attention_alignment", seed),
                              f"GPU-00000000-0000-0000-0000-{seed:012d}", tmp_path / f"synthetic-full-{seed}")
            for seed in (42, 43, 44)]


def test_read_only_plan_lists_first_attention_group_and_all_twelve_jobs(monkeypatch, capsys):
    monkeypatch.setattr(runner, "prepare_ablation", lambda *args: pytest.fail("planning must not inspect real training inputs"))
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("planning must not spawn"))
    assert launch.ablation_cli(["plan"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["read_only"] is True and record["training_execution_authorized"] is False
    assert len(record["runs"]) == 12
    assert [(r["variant"], r["seed"]) for r in record["runs"][:3]] == [("attention_alignment", seed) for seed in (42, 43, 44)]
    assert record["protocol"]["preregistered"] is False


@pytest.mark.parametrize("mutation", ["gpu_duplicate", "run_duplicate", "implicit_gpu", "bad_seed", "empty"])
def test_launcher_rejects_ambiguous_duplicate_or_unauthorized_assignments(tmp_path, mutation):
    assigned = jobs(tmp_path)
    if mutation == "gpu_duplicate": assigned[1] = launch.AssignedJob(assigned[1].spec, assigned[0].gpu, assigned[1].full_source)
    elif mutation == "run_duplicate": assigned[1] = launch.AssignedJob(assigned[0].spec, assigned[1].gpu, assigned[1].full_source)
    elif mutation == "implicit_gpu": assigned[1] = launch.AssignedJob(assigned[1].spec, "0", assigned[1].full_source)
    elif mutation == "bad_seed": assigned[1] = launch.AssignedJob(AblationRunSpec("attention_alignment", 45), assigned[1].gpu, assigned[1].full_source)
    else: assigned = []
    with pytest.raises(ValueError):
        launch.validate_jobs(assigned)


def fake_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    def prepare(spec, path):
        return SimpleNamespace(identity={"variant": spec.variant, "seed": spec.seed, "code_commit": "a" * 40},
                               output_dir=tmp_path / f"run-{spec.seed}")
    monkeypatch.setattr(runner, "prepare_ablation", prepare)
    monkeypatch.setattr(runner, "recheck_prepared", lambda p: None)


def test_explicit_gpu_environment_independent_logs_exit_codes_and_no_retry(tmp_path, monkeypatch):
    fake_preflight(tmp_path, monkeypatch)
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            seed = int(command[command.index("--seed") + 1])
            calls.append((command, kwargs))
            self.code = 7 if seed == 43 else 0
            kwargs["stdout"].write(f"synthetic process {seed}".encode())
        def wait(self, **kwargs): return self.code
        def poll(self): return self.code
    monkeypatch.setattr(launch.subprocess, "Popen", Process)
    assigned = jobs(tmp_path)
    code, output = launch.launch_jobs(assigned, authorize_training=True)
    assert code == 1 and len(calls) == 3
    for job, (command, kwargs) in zip(assigned, calls):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == job.gpu
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert kwargs["env"]["PYTHONHASHSEED"] == str(job.spec.seed)
        assert "--authorize-training" in command
    results = json.loads((output / "exit_codes.json").read_text())
    assert [r["exit_code"] for r in results["jobs"]] == [0, 7, 0]
    assert results["all_succeeded"] is False
    assert len(list(output.glob("*.log"))) == len(list(output.glob("*.exit.json"))) == 3
    with pytest.raises(FileExistsError):
        launch.launch_jobs(assigned, authorize_training=True)
    assert len(calls) == 3


def test_whole_group_preflight_failure_starts_nothing(tmp_path, monkeypatch):
    fake_preflight(tmp_path, monkeypatch)
    original = runner.prepare_ablation
    def prepare(spec, path):
        if spec.seed == 43: raise FileExistsError("already completed or partial")
        return original(spec, path)
    monkeypatch.setattr(runner, "prepare_ablation", prepare)
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must preflight all jobs before starting any"))
    with pytest.raises(FileExistsError):
        launch.launch_jobs(jobs(tmp_path), authorize_training=True)
    assert not list(tmp_path.rglob("*.log"))


def test_launcher_requires_authorization_and_cleans_up_only_its_own_children(tmp_path, monkeypatch):
    assigned = jobs(tmp_path)
    with pytest.raises(ValueError, match="authorization"):
        launch.launch_jobs(assigned)
    fake_preflight(tmp_path, monkeypatch)
    live = []
    class Process:
        def __init__(self, command, **kwargs):
            if live: raise OSError("synthetic spawn failure")
            self.code = None
            live.append(self)
        def poll(self): return self.code
        def terminate(self): self.code = -15
        def wait(self, **kwargs): return self.code
    monkeypatch.setattr(launch.subprocess, "Popen", Process)
    with pytest.raises(OSError, match="spawn failure"):
        launch.launch_jobs(assigned, authorize_training=True)
    assert live[0].code == -15
    record = json.loads(next(tmp_path.rglob("exit_codes.json")).read_text())
    assert record["interrupted"] and not record["all_succeeded"]
