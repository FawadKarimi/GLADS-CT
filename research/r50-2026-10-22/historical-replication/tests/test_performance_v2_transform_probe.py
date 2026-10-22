"""Synthetic oracle/report tests; these do not certify installed torchvision transforms."""
import hashlib
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/54_probe_performance_v2_transforms.py"
spec = importlib.util.spec_from_file_location("transform_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_integer_oracle_hand_written_asymmetric_fixture():
    raw = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    assert np.array_equal(probe.integer_geometry_reference(np, raw, 1), [[7,4,1],[8,5,2],[9,6,3]])
    assert np.array_equal(probe.integer_geometry_reference(np, raw, -1), [[3,6,9],[2,5,8],[1,4,7]])
    assert np.array_equal(probe.integer_geometry_reference(np, raw, 1, 1, -1), [[0,8,5],[0,9,6],[0,0,0]])
    assert np.array_equal(probe.integer_geometry_reference(np, raw, 0, -1, 1), [[0,0,0],[2,3,0],[5,6,0]])


def test_pixel_center_inverse_and_clockwise_direction():
    sx, sy = probe.inverse_coordinates(np, 90, 3, -5)
    # Input center (31.5,20.5) -> output (206.5,26.5).
    assert sx[26, 206] == pytest.approx(31, abs=1e-12)
    assert sy[26, 206] == pytest.approx(20, abs=1e-12)


def test_bilinear_oracle_hand_calculated_centers_and_clamping():
    raw = np.tile(np.arange(64, dtype=np.float32), (64, 1))
    result = probe.resize_reference(np, raw)
    assert result[0, 0] == 0
    assert result[100, 3] == pytest.approx(.5)
    assert result[0, -1] == 63
    assert np.array_equal(probe.resize_reference(np, np.ones((64, 64))), np.ones((224, 224)))


@pytest.mark.parametrize("v,expected", [(0.5,0), (1.5,2), (2.5,2), (-0.5,0), (-1.5,-2), (-2.5,-2)])
def test_ties_to_even(v, expected):
    assert probe.translation_pixels(v/224, -v/224) == [expected, -expected]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 256])
def test_scaling_rejects_invalid(value):
    with pytest.raises(ValueError): probe.scale_grayscale(np, np.full((64,64), value))


def test_scaling_independent_copy_and_threshold():
    raw = np.ones((64,64), np.float32)
    out = probe.scale_grayscale(np, raw)
    assert np.array_equal(out, raw) and not np.shares_memory(out, raw)
    assert np.allclose(probe.scale_grayscale(np, raw*2), 2/255, atol=1e-8, rtol=0)


@pytest.mark.parametrize("failure", ["warning", "error", "interrupt"])
def test_partial_report_preserved_and_failure_captured(tmp_path, failure):
    reporter = probe.Reporter(tmp_path/"out")
    reporter.step("success", lambda: {"independent": True})
    def fail():
        if failure == "warning": warnings.warn("invented test warning", RuntimeWarning)
        elif failure == "error": raise RuntimeError("invented test error")
        else: raise KeyboardInterrupt("invented interrupt")
    with pytest.raises(BaseException): reporter.step("failure", fail)
    data = json.loads((tmp_path/"out/report.json").read_text())
    assert data["status"] == "failed" and data["passed_steps"] == ["success"]
    assert data["failed_step"] == "failure" and data["failure_traceback"]
    assert bool(data["warnings"]) == (failure == "warning")
    assert data["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert data["model_forwards"] == 0 and data["pilot_execution_authorized"] is False


def test_no_overwrite(tmp_path):
    output = tmp_path/"out"
    probe.Reporter(output)
    before = (output/"report.json").read_bytes()
    with pytest.raises(FileExistsError): probe.Reporter(output)
    assert (output/"report.json").read_bytes() == before


def test_cpu_guard_precedes_dependency_import(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    def forbidden(*args): raise AssertionError("dependency import must not occur")
    monkeypatch.setattr(probe.importlib, "import_module", forbidden)
    assert probe.run(tmp_path/"out") == 1
    data = json.loads((tmp_path/"out/report.json").read_text())
    assert data["failed_step"] == "cpu_process_guard" and not data["passed_steps"]
