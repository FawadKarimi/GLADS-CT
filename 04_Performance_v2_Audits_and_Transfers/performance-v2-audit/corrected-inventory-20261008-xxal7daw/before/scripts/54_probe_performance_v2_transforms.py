"""Standalone synthetic CPU transform probe. No model/data/weight access or GPU API calls."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback
import warnings


CONTRACT = {
    "status": "proposed_v2_not_stage_a_equivalence",
    "resize": {"size": [224, 224], "mode": "bilinear", "align_corners": False, "antialias": True},
    "affine": {"api": "torchvision.transforms.functional.affine", "backend": "tensor",
               "interpolation": "bilinear", "scale": 1, "shear": [0, 0], "fill": 0,
               "center": [112, 112], "angle": "degrees_clockwise",
               "translation": "[horizontal, vertical], post-rotation",
               "fraction_to_pixels": "int(round(fraction * axis_length)); Python ties-to-even",
               "antialias": "not provided by affine; antialias=True applies only to resize"},
    "tolerances": {"rtol": 0, "scaling_constants_normalization_atol": 1e-6,
                   "resize_general_atol": 1e-5, "affine_atol": 5e-5,
                   "normalized_validation_chain_atol": 5e-5,
                   "normalized_train_chain_atol": 3e-4,
                   "flips_repetition_atol": 0},
    "normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]},
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def translation_pixels(horizontal, vertical, width=224, height=224):
    require(all(math.isfinite(v) for v in (horizontal, vertical)), "nonfinite translation")
    return [int(round(horizontal * width)), int(round(vertical * height))]


def scale_grayscale(np, raw):
    require(raw.shape == (64, 64), "expected grayscale 64x64")
    require(np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 255,
            "invalid grayscale finite/range")
    result = raw.astype(np.float32, copy=True)
    if result.max() > 1:
        result /= 255
    require(np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1,
            "invalid scaled range")
    return result


def resize_reference(np, source, size=224):
    """Independent pixel-center bilinear oracle for 64 -> 224 upsampling only."""
    require(source.shape == (64, 64) and size >= 64, "upsampling oracle scope")
    coords = np.clip((np.arange(size, dtype=np.float64) + .5) * 64 / size - .5, 0, 63)
    low = np.floor(coords).astype(int)
    high = np.minimum(low + 1, 63)
    frac = coords - low
    horizontal = source[:, low] * (1-frac) + source[:, high] * frac
    return horizontal[low, :] * (1-frac[:, None]) + horizontal[high, :] * frac[:, None]


def integer_geometry_reference(np, source, turns=0, dx=0, dy=0):
    """Forward integer coordinate mapping, no torchvision/torch/geometric library call."""
    h, w = source.shape
    require(h == w, "square reference required")
    result = np.zeros_like(source)
    for y in range(h):
        for x in range(w):
            if turns % 4 == 0: yy, xx = y, x
            elif turns % 4 == 1: yy, xx = x, w-1-y
            elif turns % 4 == 2: yy, xx = h-1-y, w-1-x
            else: yy, xx = h-1-x, y
            yy, xx = yy+dy, xx+dx
            if 0 <= yy < h and 0 <= xx < w:
                result[yy, xx] = source[y, x]
    return result


def inverse_coordinates(np, angle, dx, dy):
    """Output pixel centers inverted about edge-coordinate center [112,112]."""
    y, x = np.indices((224, 224), dtype=np.float64)
    x, y = x+.5-112-dx, y+.5-112-dy
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return c*x+s*y+111.5, -s*x+c*y+111.5


class Reporter:
    def __init__(self, output):
        output = Path(output)
        output.mkdir(parents=False, exist_ok=False)
        self.output = output
        self.data = dict(schema="performance_v2_synthetic_cpu_transforms_v1",
                         scope="synthetic_cpu_transforms_only", status="running",
                         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         python=platform.python_version(), executable=sys.executable,
                         runtime_versions={}, package_versions={}, passed_steps=[], warnings=[],
                         failure_traceback=None, failed_step=None, checks={}, fixture_sha256={},
                         contract=CONTRACT, model_forwards=0, training_performed=False,
                         pilot_execution_authorized=False, gpu_api_calls=0,
                         stage_a_equivalence_verified=False, transforms_checked=False)
        self.save()

    def save(self):
        raw = (json.dumps(self.data, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
        with (self.output / "report.tmp").open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(self.output / "report.tmp", self.output / "report.json")

    def step(self, name, fn):
        self.data["active_step"] = name
        self.save()
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                details = fn()
            require(not caught, "warnings encountered; no automatic continuation")
            self.data["checks"][name] = details
            self.data["passed_steps"].append(name)
        except BaseException:
            self.data["status"] = "failed"
            self.data["failed_step"] = name
            self.data["failure_traceback"] = traceback.format_exc()
            raise
        finally:
            self.data["warnings"].extend(dict(step=name, category=w.category.__name__,
                                               message=str(w.message)) for w in caught)
            self.save()


def run(output):
    report = Reporter(output)  # existing output is rejected before any dependency import
    try:
        def cpu_guard():
            require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "require CUDA_VISIBLE_DEVICES='' before startup")
            require("torch" not in sys.modules, "fresh process required")
        report.step("cpu_process_guard", cpu_guard)
        modules = {}
        for name, package in (("numpy", "numpy"), ("torch", "torch"), ("torchvision", "torchvision")):
            def load(name=name, package=package):
                modules[name] = importlib.import_module(name)
                report.data["runtime_versions"][name] = modules[name].__version__
                report.data["package_versions"][package] = importlib.metadata.version(package)
            report.step("import_"+name, load)
        np, torch, tv = modules["numpy"], modules["torch"], modules["torchvision"]
        tf = tv.transforms.functional
        report.step("cpu_threads", lambda: torch.set_num_threads(1))
        report.step("disable_gradients", lambda: torch.set_grad_enabled(False) and None)

        def tensor(array):
            return torch.tensor(array, dtype=torch.float32, device="cpu")[None, None]

        def compare(value, expected, atol):
            require(value.device.type == "cpu" and value.dtype == torch.float32, "wrong device/dtype")
            actual = value.detach().numpy()
            require(actual.shape == expected.shape and np.isfinite(actual).all(), "shape/nonfinite output")
            error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
            require(np.allclose(actual, expected, atol=atol, rtol=0), f"oracle mismatch max_abs_error={error}, atol={atol}")
            return dict(shape=list(actual.shape), dtype=str(value.dtype), device=str(value.device),
                        max_abs_error=error, atol=atol, rtol=0,
                        output_sha256=hashlib.sha256(actual.tobytes()).hexdigest())

        def affine(value, angle=0, dx=0, dy=0):
            return tf.affine(value, angle=angle, translate=[dx, dy], scale=1,
                             shear=[0, 0], interpolation=tv.transforms.InterpolationMode.BILINEAR,
                             fill=0, center=[112, 112])

        def scaling():
            raw = np.tile(np.array([0, 1, 128, 255], dtype=np.uint8), (64, 16))
            expected = np.tile(np.array([0, 1/255, 128/255, 1], dtype=np.float64), (64, 16))
            for data in (raw, raw.astype(np.float32)):
                require(np.allclose(scale_grayscale(np, data), expected, atol=1e-6, rtol=0), "scaling mismatch")
            unit = np.linspace(0, 1, 4096, dtype=np.float32).reshape(64, 64)
            require(np.array_equal(scale_grayscale(np, unit), unit), "unit scaling changed")
            for invalid in (np.full((64, 64), np.nan), np.full((64, 64), np.inf),
                            np.full((64, 64), -1), np.full((64, 64), 256), np.zeros((64, 64, 3))):
                try: scale_grayscale(np, invalid)
                except ValueError: pass
                else: raise ValueError("invalid grayscale accepted")
            return {"checks": "uint8/float32 raw and unit ranges; NaN/Inf/negative/>255/shape rejected"}
        report.step("grayscale_scaling_validation", scaling)

        y, x = np.indices((64, 64))
        inputs = {"zeros": np.zeros((64, 64), np.float32), "ones": np.ones((64, 64), np.float32),
                  "ramp": ((x+2*y)/189).astype(np.float32), "checkerboard": ((x+y)%2).astype(np.float32)}
        for name, data in inputs.items():
            report.data["fixture_sha256"][name] = hashlib.sha256(data.tobytes()).hexdigest()
            def resize_check(data=data):
                first = torch.nn.functional.interpolate(tensor(data), size=[224, 224], mode="bilinear",
                                                        align_corners=False, antialias=True)
                second = torch.nn.functional.interpolate(tensor(data), size=[224, 224], mode="bilinear",
                                                         align_corners=False, antialias=True)
                require(torch.equal(first, second), "resize repetition differs")
                require(bool((first >= 0).all() and (first <= 1).all()), "resize range")
                tolerance = 1e-6 if np.all(data == data.flat[0]) else 1e-5
                return compare(first, resize_reference(np, data)[None, None], tolerance)
            report.step("resize_"+name, resize_check)

        fixture = np.zeros((224, 224), np.float32)
        for yy, xx, value in ((0, 0, .25), (20, 31, 1), (100, 150, .5), (223, 223, .75)):
            fixture[yy, xx] = value
        report.data["fixture_sha256"]["geometry"] = hashlib.sha256(fixture.tobytes()).hexdigest()
        report.step("flip_horizontal", lambda: compare(tf.hflip(tensor(fixture)), fixture[:, ::-1][None, None], 0))
        report.step("flip_vertical", lambda: compare(tf.vflip(tensor(fixture)), fixture[::-1, :][None, None], 0))

        for name, turns, dx, dy in (("identity", 0, 0, 0), ("positive_translation", 0, 3, 5),
                                    ("negative_translation", 0, -4, -2), ("clockwise_90", 1, 0, 0),
                                    ("counterclockwise_90", -1, 0, 0), ("rotation_then_translation", 1, 3, -5)):
            def geometry(turns=turns, dx=dx, dy=dy):
                result = affine(tensor(fixture), turns*90, dx, dy)
                expected = integer_geometry_reference(np, fixture, turns, dx, dy)
                require(torch.equal(result, affine(tensor(fixture), turns*90, dx, dy)), "affine repetition differs")
                return compare(result, expected[None, None], 5e-5)
            report.step("affine_"+name, geometry)

        for angle in (-10, 10):
            def oblique(angle=angle):
                yy, xx = np.indices((224, 224))
                ramp = ((xx+2*yy)/669).astype(np.float32)
                result = affine(tensor(ramp), angle, 3, -5)
                sx, sy = inverse_coordinates(np, angle, 3, -5)
                interior = (sx >= 1) & (sx <= 222) & (sy >= 1) & (sy <= 222)
                # A linear field is reproduced by bilinear interpolation in the interior.
                expected = (sx+2*sy)/669
                details = compare(result[:, :, interior], expected[interior][None, None], 5e-5)
                exterior = (sx < -1) | (sx > 224) | (sy < -1) | (sy > 224)
                require(exterior.any() and bool((result[0, 0][exterior] == 0).all()), "zero-fill boundary mismatch")
                require(torch.equal(result, affine(tensor(ramp), angle, 3, -5)), "oblique repetition differs")
                details.update(interior_pixels=int(interior.sum()), exterior_pixels=int(exterior.sum()),
                               partially_supported_edge_pixels="not numerically certified by this fixture")
                return details
            report.step("affine_angle_"+str(angle), oblique)

        def rounding():
            for fraction, expected in ((.5/224, 0), (1.5/224, 2), (2.5/224, 2),
                                       (-.5/224, 0), (-1.5/224, -2), (-2.5/224, -2)):
                require(translation_pixels(fraction, -fraction) == [expected, -expected], "ties-to-even mismatch")
            require(translation_pixels(.08, -.08) == [18, -18], "axis translation mismatch")
            return {"ties": "ties-to-even on floating product; horizontal then vertical"}
        report.step("fraction_to_pixel_rounding", rounding)

        def normalize():
            raw = np.tile(np.array([-.2, 0, .4, 1.2], dtype=np.float32), (224, 56))
            clipped = tensor(raw).clamp(0, 1)
            channels = clipped.repeat(1, 3, 1, 1)
            require(torch.equal(channels[:, 0], channels[:, 1]) and torch.equal(channels[:, 1], channels[:, 2]), "replication mismatch")
            result = tf.normalize(channels, [.485, .456, .406], [.229, .224, .225])
            expected = (np.clip(raw.astype(np.float64), 0, 1)[None, None] - np.array([.485,.456,.406])[None,:,None,None]) / np.array([.229,.224,.225])[None,:,None,None]
            return compare(result, expected, 1e-6)
        report.step("clipping_replication_normalization", normalize)

        def chain():
            image = scale_grayscale(np, inputs["ramp"])
            resized = torch.nn.functional.interpolate(tensor(image), size=[224, 224], mode="bilinear",
                                                     align_corners=False, antialias=True)
            resized_expected = resize_reference(np, image)
            results = {}
            for mode in ("validation", "train"):
                value, expected = resized, resized_expected
                if mode == "train":
                    value = affine(tf.vflip(tf.hflip(value)), 90, 3, -5).clamp(0, 1)
                    expected = integer_geometry_reference(np, expected[::-1, ::-1], 1, 3, -5).clip(0, 1)
                value = tf.normalize(value.repeat(1, 3, 1, 1), [.485,.456,.406], [.229,.224,.225])
                expected = (expected[None,None]-np.array([.485,.456,.406])[None,:,None,None])/np.array([.229,.224,.225])[None,:,None,None]
                results[mode] = compare(value, expected, 3e-4 if mode == "train" else 5e-5)
            return results
        report.step("synthetic_train_validation_chains", chain)
        report.data.update(status="passed", transforms_checked=True, active_step=None)
    except BaseException as exc:
        if report.data["status"] != "failed":
            report.data.update(status="failed", failure_traceback=traceback.format_exc())
        report.save()
        print(json.dumps(report.data, indent=2, sort_keys=True))
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    report.save()
    print(json.dumps(report.data, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new report directory; parent must exist")
    args = parser.parse_args()
    raise SystemExit(run(args.output))
