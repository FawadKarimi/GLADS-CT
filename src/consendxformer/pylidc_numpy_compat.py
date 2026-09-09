"""Scoped NumPy legacy-alias compatibility for official pylidc==0.2.3.

The adapter temporarily replaces only pylidc modules' local ``np`` references
with a delegating proxy.  It never mutates NumPy, edits pylidc, or replaces a
pylidc algorithm.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from types import ModuleType
from typing import Any, Iterator, Sequence


class NumpyCompatibilityError(RuntimeError):
    """Raised when the scoped pylidc compatibility proxy cannot be installed."""


@dataclass(frozen=True)
class PylidcNumpyCompatibilityReport:
    required: bool
    aliases: tuple[str, ...]
    algorithm_modified: bool = False
    numpy_module_modified: bool = False
    pylidc_source_modified: bool = False
    strategy: str = "temporary_pylidc_module_numpy_proxy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "aliases": list(self.aliases),
            "algorithm_modified": self.algorithm_modified,
            "numpy_module_modified": self.numpy_module_modified,
            "pylidc_source_modified": self.pylidc_source_modified,
            "strategy": self.strategy,
        }


class _NumpyLegacyAliasProxy:
    """Delegate NumPy normally while supplying only aliases absent upstream."""

    def __init__(self, numpy_module: ModuleType, aliases: dict[str, type]) -> None:
        object.__setattr__(self, "_numpy_module", numpy_module)
        object.__setattr__(self, "_aliases", dict(aliases))

    def __getattr__(self, name: str) -> Any:
        aliases = object.__getattribute__(self, "_aliases")
        if name in aliases:
            return aliases[name]
        return getattr(object.__getattribute__(self, "_numpy_module"), name)

    def __dir__(self) -> list[str]:
        numpy_module = object.__getattribute__(self, "_numpy_module")
        aliases = object.__getattribute__(self, "_aliases")
        return sorted(set(dir(numpy_module)) | set(aliases))


# Official pylidc 0.2.3 clustering and consensus paths use these two removed
# aliases. ``np.float`` occurs in a visualization path, not in either invoked
# research path, so it is intentionally not supplied.
_INVOKED_LEGACY_ALIASES: tuple[tuple[str, type], ...] = (
    ("int", int),
    ("bool", bool),
)
_COMPATIBILITY_LOCK = threading.RLock()


def pylidc_numpy_compatibility_report(
    numpy_module: ModuleType,
) -> PylidcNumpyCompatibilityReport:
    aliases = tuple(
        name for name, _target in _INVOKED_LEGACY_ALIASES if name not in vars(numpy_module)
    )
    return PylidcNumpyCompatibilityReport(required=bool(aliases), aliases=aliases)


class PylidcNumpyCompatibility:
    """Install and restore a deterministic proxy around official pylidc calls."""

    def __init__(
        self,
        numpy_module: ModuleType,
        pylidc_modules: Sequence[ModuleType],
    ) -> None:
        if not pylidc_modules:
            raise NumpyCompatibilityError("No official pylidc modules were supplied")
        report = pylidc_numpy_compatibility_report(numpy_module)
        aliases = dict(
            (name, target)
            for name, target in _INVOKED_LEGACY_ALIASES
            if name in report.aliases
        )
        self._numpy_module = numpy_module
        self._modules = tuple(
            sorted(pylidc_modules, key=lambda module: str(getattr(module, "__name__", "")))
        )
        self._aliases = aliases
        self._proxy = _NumpyLegacyAliasProxy(numpy_module, aliases)
        self.report = report

    def _install(self) -> list[tuple[ModuleType, Any]]:
        if not self._aliases:
            return []
        originals: list[tuple[ModuleType, Any]] = []
        try:
            for module in self._modules:
                module_name = str(getattr(module, "__name__", type(module).__name__))
                if not hasattr(module, "np"):
                    raise NumpyCompatibilityError(
                        f"Official pylidc module {module_name} has no NumPy namespace"
                    )
                original = getattr(module, "np")
                if original is not self._numpy_module:
                    raise NumpyCompatibilityError(
                        f"Official pylidc module {module_name} has an unexpected NumPy namespace"
                    )
                originals.append((module, original))
                setattr(module, "np", self._proxy)
                if getattr(module, "np") is not self._proxy:
                    raise NumpyCompatibilityError(
                        f"Could not establish NumPy proxy in official pylidc module {module_name}"
                    )
            for alias, target in self._aliases.items():
                if getattr(self._proxy, alias) is not target:
                    raise NumpyCompatibilityError(
                        f"Compatibility alias np.{alias} is not the required Python {alias}"
                    )
        except Exception as exc:
            for module, original in reversed(originals):
                try:
                    setattr(module, "np", original)
                except Exception:
                    pass
            if isinstance(exc, NumpyCompatibilityError):
                raise
            raise NumpyCompatibilityError(
                f"Could not establish scoped pylidc NumPy compatibility: {exc}"
            ) from exc
        return originals

    @staticmethod
    def _restore(originals: Sequence[tuple[ModuleType, Any]]) -> None:
        failures: list[str] = []
        for module, original in reversed(originals):
            try:
                setattr(module, "np", original)
                if getattr(module, "np") is not original:
                    failures.append(str(getattr(module, "__name__", type(module).__name__)))
            except Exception:
                failures.append(str(getattr(module, "__name__", type(module).__name__)))
        if failures:
            raise NumpyCompatibilityError(
                "Could not restore NumPy namespace in official pylidc modules: "
                + ", ".join(sorted(set(failures)))
            )

    @contextmanager
    def activated(self) -> Iterator[PylidcNumpyCompatibilityReport]:
        """Temporarily expose missing aliases inside official pylidc modules."""
        with _COMPATIBILITY_LOCK:
            originals = self._install()
            try:
                yield self.report
            finally:
                self._restore(originals)

    def verify(self) -> PylidcNumpyCompatibilityReport:
        """Fail early if the compatibility context cannot be installed/restored."""
        with self.activated():
            pass
        return self.report
