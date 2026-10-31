"""Reproducibility receipts for public calculations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
import platform
import re
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class CalculationReceipt:
    """Caller inputs, data identities, code, dependencies, and solver settings."""

    request: Mapping[str, Any]
    data_digests: Mapping[str, str]
    code: Mapping[str, str | bool | None]
    dependencies: Mapping[str, str]
    solver: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe copy of the receipt."""
        return {
            "request": _plain_data(self.request),
            "data_digests": _plain_data(self.data_digests),
            "code": _plain_data(self.code),
            "dependencies": _plain_data(self.dependencies),
            "solver": _plain_data(self.solver),
        }


def build_calculation_receipt(
    *,
    request: Mapping[str, Any],
    data_digests: Mapping[str, str | None] | None = None,
    solver: Mapping[str, Any] | None = None,
) -> CalculationReceipt:
    """Build the common receipt used by each calculation entry point."""
    from openimcc import __version__

    revision, dirty = _source_identity()
    pandas = sys.modules.get("pandas")
    dependencies = {
        "python": platform.python_version(),
        "numpy": _dependency_version("numpy"),
        "scipy": _dependency_version("scipy"),
    }
    if pandas is not None:
        dependencies["pandas"] = str(getattr(pandas, "__version__", "unknown"))

    return CalculationReceipt(
        request=_freeze_data(_plain_data(request)),
        data_digests=_freeze_data({
            key: value for key, value in (data_digests or {}).items()
            if value is not None
        }),
        code=MappingProxyType({
            "version": __version__,
            "source_revision": revision,
            "source_dirty": dirty,
        }),
        dependencies=MappingProxyType(dependencies),
        solver=_freeze_data(_plain_data(solver or {})),
    )


def _dependency_version(name: str) -> str:
    module = sys.modules.get(name)
    loaded_version = getattr(module, "__version__", None)
    if loaded_version is not None:
        return str(loaded_version)
    try:
        return metadata.version(name)
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def _source_identity() -> tuple[str | None, bool | None]:
    """Read the checkout revision and dirty state once, without requiring git."""
    package_dir = Path(__file__).resolve().parent
    imported_package = (package_dir / "__init__.py").resolve()
    root = next(
        (
            ancestor
            for ancestor in (package_dir, *package_dir.parents)
            if (ancestor / "src/openimcc/__init__.py").resolve()
            == imported_package
        ),
        None,
    )
    if root is None:
        return None, None

    dot_git = root / ".git"
    if not dot_git.exists():
        return None, None
    try:
        if dot_git.is_dir():
            git_dir = dot_git
        elif dot_git.is_file():
            marker = dot_git.read_text(encoding="utf-8").strip()
            if not marker.startswith("gitdir:"):
                return None, None
            git_dir = Path(marker.removeprefix("gitdir:").strip())
            if not git_dir.is_absolute():
                git_dir = (root / git_dir).resolve()
        else:
            return None, None

        common_dir = git_dir
        common_marker = git_dir / "commondir"
        if common_marker.is_file():
            common_dir = Path(common_marker.read_text(encoding="utf-8").strip())
            if not common_dir.is_absolute():
                common_dir = (git_dir / common_dir).resolve()

        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        if head.startswith("ref:"):
            ref = head.removeprefix("ref:").strip()
            revision = next(
                (
                    value
                    for directory in (git_dir, common_dir)
                    if (value := _read_ref(directory, ref)) is not None
                ),
                None,
            )
        else:
            revision = head
        if revision is None or re.fullmatch(
            r"[0-9a-fA-F]{40,64}", revision
        ) is None:
            return None, None
    except (OSError, UnicodeError, ValueError):
        return None, None

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=1.0,
            check=False,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return revision.lower(), dirty


def _read_ref(git_dir: Path, ref: str) -> str | None:
    ref_path = git_dir / ref
    if ref_path.is_file():
        return ref_path.read_text(encoding="ascii").strip()
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="ascii").splitlines():
            if line.startswith(("#", "^")):
                continue
            value, separator, name = line.partition(" ")
            if separator and name == ref:
                return value
    return None


def _plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_data(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _plain_data(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain_data(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _freeze_data(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_data(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_data(item) for item in value)
    return value
