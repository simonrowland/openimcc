"""Suite-wide test isolation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# OPENIMCC_VAPOROCK_ROOT redirects load_gas_datapack()'s defaults to an external
# checkout's tables. Left set, every test of the packaged tables -- including
# module-scoped fixtures, which load before any function-level fixture runs --
# would silently test those tables instead. So it is removed once, before any
# test module is imported, and its value is handed to the legacy comparisons
# under a test-only name. Tests that exercise the override set it themselves
# with monkeypatch.
LEGACY_ENV = "OPENIMCC_TEST_LEGACY_VAPOROCK_ROOT"


def pytest_configure(config: pytest.Config) -> None:
    selected = os.environ.pop("OPENIMCC_VAPOROCK_ROOT", None)
    if selected and LEGACY_ENV not in os.environ:
        os.environ[LEGACY_ENV] = selected


@pytest.fixture(scope="session")
def sf04_gas_pack():
    """Load the preserved LAM parent rows with the shared JANAF gas table."""
    from openimcc.gas import load_gas_datapack

    root = Path(__file__).resolve().parents[1]
    data = root / "src/openimcc/data"
    return load_gas_datapack(
        gas_path=data / "gas/gas-shomate.csv",
        oxide_path=data / "packs/sf04-published/condensate.csv",
    )
