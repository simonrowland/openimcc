from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tools.verify_gas_binding_identity_mutations import _is_expected_red


def test_mutation_verifier_recognizes_rewritten_assertion_and_rejects_error(
    tmp_path: Path,
) -> None:
    test_file = "test_real_pytest_output.py"
    (tmp_path / test_file).write_text(
        "import pytest\n"
        "@pytest.fixture\n"
        "def setup_error():\n"
        "    raise RuntimeError('setup failed')\n"
        "def test_rewritten_assertion():\n"
        "    assert 1 == 2\n"
        "def test_setup_error_is_not_a_failure(setup_error):\n"
        "    assert setup_error\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-vv", "--color=no", "-rf"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FAILED test_real_pytest_output.py::test_rewritten_assertion - assert" in result.stdout
    assert _is_expected_red(result, test_file, "test_rewritten_assertion")
    assert not _is_expected_red(result, test_file, "test_setup_error_is_not_a_failure")
