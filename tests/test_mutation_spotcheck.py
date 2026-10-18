from __future__ import annotations

import subprocess

from tools.mutation_spotcheck import _is_expected_red
from tools.mutation_spotcheck import _run


def test_mutation_spotcheck_run_returns_pytest_process() -> None:
    result = _run("tests/test_kernel_coverage.py::test_this_node_does_not_exist")
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 4
    assert "no tests ran" in result.stdout


def test_mutation_spotcheck_accepts_a_test_failure_as_red() -> None:
    result = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="FAILED tests/test_example.py::test_case - AssertionError"
    )
    assert _is_expected_red(result)


def test_mutation_spotcheck_rejects_pytest_collection_error_as_red() -> None:
    result = subprocess.CompletedProcess(
        args=[], returncode=4, stdout="ERROR: not found: tests/test_missing.py::test_case\nno tests ran"
    )
    assert not _is_expected_red(result)
