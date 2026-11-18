from __future__ import annotations

import subprocess

from tools.verify_gas_binding_identity_mutations import _is_expected_red


def test_mutation_verifier_requires_expected_assertion_failure() -> None:
    test_file = "tests/test_gas_content_identity.py"
    test_name = "test_reordering_rows_preserves_identity_and_gas_evaluation"
    expected = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=(
            f"FAILED {test_file}::{test_name} - AssertionError: assert digest_a == digest_b"
        ),
        stderr="",
    )
    assert _is_expected_red(expected, test_file, test_name)

    setup_error = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=(
            f"ERROR {test_file}::{test_name} - RuntimeError: setup failed"
        ),
        stderr="",
    )
    assert not _is_expected_red(setup_error, test_file, test_name)

    unexpected_exception = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=(
            f"FAILED {test_file}::{test_name} - RuntimeError: setup failed"
        ),
        stderr="",
    )
    assert not _is_expected_red(unexpected_exception, test_file, test_name)

    other_test = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=f"FAILED {test_file}::test_other - AssertionError: assert False",
        stderr="",
    )
    assert not _is_expected_red(other_test, test_file, test_name)
