"""Run small, reversible mutation checks for the coverage tests.

Each mutation changes one source predicate or message, runs one focused test,
and restores the exact original text in a ``finally`` block. A red test is the
success condition: it proves the assertion is coupled to the covered behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Mutation:
    name: str
    source: Path
    old: str
    new: str
    test: str


KERNEL = ROOT / "src/openimcc/kernel.py"
MODEL = ROOT / "src/openimcc/model.py"


MUTATIONS = (
    Mutation(
        "kernel constructor dimensionality",
        KERNEL,
        "        if nu.ndim != 2:\n",
        "        if nu.ndim == 2:\n",
        "tests/test_kernel_coverage.py::test_datapack_rejects_invalid_shapes_and_values",
    ),
    Mutation(
        "kernel canonical finite-number guard",
        KERNEL,
        "            if not number.is_finite():\n",
        "            if number.is_finite():\n",
        "tests/test_kernel_coverage.py::test_canonical_serialization_preserves_json_types_and_refuses_invalid_values",
    ),
    Mutation(
        "kernel identity coverage-shape refusal",
        KERNEL,
        "    if missing or extra:\n",
        "    if not (missing or extra):\n",
        "tests/test_kernel_coverage.py::test_identity_issuer_refuses_invalid_coverage_and_published_claims",
    ),
    Mutation(
        "kernel empty-active-set refusal",
        KERNEL,
        "    if n == 0:\n",
        "    if n != 0:\n",
        "tests/test_kernel_coverage.py::test_solve_active_refuses_empty_active_parent_set",
    ),
    Mutation(
        "kernel continuation residual break",
        KERNEL,
        "        if attempt[4] > tol:\n",
        "        if attempt[4] <= tol:\n",
        "tests/test_kernel_coverage.py::test_solver_reports_a_nonconverged_continuation_stage",
    ),
    Mutation(
        "kernel positive-temperature refusal",
        KERNEL,
        "    if not math.isfinite(T) or T <= 0.0:\n",
        "    if not math.isfinite(T) or T > 0.0:\n",
        "tests/test_kernel_coverage.py::test_kernel_input_validation_reports_typed_refusals",
    ),
    Mutation(
        "kernel negative-extra refusal",
        KERNEL,
        "            if value < 0.0:\n",
        "            if value > 0.0:\n",
        "tests/test_kernel_coverage.py::test_kernel_refuses_negative_extra_components_with_stable_code",
    ),
    Mutation(
        "model unknown-nu refusal",
        MODEL,
        "    unknown_nu = sorted(set(nu) - set(parent_oxides))\n    if unknown_nu:\n",
        "    unknown_nu = sorted(set(nu) - set(parent_oxides))\n    if not unknown_nu:\n",
        "tests/test_model_coverage.py::test_numeric_parsers_report_invalid_types",
    ),
    Mutation(
        "model missing-file refusal",
        MODEL,
        "            f\"datapack file not found: {path}\"\n",
        "            f\"datapack file found: {path}\"\n",
        "tests/test_model_coverage.py::test_load_datapack_rejects_file_root_and_parent_shape_errors",
    ),
    Mutation(
        "model published-row type refusal",
        MODEL,
        "    for idx, row in enumerate(rows):\n        if not isinstance(row, dict):\n",
        "    for idx, row in enumerate(rows):\n        if isinstance(row, dict):\n",
        "tests/test_model_coverage.py::test_load_datapack_rejects_core_schema_variants[row-not-object]",
    ),
    Mutation(
        "model canonicalization-error refusal",
        MODEL,
        "    except (TypeError, ValueError) as exc:\n",
        "    except TypeError as exc:\n",
        "tests/test_model_coverage.py::test_validate_published_core_reports_canonical_serialization_failure",
    ),
    Mutation(
        "model extension-pairing refusal",
        MODEL,
        "    if sp_extension is None:\n",
        "    if sp_extension is not None:\n",
        "tests/test_model_coverage.py::test_load_datapack_rejects_model_extension_pairing[model-without-extension]",
    ),
    Mutation(
        "model extension-flag refusal",
        MODEL,
        "        if sp_extension.get(\"enable_flag\") != _SP_EXTENSION_FLAG:\n",
        "        if sp_extension.get(\"enable_flag\") == _SP_EXTENSION_FLAG:\n",
        "tests/test_model_coverage.py::test_load_datapack_rejects_extension_schema_variants[bad-flag]",
    ),
    Mutation(
        "model basis-type refusal",
        MODEL,
        "    if basis_type not in (\"mol\", \"wt\"):\n",
        "    if basis_type in (\"mol\", \"wt\"):\n",
        "tests/test_model_coverage.py::test_evaluate_rejects_unproven_and_invalid_composition_inputs",
    ),
    Mutation(
        "model unknown-component refusal",
        MODEL,
        "        if unknown:\n",
        "        if not unknown:\n",
        "tests/test_model_coverage.py::test_evaluate_rejects_unproven_and_invalid_composition_inputs",
    ),
    Mutation(
        "model extension canonical-total refusal",
        MODEL,
        "    if canonical_oxide_mol <= 0.0:\n",
        "    if canonical_oxide_mol > 0.0:\n",
        "tests/test_model_coverage.py::test_evaluate_extension_vector_guards_and_canonical_zero",
    ),
)


def _run(test: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_path
        else source_path + os.pathsep + existing_path
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", test],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    failed: list[str] = []
    for index, mutation in enumerate(MUTATIONS, 1):
        original = mutation.source.read_text()
        if original.count(mutation.old) != 1:
            print(f"ERROR {index:02d} {mutation.name}: source needle is not unique")
            failed.append(mutation.name)
            continue

        baseline = _run(mutation.test)
        if baseline.returncode != 0:
            print(
                f"ERROR {index:02d} {mutation.name}: focused test is not green "
                f"before mutation (exit {baseline.returncode})"
            )
            print("\n".join(baseline.stdout.splitlines()[-8:]))
            failed.append(mutation.name)
            continue

        mutation.source.write_text(original.replace(mutation.old, mutation.new, 1))
        try:
            mutated = _run(mutation.test)
            if mutated.returncode == 0:
                print(
                    f"NOT-RED {index:02d} {mutation.name}: "
                    "focused test still passed"
                )
                failed.append(mutation.name)
            else:
                print(
                    f"RED {index:02d} {mutation.name}: "
                    f"focused test failed as expected (exit {mutated.returncode})"
                )
        finally:
            mutation.source.write_text(original)

    if failed:
        print("FAILED mutations:", ", ".join(failed))
        return 1
    print(f"PASS {len(MUTATIONS)} mutations went red and were restored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
