"""Prove each gas binding identity regression test rejects its named defect."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = "tests/test_gas_content_identity.py"
ROW_SORT = "    rows.sort(key=_canonical_published_serialization)\n"
FILTER_ROWS = """    rows = [
        {
            column: value
            for column, value in row.items()
            if {condition}
        }
        for row in rows
    ]
    rows.sort(key=_canonical_published_serialization)
"""

MUTATIONS = (
    (
        "a-condensate-coefficient",
        "src/openimcc/gas.py",
        ROW_SORT,
        FILTER_ROWS.replace(
            "{condition}", 'table_name != "condensate" or column != "dG_C"'
        ),
        "test_mgo_supercooled_coefficient_changes_only_condensate_and_combined_identity",
    ),
    (
        "b-gas-cell",
        "src/openimcc/gas.py",
        ROW_SORT,
        FILTER_ROWS.replace(
            "{condition}", 'table_name != "gas" or column != "A"'
        ),
        "test_gas_cell_change_only_changes_gas_and_combined_identity",
    ),
    (
        "c-melt-component",
        "src/openimcc/model.py",
        "        digest=_published_datapack_manifest_hash(components),\n",
        """        digest=_published_datapack_manifest_hash(
            {
                key: value
                for key, value in components.items()
                if key != "melt_binding_digest"
            }
        ),
""",
        "test_combined_identity_includes_the_published_melt_binding_digest",
    ),
    (
        "d-path-identity",
        "src/openimcc/gas.py",
        '        return _gas_table_digest("gas", self.gas_df)\n',
        '        return hashlib.sha256(str(self.gas_path).encode("utf-8")).hexdigest()\n',
        "test_same_tables_in_another_directory_have_the_same_identity",
    ),
    (
        "e-raw-file-bytes",
        "src/openimcc/gas.py",
        '        return _gas_table_digest("gas", self.gas_df)\n',
        '        return hashlib.sha256(self.gas_path.read_bytes()).hexdigest()\n',
        "test_csv_layout_does_not_change_parsed_content_identity",
    ),
    (
        "f-row-order",
        "src/openimcc/gas.py",
        ROW_SORT,
        "    rows = list(rows)\n",
        "test_reordering_rows_preserves_identity_and_gas_evaluation",
    ),
    (
        "g-hash-seed",
        "src/openimcc/gas.py",
        '        "table": table_name,\n',
        '        "table": table_name,\n        "seed_hash": hash(table_name),\n',
        "test_gas_identity_is_stable_across_interpreter_hash_seeds",
    ),
    (
        "h-column-sensitivity",
        "src/openimcc/gas.py",
        ROW_SORT,
        FILTER_ROWS.replace("{condition}", 'column != "Ref"'),
        "test_digest_is_sensitive_to_every_column_returned_by_the_loader",
    ),
    (
        "i-duplicate-interval-order-dependence",
        "src/openimcc/gas.py",
        """    if t_mins.size > 1:
        unique_starts, start_counts = np.unique(t_mins, return_counts=True)
    else:
        return
    if start_counts.size != t_mins.size:
        start = unique_starts[np.flatnonzero(start_counts > 1)[0]]
        raise ImccGasDuplicateIntervalError(
            f"{table_name} database {table_path} has duplicate interval start "
            f"for {species!r} at T_min={start!r} K"
        )
""",
        "    return\n",
        "loader accepted duplicate interval with same identity",
    ),
    (
        "j-non-finite-interval-bound",
        "src/openimcc/gas.py",
        "    if not np.isfinite(t_mins).all():\n",
        "    if False and not np.isfinite(t_mins).all():\n",
        "test_loader_refuses_non_finite_interval_bounds",
    ),
    (
        "k-raw-bound-uniqueness",
        "src/openimcc/gas.py",
        """    _normalise_interval_bounds(gas_df, "JANAF gas", gas_display_path)
    gas_df = gas_df.set_index("species_name")
    _validate_interval_table(gas_df, "JANAF gas", gas_display_path)
""",
        """    gas_df = gas_df.set_index("species_name")
    _normalise_interval_bounds(gas_df.reset_index(), "JANAF gas", gas_display_path)
    _validate_interval_table(gas_df, "JANAF gas", gas_display_path)
""",
        "test_loader_refuses_duplicate_starts_after_float64_normalization",
    ),
    (
        "l-selector-reconversion",
        "src/openimcc/gas.py",
        "    t_mins = rows[\"T_min\"].to_numpy(copy=False)\n",
        "    t_mins = rows[\"T_min\"].astype(float).to_numpy()\n",
        "test_selector_uses_normalized_interval_values_without_reconversion",
    ),
    (
        "m-raw-bound-digest",
        "src/openimcc/gas.py",
        "        table[column] = values\n",
        "        table[column] = table[column].astype(object)\n",
        "test_numeric_bound_spellings_have_one_evaluation_and_digest",
    ),
    (
        "n-digest-skips-current-validation",
        "src/openimcc/gas.py",
        '        _validate_interval_table(self.gas_df, "JANAF gas", self.gas_path)\n',
        "        pass\n",
        "test_invalid_duplicate_starts_refuse_identity",
    ),
    (
        "o-selector-skips-duplicate-check",
        "src/openimcc/gas.py",
        "        t_mins, t_maxs, species, \"gas/condensate\", Path(\"<in-memory>\")\n",
        "        t_mins[:1], t_maxs[:1], species, \"gas/condensate\", Path(\"<in-memory>\")\n",
        "test_invalid_duplicate_starts_refuse_evaluation",
    ),
    (
        "p-selector-restores-bare-assert",
        "src/openimcc/gas.py",
        "    _validate_interval_arrays(\n        t_mins, t_maxs, species, \"gas/condensate\", Path(\"<in-memory>\")\n    )\n",
        "    assert np.isfinite(t_mins).all()\n",
        "test_invalid_direct_bounds_refuse_identity_and_evaluation",
    ),
    (
        "q-property-o2-domain-regression",
        "tests/test_gas_content_identity.py",
        '        o2_template["T_min"], o2_template["T_max"] = "0", "2000"\n',
        '        o2_template["T_min"], o2_template["T_max"] = "1500", "3000"\n',
        "test_random_accepted_tables_are_order_independent",
    ),
    (
        "r-array-validator-skips-uniqueness",
        "src/openimcc/gas.py",
        "    if start_counts.size != t_mins.size:\n",
        "    if False and start_counts.size != t_mins.size:\n",
        "test_invalid_duplicate_starts_refuse_evaluation",
    ),
)


def _run_mutation(
    name: str,
    relative_source: str,
    original: str,
    replacement: str,
    expected_failure: str,
) -> None:
    with TemporaryDirectory(prefix="openimcc-gas-identity-") as temporary:
        root = Path(temporary)
        shutil.copytree(ROOT / "src/openimcc", root / "src/openimcc")
        tests = root / "tests"
        tests.mkdir()
        shutil.copyfile(ROOT / TEST_FILE, root / TEST_FILE)
        shutil.copyfile(ROOT / "tests/conftest.py", tests / "conftest.py")

        source = root / relative_source
        content = source.read_text(encoding="utf-8")
        if content.count(original) != 1:
            raise RuntimeError(f"{name}: mutation target is not unique")
        source.write_text(content.replace(original, replacement), encoding="utf-8")

        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        environment.pop("PYTEST_ADDOPTS", None)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", TEST_FILE, "-q"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        if result.returncode == 0 or expected_failure not in output:
            tail = "\n".join(output.splitlines()[-50:])
            raise RuntimeError(
                f"{name}: expected the full identity test file to fail at "
                f"{expected_failure!r}; pytest returned {result.returncode}\n{tail}"
            )
    print(f"{name}: RED ({expected_failure})", flush=True)


def main() -> None:
    for mutation in MUTATIONS:
        _run_mutation(*mutation)
    print(f"All {len(MUTATIONS)} mutations were RED.")


if __name__ == "__main__":
    main()
