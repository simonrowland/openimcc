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
        '        object.__setattr__(self, "_gas_digest", _gas_table_digest("gas", gas_df))\n',
        '        object.__setattr__(self, "_gas_digest", hashlib.sha256(str(self.gas_path).encode("utf-8")).hexdigest())\n',
        "test_same_tables_in_another_directory_have_the_same_identity",
    ),
    (
        "e-raw-file-bytes",
        "src/openimcc/gas.py",
        '        object.__setattr__(self, "_gas_digest", _gas_table_digest("gas", gas_df))\n',
        '        object.__setattr__(self, "_gas_digest", hashlib.sha256(self.gas_path.read_bytes()).hexdigest())\n',
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
        "        if key in starts:\n",
        "        if False and key in starts:\n",
        "test_invalid_duplicate_starts_refuse_identity",
    ),
    (
        "j-non-finite-interval-bound",
        "src/openimcc/gas.py",
        "        if not math.isfinite(number):\n            raise ImccGasInvalidIntervalError(invalid_prefix)\n",
        "        if False and not math.isfinite(number):\n            raise ImccGasInvalidIntervalError(invalid_prefix)\n",
        "test_loader_refuses_non_finite_interval_bounds",
    ),
    (
        "k-numeric-bound-not-canonicalized",
        "src/openimcc/gas.py",
        "        values[position] = number\n    return values\n",
        "        values[position] = number\n    return table[column].to_numpy(copy=False)\n",
        "test_direct_exact_bounds_are_canonicalized",
    ),
    (
        "l-selector-reconversion",
        "src/openimcc/gas.py",
        '    t_mins = rows["T_min"].to_numpy(copy=False)\n',
        '    t_mins = rows["T_min"].astype(float).to_numpy()\n',
        "test_selector_uses_normalized_interval_values_without_reconversion",
    ),
    (
        "m-numeric-digest-representation",
        "src/openimcc/gas.py",
        "    return value\n\n\ndef _gas_table_digest",
        "    return str(value) if isinstance(value, (int, float)) else value\n\n\ndef _gas_table_digest",
        "test_packaged_table_content_digests_match_current_tables",
    ),
    (
        "n-condensate-canonicalization-skipped",
        "src/openimcc/gas.py",
        """        oxide_df = _canonicalize_gas_table(
            self.oxide_df,
            "condensate",
            self.oxide_path,
            _CONDENSATE_TABLE_SCHEMA,
            _CONDENSATE_NUMERIC_COLUMNS,
            _CONDENSATE_TEXT_COLUMNS,
        )
""",
        "        oxide_df = self.oxide_df\n",
        "test_evaluated_float_coefficients_and_exact_widenings_match",
    ),
    (
        "o-zero-width-interval-accepted",
        "src/openimcc/gas.py",
        "np.flatnonzero(t_mins >= t_maxs)",
        "np.flatnonzero(t_mins > t_maxs)",
        "test_loader_refuses_inverted_and_zero_width_intervals",
    ),
    (
        "p-integer-range-check-removed",
        "src/openimcc/gas.py",
        "        if isinstance(value, (int, np.integer)) and abs(int(value)) > 2**53:\n",
        "        if False and isinstance(value, (int, np.integer)) and abs(int(value)) > 2**53:\n",
        "test_integer_above_binary64_exact_range_is_refused_at_construction",
    ),
    (
        "q-property-o2-domain-regression",
        "tests/test_gas_content_identity.py",
        '        o2_template["T_min"], o2_template["T_max"] = "0", "2000"\n',
        '        o2_template["T_min"], o2_template["T_max"] = "1500", "3000"\n',
        "test_random_accepted_tables_are_order_independent",
    ),
    (
        "r-schema-check-removed",
        "src/openimcc/gas.py",
        "        or set(actual_columns) != expected_columns\n",
        "        or expected_columns - set(actual_columns)\n",
        "test_construction_requires_the_complete_fixed_schema",
    ),
    (
        "s-finiteness-limited-to-selected-rows",
        "src/openimcc/gas.py",
        "        if not math.isfinite(number):\n            raise ImccGasInvalidIntervalError(invalid_prefix)\n",
        '        if species == "Na(g)" and not math.isfinite(number):\n            raise ImccGasInvalidIntervalError(invalid_prefix)\n',
        "test_non_finite_unselected_rows_refuse_construction",
    ),
    (
        "t-label-missing-check-removed",
        "src/openimcc/gas.py",
        '    if pd.isna(value):\n        if allow_empty:\n            return ""\n        raise ValueError("missing")\n',
        '    if False and pd.isna(value):\n        if allow_empty:\n            return ""\n        raise ValueError("missing")\n',
        "test_construction_refuses_missing_empty_or_whitespace_species_labels",
    ),
    (
        "u-digest-uses-raw-frame",
        "src/openimcc/gas.py",
        '        object.__setattr__(self, "_gas_digest", _gas_table_digest("gas", gas_df))\n',
        '        object.__setattr__(self, "_gas_digest", _gas_table_digest("gas", self.gas_df))\n',
        "test_numeric_reference_float32_widening_binds_provenance_and_evaluation",
    ),
    (
        "v-evaluation-reads-raw-frame",
        "src/openimcc/gas.py",
        '        object.__setattr__(self, "gas_df", gas_df)\n',
        '        # object.__setattr__(self, "gas_df", gas_df)\n',
        "test_evaluated_float_coefficients_and_exact_widenings_match",
    ),
    (
        "w-unnamed-column-drop-removed",
        "src/openimcc/gas.py",
        "    if not unnamed_columns:\n        return table, None\n",
        "    if True:\n        return table, None\n",
        "test_vaporock_override_drops_synthetic_unnamed_columns_with_notice",
        "tests/test_gas.py",
    ),
    (
        "x-drop-widened-to-named-columns",
        "src/openimcc/gas.py",
        '        if header == ""\n',
        "        if True\n",
        "test_vaporock_override_refuses_named_extra_columns",
        "tests/test_gas.py",
    ),
    (
        "y-ingestion-notice-suppressed",
        "src/openimcc/gas.py",
        '        warnings.warn(\n            ImccGasIngestionWarning("; ".join(notice_messages)), stacklevel=2\n        )\n',
        "        pass\n",
        "test_vaporock_override_drops_synthetic_unnamed_columns_with_notice",
        "tests/test_gas.py",
    ),
    (
        "z-sentinel-setter-removed",
        "tests/test_gas.py",
        '    monkeypatch.setattr(pd.DataFrame, "__getitem__", guarded_getitem)\n',
        "",
        "test_nonfinite_temperature_refuses_before_table_access",
        "tests/test_gas.py",
    ),
    (
        "aa-sentinel-setter-removed-with-early-read",
        "tests/test_gas.py",
        '    monkeypatch.setattr(pd.DataFrame, "__getitem__", guarded_getitem)\n',
        "",
        "test_nonfinite_temperature_refuses_before_table_access",
        "tests/test_gas.py",
        (
            (
                "src/openimcc/gas.py",
                "    T = float(T_K)\n    try:\n        p_O2 = float(fO2)\n",
                '    datapack.gas_df["A"]\n    T = float(T_K)\n    try:\n        p_O2 = float(fO2)\n',
            ),
        ),
    ),
)


def _prepare_test_tree(root: Path, test_file: str) -> None:
    shutil.copytree(ROOT / "src/openimcc", root / "src/openimcc")
    tests = root / "tests"
    tests.mkdir()
    shutil.copyfile(ROOT / test_file, root / test_file)
    shutil.copyfile(ROOT / "tests/conftest.py", tests / "conftest.py")
    if test_file == "tests/test_gas.py":
        shutil.copytree(ROOT / "tests/fixtures", tests / "fixtures")
        shutil.copytree(ROOT / "data-src/janaf", root / "data-src/janaf")
        shutil.copytree(ROOT / "data-src/nasa-glenn", root / "data-src/nasa-glenn")
        (root / "tools").mkdir()
        shutil.copyfile(
            ROOT / "tools/build_gas_tables.py", root / "tools/build_gas_tables.py"
        )


def _run_test_file(root: Path, test_file: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    environment.pop("PYTEST_ADDOPTS", None)
    if test_file == "tests/test_gas.py":
        environment["GIT_DIR"] = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--absolute-git-dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        environment["GIT_WORK_TREE"] = str(root)
    return subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-q"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )


def _run_unmutated_control(test_file: str = TEST_FILE) -> None:
    with TemporaryDirectory(prefix="openimcc-gas-identity-control-") as temporary:
        root = Path(temporary)
        _prepare_test_tree(root, test_file)
        result = _run_test_file(root, test_file)
        output = result.stdout + result.stderr
        if result.returncode != 0:
            tail = "\n".join(output.splitlines()[-50:])
            raise RuntimeError(
                f"unmutated full-file control failed; pytest returned "
                f"{result.returncode}\n{tail}"
            )
        summary = next(
            (line for line in reversed(output.splitlines()) if " passed" in line),
            "pytest returned 0",
        )
        print(f"Unmutated control: GREEN ({summary})", flush=True)


def _run_mutation(
    name: str,
    relative_source: str,
    original: str,
    replacement: str,
    expected_failure: str,
    test_file: str = TEST_FILE,
    additional_changes: tuple[tuple[str, str, str], ...] = (),
) -> None:
    with TemporaryDirectory(prefix="openimcc-gas-identity-") as temporary:
        root = Path(temporary)
        _prepare_test_tree(root, test_file)

        changes = ((relative_source, original, replacement), *additional_changes)
        for relative_path, before, after in changes:
            source = root / relative_path
            content = source.read_text(encoding="utf-8")
            if content.count(before) != 1:
                raise RuntimeError(f"{name}: mutation target is not unique")
            source.write_text(content.replace(before, after), encoding="utf-8")

        result = _run_test_file(root, test_file)
        output = result.stdout + result.stderr
        if result.returncode == 0 or expected_failure not in output:
            tail = "\n".join(output.splitlines()[-50:])
            raise RuntimeError(
                f"{name}: expected the full identity test file to fail at "
                f"{expected_failure!r}; pytest returned {result.returncode}\n{tail}"
            )
    print(f"{name}: RED ({expected_failure})", flush=True)


def main() -> None:
    _run_unmutated_control()
    if any(
        len(mutation) > 5 and mutation[5] == "tests/test_gas.py"
        for mutation in MUTATIONS
    ):
        _run_unmutated_control("tests/test_gas.py")
    for mutation in MUTATIONS:
        _run_mutation(*mutation)
    print(f"All {len(MUTATIONS)} mutations were RED.")


if __name__ == "__main__":
    main()
