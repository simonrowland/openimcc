"""Content and combined identity contracts for the gas layer."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import random
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openimcc import (
    ImccGasDuplicateIntervalError as PublicDuplicateIntervalError,
    ImccGasInvalidIntervalError as PublicInvalidIntervalError,
    ImccUnprovenDatapackError,
    engine_binding_identity,
    evaluate,
    load_datapack,
)
from openimcc.gas import (
    IMCC_PARENT_OXIDES,
    ImccGasDatapack,
    ImccGasDuplicateIntervalError,
    ImccGasInvalidIntervalError,
    _nearest_interval_row,
    _gas_table_identity_cell,
    evaluate_gas,
    evaluate_gas_oxygen_balance,
    load_gas_datapack,
)


ROOT = Path(__file__).resolve().parents[1]
GAS_DATA = ROOT / "src/openimcc/data/gas"
PACK_DIR = ROOT / "src/openimcc/data/packs"
TABLE_FILES = ("gas-shomate.csv", "condensate.csv")
PACKAGED_GAS_TABLE_DIGEST = (
    "6dc7afd67397aa94b78a6a1b109c87156db2a79ac9e532146cfdcee44fb061fc"
)
PACKAGED_CONDENSATE_TABLE_DIGEST = (
    "fdfb97d72472a5b7f286ffc958a59c8eaf3b1ccd89af396b22cb9b8ace3fd0b0"
)
V1_COMPOSITION = {
    "SiO2": 0.45,
    "MgO": 0.20,
    "FeO": 0.10,
    "CaO": 0.08,
    "Al2O3": 0.08,
    "TiO2": 0.02,
    "Na2O": 0.05,
    "K2O": 0.02,
}


def _copy_tables(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in TABLE_FILES:
        path = directory / filename
        shutil.copyfile(GAS_DATA / filename, path)
        paths.append(path)
    return paths[0], paths[1]


def _load_tables(directory: Path):
    gas_path, condensate_path = _copy_tables(directory)
    return load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    *,
    lineterminator: str = "\n",
    quoting: int = csv.QUOTE_MINIMAL,
    trailing_newline: bool = True,
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        lineterminator=lineterminator,
        quoting=quoting,
    )
    writer.writeheader()
    writer.writerows(rows)
    contents = buffer.getvalue()
    if not trailing_newline:
        contents = contents.removesuffix(lineterminator)
    path.write_bytes(contents.encode("utf-8"))


def _gas_activities(temperature: float) -> dict[str, float]:
    melt_pack = load_datapack(PACK_DIR / "imcc-sf04-v1.0.2.json")
    result = evaluate(
        V1_COMPOSITION,
        temperature,
        melt_pack,
        allow_out_of_envelope=True,
        allow_extrapolation=True,
    )
    return {
        oxide: float(result.parent_activity[index])
        for index, oxide in enumerate(melt_pack.parent_oxides)
        if oxide in IMCC_PARENT_OXIDES
    }


def test_packaged_table_content_digests_match_current_tables() -> None:
    pack = load_gas_datapack()
    assert pack.gas_table_digest == PACKAGED_GAS_TABLE_DIGEST
    assert pack.condensate_table_digest == PACKAGED_CONDENSATE_TABLE_DIGEST


def test_mgo_supercooled_coefficient_changes_only_condensate_and_combined_identity(
    tmp_path: Path,
) -> None:
    melt_pack = load_datapack(PACK_DIR / "imcc-sf04-v1.0.2.json")
    original = _load_tables(tmp_path / "original")
    gas_path, _ = _copy_tables(tmp_path / "changed")
    condensate_path = tmp_path / "changed/condensate.csv"
    fields, rows = _read_csv(condensate_path)
    row = next(
        row
        for row in rows
        if row["species_name"] == "MgO(l)" and row["Ref"] == "Mg-009-SC-CP"
    )
    row["dG_C"] = format(float(row["dG_C"]) + 0.25, ".17g")
    _write_csv(condensate_path, fields, rows)
    changed = load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)

    before = engine_binding_identity(melt_pack, original)
    after = engine_binding_identity(melt_pack, changed)
    assert after.condensate_table_digest != before.condensate_table_digest
    assert after.gas_table_digest == before.gas_table_digest
    assert after.melt_binding_digest == before.melt_binding_digest
    assert after.digest != before.digest


def test_gas_cell_change_only_changes_gas_and_combined_identity(
    tmp_path: Path,
) -> None:
    melt_pack = load_datapack(PACK_DIR / "imcc-sf04-v1.0.2.json")
    original = _load_tables(tmp_path / "original")
    _, condensate_path = _copy_tables(tmp_path / "changed")
    gas_path = tmp_path / "changed/gas-shomate.csv"
    fields, rows = _read_csv(gas_path)
    row = next(row for row in rows if row["species_name"] == "Na(g)")
    row["A"] = format(float(row["A"]) + 0.25, ".17g")
    _write_csv(gas_path, fields, rows)
    changed = load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)

    before = engine_binding_identity(melt_pack, original)
    after = engine_binding_identity(melt_pack, changed)
    assert after.gas_table_digest != before.gas_table_digest
    assert after.condensate_table_digest == before.condensate_table_digest
    assert after.melt_binding_digest == before.melt_binding_digest
    assert after.digest != before.digest


def test_combined_identity_includes_the_published_melt_binding_digest() -> None:
    gas_pack = load_gas_datapack()
    v1 = load_datapack(PACK_DIR / "imcc-sf04-v1.0.2.json")
    extension = load_datapack(PACK_DIR / "imcc-sf04-ext-v4.json")

    v1_identity = engine_binding_identity(v1, gas_pack)
    extension_identity = engine_binding_identity(extension, gas_pack)
    assert v1_identity.melt_binding_digest == v1.binding_digest
    assert extension_identity.melt_binding_digest == extension.binding_digest
    assert v1_identity.gas_table_digest == extension_identity.gas_table_digest
    assert v1_identity.condensate_table_digest == extension_identity.condensate_table_digest
    assert v1_identity.digest != extension_identity.digest


def test_combined_identity_refuses_any_missing_component() -> None:
    melt_pack = load_datapack(PACK_DIR / "imcc-sf04-v1.0.2.json")
    gas_pack = load_gas_datapack()
    raw_pack = replace(
        melt_pack.kernel_datapack,
        A=np.array(melt_pack.kernel_datapack.A, copy=True),
    )
    incomplete_gas_packs = (
        SimpleNamespace(gas_table_digest=gas_pack.gas_table_digest),
        SimpleNamespace(condensate_table_digest=gas_pack.condensate_table_digest),
    )

    with pytest.raises(ImccUnprovenDatapackError):
        engine_binding_identity(raw_pack, gas_pack)
    for incomplete in incomplete_gas_packs:
        with pytest.raises(ImccUnprovenDatapackError):
            engine_binding_identity(melt_pack, incomplete)


def test_same_tables_in_another_directory_have_the_same_identity(
    tmp_path: Path,
) -> None:
    first = _load_tables(tmp_path / "first")
    second = _load_tables(tmp_path / "second")
    melt_pack = load_datapack(PACK_DIR / "imcc-sf04-v1.0.2.json")

    assert first.gas_path != second.gas_path
    assert first.oxide_path != second.oxide_path
    assert first.gas_table_digest == second.gas_table_digest
    assert first.condensate_table_digest == second.condensate_table_digest
    assert engine_binding_identity(melt_pack, first) == engine_binding_identity(
        melt_pack, second
    )


def test_csv_layout_does_not_change_parsed_content_identity(tmp_path: Path) -> None:
    original = _load_tables(tmp_path / "original")
    gas_path, condensate_path = _copy_tables(tmp_path / "variant")
    for path in (gas_path, condensate_path):
        fields, rows = _read_csv(path)
        _write_csv(
            path,
            list(reversed(fields)),
            rows,
            lineterminator="\r\n",
            quoting=csv.QUOTE_ALL,
            trailing_newline=False,
        )
    variant = load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)

    assert variant.gas_table_digest == original.gas_table_digest
    assert variant.condensate_table_digest == original.condensate_table_digest


def test_reordering_rows_preserves_identity_and_gas_evaluation(
    tmp_path: Path,
) -> None:
    original = _load_tables(tmp_path / "original")
    gas_path, condensate_path = _copy_tables(tmp_path / "reordered")
    for path in (gas_path, condensate_path):
        fields, rows = _read_csv(path)
        _write_csv(path, fields, list(reversed(rows)))
    reordered = load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)

    assert reordered.gas_table_digest == original.gas_table_digest
    assert reordered.condensate_table_digest == original.condensate_table_digest
    for table in (original.gas_df, original.oxide_df):
        rows = table.reset_index()
        assert not rows.duplicated(["species_name", "T_min"]).any()

    activities = _gas_activities(2500.0)
    for temperature in (1200.0, 1500.0, 1800.0, 2500.0, 3000.0):
        before = evaluate_gas(activities, temperature, 1.0e-8, original)
        after = evaluate_gas(activities, temperature, 1.0e-8, reordered)
        assert dict(after) == dict(before)
        assert dict(after.domain_flags) == dict(before.domain_flags)
        before_o2, before_pressures, _ = evaluate_gas_oxygen_balance(
            activities, temperature, original
        )
        after_o2, after_pressures, _ = evaluate_gas_oxygen_balance(
            activities, temperature, reordered
        )
        assert after_o2 == before_o2
        assert dict(after_pressures) == dict(before_pressures)


@pytest.mark.parametrize(
    ("filename", "species", "coefficient"),
    (
        ("gas-shomate.csv", "Na(g)", "A"),
        ("condensate.csv", "MgO(l)", "dG_A"),
    ),
)
def test_loader_refuses_duplicate_interval_starts(
    tmp_path: Path, filename: str, species: str, coefficient: str
) -> None:
    gas_path, condensate_path = _copy_tables(tmp_path)
    table_path = gas_path if filename == "gas-shomate.csv" else condensate_path
    fields, rows = _read_csv(table_path)
    original = next(row for row in rows if row["species_name"] == species)
    duplicate = dict(original)
    duplicate[coefficient] = format(float(original[coefficient]) + 1.0, ".17g")
    rows.append(duplicate)
    _write_csv(table_path, fields, rows)

    with pytest.raises(ImccGasDuplicateIntervalError, match="duplicate interval start"):
        load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)


def test_duplicate_gas_interval_probe_exposes_order_dependent_evaluation(
    tmp_path: Path,
) -> None:
    gas_path, condensate_path = _copy_tables(tmp_path / "first")
    fields, rows = _read_csv(gas_path)
    original = next(
        row
        for row in rows
        if row["species_name"] == "Na(g)" and float(row["T_min"]) == 1500.0
    )
    duplicate = dict(original)
    duplicate["A"] = format(float(original["A"]) + 1.0, ".17g")
    first_rows = rows + [duplicate]
    _write_csv(gas_path, fields, first_rows)
    try:
        first_pack = load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)
    except ImccGasDuplicateIntervalError:
        return

    reversed_gas_path, reversed_condensate_path = _copy_tables(tmp_path / "reversed")
    _write_csv(reversed_gas_path, fields, list(reversed(first_rows)))
    second_pack = load_gas_datapack(
        gas_path=reversed_gas_path, oxide_path=reversed_condensate_path
    )
    assert first_pack.gas_table_digest == second_pack.gas_table_digest
    melt_pack = load_datapack(PACK_DIR / "imcc-sf04-v1.0.2.json")
    first_identity = engine_binding_identity(melt_pack, first_pack)
    second_identity = engine_binding_identity(melt_pack, second_pack)
    assert first_identity.digest == second_identity.digest
    activities = _gas_activities(1800.0)
    first_pressures = dict(evaluate_gas(activities, 1800.0, 1.0e-8, first_pack))
    second_pressures = dict(evaluate_gas(activities, 1800.0, 1.0e-8, second_pack))
    assert first_pressures != second_pressures
    pytest.fail(
        "loader accepted duplicate interval with same identity: "
        f"gas_table_digest={first_pack.gas_table_digest}; "
        f"engine_binding_identity={first_identity.digest}; "
        f"Na pressures by row order are "
        f"{first_pressures['Na']!r} and {second_pressures['Na']!r}"
    )


@pytest.mark.parametrize(
    ("filename", "species", "bound"),
    (("gas-shomate.csv", "Na(g)", "T_min"), ("condensate.csv", "MgO(l)", "T_min")),
)
@pytest.mark.parametrize("raw_bound", ("-inf", "inf", "NaN"))
def test_loader_refuses_non_finite_interval_bounds(
    tmp_path: Path, filename: str, species: str, bound: str, raw_bound: str
) -> None:
    gas_path, condensate_path = _copy_tables(tmp_path)
    table_path = gas_path if filename == TABLE_FILES[0] else condensate_path
    fields, rows = _read_csv(table_path)
    next(row for row in rows if row["species_name"] == species)[bound] = raw_bound
    _write_csv(table_path, fields, rows)

    with pytest.raises(PublicInvalidIntervalError, match="non-finite or non-numeric"):
        load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)


@pytest.mark.parametrize("filename", TABLE_FILES)
@pytest.mark.parametrize("raw_bound", ("", "not-a-number", "NaN", "inf", "-inf"))
def test_loader_refuses_invalid_interval_bound_spellings(
    tmp_path: Path, filename: str, raw_bound: str
) -> None:
    gas_path, condensate_path = _copy_tables(tmp_path)
    table_path = gas_path if filename == TABLE_FILES[0] else condensate_path
    fields, rows = _read_csv(table_path)
    rows[0]["T_max"] = raw_bound
    _write_csv(table_path, fields, rows)

    with pytest.raises(ImccGasInvalidIntervalError):
        load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)


@pytest.mark.parametrize(
    ("filename", "species", "coefficient", "first", "second"),
    (
        ("gas-shomate.csv", "Na(g)", "A", "9007199254740992", "9007199254740993"),
        ("condensate.csv", "MgO(l)", "dG_A", "1500", "1500.0"),
    ),
)
def test_loader_refuses_duplicate_starts_after_float64_normalization(
    tmp_path: Path,
    filename: str,
    species: str,
    coefficient: str,
    first: str,
    second: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gas_path, condensate_path = _copy_tables(tmp_path)
    table_path = gas_path if filename == TABLE_FILES[0] else condensate_path
    fields, rows = _read_csv(table_path)
    original = next(row for row in rows if row["species_name"] == species)
    duplicate = dict(original)
    original["T_min"] = first
    duplicate["T_min"] = second
    original["T_max"] = duplicate["T_max"] = "9007199254740994"
    duplicate[coefficient] = format(float(original[coefficient]) + 1.0, ".17g")
    rows.append(duplicate)
    _write_csv(table_path, fields, rows)
    if filename == "condensate.csv":
        original_read_csv = pd.read_csv

        def read_csv_with_raw_bound(*args, **kwargs):
            frame = original_read_csv(*args, **kwargs)
            source = Path(args[0])
            if source == condensate_path:
                with source.open(encoding="utf-8", newline="") as stream:
                    raw_rows = list(csv.DictReader(stream))
                frame["T_min"] = [row["T_min"] for row in raw_rows]
            return frame

        monkeypatch.setattr(pd, "read_csv", read_csv_with_raw_bound)

    with pytest.raises(PublicDuplicateIntervalError, match="duplicate interval start"):
        load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)


def test_numeric_bound_spellings_have_one_evaluation_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_read_csv = pd.read_csv

    def read_csv_preserving_bound_text(*args, **kwargs):
        frame = original_read_csv(*args, **kwargs)
        source = Path(args[0])
        with source.open(encoding="utf-8", newline="") as stream:
            raw_rows = list(csv.DictReader(stream))
        for column in ("T_min", "T_max"):
            frame[column] = [row[column] for row in raw_rows]
        return frame

    monkeypatch.setattr(pd, "read_csv", read_csv_preserving_bound_text)
    packs = []
    for folder, spelling in (("integer", "1500"), ("decimal", "1500.0")):
        gas_path, condensate_path = _copy_tables(tmp_path / folder)
        fields, rows = _read_csv(gas_path)
        for row in rows:
            if row["species_name"] == "Na(g)" and float(row["T_min"]) == 1500.0:
                row["T_min"] = spelling
                row["T_max"] = "3000"
                break
        _write_csv(gas_path, fields, rows)
        packs.append(load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path))

    assert packs[0].gas_table_digest == packs[1].gas_table_digest
    assert packs[0].gas_df["T_min"].dtype == np.dtype("float64")
    activities = _gas_activities(1800.0)
    for allow_extrapolation in (True, False):
        assert dict(evaluate_gas(activities, 1800.0, 1.0e-8, packs[0], allow_extrapolation=allow_extrapolation)) == dict(
            evaluate_gas(activities, 1800.0, 1.0e-8, packs[1], allow_extrapolation=allow_extrapolation)
        )


def test_selector_uses_normalized_interval_values_without_reconversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = _load_tables(tmp_path)

    def forbidden_reconversion(*args, **kwargs):
        raise AssertionError("selector converted an already-normalized bound")

    monkeypatch.setattr(pd.Series, "astype", forbidden_reconversion)
    selected = _nearest_interval_row(pack.gas_df, "Na(g)", 1800.0)
    assert selected["T_min"] == 1500.0


def _pack_with_duplicate_start(
    tmp_path: Path, construction: str, table_name: str, reverse: bool
) -> ImccGasDatapack:
    loaded = _load_tables(tmp_path / f"{construction}-{table_name}-{reverse}")
    pack = loaded if construction == "load_then_mutate" else ImccGasDatapack(
        loaded.gas_df.copy(deep=True), loaded.oxide_df.copy(deep=True),
        loaded.gas_path, loaded.oxide_path,
    )
    frame = pack.gas_df if table_name == "gas" else pack.oxide_df
    species = "Na(g)" if table_name == "gas" else "Na2O(l)"
    bound_column = frame.columns.get_loc("T_min")
    row_positions = np.flatnonzero(frame.index == species)
    assert len(row_positions) >= 2
    frame.iloc[row_positions[1], bound_column] = frame.iloc[row_positions[0]]["T_min"]
    if reverse:
        frame = frame.iloc[::-1].copy()
        pack = ImccGasDatapack(
            frame if table_name == "gas" else pack.gas_df,
            frame if table_name == "condensate" else pack.oxide_df,
            pack.gas_path, pack.oxide_path,
        )
    return pack


@pytest.mark.parametrize("construction", ["direct", "load_then_mutate"])
@pytest.mark.parametrize("table_name", ["gas", "condensate"])
@pytest.mark.parametrize("reverse", [False, True])
def test_invalid_duplicate_starts_refuse_identity(
    tmp_path: Path, construction: str, table_name: str, reverse: bool
) -> None:
    pack = _pack_with_duplicate_start(tmp_path, construction, table_name, reverse)
    digest_name = "gas_table_digest" if table_name == "gas" else "condensate_table_digest"
    with pytest.raises(ImccGasDuplicateIntervalError):
        getattr(pack, digest_name)
    melt_pack = load_datapack(PACK_DIR / "imcc-sf04-v1.0.2.json")
    with pytest.raises(ImccGasDuplicateIntervalError):
        engine_binding_identity(melt_pack, pack)


@pytest.mark.parametrize("construction", ["direct", "load_then_mutate"])
@pytest.mark.parametrize("table_name", ["gas", "condensate"])
@pytest.mark.parametrize("reverse", [False, True])
def test_invalid_duplicate_starts_refuse_evaluation(
    tmp_path: Path, construction: str, table_name: str, reverse: bool
) -> None:
    pack = _pack_with_duplicate_start(tmp_path, construction, table_name, reverse)
    with pytest.raises(ImccGasDuplicateIntervalError):
        evaluate_gas(
            {"Na2O": 0.05}, 1800.0, 1.0e-8, pack,
            parent_oxides=("Na2O",), gas_species=("Na",), include_ions=True,
        )
    with pytest.raises(ImccGasDuplicateIntervalError):
        evaluate_gas_oxygen_balance(
            {"Na2O": 0.05}, 1800.0, pack, parent_oxides=("Na2O",),
        )


@pytest.mark.parametrize("bad_value", [float("-inf"), float("nan"), "not-a-number"])
def test_invalid_direct_bounds_refuse_identity_and_evaluation(
    bad_value: object,
) -> None:
    loaded = load_gas_datapack()
    pack = ImccGasDatapack(
        loaded.gas_df.copy(deep=True), loaded.oxide_df.copy(deep=True),
        loaded.gas_path, loaded.oxide_path,
    )
    index = pack.gas_df.index[pack.gas_df.index == "Na(g)"][0]
    if not isinstance(bad_value, float) or math.isfinite(bad_value):
        pack.gas_df["T_min"] = pack.gas_df["T_min"].astype(object)
    pack.gas_df.loc[index, "T_min"] = bad_value
    with pytest.raises(ImccGasInvalidIntervalError):
        _ = pack.gas_table_digest
    with pytest.raises(ImccGasInvalidIntervalError):
        evaluate_gas({"Na2O": 0.05}, 1800.0, 1.0e-8, pack,
                     parent_oxides=("Na2O",), gas_species=("Na",))


@pytest.mark.parametrize("dtype", ["int64", str])
def test_direct_non_float64_bound_columns_are_refused(dtype: object) -> None:
    loaded = load_gas_datapack()
    pack = ImccGasDatapack(
        loaded.gas_df.copy(deep=True), loaded.oxide_df.copy(deep=True),
        loaded.gas_path, loaded.oxide_path,
    )
    pack.gas_df["T_min"] = pack.gas_df["T_min"].astype(dtype)
    with pytest.raises(ImccGasInvalidIntervalError):
        _ = pack.gas_table_digest
    with pytest.raises(ImccGasInvalidIntervalError):
        evaluate_gas({"Na2O": 0.05}, 1800.0, 1.0e-8, pack,
                     parent_oxides=("Na2O",), gas_species=("Na",))


def test_valid_direct_pack_matches_loaded_identity_and_evaluation() -> None:
    loaded = load_gas_datapack()
    direct = ImccGasDatapack(
        loaded.gas_df.copy(deep=True), loaded.oxide_df.copy(deep=True),
        loaded.gas_path, loaded.oxide_path,
    )
    assert direct.gas_table_digest == loaded.gas_table_digest
    assert direct.condensate_table_digest == loaded.condensate_table_digest
    first = evaluate_gas({"Na2O": 0.05}, 1800.0, 1.0e-8, loaded,
                         parent_oxides=("Na2O",), gas_species=("Na",))
    second = evaluate_gas({"Na2O": 0.05}, 1800.0, 1.0e-8, direct,
                          parent_oxides=("Na2O",), gas_species=("Na",))
    assert dict(first) == dict(second)


@pytest.mark.parametrize("filename", TABLE_FILES)
def test_loader_refuses_inverted_bounds_and_accepts_equal_bounds(
    tmp_path: Path, filename: str
) -> None:
    for equal, expected_error in ((False, True), (True, False)):
        folder = tmp_path / f"{filename}-{equal}"
        gas_path, condensate_path = _copy_tables(folder)
        table_path = gas_path if filename == TABLE_FILES[0] else condensate_path
        fields, rows = _read_csv(table_path)
        row = rows[0]
        row["T_min"] = "1200"
        row["T_max"] = "1200" if equal else "1199"
        _write_csv(table_path, fields, rows)
        if expected_error:
            with pytest.raises(ImccGasInvalidIntervalError, match="T_min greater than T_max"):
                load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)
        else:
            pack = load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)
            table = pack.gas_df if filename == TABLE_FILES[0] else pack.oxide_df
            assert float(table.iloc[0]["T_min"]) == float(table.iloc[0]["T_max"])


def _order_outcome(activities, temperature, pack, allow_extrapolation):
    try:
        return ("ok", dict(evaluate_gas(
            activities, temperature, 1.0e-8, pack,
            allow_extrapolation=allow_extrapolation,
            parent_oxides=("Na2O",), gas_species=("Na",),
        )))
    except Exception as exc:
        return (type(exc), str(exc))


def test_random_accepted_tables_are_order_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = random.Random(69024)
    gas_template = _read_csv(GAS_DATA / TABLE_FILES[0])
    oxide_template = _read_csv(GAS_DATA / TABLE_FILES[1])
    tables = permutations = 0
    activities = {"Na2O": 0.05}
    selections = {
        mode: {"Na(g)": 0, "Na2O(l)": 0} for mode in ("strict", "extrapolating")
    }
    selector_globals = evaluate_gas.__globals__
    selector = selector_globals["_nearest_interval_row"]

    def count_selections(df, species, temperature, allow_extrapolation=False):
        row = selector(df, species, temperature, allow_extrapolation)
        if species in selections["strict"]:
            mode = "extrapolating" if allow_extrapolation else "strict"
            selections[mode][species] += 1
        return row

    monkeypatch.setitem(selector_globals, "_nearest_interval_row", count_selections)

    for case in range(24):
        gas_fields, gas_rows = gas_template
        oxide_fields, oxide_rows = oxide_template
        na_template = dict(next(row for row in gas_rows if row["species_name"] == "Na(g)"))
        o2_template = dict(next(row for row in gas_rows if row["species_name"] == "O2(g)"))
        oxide_template_row = dict(next(row for row in oxide_rows if row["species_name"] == "Na2O(l)"))
        gas_rows = [o2_template]
        oxide_rows = []
        # Three unique starts include a shared breakpoint and overlapping spans;
        # rows use distinct coefficient values so the selected interval matters.
        starts = [300 + rng.randrange(0, 40), 500, 500 + rng.randrange(1, 40)]
        starts = sorted(set(starts))
        if len(starts) != 3:
            starts = [300, 500, 501]
        ends = [starts[1], starts[2] + 150, starts[2] + 300]
        gas_new = []
        o2_template["T_min"], o2_template["T_max"] = "0", "2000"
        for index, (low, high) in enumerate(zip(starts, ends)):
            row = dict(na_template)
            row["T_min"], row["T_max"] = low, float(high) if index == 1 else str(high)
            row["A"] = str(float(row["A"]) + index * (case + 1))
            gas_new.append(row)
        gas_rows.extend(gas_new)
        for index, (low, high) in enumerate(zip(starts[:2], ends[:2])):
            row = dict(oxide_template_row)
            row["T_min"], row["T_max"] = (float(low), high) if index == 0 else (str(low), str(high))
            row["dG_A"] = str(float(row["dG_A"]) + index * (case + 1))
            oxide_rows.append(row)

        # O2(g) remains a single-row species; Na(g) has overlapping intervals.
        gas_path, condensate_path = _copy_tables(tmp_path / f"case-{case}")
        _write_csv(gas_path, gas_fields, gas_rows)
        _write_csv(condensate_path, oxide_fields, oxide_rows)
        original = load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)
        assert original.gas_df["T_min"].dtype == np.dtype("float64")
        orderings = [tuple(range(len(gas_rows)))]
        while len(orderings) < 5:
            candidate_order = tuple(rng.sample(range(len(gas_rows)), len(gas_rows)))
            if candidate_order not in orderings:
                orderings.append(candidate_order)
        temperatures = sorted({
            starts[0] - 1, *starts, *ends,
            *(low + (high - low) / 2 for low, high in zip(starts, ends)),
            ends[-1] + 1,
        })
        for order in orderings:
            permuted_gas = [gas_rows[i] for i in order]
            shuffled_condensate = rng.sample(oxide_rows, len(oxide_rows))
            _write_csv(gas_path, gas_fields, permuted_gas)
            _write_csv(condensate_path, oxide_fields, shuffled_condensate)
            candidate = load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)
            assert candidate.gas_table_digest == original.gas_table_digest
            assert candidate.condensate_table_digest == original.condensate_table_digest
            for temperature in temperatures:
                for extrapolate in (True, False):
                    candidate_outcome = _order_outcome(
                        activities, temperature, candidate, extrapolate
                    )
                    original_outcome = _order_outcome(
                        activities, temperature, original, extrapolate
                    )
                    assert candidate_outcome == original_outcome
                    if extrapolate:
                        assert candidate_outcome[0] == "ok"
            permutations += 1
        tables += 1

    assert tables == 24
    assert permutations == 120
    assert all(selections["strict"][species] > 0 for species in selections["strict"]), selections
    assert all(selections["extrapolating"][species] > 0 for species in selections["extrapolating"]), selections


def test_gas_identity_is_stable_across_interpreter_hash_seeds() -> None:
    code = """
import json
from pathlib import Path
from openimcc import engine_binding_identity, load_datapack
from openimcc.gas import load_gas_datapack
gas = load_gas_datapack()
packs = Path('src/openimcc/data/packs')
identities = [
    engine_binding_identity(load_datapack(packs / name), gas)
    for name in ('imcc-sf04-v1.0.2.json', 'imcc-sf04-ext-v4.json')
]
print(json.dumps([
    [item.melt_binding_digest, item.condensate_table_digest,
     item.gas_table_digest, item.digest]
    for item in identities
]))
"""
    results = []
    for seed in ("1", "98765"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(completed.stdout))
    assert results[0] == results[1]


def _change_dataframe_cell(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    changed = frame.copy(deep=True)
    if column == frame.index.name:
        labels = changed.index.tolist()
        labels[0] = f"{labels[0]}-changed"
        changed.index = pd.Index(labels, name=frame.index.name)
        return changed

    column_index = changed.columns.get_loc(column)
    row_index = next(
        index
        for index, value in enumerate(changed[column].tolist())
        if not pd.isna(value)
    )
    value = changed.iat[row_index, column_index]
    if isinstance(value, str):
        replacement = f"{value}-changed"
    elif isinstance(value, (int, np.integer)):
        replacement = int(value) + 1
    else:
        replacement = float(value) + 0.5
    changed.iat[row_index, column_index] = replacement
    return changed


def test_digest_is_sensitive_to_every_column_returned_by_the_loader() -> None:
    pack = load_gas_datapack()
    for frame_name, digest_name in (
        ("gas_df", "gas_table_digest"),
        ("oxide_df", "condensate_table_digest"),
    ):
        frame = getattr(pack, frame_name)
        columns_read = set(frame.columns) | {frame.index.name}
        assert frame.index.name == "species_name"
        original_digest = getattr(pack, digest_name)
        covered = set()
        for column in columns_read:
            changed_frame = _change_dataframe_cell(frame, column)
            changed_pack = replace(pack, **{frame_name: changed_frame})
            assert getattr(changed_pack, digest_name) != original_digest, column
            covered.add(column)
        assert covered == columns_read


@pytest.mark.parametrize("filename", TABLE_FILES)
def test_adding_or_removing_a_parsed_row_changes_its_digest(
    tmp_path: Path, filename: str
) -> None:
    original = _load_tables(tmp_path / "original")
    table_attribute = (
        "gas_table_digest" if filename == TABLE_FILES[0] else "condensate_table_digest"
    )
    for operation in ("added", "removed"):
        gas_path, condensate_path = _copy_tables(tmp_path / operation)
        path = gas_path if filename == TABLE_FILES[0] else condensate_path
        fields, rows = _read_csv(path)
        if operation == "added":
            extra = dict(rows[0])
            extra["species_name"] += "-added"
            extra["Ref"] += "-added"
            rows.append(extra)
        else:
            rows.pop()
        _write_csv(path, fields, rows)
        changed = load_gas_datapack(gas_path=gas_path, oxide_path=condensate_path)
        assert getattr(changed, table_attribute) != getattr(original, table_attribute)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (math.nan, None),
        (math.inf, {"non_finite_float": "positive_infinity"}),
        (-math.inf, {"non_finite_float": "negative_infinity"}),
    ),
)
def test_non_finite_csv_cells_have_canonical_payload_values(value, expected) -> None:
    assert _gas_table_identity_cell(value) == expected
