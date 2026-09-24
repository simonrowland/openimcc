"""Data-source, provenance, fit, and cross-check gates for the gas tables."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from openimcc.gas import (
    _GAS_PROVENANCE_AUTHORITY,
    _OXIDE_PROVENANCE_AUTHORITY,
    IMCC_GAS_CHANNEL_SPECIES,
    IMCC_SF04_WORKBOOK_GRID_K,
    R_J_MOL_K,
    _SF04_REACTIONS,
    _janaf_gibbs,
    _nearest_interval_row,
    load_gas_datapack,
)


ROOT = Path(__file__).resolve().parents[1]
GAS_DATA = ROOT / "src" / "openimcc" / "data" / "gas"
JANAF_DATA = ROOT / "data-src" / "janaf"
PROVENANCE_PATH = GAS_DATA / "PROVENANCE.yaml"

GAS_TABLE_IDS = {
    "Na": "Na-005",
    "K": "K-005",
    "SiO": "O-012",
    "Fe": "Fe-008",
    "FeO": "Fe-021",
    "Mg": "Mg-005",
    "MgO": "Mg-011",
    "SiO2": "O-040",
    "O": "O-001",
    "AlO": "Al-074",
    "AlO2": "Al-077",
    "Al2O": "Al-092",
    "Al2O2": "Al-094",
    "Na2": "Na-011",
    "NaO": "Na-008",
    "K2": "K-011",
    "KO": "K-008",
    "Si": "Si-005",
    "Al": "Al-005",
    "CaO": "Ca-030",
    "Ca": "Ca-006",
    "O2": "O-029",
}

PARENT_TABLE_IDS = {
    "Na2O": "Na-013",
    "K2O": "K-012",  # JANAF has crystal K2O only; K2O(l) is LH84 secondary.
    "MgO": "Mg-009",
    "CaO": "Ca-028",
    "Al2O3": "Al-100",
    "SiO2": "O-038",
    "FeO": "Fe-019",
}

_SOURCE_FIELDS = (
    "temperature",
    "heat_capacity",
    "entropy",
    "enthalpy_increment",
    "formation_enthalpy",
    "formation_gibbs_energy",
)


@lru_cache(maxsize=None)
def _record(table_id: str) -> dict:
    text = (JANAF_DATA / f"{table_id}.yaml").read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def _value(row: dict, field: str):
    return row[field]["value"]


@lru_cache(maxsize=None)
def _complete_rows(table_id: str) -> list[dict[str, float]]:
    rows = []
    for raw in _record(table_id)["table"]["values"]:
        values = {field: _value(raw, field) for field in _SOURCE_FIELDS}
        if all(value is not None for value in values.values()):
            rows.append({field: float(value) for field, value in values.items()})
    return rows


@lru_cache(maxsize=None)
def _ambiguous_temperatures(table_id: str) -> set[float]:
    result = set()
    for ambiguity in _record(table_id)["table"].get("parse_ambiguities", []):
        token = str(ambiguity.get("raw_line", "")).strip().split("\t", 1)[0]
        try:
            result.add(float(token))
        except ValueError:
            continue
    return result


def test_vendored_source_hashes_and_provenance_are_row_complete() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    rows = provenance["rows"]
    gas_rows = {row["species_name"]: row for row in rows if row["table"] == "gas"}
    oxide_rows = {
        row["species_name"]: row for row in rows if row["table"] == "condensate"
    }
    assert set(gas_rows) == {f"{name}(g)" for name in IMCC_GAS_CHANNEL_SPECIES}
    assert len(oxide_rows) == 8

    for species, table_id in GAS_TABLE_IDS.items():
        source = _record(table_id)
        row = gas_rows[f"{species}(g)"]
        assert row["table_id"] == table_id
        assert row["source_path"] == f"data-src/janaf/{table_id}.yaml"
        assert (ROOT / row["source_path"]).is_file()
        assert row["source_sha256"] == source["extraction"]["source_sha256"]
        assert row["authority"] == "janaf_fitted"
        assert row["method"] == "fitted"
        assert row["T_range_K"] == [1500, 3000]
        assert source["source"]["doi"] == "10.18434/T42S31"

    k_row = oxide_rows["K2O(l)"]
    assert k_row["authority"] == "secondary_transcription_unverified_primary"
    assert k_row["method"] == "transcribed_secondary_unverified_primary"
    assert k_row["source"]["primary_doi"] == "10.1063/1.555706"
    assert "VapoRock" in k_row["source"]["transcription"]


def test_runtime_provenance_mirror_matches_yaml() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    expected_gas = {}
    expected_oxide = {}
    for row in provenance["rows"]:
        if row["table"] == "gas":
            expected_gas[row["species_name"].removesuffix("(g)")] = row["authority"]
        else:
            name = row["species_name"]
            expected_oxide[name.removesuffix("(l)")] = row["authority"]

    # This is the deliberate source-of-truth boundary: the runtime remains
    # free of a YAML dependency, while this test compares every gas and oxide
    # provenance row against the checked-in YAML.  SiO2(cr) is retained in the
    # mirror for parity even though the SF04 reaction set consumes SiO2(l).
    assert _GAS_PROVENANCE_AUTHORITY == expected_gas
    assert _OXIDE_PROVENANCE_AUTHORITY == expected_oxide


def test_fitted_gas_rows_reproduce_every_complete_janaf_g_app_row() -> None:
    pack = load_gas_datapack()
    maxima: dict[str, tuple[float, float]] = {}
    for species, table_id in GAS_TABLE_IDS.items():
        source_rows = _complete_rows(table_id)
        reference = next(
            row["formation_enthalpy"]
            for row in source_rows
            if row["temperature"] == 298.15
        )
        fitted_row = pack.gas_df.loc[f"{species}(g)"]
        residuals = []
        log_residuals = []
        for source_row in source_rows:
            temperature = source_row["temperature"]
            if not 1500.0 <= temperature <= 3000.0:
                continue
            # Premise: the source apparent Gibbs value uses the 298.15 K
            # formation enthalpy plus H-H(298) minus T*S. Algebra: this is the
            # independent value against which the fitted Shomate row is tested.
            # Unit check: kJ/mol is converted to J/mol before T(K)*S(J/mol/K).
            # Sanity: the maximum residual must stay far below 0.01 log10 K.
            source_g = (
                reference + source_row["enthalpy_increment"]
            ) * 1000.0 - temperature * source_row["entropy"]
            fitted_g = _janaf_gibbs(temperature, fitted_row)
            residual = abs(fitted_g - source_g)
            residuals.append(residual)
            log_residuals.append(
                residual / (R_J_MOL_K * temperature * math.log(10.0))
            )
        maxima[species] = (max(residuals), max(log_residuals))

    assert max(value[1] for value in maxima.values()) <= 0.01
    assert max(value[0] for value in maxima.values()) < 2.1


def test_no_fitted_row_consumes_a_parse_ambiguous_source_row() -> None:
    for table_id in GAS_TABLE_IDS.values():
        selected_temperatures = {
            row["temperature"]
            for row in _complete_rows(table_id)
            if 1500.0 <= row["temperature"] <= 3000.0
        }
        assert selected_temperatures.isdisjoint(_ambiguous_temperatures(table_id))


def test_generator_is_deterministic_and_matches_packaged_output(tmp_path: Path) -> None:
    generated = tmp_path / "gas-shomate.csv"
    command = [
        sys.executable,
        str(ROOT / "tools" / "build_gas_tables.py"),
        "--output",
        str(generated),
    ]
    first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert first.stdout == second.stdout
    assert generated.read_bytes() == (GAS_DATA / "gas-shomate.csv").read_bytes()


_CONDENSATE_EXPECTED = {
    "MgO(l)": (3100, 3500, -72.34, -0.804, 5.1067, -0.3615, 0.0, 0.0, "LAM1987"),
    "CaO(l)": (2900, 3800, -76.384, 0.924, 5.922, -0.7316, 0.0436, 0.0, "LAM1987"),
    "Al2O3(l)": (
        2327,
        3000,
        -188.14,
        232.345,
        -336.622,
        193.672,
        -48.1032,
        4.4461,
        "LAM1987",
    ),
    "SiO2(l)": (1996, 3000, -109.53, 2.12, 7.6492, -1.2588, 0.0998, 0.0, "LAM1987"),
    "SiO2(cr)": (1000, 1996, -109.53, 1.937, 8.59, -1.935, 0.225, 0.0, "LAM1987"),
    "Na2O(l)": (825, 3000, -50.17, 4.82, 19.292, -5.267, 0.623, 0.0, "LAM1984"),
    "K2O(l)": (1190, 3000, -43.58, 0.8, 18.889, -4.532, 0.467, 0.0, "LAM1984"),
    "FeO(l)": (1000, 5000, -30.01, 6.72, 6.588, -1.248, 0.150, -0.007697, "JANAF"),
}


def test_condensate_rows_match_the_current_published_coefficients() -> None:
    table = pd.read_csv(GAS_DATA / "condensate.csv").set_index("species_name")
    assert set(table.index) == set(_CONDENSATE_EXPECTED)
    for species, expected in _CONDENSATE_EXPECTED.items():
        actual = table.loc[species]
        assert (int(actual["T_min"]), int(actual["T_max"])) == expected[:2]
        for column, value in zip(
            ("dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E"),
            expected[2:8],
        ):
            assert actual[column] == value
        assert actual["Ref"] == expected[8]


def _source_g_app_from_row(table_id: str, row: dict[str, float]) -> float:
    rows = _complete_rows(table_id)
    reference = next(
        row["formation_enthalpy"] for row in rows if row["temperature"] == 298.15
    )
    return (
        (reference + row["enthalpy_increment"]) * 1000.0
        - row["temperature"] * row["entropy"]
    )


def _reaction_source_series(
    species: str,
) -> tuple[int, float, list[float], list[float], list[float]]:
    parent, n_gas, n_o2 = _SF04_REACTIONS[species]
    species_id = GAS_TABLE_IDS[species]
    gas_rows = {row["temperature"]: row for row in _complete_rows(species_id)}
    oxygen_rows = {
        row["temperature"]: row for row in _complete_rows("O-029")
    }
    parent_id = PARENT_TABLE_IDS[parent] if parent else None
    parent_rows = (
        {row["temperature"]: row for row in _complete_rows(parent_id)}
        if parent_id
        else {}
    )
    common_temperatures = sorted(
        set(gas_rows)
        & set(oxygen_rows)
        & (set(parent_rows) if parent else set(gas_rows))
    )
    common_temperatures = [
        temperature
        for temperature in common_temperatures
        if 1500.0 <= temperature <= 3000.0
        and temperature not in _ambiguous_temperatures(species_id)
        and temperature not in _ambiguous_temperatures("O-029")
        and (
            not parent_id
            or temperature not in _ambiguous_temperatures(parent_id)
        )
    ]
    independent_reactions = []
    source_parent_app = []
    for source_temperature in common_temperatures:
        gas_row = gas_rows[source_temperature]
        oxygen_row = oxygen_rows[source_temperature]
        parent_row = parent_rows[source_temperature] if parent_id else None
        independent_reactions.append(
            n_gas * gas_row["formation_gibbs_energy"] * 1000.0
            + n_o2 * oxygen_row["formation_gibbs_energy"] * 1000.0
            - (
                parent_row["formation_gibbs_energy"] * 1000.0
                if parent_row
                else 0.0
            )
        )
        source_parent_app.append(
            _source_g_app_from_row(parent_id, parent_row) if parent_row else 0.0
        )
    return n_gas, n_o2, common_temperatures, independent_reactions, source_parent_app


def test_g2_reaction_convention_at_complete_janaf_nodes() -> None:
    """Check the gas convention only where every source table has a row.

    Premise: JANAF's formation-Gibbs column and the fitted apparent-Gibbs
    construction use the same elemental reference. Element-reference identity
    therefore cancels every balanced reaction's elemental baselines, leaving
    the fitted ``F - H°298`` offset, reference-state choice, and sign exposed.
    No interpolation is allowed here: both sides are evaluated at the same
    complete, non-ambiguous source node. Unit check: all values are J/mol.
    Sanity: a reference-state bias of 100 J/mol cannot hide behind a grid chord.
    """
    pack = load_gas_datapack()
    checked = 0
    maxima: dict[str, float] = {}
    for species in _SF04_REACTIONS:
        if species == "O2":
            continue
        n_gas, n_o2, temperatures, independent, source_parent_app = (
            _reaction_source_series(species)
        )
        row_max = 0.0
        for temperature, independent_reaction, independent_parent in zip(
            temperatures, independent, source_parent_app
        ):
            fitted_reaction = (
                n_gas * _janaf_gibbs(
                    temperature, pack.gas_df.loc[f"{species}(g)"]
                )
                + n_o2 * _janaf_gibbs(
                    temperature, pack.gas_df.loc["O2(g)"]
                )
                - independent_parent
            )
            row_max = max(row_max, abs(fitted_reaction - independent_reaction))
            checked += 1
        maxima[species] = row_max

    assert checked == 289
    # The measured on-node maximum is 2.728 J/mol (Fe); 10 J/mol leaves a
    # stated margin while remaining far below the old workbook-grid gate.
    assert max(maxima.values()) <= 10.0


_G2_SOURCE_HOLES = (
    ("Na2O", "Na-013", 1500.0),
    ("FeO", "Fe-019", 1700.0),
    ("SiO2", "O-038", 1700.0),
    ("MgO", "Mg-009", 2100.001),
    ("CaO", "Ca-028", 2100.0),
    ("Al2O3", "Al-100", 2400.0),
)


def test_g2_reaction_convention_on_workbook_grid() -> None:
    """Use the workbook grid as coverage evidence, including interpolation.

    This is deliberately a looser ``< 300 J/mol`` gate: it includes the linear
    interpolation error across known source holes, not only convention error.
    The named holes are Na2O/Na-013 at 1500 K, FeO/Fe-019 at 1700 K,
    SiO2/O-038 at 1700 K, MgO/Mg-009 at 2100.001 K, CaO/Ca-028 at 2100 K, and
    Al2O3/Al-100 at 2400 K. K2O's K-012 parent ends at 2000 K, so K, K2 and KO
    have no independent parent reference at 2125, 2250, 2375 or 2500 K.
    """
    for _oxide, table_id, temperature in _G2_SOURCE_HOLES:
        assert temperature in _ambiguous_temperatures(table_id)

    pack = load_gas_datapack()
    checked = 0
    missing: set[tuple[str, float]] = set()
    maxima: dict[str, float] = {}
    for species in _SF04_REACTIONS:
        if species == "O2":
            continue
        n_gas, n_o2, common_temperatures, independent_reactions, source_parent_app = (
            _reaction_source_series(species)
        )
        for temperature in IMCC_SF04_WORKBOOK_GRID_K:
            if (
                not common_temperatures
                or temperature < common_temperatures[0]
                or temperature > common_temperatures[-1]
            ):
                missing.add((species, temperature))
                continue
            # Premise: source reaction and parent apparent-G values are known
            # only at complete common nodes. Algebra: interpolate each series
            # to the workbook temperature. Unit check: the result remains
            # J/mol. Sanity: exact source nodes are returned unchanged.
            independent_reaction = float(
                np.interp(
                    temperature, common_temperatures, independent_reactions
                )
            )
            independent_parent = float(
                np.interp(temperature, common_temperatures, source_parent_app)
            )
            fitted_reaction = (
                n_gas * _janaf_gibbs(
                    temperature, pack.gas_df.loc[f"{species}(g)"]
                )
                + n_o2 * _janaf_gibbs(
                    temperature, pack.gas_df.loc["O2(g)"]
                )
                - independent_parent
            )
            discrepancy = abs(fitted_reaction - independent_reaction)
            maxima[species] = max(maxima.get(species, 0.0), discrepancy)
            checked += 1

    assert missing == {
        (species, 1500.0)
        for species in ("Na", "Na2", "NaO")
    } | {
        (species, temperature)
        for species in ("K", "K2", "KO")
        for temperature in (2125.0, 2250.0, 2375.0, 2500.0)
    }
    assert checked == 195
    assert max(maxima.values()) < 300.0


# The legacy VapoRock tables are not shipped; comparisons against them run only
# when a checkout is selected with OPENIMCC_VAPOROCK_ROOT, and skip otherwise.
# tests/conftest.py moves that value to OPENIMCC_TEST_LEGACY_VAPOROCK_ROOT so it
# cannot redirect the packaged-table tests.
_LEGACY_ROOT_ENV = os.environ.get("OPENIMCC_TEST_LEGACY_VAPOROCK_ROOT")
VAPOROCK_ROOT = Path(_LEGACY_ROOT_ENV or "/nonexistent-vaporock")
VAPOROCK_GAS = VAPOROCK_ROOT / "src" / "vaporock" / "data" / "JANAF-vapor-data-full.csv"
VAPOROCK_OXIDE = VAPOROCK_ROOT / "data" / "condensate-thermo-data.csv"


@pytest.mark.skipif(
    not _LEGACY_ROOT_ENV,
    reason="legacy VapoRock checkout is not selected",
)
def test_legacy_k_row_differs_from_janaf_source_by_about_663_j_per_mol() -> None:
    legacy_root = VAPOROCK_ROOT
    legacy_gas = legacy_root / "src" / "vaporock" / "data" / "JANAF-vapor-data-full.csv"
    legacy_oxide = legacy_root / "data" / "condensate-thermo-data.csv"
    if not (legacy_gas.is_file() and legacy_oxide.is_file()):
        pytest.skip("selected VapoRock checkout is incomplete")

    legacy_pack = load_gas_datapack(gas_path=legacy_gas, oxide_path=legacy_oxide)
    source_row = next(
        row for row in _complete_rows("K-005") if row["temperature"] == 2000.0
    )
    janaf_source_g = _source_g_app_from_row("K-005", source_row)
    legacy_g = _janaf_gibbs(
        2000.0, _nearest_interval_row(legacy_pack.gas_df, "K(g)", 2000.0)
    )
    # The legacy row's +662.8 J/mol offset explains its 0.315609 log10(p_K)
    # and the obsolete 0.316 literal; the JANAF source-column reference rounds
    # to 0.333 and is the one used by the independent regression above.
    assert legacy_g - janaf_source_g == pytest.approx(662.8, abs=1.0)


@pytest.mark.skipif(
    not (VAPOROCK_GAS.is_file() and VAPOROCK_OXIDE.is_file()),
    reason="controller-supplied VapoRock checkout is unavailable for G3",
)
def test_g3_new_tables_are_finite_against_current_vaporock() -> None:
    new_pack = load_gas_datapack()
    old_pack = load_gas_datapack(gas_path=VAPOROCK_GAS, oxide_path=VAPOROCK_OXIDE)
    compositions = [
        {oxide: 1.0 for oxide in ("SiO2", "MgO", "FeO", "CaO", "Al2O3", "Na2O", "K2O")},
        {
            "SiO2": 0.71,
            "MgO": 0.01,
            "FeO": 0.02,
            "CaO": 0.10,
            "Al2O3": 0.08,
            "Na2O": 0.06,
            "K2O": 0.02,
        },
        {
            "SiO2": 0.45,
            "MgO": 0.15,
            "FeO": 0.08,
            "CaO": 0.12,
            "Al2O3": 0.12,
            "Na2O": 0.06,
            "K2O": 0.02,
        },
    ]
    max_by_species = {species: 0.0 for species in IMCC_GAS_CHANNEL_SPECIES}
    for temperature in IMCC_SF04_WORKBOOK_GRID_K:
        for activities in compositions:
            for fugacity in (1.0, 1.0e-4, 1.0e-10):
                for species in IMCC_GAS_CHANNEL_SPECIES:
                    new = load_and_evaluate(
                        new_pack, activities, temperature, fugacity, species
                    )
                    old = load_and_evaluate(
                        old_pack, activities, temperature, fugacity, species
                    )
                    difference = math.log10(new / old)
                    assert math.isfinite(difference)
                    max_by_species[species] = max(
                        max_by_species[species], abs(difference)
                    )
    # CaO is the only >0.05 dex case: the old VapoRock CaO(g) fit starts at
    # 4500 K and this comparison intentionally evaluates its legacy extrapolation
    # down to 1500 K, while the new fit is sourced over the IMCC domain.
    assert {species for species, value in max_by_species.items() if value > 0.05} == {
        "CaO"
    }


def load_and_evaluate(pack, activities, temperature, fugacity, species):
    from openimcc.gas import evaluate_gas

    return evaluate_gas(
        activities,
        temperature,
        fugacity,
        pack,
        allow_extrapolation=True,
        gas_species=(species,),
    )[species]
