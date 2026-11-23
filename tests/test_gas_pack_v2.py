from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
import subprocess
import sys

import pytest

from openimcc.gas import (
    ImccGasSpeciesNotFoundError,
    IMCC_PARENT_OXIDES,
    default_gas_database_path,
    evaluate_gas,
    load_gas_datapack,
    species_thermo,
)
from openimcc.profiles import load_profile
from tools import build_gas_tables


ROOT = Path(__file__).resolve().parents[1]
GAS_DIR = ROOT / "src/openimcc/data/gas"
GAS_V1 = GAS_DIR / "gas-shomate-default-v1.csv"
GAS_V2 = GAS_DIR / "gas-shomate-default-v2.csv"
CONDENSATE = GAS_DIR / "condensate-default-v1.csv"


def _gas_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _hex(value: float | None) -> str | None:
    return None if value is None else float(value).hex()


def test_gas_pack_v2_is_an_append_only_version() -> None:
    old_bytes = GAS_V1.read_bytes()
    new_bytes = GAS_V2.read_bytes()
    assert new_bytes.startswith(old_bytes)

    old_rows = _gas_rows(GAS_V1)
    new_rows = _gas_rows(GAS_V2)
    appended = new_rows[len(old_rows):]
    assert len(appended) == 515
    assert len({row["species_name"] for row in appended}) == 173
    assert not ({row["species_name"] for row in old_rows}
                & {row["species_name"] for row in appended})

    intervals = {}
    for row in appended:
        intervals.setdefault(row["species_name"], set()).add(row["T_interval"])
    assert {name for name, present in intervals.items() if "1" not in present} == {
        "H2O2(g)", "S2Cl2(g)",
    }
    assert all(
        row["T_interval"] in {"1", "2", "3"}
        and float(row["T_min"]) < float(row["T_max"])
        for row in appended
    )


def test_gas_pack_v2_is_the_default_and_v1_remains_selectable() -> None:
    assert default_gas_database_path().name == GAS_V2.name
    profile = load_profile("openimcc")
    assert profile.gas_datapack.gas_path.name == GAS_V2.name
    previous = load_gas_datapack(gas_path=GAS_V1, oxide_path=CONDENSATE)
    assert previous.gas_path.name == GAS_V1.name


def test_gas_pack_v2_sources_match_the_vendored_hashes() -> None:
    sidecar = ROOT / "data-src/janaf/gas-pack-additions.sha256"
    records = [line.split() for line in sidecar.read_text().splitlines()]
    assert len(records) == 171
    for expected, relative_path in records:
        source = ROOT / relative_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected


def test_gas_pack_v2_builder_check_mode_is_current() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_gas_tables.py",
            "--gas-pack-v2",
            "--check",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "515 rows across 173 formulas; 0 fit exclusions" in result.stdout


def test_packaged_generator_check_modes_are_current() -> None:
    commands = (
        ("tools/packmanifest.py", "--check"),
        ("tools/build_coefficient_ancestry.py", "--check"),
        ("tools/rebuild_barin_aluminate_crystals.py", "--check"),
        ("tools/rebuild_rh_crystal_phases.py", "--check", "--species-only"),
        ("tools/generate_declared_assemblage_janaf.py", "--check"),
        ("tools/generate_elemental_reference_crystal_cards.py", "--check"),
    )
    for command in commands:
        result = subprocess.run(
            [sys.executable, *command],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{' '.join(command)} failed:\n{result.stdout}{result.stderr}"
        )


def test_gas_pack_v2_preserves_served_gas_values() -> None:
    base = load_gas_datapack(gas_path=GAS_V1, oxide_path=CONDENSATE)
    candidate = load_gas_datapack(gas_path=GAS_V2, oxide_path=CONDENSATE)
    numeric_fields = (
        "Cp_J_molK",
        "S_J_molK",
        "H_app_kJ_mol",
        "G_J_mol",
        "T_min",
        "T_max",
        "formation_enthalpy_298_kJ_mol",
    )

    for row in _gas_rows(GAS_V1):
        formula = row["species_name"][:-3]
        low, high = float(row["T_min"]), float(row["T_max"])
        for temperature in (low, (low + high) / 2.0, high):
            old = species_thermo(formula, "g", temperature, base)
            new = species_thermo(formula, "g", temperature, candidate)
            assert tuple(_hex(getattr(old, field)) for field in numeric_fields) == (
                tuple(_hex(getattr(new, field)) for field in numeric_fields)
            )
            assert old.source_row_id == new.source_row_id
            assert old.source_table_id == new.source_table_id
            assert old.T_interval == new.T_interval
            assert old.authority == new.authority

    for temperature in (1200.0, 1500.0, 1800.0, 3000.0):
        old = species_thermo("HO", "g", temperature, base)
        new = species_thermo("HO", "g", temperature, candidate)
        assert old == new

    new_rows = _gas_rows(GAS_V2)[len(_gas_rows(GAS_V1)):]
    for row in new_rows:
        formula = row["species_name"][:-3]
        temperature = (float(row["T_min"]) + float(row["T_max"])) / 2.0
        result = species_thermo(formula, "g", temperature, candidate)
        assert all(
            math.isfinite(getattr(result, field))
            for field in ("Cp_J_molK", "S_J_molK", "H_app_kJ_mol", "G_J_mol")
        )


def test_gas_pack_v2_does_not_move_default_gas_channels() -> None:
    base = load_gas_datapack(gas_path=GAS_V1, oxide_path=CONDENSATE)
    candidate = load_gas_datapack(gas_path=GAS_V2, oxide_path=CONDENSATE)
    activities = {
        "SiO2": 0.55,
        "MgO": 0.05,
        "FeO": 0.10,
        "Al2O3": 0.16,
    }
    assert set(activities) <= set(IMCC_PARENT_OXIDES)

    for temperature in (1000.0, 1200.0, 1500.0, 1800.0, 2200.0, 3000.0, 4000.0, 5000.0):
        old = evaluate_gas(activities, temperature, 1.0e-8, base)
        new = evaluate_gas(activities, temperature, 1.0e-8, candidate)
        assert set(old) == set(new)
        assert {
            species: float(value).hex() for species, value in old.items()
        } == {
            species: float(value).hex() for species, value in new.items()
        }
        assert dict(old.domain_flags) == dict(new.domain_flags)
        assert dict(old.provenance_class) == dict(new.provenance_class)
        assert dict(old.omitted_channels) == dict(new.omitted_channels)


def test_published_janaf_rows_are_not_treated_as_high_temperature_candidates() -> None:
    candidate = load_gas_datapack(gas_path=GAS_V2, oxide_path=CONDENSATE)
    new_species = {
        row["species_name"]
        for row in _gas_rows(GAS_V2)[len(_gas_rows(GAS_V1)):]
    }
    noticed_species = {
        species
        for _phase, species, _temperature in candidate._high_temperature_candidate_notices
    }
    assert new_species.isdisjoint(noticed_species)


def test_gas_formula_lookup_uses_only_the_explicit_spelling_alias() -> None:
    candidate = load_gas_datapack(gas_path=GAS_V2, oxide_path=CONDENSATE)
    for formula in (
        "HNC", "CNH", "HOCN", "HCNO", "OD", "SH", "DH", "C4NiO4",
    ):
        with pytest.raises(ImccGasSpeciesNotFoundError):
            species_thermo(formula, "g", 1800.0, candidate)

    for formula in ("HCN", "HNCO", "HS", "DO", "HD", "Ni(CO)4"):
        rows = candidate.gas_df.loc[f"{formula}(g)"]
        row = rows.iloc[0] if getattr(rows, "ndim", 1) == 2 else rows
        temperature = (float(row["T_min"]) + float(row["T_max"])) / 2.0
        result = species_thermo(formula, "g", temperature, candidate)
        assert result.source_row_id == str(row["Ref"])


def test_gas_pack_v2_closes_both_sis_same_reference_seams() -> None:
    rows = [row for row in _gas_rows(GAS_V2) if row["species_name"] == "SiS(g)"]
    assert len(rows) == 3
    rows.sort(key=lambda row: float(row["T_min"]))
    for left, right in zip(rows, rows[1:]):
        assert float(left["T_max"]) == float(right["T_min"])
        assert not build_gas_tables._shomate_join_open(
            left, right, float(left["T_max"])
        )
