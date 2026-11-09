from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from openimcc.gas import evaluate_gas, load_gas_datapack
from openimcc.gas_data import IMCC_PARENT_OXIDES
from openimcc.redox import evaluate_redox
from openimcc.redox_pack import load_redox_pack


ROOT = Path(__file__).resolve().parents[2]
FRAGMENT_PATH = ROOT / "src/openimcc/data/fragments/zn/rows.json"


def _fragment() -> dict:
    return json.loads(FRAGMENT_PATH.read_text(encoding="utf-8"))


def _gibbs_from_condensate_row(row: dict, temperature_k: float) -> float:
    tau = temperature_k / 1000.0
    polynomial = sum(
        float(row[f"dG_{letter}"]) * tau**power
        for power, letter in enumerate("ABCDE")
    )
    return 8.31446261815324 * 1000.0 * float(row["dH298_R"]) - (
        8.31446261815324 * temperature_k * polynomial
    )


def _row_for_temperature(record: dict, temperature_k: float) -> dict:
    for interval in record.get("intervals", []):
        low, high = interval["T_domain_K"]
        if low <= temperature_k <= high:
            return interval["row"]
    return record.get("row", {})


def test_liquid_zinc_row_matches_printed_janaf_nodes() -> None:
    fragment = _fragment()
    record = fragment["thermo_species"]["Zn(l)"]
    anchors = {
        1200.0: (27.546, 92.402),
        1300.0: (30.684, 94.914),
        1400.0: (33.822, 97.240),
        1500.0: (36.960, 99.405),
        1600.0: (40.098, 101.430),
        1700.0: (43.236, 103.332),
        1800.0: (46.374, 105.126),
        1900.0: (49.512, 106.823),
        2000.0: (52.650, 108.432),
    }
    for temperature, (enthalpy_increment, entropy) in anchors.items():
        source_gibbs = (6.519 + enthalpy_increment) * 1000.0 - temperature * entropy
        fitted_gibbs = _gibbs_from_condensate_row(record["row"], temperature)
        assert fitted_gibbs == pytest.approx(source_gibbs, abs=1.0)


def test_zno_liquid_standard_records_the_printed_fusion_step() -> None:
    fragment = _fragment()
    record = fragment["thermo_species"]["ZnO(l)"]
    construction = record["construction"]

    assert construction["formation_enthalpy_298_kJ_mol"] == -350.5
    assert construction["fusion_temperature_K"] == 2250
    assert construction["crystal_enthalpy_increment_at_fusion_kJ_mol"] == 112.162
    assert construction["liquid_enthalpy_increment_at_fusion_kJ_mol"] == 182.162
    assert (
        construction["liquid_enthalpy_increment_at_fusion_kJ_mol"]
        - construction["crystal_enthalpy_increment_at_fusion_kJ_mol"]
    ) == pytest.approx(70.0)
    assert construction["liquid_heat_capacity_J_mol_K"] == 67
    assert construction["liquid_heat_capacity_status"] == "estimated"

    # The fragment must preserve the existing default-table ZnO coefficients.
    with (ROOT / "src/openimcc/data/gas/condensate.csv").open(newline="") as source:
        shipped = [row for row in csv.DictReader(source) if row["species_name"] == "ZnO(l)"]
    assert len(shipped) == len(record["intervals"]) == 3
    for source_row, interval in zip(shipped, record["intervals"], strict=True):
        for field in ("dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E"):
            assert interval["row"][field] == float(source_row[field])


def test_willemite_row_matches_printed_formation_and_thermal_anchors() -> None:
    fragment = _fragment()
    record = fragment["thermo_species"]["Zn2SiO4(l)"]
    construction = record["construction"]

    # Printed King reaction, TSIV ZnO, and JANAF quartz values, all on a
    # common oxide reference. The source equation uses cal/mol; convert by 4.184.
    formation_enthalpy = (
        2.0 * -350.500 - 910.857 - 6990.0 * 4.184 / 1000.0
    )
    assert formation_enthalpy == pytest.approx(-1641.10316, abs=1e-8)
    assert construction["solid_formation_enthalpy_298_kJ_mol"] == pytest.approx(
        formation_enthalpy
    )
    assert abs(formation_enthalpy - (-1641.5671)) < 2.586

    # Printed Barin Cp equation, with the measured 1500°C transition and
    # 1512°C fusion enthalpy added explicitly. This test reconstructs G from
    # those source values and never invokes the row builder.
    a, b, c, d, e = (
        180.07936,
        0.013589632,
        -2528556.71904,
        -685.68429632,
        156170176.512,
    )
    t0, transition, fusion = 298.15, 1773.15, 1785.15
    transition_h_j_mol = 130.0
    fusion_h_j_mol = 106440.0
    liquid_cp = 215.39

    def crystal_h_increment(temperature: float) -> float:
        return (
            a * (temperature - t0)
            + b / 2.0 * (temperature**2 - t0**2)
            - c * (1.0 / temperature - 1.0 / t0)
            + 2.0 * d * (math.sqrt(temperature) - math.sqrt(t0))
            - e / 2.0 * (1.0 / temperature**2 - 1.0 / t0**2)
        )

    def crystal_entropy_increment(temperature: float) -> float:
        return (
            a * math.log(temperature / t0)
            + b * (temperature - t0)
            - c / 2.0 * (1.0 / temperature**2 - 1.0 / t0**2)
            - 2.0 * d * (1.0 / math.sqrt(temperature) - 1.0 / math.sqrt(t0))
            - e / 3.0 * (1.0 / temperature**3 - 1.0 / t0**3)
        )

    h_cr_fusion = crystal_h_increment(fusion) + transition_h_j_mol
    s_cr_fusion = (
        131.378
        + crystal_entropy_increment(fusion)
        + transition_h_j_mol / transition
    )
    s_liquid_fusion = s_cr_fusion + fusion_h_j_mol / fusion
    assert construction["fusion"]["temperature_K"] == fusion
    assert construction["fusion"]["enthalpy_kJ_mol"] == 106.44
    assert construction["alpha_beta_transition"]["enthalpy_kJ_mol"] == 0.13
    assert construction["liquid_heat_capacity_estimate_J_mol_K"] == liquid_cp

    for temperature in (1200.0, 1400.0, 1700.0, 1785.15, 1900.0, 2000.0):
        liquid_h_increment = (
            h_cr_fusion
            + fusion_h_j_mol
            + liquid_cp * (temperature - fusion)
        )
        liquid_entropy = s_liquid_fusion + liquid_cp * math.log(temperature / fusion)
        source_gibbs = (
            formation_enthalpy * 1000.0
            + liquid_h_increment
            - temperature * liquid_entropy
        )
        fitted_gibbs = _gibbs_from_condensate_row(record["row"], temperature)
        assert fitted_gibbs == pytest.approx(source_gibbs, abs=4.0)


def test_fragment_gives_finite_zn_gas_and_preserves_other_gas_pressures() -> None:
    pack = load_redox_pack()
    fragment = _fragment()
    packaged_reactions = {row["reaction"] for row in pack.rows}
    assert {row["reaction"] for row in fragment["rows"]} <= packaged_reactions
    temperature = 1400.0
    f_o2 = 1e-8
    inventory = {
        "Si": 0.45,
        "Mg": 0.20,
        "Fe": 0.12,
        "Ca": 0.16,
        "Al": 0.32,
        "Na": 0.04,
        "K": 0.015,
        "Ti": 0.01,
        "Zn": 1e-5,
        "O": 2 * 0.45 + 0.20 + 0.12 + 0.16 + 1.5 * 0.32 + 0.5 * 0.04 + 0.5 * 0.015 + 2 * 0.01 + 1e-5,
    }
    melt = evaluate_redox(
        inventory,
        temperature,
        "imposed",
        lambda_imposed=math.log(f_o2),
    )
    assert melt.parent_activity["ZnO"] > 0.0
    assert any(
        "imcc_zn_activity_domain_notice" in flag for flag in melt.flags
    )

    default_parents = tuple(IMCC_PARENT_OXIDES)
    common_activities = {
        parent: melt.parent_activity[parent] for parent in default_parents
    }
    extended_activities = {
        **common_activities,
        "ZnO": melt.parent_activity["ZnO"],
    }
    gas_datapack = load_gas_datapack()
    baseline = evaluate_gas(
        common_activities,
        temperature,
        f_o2,
        datapack=gas_datapack,
        parent_oxides=default_parents,
    )
    extended = evaluate_gas(
        extended_activities,
        temperature,
        f_o2,
        datapack=gas_datapack,
        parent_oxides=(*default_parents, "ZnO"),
    )

    assert math.isfinite(extended["Zn"]) and extended["Zn"] > 0.0
    assert math.isfinite(extended["ZnO"]) and extended["ZnO"] > 0.0
    assert all(extended[name] == baseline[name] for name in baseline)
