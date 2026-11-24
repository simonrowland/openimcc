from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from openimcc.gas import load_gas_datapack, species_thermo


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "src/openimcc/data/packs/imcc-sf04-d066-p2o5-candidate-v1.json"
SOURCES = ROOT / "data-src/p2o5-candidate/thermochemistry.json"
JUNG_HUDON = ROOT / "data-src/published-thermo/jung-hudon-p2o5.json"
R_J_MOL_K = 8.31446261815324
LN10 = math.log(10.0)
CALCIUM_BOILING_K = 1757.0
MAGNESIUM_BOILING_K = 1363.0


def _janaf_value(table_id: str, temperature: float, quantity: str) -> float:
    table = yaml.safe_load(
        (ROOT / "data-src/janaf" / f"{table_id}.yaml").read_text(encoding="utf-8")
    )["table"]
    points = sorted(
        (
            float(row["temperature"]["value"]),
            float(row[quantity]["value"]),
        )
        for row in table["values"]
        if row[quantity]["value"] is not None
    )
    for t, value in points:
        if t == temperature:
            return value
    for (low_t, low_v), (high_t, high_v) in zip(points, points[1:]):
        if low_t < temperature < high_t:
            fraction = (temperature - low_t) / (high_t - low_t)
            return low_v + fraction * (high_v - low_v)
    raise ValueError(f"JANAF {table_id} has no {quantity} value at {temperature:g} K")


def _mg_source_value(source: dict, column: int, temperature: float) -> float:
    rows = source["janaf_pages"]["Mg-039"]["values"]
    points = sorted(
        (float(row[0]), float(row[column]))
        for row in rows
        if row[column] is not None
    )
    for t, value in points:
        if t == temperature:
            return value
    for (low_t, low_v), (high_t, high_v) in zip(points, points[1:]):
        if low_t < temperature < high_t:
            fraction = (temperature - low_t) / (high_t - low_t)
            return low_v + fraction * (high_v - low_v)
    raise ValueError(f"Mg-039 has no column {column} value at {temperature:g} K")


def _element(species: str, temperature: float, gas_pack):
    if species == "Ca":
        phase = "l" if temperature < CALCIUM_BOILING_K else "g"
    elif species == "Mg":
        phase = "l" if temperature < MAGNESIUM_BOILING_K else "g"
    else:
        phase = "g"
    return species_thermo(species, phase, temperature, gas_pack)


def _absolute_p2o5(temperature: float, gas_pack) -> tuple[float, float, float]:
    jung = json.loads(JUNG_HUDON.read_text(encoding="utf-8"))
    equation = jung["delta_g_standard_j_mol"]
    p2 = _element("P2", temperature, gas_pack)
    oxygen = _element("O2", temperature, gas_pack)
    # Jung–Hudon is inferred at 1 atm. For Δνgas=-3.5, converting to 1 bar
    # adds -3.5*R*T*ln(1/1.01325), in J/mol; its T slope contributes to ΔS.
    pressure_slope = 3.5 * R_J_MOL_K * math.log(1.01325)
    reaction_h = float(equation["intercept"])
    reaction_s = -(float(equation["slope"]) + pressure_slope)
    # Add absolute gas H/S to the product-side reaction H/S, then use G=H-TS.
    # H and G are J/mol; S is J/(mol K).
    enthalpy = (
        p2.H_app_kJ_mol * 1000.0
        + 2.5 * oxygen.H_app_kJ_mol * 1000.0
        + reaction_h
    )
    entropy = p2.S_J_molK + 2.5 * oxygen.S_J_molK + reaction_s
    gibbs = enthalpy - temperature * entropy
    return enthalpy, entropy, gibbs


def _mg_reaction_routes(
    temperature: float, source: dict, gas_pack
) -> tuple[float, float, float, float]:
    mg = _element("Mg", temperature, gas_pack)
    p2 = _element("P2", temperature, gas_pack)
    oxygen = _element("O2", temperature, gas_pack)
    product_hf = _mg_source_value(source, 5, temperature) * 1000.0
    product_gf = _mg_source_value(source, 6, temperature) * 1000.0
    product_s = _mg_source_value(source, 2, temperature)
    mgo_hf = _janaf_value("Mg-009", temperature, "formation_enthalpy") * 1000.0
    mgo_gf = _janaf_value("Mg-009", temperature, "formation_gibbs_energy") * 1000.0
    mgo_s = _janaf_value("Mg-009", temperature, "entropy")
    p2o5_h, p2o5_s, p2o5_g = _absolute_p2o5(temperature, gas_pack)

    # The elemental H and S terms make each species absolute; their coefficients
    # cancel in the balanced oxide reaction at every temperature.
    product_h = product_hf + 3.0 * mg.H_app_kJ_mol * 1000.0 + p2.H_app_kJ_mol * 1000.0 + 4.0 * oxygen.H_app_kJ_mol * 1000.0
    mgo_h = mgo_hf + mg.H_app_kJ_mol * 1000.0 + 0.5 * oxygen.H_app_kJ_mol * 1000.0
    delta_h_abs = product_h - 3.0 * mgo_h - p2o5_h
    delta_s_abs = product_s - 3.0 * mgo_s - p2o5_s

    product_s_f = product_s - 3.0 * mg.S_J_molK - p2.S_J_molK - 4.0 * oxygen.S_J_molK
    mgo_s_f = mgo_s - mg.S_J_molK - 0.5 * oxygen.S_J_molK
    p2o5_s_f = p2o5_s - p2.S_J_molK - 2.5 * oxygen.S_J_molK
    delta_s_formation = product_s_f - 3.0 * mgo_s_f - p2o5_s_f

    # Put the Jung–Hudon absolute liquid on JANAF's elemental reference before
    # combining it with the JANAF formation Gibbs rows for phosphate and MgO.
    p2o5_gf = p2o5_g - p2.G_J_mol - 2.5 * oxygen.G_J_mol
    delta_g_formation = product_gf - 3.0 * mgo_gf - p2o5_gf
    log_k_formation = -delta_g_formation / (R_J_MOL_K * temperature * LN10)
    log_k_absolute = -(delta_h_abs - temperature * delta_s_abs) / (
        R_J_MOL_K * temperature * LN10
    )
    return log_k_formation, log_k_absolute, delta_s_formation, delta_s_abs


def _ca_reaction_routes(
    temperature: float, source: dict, gas_pack
) -> tuple[float, float, float, float]:
    melting = float(source["ca3_po4_2_melting"]["temperature_K"])
    fusion_s = 13.0 * float(source["class_c_fusion_entropy_J_mol_K_atom"])
    row_1600 = source["whitlockite"]["values"][-1]
    anchor_t = float(row_1600[0])
    anchor_s = float(row_1600[2])

    # R&H gives formation G and absolute S at 1600 K. Add the absolute JANAF
    # elemental G values, then derive H=G+T*S before continuing with absolute Cp.
    ca0 = _element("Ca", anchor_t, gas_pack)
    p20 = _element("P2", anchor_t, gas_pack)
    o20 = _element("O2", anchor_t, gas_pack)
    crystal_g0 = (
        float(row_1600[6]) * 1000.0
        + 3.0 * ca0.G_J_mol
        + p20.G_J_mol
        + 4.0 * o20.G_J_mol
    )
    crystal_h0 = crystal_g0 + anchor_t * anchor_s
    cp = float(row_1600[1])

    def crystal_h_s(t: float) -> tuple[float, float]:
        h = crystal_h0 + cp * (t - anchor_t)
        s = anchor_s + cp * math.log(t / anchor_t)
        return h, s

    crystal_h, crystal_s = crystal_h_s(melting)
    cao_m = species_thermo("CaO", "l", melting, gas_pack)
    p2o5_m = species_thermo("P2O5", "l", melting, gas_pack)
    delta_h_crystal = (
        crystal_h - 3.0 * cao_m.H_app_kJ_mol * 1000.0 - p2o5_m.H_app_kJ_mol * 1000.0
    )
    delta_s_crystal = crystal_s - 3.0 * cao_m.S_J_molK - p2o5_m.S_J_molK
    # At fusion, Hliq=Hcr+Tm*ΔSfus and Sliq=Scr+ΔSfus. Entropy is J/(mol K);
    # multiplying it by Tm yields the J/mol enthalpy increment.
    delta_h_liquid = delta_h_crystal + melting * fusion_s
    delta_s_liquid = delta_s_crystal + fusion_s
    delta_g_absolute = delta_h_liquid - temperature * delta_s_liquid
    log_k_absolute = -delta_g_absolute / (R_J_MOL_K * temperature * LN10)

    cao = species_thermo("CaO", "l", temperature, gas_pack)
    p2o5 = species_thermo("P2O5", "l", temperature, gas_pack)
    ca = _element("Ca", temperature, gas_pack)
    p2 = _element("P2", temperature, gas_pack)
    oxygen = _element("O2", temperature, gas_pack)
    product_g = delta_g_absolute + 3.0 * cao.G_J_mol + p2o5.G_J_mol
    product_s = delta_s_liquid + 3.0 * cao.S_J_molK + p2o5.S_J_molK

    # Convert every species to formation properties using the same temperature-
    # dependent element references; the Ca liquid-to-gas transition is shared.
    product_g_f = product_g - 3.0 * ca.G_J_mol - p2.G_J_mol - 4.0 * oxygen.G_J_mol
    cao_g_f = cao.G_J_mol - ca.G_J_mol - 0.5 * oxygen.G_J_mol
    p2o5_g_f = p2o5.G_J_mol - p2.G_J_mol - 2.5 * oxygen.G_J_mol
    delta_g_formation = product_g_f - 3.0 * cao_g_f - p2o5_g_f

    product_s_f = product_s - 3.0 * ca.S_J_molK - p2.S_J_molK - 4.0 * oxygen.S_J_molK
    cao_s_f = cao.S_J_molK - ca.S_J_molK - 0.5 * oxygen.S_J_molK
    p2o5_s_f = p2o5.S_J_molK - p2.S_J_molK - 2.5 * oxygen.S_J_molK
    delta_s_formation = product_s_f - 3.0 * cao_s_f - p2o5_s_f
    log_k_formation = -delta_g_formation / (R_J_MOL_K * temperature * LN10)
    return log_k_formation, log_k_absolute, delta_s_formation, delta_s_liquid


@pytest.mark.parametrize("row_index", (38, 39))
def test_associates_match_absolute_and_temperature_consistent_formation_routes(
    row_index: int,
) -> None:
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    source = json.loads(SOURCES.read_text(encoding="utf-8"))
    gas_pack = load_gas_datapack()
    row = pack["rows"][row_index]
    tolerance_dex = float(row["standard_state_refit"]["max_abs_residual_dex"]) + 1.0e-6
    # An entropy-route logK difference is |ΔS|/(R ln 10), so convert the dex
    # residual to the corresponding J/(mol K) tolerance.
    entropy_tolerance = R_J_MOL_K * LN10 * tolerance_dex

    for temperature in (1700.0, 2000.0, 2300.0):
        if row_index == 38:
            log_form, log_absolute, delta_s_form, delta_s_absolute = _mg_reaction_routes(
                temperature, source, gas_pack
            )
        else:
            log_form, log_absolute, delta_s_form, delta_s_absolute = _ca_reaction_routes(
                temperature, source, gas_pack
            )
        log_pack = row["A"] + row["B"] / temperature

        assert log_form > 0.0
        assert log_absolute > 0.0
        assert abs(log_form - log_absolute) <= tolerance_dex
        assert abs(log_pack - log_form) <= tolerance_dex
        assert abs(delta_s_form - delta_s_absolute) <= entropy_tolerance


def test_ca_absolute_reference_fit_has_the_correct_temperature_slope() -> None:
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    row = pack["rows"][39]
    logs = [row["A"] + row["B"] / t for t in (1700.0, 2000.0, 2300.0)]

    assert row["A"] == pytest.approx(2.90, abs=0.02)
    assert row["B"] == pytest.approx(33651.0, abs=50.0)
    assert logs == pytest.approx([22.69, 19.72, 17.53], abs=0.03)
    assert logs[0] > logs[1] > logs[2]


def test_magnesium_hess_check_and_pack_enthalpy_path() -> None:
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    source = json.loads(SOURCES.read_text(encoding="utf-8"))
    mg_rows = source["janaf_pages"]["Mg-038"]["values"]
    mg_hf = next(row[5] for row in mg_rows if row[0] == 298.15)
    mgo_hf = _janaf_value("Mg-008", 298.15, "formation_enthalpy")
    p2o5 = source["janaf_reference_checks"]["p2o5_crystal_298"]
    p2o5_hf = (
        p2o5["formation_enthalpy_kJ_mol"]
        / p2o5["p2o5_formula_units_per_table_formula"]
    )
    delta_h_crystal = mg_hf - 3.0 * mgo_hf - p2o5_hf

    # The JANAF pages give no numeric uncertainties; lock printed precision and
    # the rounded approximately -435 kJ/mol independent Hess value.
    assert delta_h_crystal == pytest.approx(-436.407, abs=0.01)
    assert abs(delta_h_crystal - (-435.0)) < 2.0

    mg_row = pack["rows"][38]
    delta_h_from_pack = -mg_row["B"] * R_J_MOL_K * LN10 / 1000.0
    assert delta_h_from_pack == pytest.approx(-414.63, abs=0.02)
    assert abs(delta_h_from_pack - delta_h_crystal) < 50.0
