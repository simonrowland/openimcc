from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from openimcc._constants import MOLAR_MASS_G_MOL
from openimcc.gas import IMCC_PARENT_OXIDES, evaluate_gas, load_gas_datapack
from openimcc.redox import evaluate_redox
from openimcc.redox_pack import load_redox_pack, species_gibbs_j_mol


ROOT = Path(__file__).resolve().parents[2]
FRAGMENT_PATH = ROOT / "src/openimcc/data/fragments/yb/rows.json"
R_J_MOL_K = 8.31446261815324
T_REFERENCE_K = 298.15


def _fragment() -> dict:
    return json.loads(FRAGMENT_PATH.read_text(encoding="utf-8"))


def _manual_condensate_gibbs_kj(standard: dict, temperature_k: float) -> float:
    """Integrate the printed crystal anchor, fusion entropy, and liquid Cp."""
    fusion_k = float(standard["fusion_temperature_K"])
    cp_solid = float(standard["solid_Cp_J_mol_K"])
    cp_liquid = float(standard["liquid_Cp_J_mol_K"])
    entropy_fusion = float(standard["fusion_entropy_J_mol_K"])
    entropy_crystal_298 = float(standard["selected_S298_J_mol_K"])
    enthalpy_crystal_298 = float(standard["selected_dfH298_kJ_mol"])
    h_increment = (
        cp_solid * (fusion_k - T_REFERENCE_K) / 1000.0
        + fusion_k * entropy_fusion / 1000.0
        + cp_liquid * (temperature_k - fusion_k) / 1000.0
    )
    entropy = (
        entropy_crystal_298
        + cp_solid * math.log(fusion_k / T_REFERENCE_K)
        + entropy_fusion
        + cp_liquid * math.log(temperature_k / fusion_k)
    )
    return enthalpy_crystal_298 + h_increment - temperature_k * entropy / 1000.0


def _manual_yb2o3_gibbs_j(temperature_k: float) -> float:
    """Integrate the printed C/H/liquid phase chain without a row builder."""
    t0 = T_REFERENCE_K
    transition_k = 2687.0
    fusion_k = 2707.0
    a, b, e = 130.6438, 3.34628e-3, -1_448_200.0
    h_c = a * (transition_k - t0)
    h_c += 0.5 * b * (transition_k**2 - t0**2)
    h_c -= e * (1.0 / transition_k - 1.0 / t0)
    s_c = a * math.log(transition_k / t0)
    s_c += b * (transition_k - t0)
    s_c += 0.5 * e * (1.0 / t0**2 - 1.0 / transition_k**2)
    h_increment = (
        h_c
        + 27_000.0
        + 134.0 * (fusion_k - transition_k)
        + 84_000.0
        + 146.0 * (temperature_k - fusion_k)
    )
    entropy = (
        133.1
        + s_c
        + 27_000.0 / transition_k
        + 134.0 * math.log(fusion_k / transition_k)
        + 84_000.0 / fusion_k
        + 146.0 * math.log(temperature_k / fusion_k)
    )
    return -1_814_500.0 + h_increment - temperature_k * entropy


def test_fragment_rows_reproduce_printed_thermochemical_anchors():
    fragment = _fragment()
    standards = fragment["standards"]

    # Konings et al., Table 88: the 2687 K transition and 2707 K fusion
    # increments sum to 111 kJ/mol from the C crystal to the liquid. Fyhrie
    # et al., Table 2, independently report 102 +/- 10 kJ/mol by DnC.
    oxide = standards["Yb2O3(l)"]
    assert oxide["solid_transition_enthalpy_kJ_mol"] == 27.0
    assert oxide["H_to_liquid_fusion_enthalpy_kJ_mol"] == 84.0
    assert 27.0 + 84.0 == oxide["C_to_liquid_total_enthalpy_kJ_mol"] == 111.0
    assert abs(111.0 - 102.0) < math.sqrt(5.0**2 + 8.0**2 + 10.0**2)
    assert oxide["fusion_temperature_K"] == 2707.0
    assert oxide["liquid_Cp_J_mol_K"] == 146.0
    assert oxide["selected_dfH298_kJ_mol"] == -1814.5
    assert oxide["selected_S298_J_mol_K"] == 133.1
    assert oxide["anchor_phase"].startswith("C-type crystal")
    assert oxide["derivation"]

    # Liang et al. measured both elemental and oxide formation anchors for
    # Yb2SiO5. The current JANAF high-cristobalite and Konings crystal
    # references give the same crystalline reaction cycle used by the row.
    mono = standards["Yb2SiO5(l)"]
    mono_from_oxides = -1_814.5 - 905.489 - 49.45
    assert mono["source_dfH298_from_oxides_kJ_mol"] == -49.45
    assert mono["source_dfH298_from_elements_kJ_mol"] == -2774.75
    assert mono["selected_dfH298_kJ_mol"] == pytest.approx(mono_from_oxides)
    assert abs(mono_from_oxides - (-2774.75)) < math.hypot(8.35, 8.21)
    assert mono["fusion_temperature_K"] == 1950.0 + 273.15
    expected_mono_fusion_s = (151_000.0 / 2123.15) * (8.0 / 11.0)
    expected_mono_fusion_h = (
        expected_mono_fusion_s * mono["fusion_temperature_K"] / 1000.0
    )
    assert mono["fusion_entropy_J_mol_K"] == pytest.approx(expected_mono_fusion_s)
    assert mono["fusion_enthalpy_kJ_mol"] == pytest.approx(expected_mono_fusion_h)
    assert mono["solid_Cp_J_mol_K"] == pytest.approx(3.0 * 8.0 * R_J_MOL_K)
    assert mono["liquid_Cp_J_mol_K"] == pytest.approx(146.0 * 8.0 / 5.0)
    assert mono["fusion_enthalpy_status"] == "provisional estimate"

    # Costa et al., Table 3, 42 mol% SiO2: combine the printed solution
    # enthalpies as a solid-oxide Hess cycle. Their Table 4 supplies fusion.
    disilicate = standards["Yb2Si2O7(l)"]
    hess = 2.41 + 2.0 * (-12.53) - 136.22
    hess_spread = math.sqrt(13.44**2 + 4.0 * 6.40**2 + 7.13**2)
    disilicate_crystal_h = -1_814.5 + 2.0 * (-905.489) + hess
    assert disilicate["source_dfH298_from_oxides_kJ_mol"] == pytest.approx(hess)
    assert disilicate["source_dfH298_from_oxides_spread_kJ_mol"] == pytest.approx(
        hess_spread, abs=0.005
    )
    assert disilicate["selected_dfH298_kJ_mol"] == pytest.approx(
        disilicate_crystal_h
    )
    assert disilicate["fusion_enthalpy_kJ_mol"] == 151.0
    assert disilicate["fusion_enthalpy_spread_kJ_mol"] == 8.0
    assert disilicate["fusion_temperature_K"] == 1850.0 + 273.15
    assert disilicate["fusion_entropy_J_mol_K"] == pytest.approx(
        151_000.0 / disilicate["fusion_temperature_K"]
    )
    assert disilicate["solid_Cp_J_mol_K"] == pytest.approx(
        3.0 * 11.0 * R_J_MOL_K
    )
    assert disilicate["liquid_Cp_J_mol_K"] == pytest.approx(146.0 * 11.0 / 5.0)

    # Each condensate polynomial retains the printed 298 K crystal formation
    # enthalpy anchor; evaluating the solid-oxide Hess cycles never calls a
    # builder or calibrates a parameter to activity/battery rows.
    for species, standard in standards.items():
        evaluator = fragment["thermo_species"][species]
        for interval in evaluator["intervals"]:
            assert interval["row"]["dH298_R"] * R_J_MOL_K == pytest.approx(
                standard["selected_dfH298_kJ_mol"], abs=1e-6
            )


def test_fragment_loads_with_default_and_unlocks_finite_yb_gas():
    pack = load_redox_pack()
    fragment = _fragment()
    packaged_reactions = {row["reaction"] for row in pack.rows}
    assert {row["reaction"] for row in fragment["rows"]} <= packaged_reactions

    # The fitted evaluator is checked against a direct integration of its
    # printed crystal-Cp/fusion/liquid-Cp construction, not its builder.
    oxide_g = species_gibbs_j_mol(pack, "Yb2O3(l)", 2300.0)
    assert oxide_g == pytest.approx(_manual_yb2o3_gibbs_j(2300.0), abs=1.0)
    for species in ("Yb2SiO5(l)", "Yb2Si2O7(l)"):
        standard = fragment["standards"][species]
        expected = _manual_condensate_gibbs_kj(standard, 1800.0) * 1000.0
        assert species_gibbs_j_mol(pack, species, 1800.0) == pytest.approx(
            expected, abs=2.0
        )

    # Premise: oxide mass fractions are on an arbitrary 1 g basis. Algebra:
    # n(formula)=mass/M, then accumulate each formula's elemental atom moles.
    # The common basis scale cancels in the melt solve. A trace Yb2O3 term
    # enables the opt-in parent activity while retaining a regolith-like feed.
    oxide_mass = {
        "SiO2": 0.45,
        "MgO": 0.20,
        "FeO": 0.10,
        "CaO": 0.10,
        "Al2O3": 0.10,
        "TiO2": 0.01,
        "Na2O": 0.02,
        "K2O": 0.01,
        "Yb2O3": 0.0001,
    }
    formulas = {
        "SiO2": {"Si": 1, "O": 2},
        "MgO": {"Mg": 1, "O": 1},
        "FeO": {"Fe": 1, "O": 1},
        "CaO": {"Ca": 1, "O": 1},
        "Al2O3": {"Al": 2, "O": 3},
        "TiO2": {"Ti": 1, "O": 2},
        "Na2O": {"Na": 2, "O": 1},
        "K2O": {"K": 2, "O": 1},
        "Yb2O3": {"Yb": 2, "O": 3},
    }
    inventory: dict[str, float] = {}
    for oxide, mass_g in oxide_mass.items():
        molar_mass = sum(
            amount * MOLAR_MASS_G_MOL[element]
            for element, amount in formulas[oxide].items()
        )
        formula_moles = mass_g / molar_mass
        for element, amount in formulas[oxide].items():
            inventory[element] = inventory.get(element, 0.0) + amount * formula_moles

    temperature_k = 2300.0
    f_o2 = 1e-10
    melt = evaluate_redox(
        inventory,
        temperature_k,
        "imposed",
        math.log(f_o2),
    )
    activities = dict(melt.parent_activity)
    yb2o3_activity = activities.pop("Yb2O3")
    assert math.isfinite(yb2o3_activity) and yb2o3_activity > 0.0

    gas_pack = load_gas_datapack()
    baseline = evaluate_gas(
        activities,
        temperature_k,
        f_o2,
        datapack=gas_pack,
        parent_oxides=IMCC_PARENT_OXIDES,
    )
    with_yb = evaluate_gas(
        {**activities, "Yb2O3": yb2o3_activity},
        temperature_k,
        f_o2,
        datapack=gas_pack,
        parent_oxides=(*IMCC_PARENT_OXIDES, "Yb2O3"),
    )
    assert math.isfinite(with_yb["Yb"]) and with_yb["Yb"] > 0.0
    assert math.isfinite(with_yb["YbO"]) and with_yb["YbO"] > 0.0
    assert len(baseline) == 32
    assert all(with_yb[name].hex() == baseline[name].hex() for name in baseline)
