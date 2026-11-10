import json
import math
from pathlib import Path

import pytest

from openimcc import evaluate, load_datapack
from openimcc._constants import R_J_MOL_K
from openimcc.gas import evaluate_gas, load_gas_datapack
from openimcc.ni import evaluate_ni_activity
from openimcc.redox_pack import load_redox_pack, species_gibbs_j_mol


FRAGMENT = (
    Path(__file__).resolve().parents[2]
    / "src/openimcc/data/fragments/ni/rows.json"
)


def test_fragment_row_matches_printed_holzheid_anchors() -> None:
    fragment = json.loads(FRAGMENT.read_text(encoding="utf-8"))
    row = fragment["rows"][0]

    # Holzheid et al. (1997), abstract, pp. 21-38.
    printed_gamma = 2.70
    printed_spread = 0.52
    printed_count = 77
    assert row["gamma_law"]["gamma_point"] == printed_gamma
    assert row["gamma_law"]["gamma_endpoints"] == [2.18, 3.22]
    assert row["reported_gamma_spread"]["value"] == printed_spread
    assert row["experiment_count"] == printed_count
    assert row["temperature_window_K"] == [1300 + 273.15, 1600 + 273.15]
    assert row["log10_fO2_source"]["range"] == [-3.0, 1.5]
    assert row["composition_windows_wt_pct"]["FeO"]["range"] == [0.0, 12.0]
    assert row["composition_windows_wt_pct"]["MgO"]["range"] == [4.0, 30.0]

    # Gurvich et al., TSIV Volume IV, Table 2026.
    source_ni_o = fragment["standards"][0]["source_anchors"]
    assert source_ni_o["fusion_temperature_K"] == 2228
    assert source_ni_o["fusion_enthalpy_kJ_mol"] == 42.0
    assert source_ni_o["liquid_heat_capacity_j_mol_K"] == 67.0

    # Direct simple-oxide mole-fraction calculation, independent of the helper.
    printed_x_nio = 0.001
    assert printed_gamma * printed_x_nio == pytest.approx(0.00270)


def test_helper_uses_the_fragment_and_flags_conditions_outside_its_domain() -> None:
    result = evaluate_ni_activity(
        {"SiO2": 50.0, "MgO": 8.0, "FeO": 6.0, "CaO": 10.0, "NiO": 0.01},
        1500.0,
        2.0,
    )
    expected_x = 0.01 / (50.0 + 8.0 + 6.0 + 10.0 + 0.01)
    assert result.gamma == 2.70
    assert result.x_nio == pytest.approx(expected_x)
    assert result.activity == pytest.approx(2.70 * expected_x)
    assert result.notice.code == "imcc_ni_activity_domain_notice"
    assert "temperature_outside_1573_1873_K" in result.notice.violated_axes
    assert "oxygen_fugacity_outside_IW_minus_3_to_plus_1_5" in result.notice.violated_axes


def test_fragment_adds_finite_ni_gas_without_changing_other_gas_outputs() -> None:
    temperature = 1800.0
    composition = {
        "SiO2": 45.0,
        "MgO": 10.0,
        "FeO": 6.0,
        "CaO": 9.0,
        "Al2O3": 10.0,
        "TiO2": 0.8,
        "Na2O": 1.5,
        "K2O": 0.05,
        "NiO": 0.02,
    }
    melt_composition = {key: value for key, value in composition.items() if key != "NiO"}
    melt = evaluate(melt_composition, temperature, load_datapack())
    activities = dict(zip(melt.parent_oxides, melt.parent_activity))
    gas_pack = load_gas_datapack()

    # Set the gas oxygen fugacity to IW-2 from the same default redox standards
    # that define the fragment's source window.
    redox_pack = load_redox_pack()
    delta_g_iw = (
        species_gibbs_j_mol(redox_pack, "FeO(l)", temperature)
        - species_gibbs_j_mol(redox_pack, "Fe(cr)", temperature)
        - 0.5 * species_gibbs_j_mol(redox_pack, "O2(g)", temperature)
    )
    log10_fO2_iw = 2.0 * delta_g_iw / (R_J_MOL_K * temperature * math.log(10.0))
    fO2 = 10.0 ** (log10_fO2_iw - 2.0)
    ni = evaluate_ni_activity(composition, temperature, -2.0)
    assert ni.notice.violated_axes == ()

    baseline = dict(evaluate_gas(activities, temperature, fO2, gas_pack))
    activities["NiO"] = ni.activity
    augmented = dict(evaluate_gas(activities, temperature, fO2, gas_pack))

    ni_pressures = {species: value for species, value in augmented.items() if species.startswith("Ni")}
    assert ni_pressures
    assert all(math.isfinite(value) and value > 0.0 for value in ni_pressures.values())
    assert {
        species: value for species, value in augmented.items() if not species.startswith("Ni")
    } == baseline
