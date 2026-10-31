import math
from pathlib import Path

import pytest

from openimcc.gas import R_J_MOL_K, load_gas_datapack, species_thermo


ROOT = Path(__file__).resolve().parents[1]
GAS = ROOT / "src/openimcc/data/gas"
CRYSTAL_CANDIDATE = GAS / "condensate-sio2-janaf-cr-candidate.csv"


def test_janaf_cristobalite_candidate_closes_against_the_existing_liquid() -> None:
    default = load_gas_datapack()
    candidate = load_gas_datapack(
        gas_path=GAS / "gas-shomate.csv", oxide_path=CRYSTAL_CANDIDATE
    )

    assert default.condensate_table_digest != candidate.condensate_table_digest
    assert species_thermo("SiO2", "cr", 1996.0, default).source_row_id == "LAM1987"
    crystal_row = species_thermo("SiO2", "cr", 1996.0, candidate)
    assert crystal_row.source_row_id == "O-035"
    assert crystal_row.source_table_id == "O-035"
    assert (crystal_row.T_min, crystal_row.T_max) == (1000.0, 1996.0)

    expected_default = {
        1823.0: -0.103280037,
        1873.0: -0.095105132,
        1900.0: -0.090791580,
        1996.0: -0.076078659,
    }
    expected_candidate = {
        1823.0: -0.022282118,
        1873.0: -0.015746705,
        1900.0: -0.012245435,
        1996.0: 0.0,
    }
    for temperature in expected_default:
        default_delta_g = (
            species_thermo("SiO2", "l", temperature, default).G_J_mol
            - species_thermo("SiO2", "cr", temperature, default).G_J_mol
        )
        candidate_delta_g = (
            species_thermo("SiO2", "l", temperature, candidate).G_J_mol
            - species_thermo("SiO2", "cr", temperature, candidate).G_J_mol
        )
        default_conversion = -default_delta_g / (
            R_J_MOL_K * temperature * math.log(10.0)
        )
        candidate_conversion = -candidate_delta_g / (
            R_J_MOL_K * temperature * math.log(10.0)
        )
        assert default_conversion == pytest.approx(
            expected_default[temperature], abs=1e-9
        )
        assert candidate_conversion == pytest.approx(
            expected_candidate[temperature], abs=1e-9
        )

    liquid_g = species_thermo("SiO2", "l", 1996.0, candidate).G_J_mol
    crystal_g = crystal_row.G_J_mol
    delta_g_fus = liquid_g - crystal_g

    # Premise: at phase equality, G°_l + RT ln(a_l) = G°_cr + RT ln(a_cr).
    # Algebra: log10(a_l/a_cr) = -(G°_l-G°_cr)/(RT ln 10).
    # Units: delta_g_fus and RT are both J/mol, so the ratio is dimensionless.
    # Sanity: at Tm the candidate pair has delta_g_fus=0 and the conversion is 0.
    conversion = -delta_g_fus / (R_J_MOL_K * 1996.0 * math.log(10.0))
    assert delta_g_fus == pytest.approx(0.0, abs=0.05)
    assert conversion == pytest.approx(0.0, abs=1e-5)
