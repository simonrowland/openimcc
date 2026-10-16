from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from openimcc import evaluate_redox
from openimcc.redox_pack import (
    RedoxCompositionTermError,
    canonical_redox_digest,
    load_redox_pack,
)


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "src/openimcc/data/packs"
TI = PACK_DIR / "openimcc-redox-ti-candidate.json"
CR = PACK_DIR / "openimcc-redox-cr-candidate.json"
CR_TI = PACK_DIR / "openimcc-redox-cr-ti-candidate.json"
DEFAULT = PACK_DIR / "openimcc-redox-v1.json"


_MW = {
    "SiO2": 60.0843,
    "Al2O3": 101.9613,
    "MgO": 40.3044,
    "CaO": 56.0774,
    "TiO2": 79.866,
    "Na2O": 61.9789,
    "FeO": 71.844,
}
_OPTICAL = {
    "SiO2": (1, 2, 0.48),
    "Al2O3": (2, 3, 0.61),
    "MgO": (1, 1, 0.78),
    "CaO": (1, 1, 1.00),
    "TiO2": (1, 2, 1.10),
    "Na2O": (2, 1, 1.15),
    "FeO": (1, 1, 1.00),
}


def _inventory(oxide_wt: dict[str, float]) -> dict[str, float]:
    inventory = {element: 0.0 for element in ("Si", "Al", "Mg", "Ca", "Ti", "Na", "Fe")}
    cation = {
        "SiO2": "Si", "Al2O3": "Al", "MgO": "Mg", "CaO": "Ca",
        "TiO2": "Ti", "Na2O": "Na", "FeO": "Fe",
    }
    for oxide, amount in oxide_wt.items():
        count = _OPTICAL[oxide][0]
        inventory[cation[oxide]] += count * amount / _MW[oxide]
    # The redox API needs a finite Fe couple; this trace amount is below the
    # composition precision and keeps the equation check effectively Fe-free.
    inventory["Fe"] = max(inventory["Fe"], 1e-12)
    inventory["O"] = 0.0
    return inventory


def _independent_lambda(oxide_wt: dict[str, float]) -> float:
    # Independent hand calculation of Berry's Table 3 recipe:
    # Lambda = sum(Lambda_i*m_i*n_i) / sum(m_i*n_i).
    oxygen_weighted = {
        oxide: amount / _MW[oxide] * _OPTICAL[oxide][1]
        for oxide, amount in oxide_wt.items()
    }
    return sum(
        _OPTICAL[oxide][2] * amount
        for oxide, amount in oxygen_weighted.items()
    ) / sum(oxygen_weighted.values())


def _berry_fraction(temperature: float, log10_f_o2: float, lam: float) -> float:
    log_k_prime = -1.84 + 3.44 * lam + 6262.5 / temperature
    ratio = 10.0 ** (-log_k_prime - 0.25 * log10_f_o2)
    return ratio / (1.0 + ratio)


def _imposed(inventory: dict[str, float], temperature: float, logf: float, pack=TI):
    return evaluate_redox(
        inventory, temperature, "imposed", logf * math.log(10.0), redox_pack_path=pack
    )


def _legacy_output_digest(result) -> str:
    old_fields = (
        result.temperature_K, result.mode, result.lambda_ln_f_o2, result.lambda_sat,
        tuple(result.parent_moles.items()), tuple(result.parent_activity.items()),
        tuple(result.species_moles.items()), result.ferric_fe_mol,
        result.ferrous_fe_mol, result.oxygen_inventory_mol, result.oxygen_mol,
        result.oxygen_residual_mol, result.oxygen_uptake_mol,
        result.oxygen_capacity_mol_per_ln_f_o2, result.metal_moles,
        result.metal_buffered, result.endpoint_status, result.flags,
        result.last_resort_ideal_parents, result.extrapolated_redox_rows,
        result.active_row_band_intersection_K, result.solver_path,
        result.fallback_fired, result.warm_start,
        tuple((element, tuple(states.items())) for element, states
              in result.element_moles_by_oxidation_state.items()),
        result.unmodelled_redox_couples, result.redox_pack_digest,
        result.melt_pack_binding_digest, result.chromous_cr_mol,
        result.chromic_cr_mol, result.cr_oxygen_capacity_mol,
        result.composition_redox_iterations,
    )
    return hashlib.sha256(repr(old_fields).encode()).hexdigest()


def test_ti_row_matches_independent_equation_at_three_glass_compositions():
    glasses = [
        ({"SiO2": 49.96, "Al2O3": 15.44, "MgO": 10.59, "CaO": 23.27, "TiO2": 1.03}, 1673.15, -10.7),
        ({"SiO2": 49.15, "Al2O3": 13.19, "MgO": 17.41, "CaO": 19.83, "TiO2": 1.04}, 1773.0, -9.2),
        ({"SiO2": 51.49, "Al2O3": 6.25, "MgO": 4.27, "CaO": 37.43, "TiO2": 1.02}, 1843.0, -9.0),
    ]
    pack = load_redox_pack(TI)
    ti_row = next(row for row in pack.rows if row["complex"] == "TiO1.5(l)-effective")
    assert ti_row["external_oxygen_stoich_product_positive"] == 0.25
    for oxides, temperature, logf in glasses:
        lam = _independent_lambda(oxides)
        result = _imposed(_inventory(oxides), temperature, logf)
        total_ti = result.titanous_ti_mol + result.titanic_ti_mol
        fraction = result.titanous_ti_mol / total_ti
        expected = _berry_fraction(temperature, logf, lam)
        assert fraction == pytest.approx(expected, abs=2e-9)
        assert result.element_moles_by_oxidation_state["Ti"][3] == result.titanous_ti_mol
        assert result.element_moles_by_oxidation_state["Ti"][4] == result.titanic_ti_mol


def test_apollo_14072_fraction_matches_reported_approximately():
    oxides = {
        "SiO2": 45.15, "TiO2": 2.57, "Al2O3": 11.07, "FeO": 17.82,
        "MgO": 12.16, "CaO": 9.84, "Na2O": 0.32,
    }
    temperature = 1673.15
    # O'Neill-Pownceby IW-1; derive the buffer from the published expression.
    logf_iw = 14.056 - 28776.5 / temperature - 2.0384 * math.log10(temperature)
    logf = logf_iw - 1.0
    lam = _independent_lambda(oxides)
    result = _imposed(_inventory(oxides), temperature, logf)
    fraction = result.titanous_ti_mol / (result.titanous_ti_mol + result.titanic_ti_mol)
    assert lam == pytest.approx(0.631, abs=0.001)
    assert fraction == pytest.approx(_berry_fraction(temperature, logf, lam), abs=2e-9)
    # The report rounds Berry's equation 6 result to about 0.037.
    assert fraction == pytest.approx(0.037, abs=0.002)
    assert result.ti_oxygen_capacity_mol == pytest.approx(0.5 * result.titanous_ti_mol)


def test_ratio_has_quarter_decade_fugacity_slope():
    oxides = {"SiO2": 49.96, "Al2O3": 15.44, "MgO": 10.59, "CaO": 23.27, "TiO2": 1.03}
    low = _imposed(_inventory(oxides), 1673.15, -11.0)
    high = _imposed(_inventory(oxides), 1673.15, -10.0)
    low_ratio = low.titanous_ti_mol / low.titanic_ti_mol
    high_ratio = high.titanous_ti_mol / high.titanic_ti_mol
    assert math.log10(high_ratio / low_ratio) == pytest.approx(-0.25, abs=2e-8)


def test_tranell_cst4_is_a_held_out_comparison_not_a_fit():
    oxides = {"SiO2": 60.3, "CaO": 31.8, "TiO2": 7.33}
    lam = _independent_lambda(oxides)
    result = _imposed(_inventory(oxides), 1843.0, -9.0)
    fraction = result.titanous_ti_mol / (result.titanous_ti_mol + result.titanic_ti_mol)
    expected = _berry_fraction(1843.0, -9.0, lam)
    assert fraction == pytest.approx(expected, abs=2e-9)
    # Tranell's independently measured CST4 value is 0.090 at these conditions.
    # This assertion records the visible disagreement; the measurement never sets K.
    assert fraction < 0.090


def test_temperature_and_optical_basicity_extrapolations_are_flagged():
    oxides = {"SiO2": 80.0, "CaO": 10.0, "TiO2": 10.0}
    result = _imposed(_inventory(oxides), 2000.0, -9.0)
    assert any("redox_row_temperature_extrapolation" in flag for flag in result.flags)
    assert any("optical_basicity_outside_calibrated_range" in flag for flag in result.flags)


def test_total_oxygen_capacity_matches_finite_difference_with_fe_coupling():
    oxides = {
        "SiO2": 45.15, "TiO2": 2.57, "Al2O3": 11.07, "FeO": 17.82,
        "MgO": 12.16, "CaO": 9.84, "Na2O": 0.32,
    }
    feed = _inventory(oxides)
    temperature = 1673.15
    center_logf = 14.056 - 28776.5 / temperature - 2.0384 * math.log10(temperature) - 1
    h = 1e-4
    center = _imposed(feed, temperature, center_logf)
    plus = _imposed(feed, temperature, center_logf + h / math.log(10.0))
    minus = _imposed(feed, temperature, center_logf - h / math.log(10.0))
    finite_difference = (plus.oxygen_mol - minus.oxygen_mol) / (2.0 * h)
    assert center.oxygen_capacity_mol_per_ln_f_o2 == pytest.approx(
        finite_difference, rel=1e-4
    )


def test_tagged_linear_term_preserves_cr_candidate_result(tmp_path):
    content = json.loads(CR.read_text())
    cr_row = content["rows"][-1]
    coefficients = cr_row.pop("composition_coefficients")
    cr_row["composition_term"] = {
        "kind": "linear_cation_fractions", "coefficients": coefficients
    }
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path = tmp_path / "cr-tagged.json"
    path.write_text(json.dumps(content))
    tagged = load_redox_pack(path)
    assert tagged.rows[-1]["composition_term"]["kind"] == "linear_cation_fractions"
    inventory = {"Si": 0.3, "Mg": 0.1, "Fe": 0.1, "Ca": 0.15, "Al": 0.1,
                 "Ti": 0.01, "Na": 0.01, "K": 0.001, "Cr": 0.002, "O": 1.05}
    before = evaluate_redox(inventory, 1700.0, "imposed", -8 * math.log(10), redox_pack_path=CR)
    after = evaluate_redox(inventory, 1700.0, "imposed", -8 * math.log(10), redox_pack_path=path)
    assert before.species_moles == after.species_moles
    assert before.oxygen_mol == after.oxygen_mol
    assert before.oxygen_capacity_mol_per_ln_f_o2 == after.oxygen_capacity_mol_per_ln_f_o2
    assert before.element_moles_by_oxidation_state == after.element_moles_by_oxidation_state


def test_cr_candidate_legacy_outputs_are_bit_identical_to_base_commit():
    feed = {"Si": 0.3, "Mg": 0.1, "Fe": 0.1, "Ca": 0.15, "Al": 0.1,
            "Ti": 0.01, "Na": 0.01, "K": 0.001, "Cr": 0.002, "O": 1.05}
    expected = (
        "d9be283f274f6340eee15c5223eebbba92dd01a38720e0fee606637df011ee71",
        "7a5f098c6e1354e4ca4c1fda0b37bd8fa36b06e3b0087a3168cd75bdf12a2027",
        "f141153bab1b73d235ed5713045d2e33964222813785763cff199d5c745fca30",
    )
    actual = []
    for temperature, logf in ((1673.0, -10.0), (1700.0, -8.0), (1773.0, -6.0)):
        result = evaluate_redox(
            feed, temperature, "imposed", logf * math.log(10.0), redox_pack_path=CR
        )
        actual.append(_legacy_output_digest(result))
    assert tuple(actual) == expected


def test_default_outputs_are_bit_identical_to_base_commit():
    cases = [
        (1800.0, {"Fe": 1.0, "O": 1.25}),
        (2200.0, {"Fe": 1.0, "Mn": 0.1, "Si": 0.2, "Al": 0.1, "O": 1.65}),
        (2500.0, {"Fe": 0.7, "Si": 0.2, "Mg": 0.1, "O": 1.0}),
    ]
    expected = (
        "322c95c29ab2f48970690f09978b6f82491a38ce99f655d8dc065cf296905a61",
        "ed556c3da7e16d08bb8bf3d9e75659759dcf0387e1660d3a52e8c6b06b067952",
        "f938b462405d6de6127371854007c8d1f4cfee05b67b25a3929c4d36c90dbd42",
    )
    actual = [
        _legacy_output_digest(evaluate_redox(feed, temperature, "closed"))
        for temperature, feed in cases
    ]
    assert tuple(actual) == expected


@pytest.mark.parametrize("mutation", ["unknown-kind", "unknown-field"])
def test_optical_basicity_term_rejects_unknown_kind_or_fields(tmp_path, mutation):
    content = json.loads(TI.read_text())
    term = next(row for row in content["rows"] if row["complex"] == "TiO1.5(l)-effective")["composition_term"]
    if mutation == "unknown-kind":
        term["kind"] = "fitted_residual"
    else:
        term["extra"] = 1.0
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path = tmp_path / "invalid-candidate.json"
    path.write_text(json.dumps(content))
    with pytest.raises(RedoxCompositionTermError):
        load_redox_pack(path)


def test_default_pack_rejects_the_new_composition_term(tmp_path):
    content = json.loads(DEFAULT.read_text())
    content["rows"][1]["composition_term"] = {
        "kind": "optical_basicity", "A": -1.84, "B_lambda": 3.44,
        "C_over_T": 6262.5, "lambda_range": [0.586, 0.635],
    }
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path = tmp_path / "default-with-candidate-term.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="candidate-pack only"):
        load_redox_pack(path)


def test_new_candidate_packs_select_explicitly_and_default_pack_is_unchanged():
    assert load_redox_pack(TI).version == "1.5.0-ti-candidate"
    assert load_redox_pack(CR_TI).version == "1.6.0-cr-ti-candidate"
    combined = _imposed(
        {"Si": 0.3, "Mg": 0.1, "Fe": 0.1, "Ca": 0.15, "Al": 0.1,
         "Ti": 0.01, "Na": 0.01, "K": 0.001, "Cr": 0.002, "O": 1.05},
        1700.0, -8.0, CR_TI,
    )
    assert combined.titanous_ti_mol > 0.0
    assert combined.chromous_cr_mol > 0.0
    default_bytes = hashlib.sha256(DEFAULT.read_bytes()).hexdigest()
    for temperature, feed in [
        (1800.0, {"Fe": 1.0, "O": 1.25}),
        (2200.0, {"Fe": 1.0, "Mn": 0.1, "Si": 0.2, "Al": 0.1, "O": 1.65}),
        (2500.0, {"Fe": 0.7, "Si": 0.2, "Mg": 0.1, "O": 1.0}),
    ]:
        implicit = evaluate_redox(feed, temperature, "closed")
        explicit = evaluate_redox(feed, temperature, "closed", redox_pack_path=DEFAULT)
        assert implicit == explicit
    assert hashlib.sha256(DEFAULT.read_bytes()).hexdigest() == default_bytes
