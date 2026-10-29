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
TI = PACK_DIR / "openimcc-redox-ti-candidate-v4.json"
CR = PACK_DIR / "openimcc-redox-cr-candidate-v4.json"
CR_TI = PACK_DIR / "openimcc-redox-cr-ti-candidate-v4.json"
DEFAULT = PACK_DIR / "openimcc-redox-v4.json"


_MW = {
    "SiO2": 60.0843,
    "Al2O3": 101.9613,
    "MgO": 40.3044,
    "CaO": 56.0774,
    "TiO2": 79.866,
    "Na2O": 61.9789,
    "FeO": 71.844,
    "K2O": 94.196,
    "Cr2O3": 151.9904,
}
_OPTICAL = {
    "SiO2": (1, 2, 0.48),
    "Al2O3": (2, 3, 0.61),
    "Cr2O3": (2, 3, None),
    "MgO": (1, 1, 0.78),
    "CaO": (1, 1, 1.00),
    "TiO2": (1, 2, 1.10),
    "Na2O": (2, 1, 1.15),
    "K2O": (2, 1, 1.40),
    "FeO": (1, 1, 1.00),
}


def _inventory(oxide_wt: dict[str, float]) -> dict[str, float]:
    inventory = {
        element: 0.0
        for element in ("Si", "Al", "Cr", "Mg", "Ca", "Ti", "Na", "K", "Fe")
    }
    cation = {
        "SiO2": "Si", "Al2O3": "Al", "Cr2O3": "Cr", "MgO": "Mg",
        "CaO": "Ca", "TiO2": "Ti", "Na2O": "Na", "K2O": "K", "FeO": "Fe",
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
    # Independent hand calculation of Berry's oxygen-weighted recipe:
    # Lambda = sum(Lambda_i*m_i*n_i) / sum(m_i*n_i).
    oxygen_weighted = {
        oxide: amount / _MW[oxide] * _OPTICAL[oxide][1]
        for oxide, amount in oxide_wt.items()
        if _OPTICAL[oxide][2] is not None
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
    """Hash every numeric output field for the exact-bit default redox pin."""
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
        result.unmodelled_redox_couples, result.chromous_cr_mol,
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


def test_ti_basicity_reproduces_berry_table_1_and_anchor_envelope():
    from openimcc.redox import _berry_optical_basicity

    glasses = [
        ({"SiO2": 49.96, "Al2O3": 15.44, "MgO": 10.59, "CaO": 23.27, "TiO2": 1.03}, 0.6094),
        ({"SiO2": 49.15, "Al2O3": 13.19, "MgO": 17.41, "CaO": 19.83, "TiO2": 1.04}, 0.6127),
        ({"SiO2": 51.49, "Al2O3": 6.25, "MgO": 4.27, "CaO": 37.43, "TiO2": 1.02}, 0.6345),
        ({"SiO2": 54.09, "Al2O3": 9.53, "MgO": 21.66, "CaO": 14.31, "TiO2": 1.06}, 0.5986),
        ({"SiO2": 65.86, "Al2O3": 10.03, "MgO": 6.98, "CaO": 15.30, "TiO2": 1.13}, 0.5633),
    ]
    calculated = []
    for oxides, published_lambda in glasses:
        result = _imposed(_inventory(oxides), 1673.15, -10.0)
        value = _berry_optical_basicity(result.element_moles_by_oxidation_state)
        assert value == pytest.approx(published_lambda, abs=0.002)
        calculated.append(value)
        expected = _berry_fraction(1673.15, -10.0, _independent_lambda(oxides))
        fraction = result.titanous_ti_mol / (result.titanous_ti_mol + result.titanic_ti_mol)
        assert fraction == pytest.approx(expected, abs=2e-9)

    assert min(calculated) >= 0.5633
    assert max(calculated) <= 0.6353


@pytest.mark.parametrize(("element", "cations", "published_lambda"), [
    ("K", 2, 1.40),
])
def test_berry_optical_basicity_uses_published_k_value(
    element: str, cations: int, published_lambda: float
):
    from openimcc.redox import _berry_optical_basicity

    assert _berry_optical_basicity({element: {1: cations}}) == pytest.approx(
        published_lambda
    )


def test_berry_optical_basicity_excludes_unsourced_chromium():
    from openimcc.redox import _berry_optical_basicity

    assert _berry_optical_basicity(
        {"Si": {4: 1.0}, "Cr": {3: 2.0}}
    ) == pytest.approx(0.48)


@pytest.mark.parametrize(
    ("pack", "additions"),
    [
        pytest.param(TI, {"K2O": 2.0}, id="ti-k-bearing"),
        pytest.param(TI, {"Cr2O3": 2.0}, id="ti-cr-bearing"),
        pytest.param(TI, {"K2O": 1.0, "Cr2O3": 1.0}, id="ti-k-and-cr-bearing"),
        pytest.param(CR_TI, {"K2O": 2.0}, id="cr-ti-k-bearing"),
        pytest.param(CR_TI, {"Cr2O3": 2.0}, id="cr-ti-cr-bearing"),
        pytest.param(CR_TI, {"K2O": 1.0, "Cr2O3": 1.0}, id="cr-ti-k-and-cr-bearing"),
    ],
)
def test_ti_candidate_lambda_excludes_unsourced_cr_and_flags_it(
    pack: Path, additions: dict[str, float]
):
    from openimcc.redox import _berry_optical_basicity

    oxides = {
        "SiO2": 49.96, "Al2O3": 15.44, "MgO": 10.59,
        "CaO": 23.27, "TiO2": 1.03, **additions,
    }
    result = _imposed(_inventory(oxides), 1673.15, -10.0, pack)
    lambda_value = _berry_optical_basicity(result.element_moles_by_oxidation_state)
    expected_lambda = _independent_lambda(oxides)
    ti_fraction = result.titanous_ti_mol / (
        result.titanous_ti_mol + result.titanic_ti_mol
    )

    assert lambda_value == pytest.approx(expected_lambda, abs=2e-12)
    assert ti_fraction == pytest.approx(
        _berry_fraction(1673.15, -10.0, expected_lambda), abs=2e-9
    )
    if "Cr2O3" in additions:
        assert any(
            "Cr-bearing melt is outside the candidate optical-basicity basis" in flag
            for flag in result.flags
        )
    else:
        assert not any(
            "Cr-bearing melt is outside the candidate optical-basicity basis" in flag
            for flag in result.flags
        )


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


@pytest.mark.parametrize(
    ("pack", "absent_element"),
    [
        (CR, "Cr"),
        (CR, "Ti"),
        (TI, "Cr"),
        (TI, "Ti"),
        (CR_TI, "Cr"),
        (CR_TI, "Ti"),
    ],
)
def test_candidate_pack_treats_absent_trace_elements_as_inactive(pack, absent_element):
    inventory = {
        "Si": 0.60,
        "Al": 0.20,
        "Mg": 0.15,
        "Ca": 0.25,
        "Fe": 1.0e-12,
    }
    present_element = "Ti" if absent_element == "Cr" else "Cr"
    inventory[present_element] = 0.01

    result = evaluate_redox(
        inventory,
        1843.0,
        "imposed",
        -9.0 * math.log(10.0),
        redox_pack_path=pack,
    )

    assert absent_element not in result.element_moles_by_oxidation_state
    assert absent_element not in {
        record["element"] for record in result.unmodelled_redox_couples
    }
    assert result.element_moles_by_oxidation_state["Fe"][2] + result.element_moles_by_oxidation_state["Fe"][3] == pytest.approx(
        1.0e-12
    )


def test_temperature_and_optical_basicity_extrapolations_are_flagged():
    oxides = {"SiO2": 80.0, "CaO": 10.0, "TiO2": 10.0}
    result = _imposed(_inventory(oxides), 2000.0, -9.0)
    assert any("redox_row_temperature_extrapolation" in flag for flag in result.flags)
    assert any("optical_basicity_outside_calibrated_range" in flag for flag in result.flags)


@pytest.mark.parametrize("pack", [TI, CR_TI])
def test_ti_anchor_domain_flags_are_advisory_and_do_not_change_prediction(pack):
    def feed(oxides):
        result = _inventory(oxides)
        if pack == CR_TI:
            result["Cr"] = 1e-4
        return result

    inside = {
        "SiO2": 49.96, "Al2O3": 15.44, "MgO": 10.59,
        "CaO": 23.27, "TiO2": 1.02,
    }
    expected_inside = dict(inside)
    if pack == CR_TI:
        expected_inside["Cr2O3"] = 1e-4 / 2 * _MW["Cr2O3"]
    inside_result = _imposed(feed(inside), 1673.15, -10.7, pack)
    assert not any("Ti candidate outside Berry anchor" in flag
                   or "TiO2 exceeds Berry anchor" in flag
                   or "Ti candidate temperature outside Berry calibration" in flag
                   for flag in inside_result.flags)
    expected_inside_fraction = _berry_fraction(
        1673.15, -10.7, _independent_lambda(expected_inside)
    )
    assert inside_result.titanous_ti_mol / inside_result.titanic_ti_mol == pytest.approx(
        expected_inside_fraction / (1 - expected_inside_fraction),
        abs=2e-9,
    )

    outside = {"SiO2": 60.3, "CaO": 31.8, "TiO2": 7.33}
    expected_outside = dict(outside)
    if pack == CR_TI:
        expected_outside["Cr2O3"] = 1e-4 / 2 * _MW["Cr2O3"]
    outside_result = _imposed(feed(outside), 1843.0, -9.0, pack)
    ti_flags = [flag for flag in outside_result.flags if "Tranell held-out titrations" in flag]
    assert len(ti_flags) == 2
    assert not any("outside Berry anchor Lambda" in flag for flag in ti_flags)
    assert any("1.03 wt%" in flag for flag in ti_flags)
    assert any("1673-1674 K" in flag for flag in ti_flags)
    fraction = outside_result.titanous_ti_mol / (
        outside_result.titanous_ti_mol + outside_result.titanic_ti_mol
    )
    assert fraction == pytest.approx(
        _berry_fraction(1843.0, -9.0, _independent_lambda(expected_outside)), abs=2e-9
    )


def test_combined_candidate_locks_cr_ti_fixed_point_coupling():
    feed = {
        "Si": 0.3, "Mg": 0.1, "Fe": 0.1, "Ca": 0.15, "Al": 0.1,
        "Ti": 0.01, "Na": 0.01, "K": 0.001, "Cr": 0.002, "O": 1.05,
    }
    temperature, logf = 1700.0, -8.0
    result = _imposed(feed, temperature, logf, CR_TI)
    total_cations = sum(amount for element, amount in feed.items() if element != "O")
    composition = sum(coefficient * amount / total_cations for coefficient, amount in [
        (2.00, feed["Na"]), (1.00, feed["Mg"]), (1.00, result.ferrous_fe_mol),
        (0.37, feed["Al"]), (0.37, result.ferric_fe_mol), (2.12, feed["K"]),
        (2.44, feed["Ca"]), (3.69, feed["Ti"]),
    ])
    qfm = 8.58 - 25050.0 / temperature
    log10_cr3_cr2 = 0.25 * (logf - qfm) + 3031.0 / temperature - 2.26 + composition
    expected_cr2_fraction = 1.0 / (1.0 + 10.0**log10_cr3_cr2)
    actual_cr2_fraction = result.chromous_cr_mol / (
        result.chromous_cr_mol + result.chromic_cr_mol
    )
    lambda_value = _independent_lambda({
        "SiO2": feed["Si"] * _MW["SiO2"],
        "Al2O3": feed["Al"] / 2 * _MW["Al2O3"],
        "MgO": feed["Mg"] * _MW["MgO"],
        "CaO": feed["Ca"] * _MW["CaO"],
        "TiO2": feed["Ti"] * _MW["TiO2"],
        "Na2O": feed["Na"] / 2 * _MW["Na2O"],
        "FeO": (result.ferrous_fe_mol + result.ferric_fe_mol) * _MW["FeO"],
        "K2O": feed["K"] / 2 * _MW["K2O"],
        "Cr2O3": feed["Cr"] / 2 * _MW["Cr2O3"],
    })
    expected_ti_fraction = _berry_fraction(temperature, logf, lambda_value)
    actual_ti_fraction = result.titanous_ti_mol / (
        result.titanous_ti_mol + result.titanic_ti_mol
    )
    assert actual_cr2_fraction == pytest.approx(expected_cr2_fraction, abs=2e-10)
    assert actual_ti_fraction == pytest.approx(expected_ti_fraction, abs=2e-9)
    assert result.composition_redox_iterations >= 2


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


def test_cr_candidate_v3_outputs_pin_corrected_homologue():
    """Pin corrected Cr outputs with a tolerance grounded in model precision.

    The solver's 1e-12 balance residual certifies its residual criterion; it
    does not bound error in each reported output. The 1e-12 relative,
    1e-15 absolute output tolerance is chosen far below model-input precision
    and above the observed cross-platform roundoff; the largest measured
    cross-platform delta was 1.518e-18 absolute (1.002e-15 relative). The test
    reports the largest observed delta so future platform results remain
    reviewable.
    """
    feed = {"Si": 0.3, "Mg": 0.1, "Fe": 0.1, "Ca": 0.15, "Al": 0.1,
            "Ti": 0.01, "Na": 0.01, "K": 0.001, "Cr": 0.002, "O": 1.05}
    # The default melt component is D066 v5. Its five association A refits
    # change the composition-dependent redox inputs; the Cr candidate rows
    # and redox equations are unchanged.
    expected = (
        (0.0015149535635929346, 0.00048504643640706716, 0.0007574767817964673),
        (0.0010971972025757852, 0.0009028027974242138, 0.0005485986012878926),
        (0.0007885946163817768, 0.0012114053836182219, 0.0003942973081908884),
    )
    actual = []
    for temperature, logf in ((1673.0, -10.0), (1700.0, -8.0), (1773.0, -6.0)):
        result = evaluate_redox(
            feed, temperature, "imposed", logf * math.log(10.0), redox_pack_path=CR
        )
        actual.append((
            result.chromous_cr_mol,
            result.chromic_cr_mol,
            result.cr_oxygen_capacity_mol,
        ))
        assert result.mode == "imposed"
        assert result.solver_path == "newton"
        assert result.fallback_fired is False
    deltas = [
        abs(observed_value - stored_value)
        for observed, stored in zip(actual, expected)
        for observed_value, stored_value in zip(observed, stored)
    ]
    relative_deltas = [
        abs(observed_value - stored_value) / abs(stored_value)
        for observed, stored in zip(actual, expected)
        for observed_value, stored_value in zip(observed, stored)
    ]
    print(
        f"corrected Cr output max delta: absolute={max(deltas):.3e}; "
        f"relative={max(relative_deltas):.3e}"
    )
    for observed, stored in zip(actual, expected):
        assert observed == pytest.approx(stored, rel=1e-12, abs=1e-15)


def test_default_output_digests_pin_corrected_redox_homologue():
    cases = [
        (1800.0, {"Fe": 1.0, "O": 1.25}),
        (2200.0, {"Fe": 1.0, "Mn": 0.1, "Si": 0.2, "Al": 0.1, "O": 1.65}),
        (2500.0, {"Fe": 0.7, "Si": 0.2, "Mg": 0.1, "O": 1.0}),
    ]
    # D066 v5's five association A refits change the mixed Fe-Mg-Si endpoint;
    # the redox datapack and equations remain unchanged.
    expected = (
        "22aa16d3e716accfa736676672a97b25cd503eeef5183fa07e9d79f7bec819a1",
        "18340e4531b21326f5e21005c3565dd2f9f377d98d01ab18e409d97dc473391b",
        "0970c5620b101dcc24fc04cc0b10221c0d4121ff096a327f6fa9bc2922e9f288",
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
    assert load_redox_pack(TI).version == "1.8.0-ti-candidate"
    assert load_redox_pack(CR_TI).version == "1.9.0-cr-ti-candidate"
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
