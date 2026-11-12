import math

import pytest

from openimcc import (
    ImccComponentOutsideDomainError,
    ImccCompositionIncompleteError,
    ImccTOutsideDatapackDomainError,
    evaluate,
    evaluate_mgo_sio2_mqm,
)
from openimcc._constants import R_J_MOL_K
from openimcc.gas import load_gas_datapack, species_thermo
from openimcc.mqm_mgo_sio2 import _cp_integrals, _wu_liquid_gibbs_j_mol


def _wu_solid_gibbs_j_mol(name: str, temperature_K: float) -> float:
    """Wu Table 1, p. 28, forsterite and protoenstatite solid standards."""
    # Table 1 uses H=A+∫Cp dT and S=B+∫Cp/T dT, with Cp in cal/mol/K.
    rows = {
        "forsterite": (
            -520482.62, 22.468913,
            (57.036654, 0.0, 0.0, -478.31286, -0.27782811),
        ),
        "protoenstatite": (
            -368906.91, 16.117975,
            (39.813456, 0.0, -5.4267686, -286.94742, 0.66718532),
        ),
    }
    a_h, b_s, cp = rows[name]
    h_integral, s_integral = _cp_integrals(
        temperature_K, reference_K=298.15,
        a=cp[0], b=cp[1], c=cp[2], d=cp[3], e=cp[4],
    )
    return (a_h + h_integral - temperature_K * (b_s + s_integral)) * 4.184


def test_wu_table_2_excess_gibbs_at_x_half_after_openimcc_standard_conversion() -> None:
    # Wu et al. Table 2, p. 32, prints RT ln(gamma) to 1 J/mol at xSiO2=0.5.
    # The 5 J/mol gate allows table rounding plus the central-difference
    # derivative used by this standalone evaluator.
    printed_wu_j_mol = {
        2073.15: {"MgO": -41605.0, "SiO2": 4317.0},
        2273.15: {"MgO": -37237.0, "SiO2": 2759.0},
        2473.15: {"MgO": -33116.0, "SiO2": 1352.0},
    }
    standards = load_gas_datapack()
    for temperature, expected_by_oxide in printed_wu_j_mol.items():
        result = evaluate_mgo_sio2_mqm({"MgO": 0.5, "SiO2": 0.5}, temperature)
        assert standards.condensate_table_digest in result.standard_state
        for oxide, activity in (("MgO", result.a_mgo), ("SiO2", result.a_sio2)):
            open_g = species_thermo(oxide, "l", temperature, standards).G_J_mol
            wu_g = _wu_liquid_gibbs_j_mol(oxide, temperature)
            observed_open = R_J_MOL_K * temperature * math.log(activity / 0.5)
            expected_open = expected_by_oxide[oxide] + wu_g - open_g
            assert observed_open == pytest.approx(expected_open, abs=5.0)


def test_arutyunyan_held_out_beta_cristobalite_activities_and_v4_residuals() -> None:
    # Arutyunyan et al. (2005), Table 2, printed p. 394; standard is beta
    # cristobalite. Convert with μ=G°+RT ln(a): ln(a_liq)=ln(a_beta)
    # +(G°_beta−G°_liquid)/(RT). The predeclared gate is 0.015 dex, covering
    # source-table rounding and the high-cristobalite Gibbs fit residual. The
    # x=0.486 point is starred melt + Mg2SiO4; the paper's liquidus values are
    # x=0.488 and 0.563. a(MgO) is versus crystalline MgO and is not scored.
    temperature = 1873.15
    rows = (
        (0.486, 0.693), (0.498, 0.737), (0.511, 0.797),
        (0.528, 0.865), (0.550, 0.951), (0.563, 1.000),
    )
    standards = load_gas_datapack()
    g_beta = species_thermo("SiO2", "cr", temperature, standards).G_J_mol
    g_liquid = species_thermo("SiO2", "l", temperature, standards).G_J_mol
    standard_shift = (g_beta - g_liquid) / (R_J_MOL_K * temperature)
    signed_candidate_residuals = []
    for x_sio2, beta_activity in rows:
        candidate = evaluate_mgo_sio2_mqm(
            {"MgO": 1.0 - x_sio2, "SiO2": x_sio2}, temperature
        )
        converted_activity = beta_activity * math.exp(standard_shift)
        residual = math.log10(candidate.a_sio2 / converted_activity)
        signed_candidate_residuals.append(residual)
        assert residual == pytest.approx(
            (0.0032, 0.0090, 0.0070, 0.0084, 0.0079, 0.0066)[len(signed_candidate_residuals)-1],
            abs=0.001,
        )
        assert abs(residual) <= 0.015
        assert candidate.stability == "homogeneous"
        assert candidate.two_liquid_field_x_bounds is not None

        # The unchanged default model retains its approximately −0.38 dex
        # residual over these rows; it is reported alongside the candidate.
        v4 = evaluate(
            {"MgO": 1.0 - x_sio2, "SiO2": x_sio2},
            temperature, basis_type="mol",
        )
        v4_residual = math.log10(v4.activity("SiO2") / converted_activity)
        assert -0.42 <= v4_residual <= -0.34


def test_chen_presnall_peritectic_consistency_with_wu_fit() -> None:
    # Chen & Presnall (1975) remeasured the Bowen & Andersen peritectic Wu
    # fitted, including the Table 1 solid enthalpy adjustments; Chen is absent
    # from Wu's references. Use 1830 K so the printed protoenstatite standard
    # range is not extrapolated. This is a consistency check, not held-out data.
    x_chen = (60.6 / 60.0843) / (60.6 / 60.0843 + 39.4 / 40.3044)
    temperature = 1830.0
    assert x_chen == pytest.approx(0.507809651, abs=1e-9)

    standards = load_gas_datapack()
    g_mgo = species_thermo("MgO", "l", temperature, standards).G_J_mol
    g_sio2 = species_thermo("SiO2", "l", temperature, standards).G_J_mol
    for x_sio2, tolerance, expected_indices in (
        (0.503, 1e-4, None),
        (x_chen, 0.02, (-0.0127, -0.0002)),
    ):
        result = evaluate_mgo_sio2_mqm(
            {"MgO": 1.0 - x_sio2, "SiO2": x_sio2}, temperature
        )
        assert result.stability == "homogeneous"
        saturation_indices = []
        for phase, stoichiometry in (
            ("forsterite", {"MgO": 2.0, "SiO2": 1.0}),
            ("protoenstatite", {"MgO": 1.0, "SiO2": 1.0}),
        ):
            solid_g = _wu_solid_gibbs_j_mol(phase, temperature)
            delta_g_dissolution = (
                stoichiometry["MgO"] * g_mgo
                + stoichiometry["SiO2"] * g_sio2
                - solid_g
            )
            log_q = (
                stoichiometry["MgO"] * math.log10(result.a_mgo)
                + stoichiometry["SiO2"] * math.log10(result.a_sio2)
            )
            saturation_index = log_q + delta_g_dissolution / (
                R_J_MOL_K * temperature * math.log(10.0)
            )
            saturation_indices.append(saturation_index)
            assert abs(saturation_index) <= tolerance
        if expected_indices is not None:
            assert saturation_indices == pytest.approx(expected_indices, abs=5e-5)


def test_silica_rich_two_liquid_field_and_consolute_bracket() -> None:
    at_monotectic_scale = evaluate_mgo_sio2_mqm(
        {"MgO": 0.3, "SiO2": 0.7}, 1960.0
    )
    assert at_monotectic_scale.stability == "inside_two_liquid_field"
    assert at_monotectic_scale.two_liquid_field_x_bounds == pytest.approx(
        (0.596, 0.988), abs=0.001
    )

    below_critical = evaluate_mgo_sio2_mqm(
        {"MgO": 0.5, "SiO2": 0.5}, 2214.0
    )
    above_critical = evaluate_mgo_sio2_mqm(
        {"MgO": 0.5, "SiO2": 0.5}, 2224.0
    )
    assert below_critical.two_liquid_field_x_bounds == pytest.approx(
        (0.835, 0.846), abs=0.001
    )
    assert above_critical.two_liquid_field_x_bounds is None


def test_kambayashi_is_reported_as_not_independent() -> None:
    # Kambayashi & Kato 1984 Table 2 supplied activities fitted by Wu et al.;
    # this is a reproducibility check, never a held-out score.
    temperature = 1973.15
    standards = load_gas_datapack()
    g_beta = species_thermo("SiO2", "cr", temperature, standards).G_J_mol
    g_liquid = species_thermo("SiO2", "l", temperature, standards).G_J_mol
    shift = (g_beta - g_liquid) / (R_J_MOL_K * temperature)
    rows = ((0.476, 0.567), (0.438, 0.363), (0.547, 0.870), (0.516, 0.760))
    residuals = []
    for x_sio2, beta_activity in rows:
        result = evaluate_mgo_sio2_mqm(
            {"MgO": 1.0 - x_sio2, "SiO2": x_sio2}, temperature
        )
        residuals.append(math.log10(result.a_sio2 / (beta_activity * math.exp(shift))))
    # Preserve the observed signed residuals; no accuracy threshold is applied
    # because these activities were inputs to Wu's assessment.
    assert residuals == pytest.approx(
        (0.0203, 0.0855, -0.0118, -0.0096), abs=0.001
    )


def test_binary_domain_frozen_result_and_default_bit_identity() -> None:
    composition = {"MgO": 0.5, "SiO2": 0.5}
    before = evaluate(composition, 1800.0, basis_type="mol")
    before_bits = (
        tuple(value.hex() for value in before.parent_activity),
        tuple(value.hex() for value in before.complex_x),
        before.D.hex(),
    )
    result = evaluate_mgo_sio2_mqm(composition, 1800.0)
    after = evaluate(composition, 1800.0, basis_type="mol")
    after_bits = (
        tuple(value.hex() for value in after.parent_activity),
        tuple(value.hex() for value in after.complex_x),
        after.D.hex(),
    )
    assert before_bits == after_bits
    assert result.pair_fractions == pytest.approx(
        (0.003533358, 0.659599950, 0.336866691), abs=1e-8
    )
    assert math.fsum(result.pair_fractions) == pytest.approx(1.0, abs=1e-12)
    assert result.experimental_status.startswith("EXPERIMENTAL")
    assert result.candidate_id == "wu-1993-mgo-sio2-mqm-openimcc-liquid-standards-v1"
    with pytest.raises(AttributeError):
        result.a_sio2 = 0.0

    with pytest.raises(ImccComponentOutsideDomainError):
        evaluate_mgo_sio2_mqm({"MgO": 1, "SiO2": 1, "CaO": 1e-6}, 1800)
    with pytest.raises(ImccCompositionIncompleteError):
        evaluate_mgo_sio2_mqm({"MgO": 0, "SiO2": 0}, 1800)
    with pytest.raises(ImccTOutsideDatapackDomainError):
        evaluate_mgo_sio2_mqm(composition, 1199.0)
    with pytest.raises(ImccTOutsideDatapackDomainError):
        evaluate_mgo_sio2_mqm(composition, 3001.0)
