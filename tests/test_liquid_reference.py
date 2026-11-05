from __future__ import annotations

import re
from collections import Counter

import pytest

from openimcc import (
    CertificationStatus,
    AuthorityWindow,
    ImccGasResult,
    LiquidReferenceAuthority,
    FeOLiquidStandard,
    FeOLiquidStandardOffset,
    LiquidReferenceNotice,
    LiquidReferenceRecord,
    evaluate,
    evaluate_gas,
    feo_liquid_standard,
    feo_liquid_standard_offset,
    load_datapack,
    liquid_reference_authority,
    liquid_reference_records,
)


def test_authority_enum_and_inclusive_window_boundaries() -> None:
    with pytest.raises(ValueError, match="unknown liquid-reference authority"):
        LiquidReferenceRecord("X(l)", "unknown", None, None, None, (), source_ids=())  # type: ignore[arg-type]
    assert liquid_reference_authority("SiO2(l)", 900).authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert liquid_reference_authority("SiO2(l)", 1996).authority is LiquidReferenceAuthority.CORROBORATED_CALCULATION
    assert liquid_reference_authority("SiO2(l)", 899.999).authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert liquid_reference_authority("SiO2(l)", 3000).authority is LiquidReferenceAuthority.CORROBORATED_CALCULATION
    assert liquid_reference_authority("SiO2(l)", 3000.001).authority is LiquidReferenceAuthority.EXTRAPOLATED


def test_silica_authority_windows_and_boundaries() -> None:
    expected = {
        899: (LiquidReferenceAuthority.EXTRAPOLATED, (1500, 1800), None),
        900: (LiquidReferenceAuthority.EXTRAPOLATED, (900, 1200), 0.18),
        1199: (LiquidReferenceAuthority.EXTRAPOLATED, (900, 1200), 0.18),
        1200: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1200, 1500), 0.05),
        1201: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1200, 1500), 0.05),
        1499: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1200, 1500), 0.05),
        1500: (LiquidReferenceAuthority.CERTIFIED, (1500, 1800), None),
        1800: (LiquidReferenceAuthority.CERTIFIED, (1500, 1800), None),
        1801: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1800, 3000), 0.06),
        1996: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1800, 3000), 0.06),
        2473: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1800, 3000), 0.06),
        3000: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1800, 3000), 0.06),
        3001: (LiquidReferenceAuthority.EXTRAPOLATED, (1500, 1800), None),
    }
    for temperature, (authority, window, band) in expected.items():
        status = liquid_reference_authority("SiO2(l)", temperature)
        assert status.authority is authority
        assert status.certified_T_K == window
        assert status.band_value == band
        if band is not None:
            assert status.band_unit == "dex log10 pressure"

    certified = liquid_reference_authority("SiO2(l)", 1600)
    evidence = " ".join(item.basis for item in certified.evidence_ref)
    assert "partial-molar Cp" in evidence
    assert "not pure-liquid SiO2 Cp" in evidence


def test_alumina_authority_windows_and_boundaries() -> None:
    expected = {
        1799: LiquidReferenceAuthority.EXTRAPOLATED,
        1801: LiquidReferenceAuthority.CORROBORATED_CALCULATION,
        2120: LiquidReferenceAuthority.CORROBORATED_CALCULATION,
        2326: LiquidReferenceAuthority.CORROBORATED_CALCULATION,
        2327: LiquidReferenceAuthority.CERTIFIED,
        2500: LiquidReferenceAuthority.CERTIFIED,
        2873: LiquidReferenceAuthority.CERTIFIED,
        2874: LiquidReferenceAuthority.CORROBORATED_CALCULATION,
        3000: LiquidReferenceAuthority.CORROBORATED_CALCULATION,
        3001: LiquidReferenceAuthority.EXTRAPOLATED,
    }
    for temperature, authority in expected.items():
        assert liquid_reference_authority("Al2O3(l)", temperature).authority is authority
    lower = liquid_reference_authority("Al2O3(l)", 1900)
    assert lower.certified_T_K == (1801, 2327)
    assert lower.band_value == pytest.approx(0.05)
    assert lower.band_unit == "dex log10 pressure"
    assert "corroborated by independent calculation" in " ".join(item.basis for item in lower.evidence_ref)
    assert "emissivity" in " ".join(item.basis for item in lower.evidence_ref)
    certified = liquid_reference_authority("Al2O3(l)", 2327)
    assert certified.certified_T_K == (2327, 2873)
    assert certified.band_value == pytest.approx(0.036)


def test_alumina_high_temperature_calculated_continuation() -> None:
    expected = {
        2872: (LiquidReferenceAuthority.CERTIFIED, (2327, 2873), 0.036),
        2873: (LiquidReferenceAuthority.CERTIFIED, (2327, 2873), 0.036),
        2874: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (2873, 3000), 0.036),
        3000: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (2873, 3000), 0.036),
    }
    for temperature, (authority, window, band) in expected.items():
        status = liquid_reference_authority("Al2O3(l)", temperature)
        assert (status.authority, status.certified_T_K, status.band_value) == (authority, window, band)
        assert status.band_unit == "dex log10 pressure"
    evidence = " ".join(
        item.basis for item in liquid_reference_authority("Al2O3(l)", 2874).evidence_ref
    )
    assert "calculated continuation" in evidence
    assert "2.13 kJ/mol" in evidence


def test_alumina_low_temperature_extrapolation_envelope() -> None:
    expected = {
        1322: (LiquidReferenceAuthority.EXTRAPOLATED, (2327, 2873), None),
        1323: (LiquidReferenceAuthority.EXTRAPOLATED, (1323, 1801), 0.17),
        1800: (LiquidReferenceAuthority.EXTRAPOLATED, (1323, 1801), 0.17),
        1801: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1801, 2327), 0.05),
    }
    for temperature, (authority, window, band) in expected.items():
        status = liquid_reference_authority("Al2O3(l)", temperature)
        assert status.authority is authority
        assert status.certified_T_K == window
        assert status.band_value == band
        if band is not None:
            assert status.band_unit == "dex log10 pressure"
    evidence = " ".join(
        item.basis for item in liquid_reference_authority("Al2O3(l)", 1500).evidence_ref
    )
    assert "model envelope" in evidence
    assert "not corroboration" in evidence


def test_authority_windows_reject_overlaps() -> None:
    evidence = ()
    extra = AuthorityWindow(LiquidReferenceAuthority.CERTIFIED, (2, 4), None, None, evidence)
    with pytest.raises(ValueError, match="windows overlap"):
        LiquidReferenceRecord("X(l)", LiquidReferenceAuthority.CERTIFIED,
            (1, 3), None, None, evidence, source_ids=(), additional_windows=(extra,))
    first = AuthorityWindow(LiquidReferenceAuthority.CERTIFIED, (1, 3), None, None, evidence)
    second = AuthorityWindow(LiquidReferenceAuthority.CERTIFIED, (2, 4), None, None, evidence)
    with pytest.raises(ValueError, match="windows overlap"):
        LiquidReferenceRecord("X(l)", LiquidReferenceAuthority.EXTRAPOLATED,
            None, None, None, evidence, source_ids=(), additional_windows=(first, second))


def test_composition_scoped_k2o_authority() -> None:
    in_scope = liquid_reference_authority("K2O(l)", 1500, {"K2O_wt_pct": 25})
    out_scope = liquid_reference_authority("K2O(l)", 1500, {"K2O_wt_pct": 20})
    no_composition = liquid_reference_authority("K2O(l)", 1500)
    assert in_scope.authority is LiquidReferenceAuthority.CERTIFIED
    assert in_scope.scope_applied
    assert out_scope.authority is LiquidReferenceAuthority.CERTIFIED
    assert out_scope.in_composition_scope is False
    assert "scored with notice" in out_scope.reason
    assert no_composition.authority is LiquidReferenceAuthority.CERTIFIED
    assert not no_composition.scope_applied
    assert no_composition.composition_scope == {"K2O_wt_pct": (25, 40)}
    assert liquid_reference_authority("K2O(l)", 1500, {"K2O_wt_pct": 40}).authority is LiquidReferenceAuthority.CERTIFIED
    outside_upper = liquid_reference_authority("K2O(l)", 1500, {"K2O_wt_pct": 40.001})
    assert outside_upper.authority is LiquidReferenceAuthority.CERTIFIED
    assert outside_upper.in_composition_scope is False
    assert liquid_reference_authority("K2O(l)", 1765).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("FeO(l)", 1649).authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert liquid_reference_authority("FeO(l)", 1649.15).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("Al2O3(l)", 2400).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("Al2O3(l)", 2500).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("Na2O(l)", 1800).authority is LiquidReferenceAuthority.CORROBORATED_CALCULATION
    assert liquid_reference_authority("Na2O(l)", 1800.001).authority is LiquidReferenceAuthority.CORROBORATED_CALCULATION
    assert liquid_reference_authority("TiO2(l)", 2000).authority is LiquidReferenceAuthority.EXTRAPOLATED
    tio2 = liquid_reference_authority("TiO2(l)", 2130)
    assert tio2.authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert tio2.certified_T_K is None
    assert tio2.band_value is None
    evidence = " ".join(item.basis for item in tio2.evidence_ref)
    assert "assumed fusion entropy 7.5 cal/mol/K by analogy with V2O4" in evidence
    assert "estimated liquid Cp 8 cal per g-atom" in evidence
    assert "no measured pure-liquid data" in evidence


def test_k2o_high_temperature_retains_mixture_scope_and_notice(packaged_gas_pack) -> None:
    outside = liquid_reference_authority(
        "K2O(l)", 2000, {"K2O_wt_pct": 15}
    )
    assert outside.authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert outside.certified_T_K == (1765, 2473)
    assert outside.band_value == pytest.approx(0.33)
    assert outside.band_unit == "dex log10 pK"
    assert outside.composition_scope == {"K2O_wt_pct": (25, 40)}
    assert outside.scope_applied
    assert outside.in_composition_scope is False
    assert "scored with notice" in outside.reason
    plante = outside.evidence_ref[0]
    assert plante.author == "Plante"
    assert "-0.13 dex at 25-40 wt% K2O" in plante.basis
    assert "-0.38 dex at 12-20 wt% K2O" in plante.basis

    outside_result = evaluate_gas(
        {"K2O": 1.0}, 2000, 1e-8, packaged_gas_pack,
        gas_species=("K", "K2O", "KO"),
        composition_wt_pct={"K2O": 15},
    )
    outside_codes = {notice.code for notice in outside_result.notices}
    assert "liquid-reference-source-construction-envelope" in outside_codes
    assert "liquid-reference-outside-evidence-composition-scope" in outside_codes
    scope_notice = next(
        notice for notice in outside_result.notices
        if notice.code == "liquid-reference-outside-evidence-composition-scope"
    )
    assert "K2O_wt_pct 25–40" in scope_notice.message
    assert "-0.38 dex at 12-20 wt% K2O" in scope_notice.message

    inside = liquid_reference_authority(
        "K2O(l)", 2000, {"K2O_wt_pct": 30}
    )
    assert inside.authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert inside.band_value == pytest.approx(0.33)
    assert inside.composition_scope == {"K2O_wt_pct": (25, 40)}
    assert inside.scope_applied
    assert inside.in_composition_scope is True
    inside_result = evaluate_gas(
        {"K2O": 1.0}, 2000, 1e-8, packaged_gas_pack,
        gas_species=("K", "K2O", "KO"),
        composition_wt_pct={"K2O": 30},
    )
    inside_codes = {notice.code for notice in inside_result.notices}
    assert "liquid-reference-source-construction-envelope" in inside_codes
    assert "liquid-reference-outside-evidence-composition-scope" not in inside_codes


def test_k2o_high_temperature_source_envelope_and_numeric_outputs(packaged_gas_pack) -> None:
    gas_pressures = {
        1500: (1.1833078684853735, 0.0012060311882495067, 2.6363248391966292e-05),
        1800: (24.393929477864923, 0.021214825586168343, 0.00042925631461085715),
        2200: (353.1100912216404, 0.24333402028900594, 0.0050043963016291985),
    }
    for temperature, expected in gas_pressures.items():
        result = evaluate_gas(
            {"K2O": 1.0}, temperature, 1e-8, packaged_gas_pack,
            gas_species=("K", "K2O", "KO"),
            composition_wt_pct={"K2O": 30.0},
        )
        assert tuple(result.values()) == pytest.approx(expected, rel=1e-12)
        status = result.liquid_reference_status["K"]
        if temperature == 1500:
            assert status.authority is LiquidReferenceAuthority.CERTIFIED
            assert status.certified_T_K == (1259, 1765)
            assert status.band_value == pytest.approx(0.13)
            assert status.in_composition_scope is True
            assert not result.notices
            continue

        assert status.authority is LiquidReferenceAuthority.EXTRAPOLATED
        assert status.certified_T_K == (1765, 2473)
        assert status.band_value == pytest.approx(0.33)
        assert status.band_unit == "dex log10 pK"
        evidence = " ".join(item.basis for item in status.evidence_ref)
        assert "NASA/TP-2002-211556" in evidence
        assert "Vol. IV, Book 1 p. 377" in evidence
        assert "Table 2 p. 154" in evidence
        assert "compilation bridge" in evidence
        assert "envelope_dex=-0.15/+0.33" in evidence
        assert all("Burcat" not in item.author for item in status.evidence_ref)
        notice, = [
            item for item in result.notices
            if item.code == "liquid-reference-source-construction-envelope"
        ]
        assert isinstance(notice, LiquidReferenceNotice)
        assert notice.status is status
        assert "Reason: source/construction spread is asymmetric" in notice.message
        assert "envelope_dex=-0.15/+0.33 log10 pK" in notice.message
        assert "symmetric outward bound" in notice.message
        assert "not a confidence interval" in notice.message
        assert any("band ±0.33 dex log10 pK" in (flag or "") for flag in result.domain_flags.values())

    expected_k2o_activities = {
        1800: 3.2742709993392502e-09,
        2200: 7.92765095987809e-08,
    }
    melt_pack = load_datapack()
    for temperature, expected in expected_k2o_activities.items():
        melt = evaluate(
            {"K2O": 30.0, "SiO2": 70.0}, temperature,
            melt_pack, basis_type="wt",
        )
        index = melt.parent_oxides.index("K2O")
        assert melt.parent_activity[index] == pytest.approx(expected, rel=1e-10)


def test_sodium_oxide_authority_windows_and_boundaries() -> None:
    expected = {
        1299: (LiquidReferenceAuthority.EXTRAPOLATED, (1405.2, 2473), None),
        1300: (LiquidReferenceAuthority.EXTRAPOLATED, (1300, 1405.2), 0.21),
        1405.2: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1405.2, 2473), 0.21),
        1800: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1405.2, 2473), 0.21),
        2473: (LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1405.2, 2473), 0.21),
        2474: (LiquidReferenceAuthority.EXTRAPOLATED, (1405.2, 2473), None),
    }
    for temperature, (authority, window, band) in expected.items():
        status = liquid_reference_authority("Na2O(l)", temperature)
        assert status.authority is authority
        assert status.certified_T_K == window
        assert status.band_value == band
        if band is not None:
            assert status.band_unit == "dex log10 pressure"
    evidence = " ".join(
        item.basis for item in liquid_reference_authority("Na2O(l)", 1800).evidence_ref
    )
    assert "partial-molar Cp" in evidence
    assert "not pure-liquid Cp" in evidence
    assert "47.7 +/- 2.5 kJ/mol" in evidence


def test_public_records_round_trip_and_gas_result_notice(packaged_gas_pack) -> None:
    records = liquid_reference_records()
    assert {record.species for record in records} == {
        "SiO2(l)", "K2O(l)", "CaO(l)", "MgO(l)", "FeO(l)",
        "Al2O3(l)", "TiO2(l)", "Na2O(l)",
    }
    for record in records:
        temperature = (
            sum(record.certified_T_K) / 2
            if record.certified_T_K is not None else 1000
        )
        resolved = liquid_reference_authority(record.species, temperature)
        assert resolved.certified_T_K == record.certified_T_K
        assert resolved.evidence_ref == record.evidence_ref
    pack = packaged_gas_pack
    certified = evaluate_gas({"SiO2": 1.0}, 1500, 1e-8, pack)
    assert all("SiO2(l)" not in (flag or "") for flag in certified.domain_flags.values())
    binary_alkali_silica = evaluate_gas(
        {"K2O": 0.333, "SiO2": 0.667}, 1500, 1e-8, pack
    )
    assert {status.species for status in binary_alkali_silica.liquid_reference_status.values()} == {
        "K2O(l)", "SiO2(l)"
    }
    assert all(
        not any(parent in (flag or "") for parent in ("MgO(l)", "CaO(l)", "Al2O3(l)"))
        for flag in binary_alkali_silica.domain_flags.values()
    )
    calcium = evaluate_gas({"CaO": 1.0}, 2000, 1e-8, pack)
    ca_status = next(s for s in calcium.liquid_reference_status.values() if s.species == "CaO(l)")
    assert ca_status.authority is LiquidReferenceAuthority.CERTIFIED
    assert ca_status.band_value == pytest.approx(0.07)
    assert not any(
        n.code == "liquid-reference-corroborated-calculation"
        and n.status.species == "CaO(l)"
        for n in calcium.notices
    )
    assert all("CaO(l)" not in (flag or "") for flag in calcium.domain_flags.values())
    alumina_calc = evaluate_gas({"Al2O3": 1.0}, 1900, 1e-8, pack)
    alumina_status = next(
        s for s in alumina_calc.liquid_reference_status.values()
        if s.species == "Al2O3(l)"
    )
    assert alumina_status.authority is LiquidReferenceAuthority.CORROBORATED_CALCULATION
    assert alumina_status.band_value == pytest.approx(0.05)
    assert not any("Al2O3(l)" in (flag or "") for flag in alumina_calc.domain_flags.values())
    alumina_notice, = [n for n in alumina_calc.notices
                       if n.code == "liquid-reference-corroborated-calculation"]
    assert alumina_notice.status.band_value == pytest.approx(0.05)
    alumina_measured = evaluate_gas({"Al2O3": 1.0}, 2200, 1e-8, pack)
    assert all("Al2O3(l)" not in (flag or "") for flag in alumina_measured.domain_flags.values())
    corroborated_status = CertificationStatus(
        "Synthetic(l)",
        LiquidReferenceAuthority.CORROBORATED_CALCULATION,
        (1, 2),
        0.07,
        "dex pO2",
        (),
        None,
        False,
        "independent calculation",
    )
    corroborated = ImccGasResult(
        {}, "bar", {}, {}, liquid_reference_status={"Synthetic": corroborated_status}
    )
    notice = corroborated.notices[0]
    assert notice.code == "liquid-reference-corroborated-calculation"
    assert notice.status is corroborated_status
    assert notice.status.band_value == pytest.approx(0.07)
    extrapolated = evaluate_gas({"MgO": 1.0}, 2000, 1e-8, pack)
    assert any(
        status.species == "MgO(l)"
        and status.authority is LiquidReferenceAuthority.EXTRAPOLATED
        for status in extrapolated.liquid_reference_status.values()
    )
    assert any("MgO(l)" in (flag or "") for flag in extrapolated.domain_flags.values())
    assert any("±0.76 dex pO2" in (flag or "") for flag in extrapolated.domain_flags.values())
    potassium_no_composition = evaluate_gas({"K2O": 1.0}, 1500, 1e-8, pack)
    no_composition_status = next(
        status for status in potassium_no_composition.liquid_reference_status.values()
        if status.species == "K2O(l)"
    )
    assert not no_composition_status.scope_applied
    assert no_composition_status.in_composition_scope is None
    assert any("K2O(l)" in (flag or "") for flag in potassium_no_composition.domain_flags.values())
    potassium_in_scope = evaluate_gas(
        {"K2O": 1.0}, 1500, 1e-8, pack,
        composition_wt_pct={"K2O": 30.0},
    )
    in_scope_status = next(
        status for status in potassium_in_scope.liquid_reference_status.values()
        if status.species == "K2O(l)"
    )
    assert in_scope_status.scope_applied
    assert in_scope_status.authority is LiquidReferenceAuthority.CERTIFIED
    assert all("K2O(l)" not in (flag or "") for flag in potassium_in_scope.domain_flags.values())
    potassium_out_scope = evaluate_gas(
        {"K2O": 1.0}, 1500, 1e-8, pack,
        composition_wt_pct={"K2O": 20.0},
    )
    out_scope_status = next(
        status for status in potassium_out_scope.liquid_reference_status.values()
        if status.species == "K2O(l)"
    )
    assert out_scope_status.authority is LiquidReferenceAuthority.CERTIFIED
    assert out_scope_status.in_composition_scope is False
    assert all("K2O(l)" not in (flag or "") for flag in potassium_out_scope.domain_flags.values())
    scope_notice = next(
        notice for notice in potassium_out_scope.notices
        if notice.code == "liquid-reference-outside-evidence-composition-scope"
    )
    assert "outside the evidence range K2O_wt_pct 25–40" in scope_notice.message
    assert "Plante (1979)" in scope_notice.message
    assert "-0.38 dex at 12-20 wt% K2O" in scope_notice.message


def test_halide_side_parent_uses_liquid_authority_lookup(packaged_gas_pack) -> None:
    pack = packaged_gas_pack
    result = evaluate_gas(
        {"Na2O": 1.0, "Cl2": 1.0}, 1600, 1e-8, pack,
        parent_oxides=("Na2O", "Cl2"), gas_species=("NaCl",),
        composition_wt_pct={"Na2O": 30.0},
    )
    status = result.liquid_reference_status["NaCl"]
    assert status.species == "Na2O(l)"
    assert status.authority is LiquidReferenceAuthority.CORROBORATED_CALCULATION
    assert status.band_value == pytest.approx(0.21)
    assert result.domain_flags["NaCl"] is None


def test_extrapolated_window_and_corroborated_window_notice(packaged_gas_pack) -> None:
    pack = packaged_gas_pack
    extrapolated = evaluate_gas(
        {"SiO2": 1.0}, 1000, 1e-8, pack,
        gas_species=("SiO",), allow_extrapolation=True,
    )
    assert extrapolated.liquid_reference_status["SiO"].authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert extrapolated.liquid_reference_status["SiO"].band_value == pytest.approx(0.18)
    assert "outside declared G(T) interval" in extrapolated.domain_flags["SiO"]

    corroborated = evaluate_gas({"SiO2": 1.0}, 1300, 1e-8, pack)
    assert corroborated.liquid_reference_status["SiO2"].authority is LiquidReferenceAuthority.CORROBORATED_CALCULATION
    assert any(
        notice.code == "liquid-reference-corroborated-calculation"
        and notice.status.species == "SiO2(l)"
        for notice in corroborated.notices
    )


def test_window_band_does_not_apply_outside_every_window() -> None:
    below = liquid_reference_authority("Al2O3(l)", 1200)
    assert below.authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert below.band_value is None and below.band_unit is None
    mgo = liquid_reference_authority("MgO(l)", 1000)
    assert mgo.authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert mgo.band_value is None and mgo.band_unit is None


def test_mgo_extrapolated_source_band_window() -> None:
    for temperature in (1323, 1800, 2473):
        status = liquid_reference_authority("MgO(l)", temperature)
        assert status.authority is LiquidReferenceAuthority.EXTRAPOLATED
        assert status.certified_T_K == (1323, 2473)
        assert status.band_value == pytest.approx(0.76)
        assert status.band_unit == "dex pO2"

    for temperature in (1000, 1322.999, 2473.001, 3000):
        status = liquid_reference_authority("MgO(l)", temperature)
        assert status.authority is LiquidReferenceAuthority.EXTRAPOLATED
        assert status.band_value is None and status.band_unit is None


def test_tio2_extrapolated_cross_assessment_window() -> None:
    for temperature in (2185, 2300, 2473):
        status = liquid_reference_authority("TiO2(l)", temperature)
        assert status.authority is LiquidReferenceAuthority.EXTRAPOLATED
        assert status.certified_T_K == (2185, 2473)
        assert status.band_value == pytest.approx(0.11)
        assert status.band_unit == "dex log10 pressure"
        evidence = " ".join(item.basis for item in status.evidence_ref)
        assert "cross-assessment envelope" in evidence
        assert "one-sided" in evidence
        assert "not a measurement or confidence interval" in evidence

    for temperature in (2000, 2184.999, 2473.001):
        status = liquid_reference_authority("TiO2(l)", temperature)
        assert status.authority is LiquidReferenceAuthority.EXTRAPOLATED
        assert status.band_value is None and status.band_unit is None


def test_feo_authority_window_and_band_are_unchanged() -> None:
    inside = liquid_reference_authority("FeO(l)", 1700)
    assert inside.authority is LiquidReferenceAuthority.CERTIFIED
    assert inside.certified_T_K == (1649.15, 1766.15)
    assert inside.band_value is None and inside.band_unit is None

    for temperature in (1649.149, 1766.151):
        outside = liquid_reference_authority("FeO(l)", temperature)
        assert outside.authority is LiquidReferenceAuthority.EXTRAPOLATED
        assert outside.band_value is None and outside.band_unit is None


def test_feo_liquid_standard_declares_profile_row_and_condensate_digest() -> None:
    expected = {
        "openimcc": "DG46-MASSON65",
        "published": "Fe-019",
    }
    for profile, standard_id in expected.items():
        standard = feo_liquid_standard(profile)
        assert isinstance(standard, FeOLiquidStandard)
        assert standard.standard_id == standard_id
        assert standard.certified_T_K == (1649.15, 1766.15)
        assert standard.source_citations
        result = evaluate_gas({"FeO": 1.0}, 1700.0, 1e-10, profile=profile)
        assert standard.condensate_table_digest == result.condensate_table_digest

    assert feo_liquid_standard().standard_id == "DG46-MASSON65"


def test_feo_liquid_standard_offsets_match_anchor_table_targets() -> None:
    # The candidate row is a fit to the three D&G targets. The documented
    # candidate-minus-target residuals are +47, -97, and +50 J/mol; adding
    # table rounding requires just over 97 J/mol, so use the outward 100 J/mol
    # tolerance rather than treating the rounded target as an exact row value.
    expected_J_per_mol = {
        1649.15: 9654.6,
        1709.15: 9140.8,
        1766.15: 8369.6,
    }
    for temperature, target in expected_J_per_mol.items():
        offset = feo_liquid_standard_offset(temperature)
        assert isinstance(offset, FeOLiquidStandardOffset)
        assert offset.value_J_per_mol == pytest.approx(target, abs=100.0)
        assert offset.authority == "anchored"


def test_feo_liquid_standard_offset_is_zero_for_janaf_profile() -> None:
    for temperature in (1300.0, 1649.15, 1766.15, 2500.0):
        offset = feo_liquid_standard_offset(temperature, profile="published")
        assert offset.value_J_per_mol == pytest.approx(0.0, abs=1e-9)


def test_feo_liquid_standard_offset_is_continuous_and_labels_authority() -> None:
    temperatures = tuple(float(value) for value in range(1300, 2501, 50))
    offsets = [feo_liquid_standard_offset(value) for value in temperatures]
    values = [offset.value_J_per_mol for offset in offsets]

    assert all(
        offset.authority == (
            "anchored" if 1649.15 <= temperature <= 1766.15 else "extrapolated"
        )
        for temperature, offset in zip(temperatures, offsets)
    )
    assert all(right < left for left, right in zip(values, values[1:]))
    assert all(abs(right - left) < 1000.0 for left, right in zip(values, values[1:]))
    assert feo_liquid_standard_offset(1649.15).authority == "anchored"
    assert feo_liquid_standard_offset(1766.15).authority == "anchored"


def test_authority_requires_the_source_row_its_evidence_assessed() -> None:
    feo = liquid_reference_authority(
        "FeO(l)", 1700, selected_source_id="DG46-MASSON65"
    )
    assert feo is not None
    assert feo.authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority(
        "FeO(l)", 1700, selected_source_id="Fe-019"
    ) is None
    assert liquid_reference_authority(
        "Al2O3(l)", 2500, selected_source_id="LAM1987"
    ) is None


def test_evidence_dois_are_structured_without_changing_identifiers() -> None:
    old_doi_strings = (
        "DOI 10.1016/0016-7037(82)90383-0",
        "DOI 10.1111/j.1151-2916.1984.tb19155.x",
        "DOI 10.1063/1.5141841",
        "DOI 10.1021/acs.jpcc.4c04752",
        "DOI 10.1021/ja01209a030 and 10.1098/rspa.1965.0176",
        "DOI 10.1016/j.oceram.2026.101029",
        "DOI 10.1111/j.1151-2916.1984.tb19155.x",
        "DOI 10.18434/T42S31",
        "DOI 10.1016/j.jct.2017.04.003",
        "DOI 10.1002/2016JE005059",
        "DOI 10.1002/2016JE005059",
        "DOI 10.1111/j.1151-2916.1992.tb04216.x",
        "DOI 10.1143/JJAP.43.1496",
        "DOI 10.1016/j.oceram.2026.101029",
        "DOI 10.1063/1.555706",
        "DOI 10.1016/j.oceram.2026.101029",
        "DOI 10.1111/j.1151-2916.1992.tb04216.x",
        "DOI 10.1143/JJAP.43.1496",
        "DOI 10.1016/j.oceram.2026.101029",
        "DOI 10.1016/S0022-0248(02)01438-0",
        "DOI 10.18434/T42S31",
        "DOI 10.1016/j.jct.2017.04.003",
    )
    expected_dois = [
        doi
        for old_string in old_doi_strings
        for doi in re.findall(r"10[.]\d{4,9}/[^\s;]+", old_string)
    ]
    references = []
    for record in liquid_reference_records():
        references.extend(record.evidence_ref)
        for window in record.additional_windows:
            references.extend(window.evidence_ref)
    structured_dois = [
        doi for evidence in references for doi in (evidence.doi or ())
    ]

    assert Counter(structured_dois) == Counter(expected_dois)
    assert all("DOI " not in evidence.basis for evidence in references)
    assert {evidence.locator for evidence in references if evidence.locator} == {
        "run 17c3-2", "NASA/TP-2002-211556",
        "Vol. IV, Book 1 p. 377; Book 2 p. 406",
        "Table 2 p. 154",
    }


def test_citation_text_with_doi_has_structured_doi() -> None:
    references = []
    for record in liquid_reference_records():
        references.extend(record.evidence_ref)
        for window in record.additional_windows:
            references.extend(window.evidence_ref)
    references.extend(feo_liquid_standard("published").source_citations)

    for evidence in references:
        citation = f"{evidence.author} ({evidence.year}): {evidence.basis}"
        if "doi" in citation.lower():
            assert evidence.doi, citation
