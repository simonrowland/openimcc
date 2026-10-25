from __future__ import annotations

import pytest

from openimcc import (
    CertificationStatus,
    AuthorityWindow,
    ImccGasResult,
    LiquidReferenceAuthority,
    LiquidReferenceRecord,
    evaluate_gas,
    liquid_reference_authority,
    liquid_reference_records,
    load_gas_datapack,
)


def test_authority_enum_and_inclusive_window_boundaries() -> None:
    with pytest.raises(ValueError, match="unknown liquid-reference authority"):
        LiquidReferenceRecord("X(l)", "unknown", None, None, None, ())  # type: ignore[arg-type]
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
        1801: LiquidReferenceAuthority.EXTRAPOLATED,
        2120: LiquidReferenceAuthority.CORROBORATED_CALCULATION,
        2326: LiquidReferenceAuthority.CORROBORATED_CALCULATION,
        2327: LiquidReferenceAuthority.CERTIFIED,
        2500: LiquidReferenceAuthority.CERTIFIED,
        3000: LiquidReferenceAuthority.CERTIFIED,
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
    assert certified.certified_T_K == (2327, 3000)
    assert certified.band_value == pytest.approx(0.036)


def test_alumina_low_temperature_extrapolation_envelope() -> None:
    expected = {
        1322: (LiquidReferenceAuthority.EXTRAPOLATED, (2327, 3000), None),
        1323: (LiquidReferenceAuthority.EXTRAPOLATED, (1323, 1801), 0.17),
        1800: (LiquidReferenceAuthority.EXTRAPOLATED, (1323, 1801), 0.17),
        1801: (LiquidReferenceAuthority.EXTRAPOLATED, (1323, 1801), 0.17),
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
            (1, 3), None, None, evidence, (extra,))
    first = AuthorityWindow(LiquidReferenceAuthority.CERTIFIED, (1, 3), None, None, evidence)
    second = AuthorityWindow(LiquidReferenceAuthority.CERTIFIED, (2, 4), None, None, evidence)
    with pytest.raises(ValueError, match="windows overlap"):
        LiquidReferenceRecord("X(l)", LiquidReferenceAuthority.EXTRAPOLATED,
            None, None, None, evidence, (first, second))


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
    assert liquid_reference_authority("FeO(l)", 1649).authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert liquid_reference_authority("FeO(l)", 1649.15).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("Al2O3(l)", 2400).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("Al2O3(l)", 2500).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("Na2O(l)", 1800).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("Na2O(l)", 1800.001).authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert liquid_reference_authority("TiO2(l)", 2000).authority is LiquidReferenceAuthority.EXTRAPOLATED
    tio2 = liquid_reference_authority("TiO2(l)", 2130)
    assert tio2.authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert tio2.certified_T_K is None
    assert tio2.band_value is None
    evidence = " ".join(item.basis for item in tio2.evidence_ref)
    assert "assumed fusion entropy 7.5 cal/mol/K by analogy with V2O4" in evidence
    assert "estimated liquid Cp 8 cal per g-atom" in evidence
    assert "no measured pure-liquid data" in evidence


def test_public_records_round_trip_and_gas_result_notice() -> None:
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
    pack = load_gas_datapack()
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
    assert any("±0.3 dex pO2" in (flag or "") for flag in extrapolated.domain_flags.values())
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


def test_halide_side_parent_uses_liquid_authority_lookup() -> None:
    pack = load_gas_datapack()
    result = evaluate_gas(
        {"Na2O": 1.0, "Cl2": 1.0}, 1600, 1e-8, pack,
        parent_oxides=("Na2O", "Cl2"), gas_species=("NaCl",),
        composition_wt_pct={"Na2O": 30.0},
    )
    status = result.liquid_reference_status["NaCl"]
    assert status.species == "Na2O(l)"
    assert status.authority is LiquidReferenceAuthority.CERTIFIED
    assert result.domain_flags["NaCl"] is None


def test_extrapolated_window_and_corroborated_window_notice() -> None:
    pack = load_gas_datapack()
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
    mgo = liquid_reference_authority("MgO(l)", 1700)
    assert mgo.band_value == 0.3
