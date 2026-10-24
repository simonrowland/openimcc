from __future__ import annotations

import pytest

from openimcc import (
    CertificationStatus,
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
    assert liquid_reference_authority("SiO2(l)", 900).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("SiO2(l)", 1996).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("SiO2(l)", 899.999).authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert liquid_reference_authority("SiO2(l)", 1996.001).authority is LiquidReferenceAuthority.EXTRAPOLATED


def test_composition_scoped_k2o_authority() -> None:
    in_scope = liquid_reference_authority("K2O(l)", 1500, {"K2O_wt_pct": 25})
    out_scope = liquid_reference_authority("K2O(l)", 1500, {"K2O_wt_pct": 20})
    no_composition = liquid_reference_authority("K2O(l)", 1500)
    assert in_scope.authority is LiquidReferenceAuthority.CERTIFIED
    assert in_scope.scope_applied
    assert out_scope.authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert no_composition.authority is LiquidReferenceAuthority.CERTIFIED
    assert not no_composition.scope_applied
    assert no_composition.composition_scope == {"K2O_wt_pct": (25, 40)}
    assert liquid_reference_authority("K2O(l)", 1500, {"K2O_wt_pct": 40}).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("K2O(l)", 1500, {"K2O_wt_pct": 40.001}).authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert liquid_reference_authority("FeO(l)", 1649).authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert liquid_reference_authority("FeO(l)", 1649.15).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("Al2O3(l)", 2400).authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert liquid_reference_authority("Al2O3(l)", 2500).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("Na2O(l)", 1800).authority is LiquidReferenceAuthority.CERTIFIED
    assert liquid_reference_authority("Na2O(l)", 1800.001).authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert liquid_reference_authority("TiO2(l)", 2000).authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert liquid_reference_authority("TiO2(l)", 2130).authority is LiquidReferenceAuthority.CERTIFIED


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
    calcium = evaluate_gas({"CaO": 1.0}, 2000, 1e-8, pack)
    ca_status = next(s for s in calcium.liquid_reference_status.values() if s.species == "CaO(l)")
    assert ca_status.authority is LiquidReferenceAuthority.CERTIFIED
    assert ca_status.band_value == pytest.approx(0.07)
    assert not any(n.code == "liquid-reference-corroborated-calculation" for n in calcium.notices)
    assert all("CaO(l)" not in (flag or "") for flag in calcium.domain_flags.values())
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
    assert out_scope_status.authority is LiquidReferenceAuthority.EXTRAPOLATED
    assert any("±0.13 dex pO2" in (flag or "") for flag in potassium_out_scope.domain_flags.values())


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


def test_certified_window_keeps_condensate_interval_flag() -> None:
    pack = load_gas_datapack()
    result = evaluate_gas(
        {"SiO2": 1.0}, 1000, 1e-8, pack,
        gas_species=("SiO",), allow_extrapolation=True,
    )
    assert result.liquid_reference_status["SiO"].authority is LiquidReferenceAuthority.CERTIFIED
    assert "outside declared G(T) interval" in result.domain_flags["SiO"]
