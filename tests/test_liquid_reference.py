from __future__ import annotations

import pytest

from openimcc import (
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
    corroborated = evaluate_gas({"CaO": 1.0}, 2000, 1e-8, pack)
    status = next(
        status for status in corroborated.liquid_reference_status.values()
        if status.species == "CaO(l)"
    )
    assert status.authority is LiquidReferenceAuthority.CORROBORATED_CALCULATION
    notice = next(n for n in corroborated.notices if n.code == "liquid-reference-corroborated-calculation")
    assert notice.status is status
    assert status.band_value == pytest.approx(0.07)
    extrapolated = evaluate_gas({"MgO": 1.0}, 2000, 1e-8, pack)
    assert any(
        status.species == "MgO(l)"
        and status.authority is LiquidReferenceAuthority.EXTRAPOLATED
        for status in extrapolated.liquid_reference_status.values()
    )
    assert any("MgO(l)" in (flag or "") for flag in extrapolated.domain_flags.values())
    assert any("±0.3 dex pO2" in (flag or "") for flag in extrapolated.domain_flags.values())
    potassium = evaluate_gas({"K2O": 1.0}, 1500, 1e-8, pack)
    k_status = next(
        status for status in potassium.liquid_reference_status.values()
        if status.species == "K2O(l)"
    )
    assert not k_status.scope_applied
    assert any("K2O(l)" in (flag or "") for flag in potassium.domain_flags.values())
