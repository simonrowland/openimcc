"""Runtime gates for the packaged IMCC-SF04 gas mass-action layer."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from openimcc.gas import (
    IMCC_GAS_CHANNEL_SPECIES,
    IMCC_GAS_INCOMPLETE_PARENT_SPECIES,
    IMCC_GAS_NO_JANAF_ROWS,
    IMCC_GAS_UNAVAILABLE_SPECIES,
    IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS,
    IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES,
    IMCC_SF04_WORKBOOK_GRID_K,
    ImccGasDatapack,
    ImccGasInvalidFugacityError,
    ImccGasResult,
    ImccGasSpeciesNotFoundError,
    ImccGasTemperatureOutsideDomainError,
    default_condensate_database_path,
    default_gas_database_path,
    evaluate_gas,
    gas_species_provenance,
    load_gas_datapack,
)


# These are independent lg10(p/bar) references, rounded to three decimals.
# For every gas record, the source-side calculation uses the JANAF record's
# printed ``formation_enthalpy`` at 298.15 K, ``enthalpy_increment`` and
# ``entropy`` at the test temperature; it never reads gas-shomate.csv.  The
# parent side uses the named shipped condensate.csv row.  The source mappings
# are: Na-005+Na2O(l), K-005+K2O(l), O-012+SiO2(l), Fe-008+FeO(l),
# Fe-021+FeO(l), Mg-005+MgO(l), Mg-011+MgO(l), O-040+SiO2(l), and O-029
# for the caller-pinned O2 reference.
#
# Premise: at unit parent activity and fO2, the reaction equilibrium relation is
# log10(p_gas) = -(n_gas*G_gas + n_O2*G_O2 - G_parent) /
# (n_gas*R*T*ln(10)). Algebra: source kJ/mol values are converted to J/mol
# before the stoichiometric sum. Unit check: the denominator is J/mol. Sanity:
# SiO2(l) -> SiO2(g) at 2000 K rounds to -6.669 (n_O2 = 0); SiO2(l) ->
# SiO(g) + 1/2 O2(g) rounds to -7.759.
_RUNG2B_AGAINST_JANAF_PRINTED_GAS_COLUMNS = {
    2000.0: {
        "Na": -1.890,
        "K": 0.333,
        "SiO": -7.759,
        "Fe": -7.307,
        "FeO": -5.465,
        "Mg": -7.856,
        "MgO": -7.177,
        "SiO2": -6.669,
        "O2": 0.0,
    }
}

# T-625 source mappings are Al-074/Al-077/Al-092/Al-094/Al-005 plus
# Al2O3(l), Na-011/Na-008 plus Na2O(l), K-011/K-008 plus K2O(l),
# Si-005 plus SiO2(l), Ca-030/Ca-006 plus CaO(l), and O-001 without a parent.
_T625_AGAINST_JANAF_PRINTED_GAS_COLUMNS = {
    2500.0: {
        "O": -1.839,
        "AlO": -5.865,
        "AlO2": -5.932,
        "Al2O": -9.964,
        "Al2O2": -8.422,
        "Na2": -3.725,
        "NaO": -1.414,
        "K2": -0.030,
        "KO": 0.534,
        "Si": -11.905,
        "Al": -8.695,
        "CaO": -5.475,
        "Ca": -6.258,
    }
}


@pytest.fixture(scope="module")
def gas_pack() -> ImccGasDatapack:
    return load_gas_datapack()


@pytest.fixture
def unit_activities() -> dict[str, float]:
    return {
        "SiO2": 1.0,
        "MgO": 1.0,
        "FeO": 1.0,
        "CaO": 1.0,
        "Al2O3": 1.0,
        "TiO2": 1.0,
        "Na2O": 1.0,
        "K2O": 1.0,
    }


def test_default_tables_are_packaged_and_load_without_environment(
    monkeypatch: pytest.MonkeyPatch, gas_pack: ImccGasDatapack
) -> None:
    monkeypatch.delenv("OPENIMCC_VAPOROCK_ROOT", raising=False)
    gas_path = default_gas_database_path()
    oxide_path = default_condensate_database_path()
    assert gas_path.name == "gas-shomate.csv"
    assert oxide_path.name == "condensate.csv"
    assert gas_path.is_file()
    assert oxide_path.is_file()
    assert len(gas_pack.gas_df) == 22
    assert len(gas_pack.oxide_df) == 8


def test_explicit_vaporock_override_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
    gas_pack: ImccGasDatapack,
    tmp_path: Path,
) -> None:
    root = tmp_path / "vaporock"
    gas_target = root / "src" / "vaporock" / "data"
    oxide_target = root / "data"
    gas_target.mkdir(parents=True)
    oxide_target.mkdir(parents=True)
    shutil.copy2(gas_pack.gas_path, gas_target / "JANAF-vapor-data-full.csv")
    shutil.copy2(gas_pack.oxide_path, oxide_target / "condensate-thermo-data.csv")
    monkeypatch.setenv("OPENIMCC_VAPOROCK_ROOT", str(root))

    overridden = load_gas_datapack()
    assert overridden.gas_path == gas_target / "JANAF-vapor-data-full.csv"
    assert overridden.oxide_path == oxide_target / "condensate-thermo-data.csv"
    assert overridden.gas_df.equals(gas_pack.gas_df)
    assert overridden.oxide_df.equals(gas_pack.oxide_df)


def test_runtime_schemas_and_intervals_are_unchanged(gas_pack: ImccGasDatapack) -> None:
    assert tuple(gas_pack.gas_df.columns) == (
        "state",
        "T_interval",
        "cation",
        "cat_num",
        "oxy_num",
        "T_min",
        "T_max",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "Ref",
    )
    assert tuple(gas_pack.oxide_df.columns) == (
        "state",
        "cation",
        "cat_num",
        "oxy_num",
        "T_min",
        "T_max",
        "dH298_R",
        "dG_A",
        "dG_B",
        "dG_C",
        "dG_D",
        "dG_E",
        "Ref",
    )
    assert set(gas_pack.gas_df.index) == {
        f"{species}(g)" for species in IMCC_GAS_CHANNEL_SPECIES
    }
    assert (gas_pack.gas_df["T_min"] == 1500).all()
    assert (gas_pack.gas_df["T_max"] == 3000).all()


def test_channel_coverage_ledger_is_closed() -> None:
    implemented = set(IMCC_GAS_CHANNEL_SPECIES)
    unavailable = set(IMCC_GAS_UNAVAILABLE_SPECIES)
    in_domain = set(IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES)
    extrapolated = set(IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS)
    assert len(implemented) == 22
    assert len(unavailable) == 10
    assert implemented.isdisjoint(unavailable)
    assert in_domain.isdisjoint(extrapolated)
    assert in_domain | extrapolated == implemented


_EXTRAPOLATION_REFUSAL_CASES = (
    ("SiO", 1900.0, "SiO2(l)"),
    ("Mg", 2500.0, "MgO(l)"),
    ("MgO", 2500.0, "MgO(l)"),
    ("SiO2", 1900.0, "SiO2(l)"),
    ("AlO", 2000.0, "Al2O3(l)"),
    ("AlO2", 2000.0, "Al2O3(l)"),
    ("Al2O", 2000.0, "Al2O3(l)"),
    ("Al2O2", 2000.0, "Al2O3(l)"),
    ("Si", 1900.0, "SiO2(l)"),
    ("Al", 2000.0, "Al2O3(l)"),
    ("CaO", 2500.0, "CaO(l)"),
    ("Ca", 2500.0, "CaO(l)"),
)


@pytest.mark.parametrize(
    ("species", "temperature", "source_species"), _EXTRAPOLATION_REFUSAL_CASES
)
def test_parent_domain_gaps_refuse_with_typed_errors(
    gas_pack: ImccGasDatapack,
    unit_activities: dict[str, float],
    species: str,
    temperature: float,
    source_species: str,
) -> None:
    assert set(IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS) == {
        case[0] for case in _EXTRAPOLATION_REFUSAL_CASES
    }
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as exc:
        evaluate_gas(
            unit_activities,
            temperature,
            1.0,
            gas_pack,
            gas_species=(species,),
            allow_extrapolation=False,
        )
    assert exc.value.code == "imcc_gas_T_outside_domain"
    assert source_species in str(exc.value)


def test_gas_domain_refusal_is_typed(gas_pack: ImccGasDatapack) -> None:
    """A selected gas-row gap still refuses before evaluating the reaction."""
    gas_df = gas_pack.gas_df.copy()
    # Keep O2 available at the diagnostic temperature so the refusal names the
    # selected Fe(g) row rather than the caller-pinned reference row.
    gas_df.loc["O2(g)", "T_min"] = 1000.0
    diagnostic_pack = ImccGasDatapack(
        gas_df=gas_df,
        oxide_df=gas_pack.oxide_df,
        gas_path=gas_pack.gas_path,
        oxide_path=gas_pack.oxide_path,
    )
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as exc:
        evaluate_gas(
            {"FeO": 1.0},
            1400.0,
            fO2=1.0,
            datapack=diagnostic_pack,
            gas_species=("Fe",),
            allow_extrapolation=False,
        )
    assert exc.value.code == "imcc_gas_T_outside_domain"
    assert "Fe(g)" in str(exc.value)


def test_sf04_basalt_at_1800_predicts_and_flags_source_rows(
    gas_pack: ImccGasDatapack,
) -> None:
    from openimcc import evaluate as evaluate_imcc
    from openimcc import load_datapack

    composition = {
        "SiO2": 51.85068,
        "MgO": 4.78527,
        "FeO": 13.77307,
        "CaO": 9.02862,
        "Al2O3": 14.80572,
        "TiO2": 1.73824,
        "Na2O": 3.23108,
        "K2O": 0.78732,
    }
    imcc_pack = load_datapack(Path("src/openimcc/data/packs/imcc-sf04-v1.0.2.json"))
    imcc_result = evaluate_imcc(
        composition,
        T_K=1800.0,
        pack=imcc_pack,
        basis_type="wt",
        allow_extrapolation=True,
    )
    activities = dict(zip(imcc_result.parent_oxides, imcc_result.parent_activity))

    result = evaluate_gas(activities, 1800.0, 1.0e-10, gas_pack)

    assert isinstance(result, ImccGasResult)
    assert set(result) == set(IMCC_GAS_CHANNEL_SPECIES)
    assert all(math.isfinite(value) and value >= 0.0 for value in result.values())
    flagged = {
        "SiO",
        "SiO2",
        "Si",
        "Mg",
        "MgO",
        "CaO",
        "Ca",
        "AlO",
        "AlO2",
        "Al2O",
        "Al2O2",
        "Al",
    }
    assert all(result.domain_flags[species] is not None for species in flagged)
    assert all(
        result.domain_flags[species] is None
        for species in set(result) - flagged
    )
    assert "MgO(l)" in result.domain_flags["Mg"]
    assert "[3100, 3500] K" in result.domain_flags["Mg"]
    assert "T=1800.0 K" in result.domain_flags["SiO"]


def test_o2_pin_is_never_domain_flagged_at_3200_k(
    gas_pack: ImccGasDatapack,
    unit_activities: dict[str, float],
) -> None:
    o2_only = evaluate_gas(
        unit_activities,
        3200.0,
        1.0e-8,
        gas_pack,
        gas_species=("O2",),
    )
    full = evaluate_gas(unit_activities, 3200.0, 1.0e-8, gas_pack)

    assert o2_only.domain_flags["O2"] is None
    assert full.domain_flags["O2"] is None
    # The same O2(g) row remains visible on reactions that actually consume it.
    assert "O2(g)" in full.domain_flags["SiO"]


def test_gas_result_is_frozen_bar_mapping_and_preserves_numbers(
    gas_pack: ImccGasDatapack,
) -> None:
    activities = {
        "SiO2": 1.0,
        "MgO": 1.0,
        "FeO": 1.0,
        "CaO": 1.0,
        "Al2O3": 1.0,
        "Na2O": 1.0,
        "K2O": 1.0,
    }
    result = evaluate_gas(
        activities,
        2000.0,
        1.0,
        gas_pack,
        allow_extrapolation=True,
        gas_species=("Na", "K", "SiO", "O2"),
    )

    assert result.unit == "bar"
    assert dict(result) == {
        "Na": 0.012891128142910052,
        "K": 2.1523860349175448,
        "SiO": 1.741275855555553e-08,
        "O2": 1.0,
    }
    assert result.domain_flags["Na"] is None
    assert result.provenance_class["Na"] == "lam1984_transcribed"
    with pytest.raises(TypeError):
        result.domain_flags["Na"] = "mutated"  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        result.unit = "Pa"  # type: ignore[misc]


def test_negative_fugacity_refuses_with_bar_relative_hint(
    gas_pack: ImccGasDatapack,
) -> None:
    with pytest.raises(
        ImccGasInvalidFugacityError,
        match=r"p_O2/p°.*bar-relative.*not log10 fO2",
    ) as exc:
        evaluate_gas(
            {"Na2O": 1.0},
            1800.0,
            -8.0,
            gas_pack,
            gas_species=("Na",),
        )
    assert exc.value.code == "imcc_gas_invalid_fO2"


def test_full_workbook_grid_runs_for_in_domain_channels(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    for species in IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES:
        for temperature in IMCC_SF04_WORKBOOK_GRID_K:
            result = evaluate_gas(
                unit_activities,
                temperature,
                1.0,
                gas_pack,
                gas_species=(species,),
            )
            assert result[species] >= 0.0


def test_all_channels_compute_when_extrapolation_is_explicit(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    result = evaluate_gas(
        unit_activities,
        2500.0,
        1.0e-10,
        gas_pack,
        allow_extrapolation=True,
    )
    assert set(result) == set(IMCC_GAS_CHANNEL_SPECIES)
    assert all(value >= 0.0 for value in result.values())
    assert result["O2"] == 1.0e-10


def test_p1_g1_against_janaf_printed_gas_columns(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    """The 2000 K rung-2b sample is independent of the fitted gas rows."""
    T = 2000.0
    pressures = evaluate_gas(
        unit_activities, T, fO2=1.0, datapack=gas_pack, allow_extrapolation=True
    )
    # Gate derivation: half the last stored digit is 0.0005 dex; the maximum
    # evaluate_gas residual against the unrounded reference is 5.2e-5 dex,
    # leaving 0.000448 dex of margin to 0.001 dex. Today's worst case is
    # 0.000508 dex (Si), so 0.0005 dex would false-fail. At 2000 K, 0.001 dex
    # is 38.3 J/mol on G_gas. These tests prove the evaluate_gas assembly and
    # stoichiometry path, including the condensate parent; they are not an
    # independent fit (G1 covers the fit).
    worst = 0.0
    nonzero_errors = 0
    for species, expected_log10 in _RUNG2B_AGAINST_JANAF_PRINTED_GAS_COLUMNS[T].items():
        actual_log10 = math.log10(pressures[species])
        error = abs(actual_log10 - expected_log10)
        worst = max(worst, error)
        nonzero_errors += error > 0.0
        assert error <= 0.001, (
            f"{species} at {T} K: |{actual_log10:.6f} - {expected_log10:.6f}| "
            f"= {error:.6f} dex > 0.001 dex"
        )
    assert nonzero_errors >= 1
    assert worst <= 0.001


def test_t625_against_janaf_printed_gas_columns(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    """The T-625 additions agree with independently rounded printed lgK."""
    T = 2500.0
    refs = _T625_AGAINST_JANAF_PRINTED_GAS_COLUMNS[T]
    pressures = evaluate_gas(
        unit_activities,
        T,
        fO2=1.0,
        datapack=gas_pack,
        allow_extrapolation=True,
        gas_species=tuple(refs),
    )
    assert set(pressures) == set(refs)
    for species, expected_log10 in refs.items():
        actual_log10 = math.log10(pressures[species])
        error = abs(actual_log10 - expected_log10)
        assert 0.0 < error <= 0.001, (
            f"{species} at {T} K: independently rounded reference must have "
            f"nonzero error <= 0.001 dex, got {error:.6f}"
        )


def test_kp_derivation_spot_check_sio(gas_pack: ImccGasDatapack) -> None:
    """Recompute SiO2(l) -> SiO(g) + 1/2 O2(g) at 2000 K by hand."""
    R = 8.314462618
    T = 2000.0
    G_sio2_l = -1129878.93
    G_sio_g = -593626.03
    G_o2_g = -478320.38
    dG = G_sio_g + 0.5 * G_o2_g - G_sio2_l
    Kp_hand = math.exp(-dG / (R * T))

    p_sio = evaluate_gas(
        {"SiO2": 1.0}, T, fO2=1.0, datapack=gas_pack, allow_extrapolation=True
    )["SiO"]

    # The new fit's measured relative residual is 4.27e-5 against these
    # rounded hand values; retain the original 1e-4 check margin.
    assert p_sio == pytest.approx(Kp_hand, rel=1e-4)


def test_authority_query_carries_secondary_k2o_flag() -> None:
    assert gas_species_provenance("Na")["authority"] == "lam1984_transcribed"
    for species in ("K", "K2", "KO"):
        provenance = gas_species_provenance(species)
        assert provenance["authority"] == "secondary_transcription_unverified_primary"
        assert (
            provenance["condensate_authority"]
            == "secondary_transcription_unverified_primary"
        )
    assert gas_species_provenance("O2")["authority"] == "janaf_fitted"
    with pytest.raises(ImccGasSpeciesNotFoundError):
        gas_species_provenance("not-a-channel")


def test_activity_and_fugacity_scaling(
    gas_pack: ImccGasDatapack,
) -> None:
    # Premise: SiO2 -> SiO + 1/2 O2 has n_gas=1, so (5) is linear in a_SiO2.
    # Algebra: p(a/2) = K*(a/2) = p(a)/2 and p(2a) = 2*p(a). Unit check:
    # both activities and p/p° are dimensionless. Sanity: halving a unit
    # activity halves p_SiO.
    p_full = evaluate_gas(
        {"SiO2": 1.0}, 2000.0, 1.0, gas_pack, gas_species=("SiO",)
    )["SiO"]
    p_half = evaluate_gas(
        {"SiO2": 0.5}, 2000.0, 1.0, gas_pack, gas_species=("SiO",)
    )["SiO"]
    p_double = evaluate_gas(
        {"SiO2": 2.0}, 2000.0, 1.0, gas_pack, gas_species=("SiO",)
    )["SiO"]
    assert p_half == pytest.approx(0.5 * p_full, rel=1e-12)
    assert p_double == pytest.approx(2.0 * p_full, rel=1e-12)

    # Premise: K2O -> 2 K + 1/2 O2 has n_gas=2, so (5) is proportional to
    # a_K2O**(1/2). Algebra: p(a/2)/p(a) = sqrt(1/2), and
    # p(2a)/p(a) = sqrt(2). Unit check: the exponent applies to the
    # dimensionless parent activity, not a single-cation conversion. Sanity:
    # a_K2O=0.25 gives the ratios below exactly.
    p_k = evaluate_gas(
        {"K2O": 0.25}, 1800.0, 1.0e-10, gas_pack, gas_species=("K",)
    )["K"]
    p_k_half = evaluate_gas(
        {"K2O": 0.125}, 1800.0, 1.0e-10, gas_pack, gas_species=("K",)
    )["K"]
    p_k_double = evaluate_gas(
        {"K2O": 0.5}, 1800.0, 1.0e-10, gas_pack, gas_species=("K",)
    )["K"]
    assert p_k_half / p_k == pytest.approx(2.0**-0.5, rel=1e-12)
    assert p_k_double / p_k == pytest.approx(2.0**0.5, rel=1e-12)

    # Premise: Na2O -> 2 Na + 1/2 O2 has n_O2/n_gas=1/4. Algebra: changing
    # fO2 from 1 to 1e-4 multiplies p_Na by (1e-4)^(-1/4)=10. Unit check:
    # fO2 is p_O2/p°, hence dimensionless. Sanity: p_low = 10*p_ref.
    p_ref = evaluate_gas(
        {"Na2O": 1.0}, 2000.0, 1.0, gas_pack, allow_extrapolation=True
    )["Na"]
    p_low = evaluate_gas(
        {"Na2O": 1.0}, 2000.0, 1.0e-4, gas_pack, allow_extrapolation=True
    )["Na"]
    assert p_low == pytest.approx(10.0 * p_ref, rel=1e-12)


def test_runtime_evaluation_is_deterministic(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    run1 = evaluate_gas(
        unit_activities,
        2000.0,
        1.0,
        datapack=gas_pack,
        allow_extrapolation=True,
    )
    run2 = evaluate_gas(
        unit_activities,
        2000.0,
        1.0,
        datapack=gas_pack,
        allow_extrapolation=True,
    )
    assert run1 == run2


def test_missing_gas_row_is_typed_refusal(gas_pack: ImccGasDatapack) -> None:
    stripped = gas_pack.gas_df.drop(index="K(g)")
    stripped_pack = ImccGasDatapack(
        gas_df=stripped,
        oxide_df=gas_pack.oxide_df,
        gas_path=gas_pack.gas_path,
        oxide_path=gas_pack.oxide_path,
    )
    with pytest.raises(ImccGasSpeciesNotFoundError) as exc:
        evaluate_gas(
            {"K2O": 1.0},
            2000.0,
            1.0,
            stripped_pack,
            gas_species=("K",),
            allow_extrapolation=True,
        )
    assert exc.value.code == "imcc_gas_species_not_found"


def test_unavailable_species_ledger_names_the_closing_source(
    gas_pack: ImccGasDatapack,
) -> None:
    assert set(IMCC_GAS_NO_JANAF_ROWS) == {
        "Na2O",
        "K2O",
        "Na+",
        "K+",
        "e-",
        "Zn",
        "ZnO",
    }
    assert set(IMCC_GAS_INCOMPLETE_PARENT_SPECIES) == {"Ti", "TiO", "TiO2"}
    assert all(
        source.startswith("needs") or "; needs" in source
        for source in IMCC_GAS_UNAVAILABLE_SPECIES.values()
    )
    assert all(
        f"{species}(g)" not in gas_pack.gas_df.index
        for species in IMCC_GAS_NO_JANAF_ROWS
    )
    # The fitted package contains the 22 retained channels only; the Ti rows
    # are deliberately not shipped because their parent path is incomplete.
    assert all(
        f"{species}(g)" not in gas_pack.gas_df.index
        for species in IMCC_GAS_INCOMPLETE_PARENT_SPECIES
    )
    assert "TiO2(l)" not in gas_pack.oxide_df.index


def test_imcc_adapter_activities_reach_all_channels(
    gas_pack: ImccGasDatapack,
) -> None:
    from openimcc import evaluate as evaluate_imcc
    from openimcc import load_datapack

    imcc_pack = load_datapack(Path("src/openimcc/data/packs/imcc-sf04-v1.0.2.json"))
    composition = {
        "SiO2": 71.39,
        "MgO": 0.27,
        "FeO": 0.04,
        "CaO": 10.75,
        "Al2O3": 2.78,
        "TiO2": 0.0,
        "Na2O": 12.75,
        "K2O": 2.02,
    }
    imcc_result = evaluate_imcc(
        composition,
        T_K=2500.0,
        pack=imcc_pack,
        basis_type="wt",
        allow_extrapolation=True,
    )
    activities = dict(zip(imcc_result.parent_oxides, imcc_result.parent_activity))
    result = evaluate_gas(
        activities,
        2500.0,
        1.0e-10,
        gas_pack,
        allow_extrapolation=True,
    )
    assert set(result) == set(IMCC_GAS_CHANNEL_SPECIES)
    assert all(value >= 0.0 for value in result.values())
    assert result["O2"] == 1.0e-10
    assert result["Na"] > 0.0
    assert result["K"] > 0.0


def test_gas_imports_without_pandas_and_refuses_at_load() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys",
                    'sys.modules["pandas"] = None',
                    "import openimcc.gas as gas",
                    "from openimcc import ImccRefusal",
                    "try:",
                    "    gas.load_gas_datapack()",
                    "except ImccRefusal as exc:",
                    '    assert \'install "openimcc[gas]" (pandas import failed:\' in str(exc)',
                    "else:",
                    "    raise AssertionError(\"load_gas_datapack unexpectedly succeeded\")",
                ]
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
    assert probe.returncode == 0, probe.stderr


def _unread_gas_pack() -> ImccGasDatapack:
    return ImccGasDatapack(
        gas_df=pd.DataFrame(),
        oxide_df=pd.DataFrame(),
        gas_path=Path("."),
        oxide_path=Path("."),
    )


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_temperature_refuses_before_table_access(temperature: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_gas(
            {"Na2O": 1.0},
            temperature,
            1.0,
            _unread_gas_pack(),
            gas_species=("Na",),
        )


@pytest.mark.parametrize("fugacity", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_fugacity_refuses_before_table_access(fugacity: float) -> None:
    with pytest.raises(
        ImccGasInvalidFugacityError,
        match=r"finite and positive p_O2/p°.*not log10 fO2",
    ):
        evaluate_gas(
            {"Na2O": 1.0},
            2000.0,
            fugacity,
            _unread_gas_pack(),
            gas_species=("Na",),
        )


@pytest.mark.parametrize("activity", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_activity_refuses(activity: float) -> None:
    with pytest.raises(ValueError, match="finite and >= 0"):
        evaluate_gas(
            {"Na2O": activity},
            2000.0,
            1.0,
            _unread_gas_pack(),
            gas_species=("Na",),
        )
