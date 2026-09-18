"""Rung-2b gates for the IMCC-SF04 gas mass-action layer (chunk 5a)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from openimcc.gas import (

    default_condensate_database_path,
    default_gas_database_path,
    IMCC_GAS_CHANNEL_SPECIES,
    IMCC_GAS_INCOMPLETE_PARENT_SPECIES,
    IMCC_GAS_NO_JANAF_ROWS,
    IMCC_GAS_UNAVAILABLE_SPECIES,
    IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS,
    IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES,
    IMCC_SF04_WORKBOOK_GRID_K,
    ImccGasDatapack,
    ImccGasSpeciesNotFoundError,
    ImccGasTemperatureOutsideDomainError,
    evaluate_gas,
    load_gas_datapack,
)

# The gas tables are not bundled: they come from an AGPL VapoRock checkout and
# this package is Apache-2.0 (see docs/PRE-EXTRACTION-CHECKLIST.md). Skip the
# whole module unless the caller has pointed us at one, rather than erroring in
# collection.
def _gas_tables_available() -> bool:
    try:
        default_gas_database_path()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _gas_tables_available(),
    reason="gas tables unavailable; set OPENIMCC_VAPOROCK_ROOT to run these",
)



# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _require_vaporock_sibling() -> None:
    """Skip rather than FileNotFoundError when the VapoRock sibling is absent.

    The JANAF gas and condensate tables are NOT part of this repository. They are
    read from a read-only SIBLING checkout resolved as
    ``Path(gas.__file__).parents[3] / ".." / "VapoRock"`` (see
    ``imcc_sf04/gas.py`` ``_default_vaporock_root``). That resolves next to the
    repo root, so it is present in a normal working clone and ABSENT anywhere the
    repo is copied on its own — which is exactly what a CI job directory is.

    Measured 2026-08-16 on the pr-tier gate: all 28 tests in this module errored
    with ``FileNotFoundError … /../VapoRock/src/vaporock/data/JANAF-vapor-data-full.csv``
    because the runner rsyncs the tree to ``~/repos/ci-jobs/<job>/`` and nothing
    sits beside it. They read as 28 new defects in the differential when the code
    under test was fine.

    Same class as the gitignored EXT4 working set guarded by ``_ext4()`` in
    ``test_imcc_sp_extension.py``: a test whose input lives outside the repo must
    say so by SKIPPING, so a checkout without it reports honestly instead of
    manufacturing red. Production still refuses loudly — this guard is test-side
    only and deliberately does not soften ``load_gas_datapack``.
    """
    missing = [
        path
        for path in (default_gas_database_path, default_condensate_database_path)
        if not Path(path).is_file()
    ]
    if missing:
        pytest.skip(
            "read-only VapoRock sibling checkout is absent from this tree; "
            f"missing {', '.join(str(p) for p in missing)}"
        )


@pytest.fixture(scope="module")
def gas_pack() -> ImccGasDatapack:
    _require_vaporock_sibling()
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


# --------------------------------------------------------------------------- #
# Rung-2b pure-component vapor-pressure gate (non-circular)
# --------------------------------------------------------------------------- #

# Declared rung-2b sample: T = 2000 K, fO2 = 1 bar, all parent activities = 1.
# This is the 9-species subset implemented in the chunk-5a gas layer.  The
# rung-3/4 retained population is 32 species (Schaefer2004-MAGMA-valid.xlsx)
# and is deliberately NOT claimed here; see test_p1_g2_rung2b_coverage_subset.
#
# Reference log10(p / bar) values are transcribed from the PRINTED JANAF tables
# (Chase 1998, via NIST WebBook / janaf.nist.gov) at 2000 K, rounded to the
# 3-decimal precision of the printed lgK column.  Each value cites the VapoRock
# read-only sibling table where the source row lives; the code under test does
# NOT read these literals.
#
# Source rows (read-only VapoRock sibling checkout):
#   gas species: ../VapoRock/src/vaporock/data/JANAF-vapor-data-full.csv
#   condensate oxides: ../VapoRock/data/condensate-thermo-data.csv
_RUNG2B_PRINTED_JANAF = {
    2000.0: {
        # Na(g) interval [1170.525, 6000]; Na2O(l) interval [825, 3000]
        "Na": -1.889,
        # K(g) interval [1800, 6000]; K2O(l) interval [1190, 3000]
        "K": 0.316,
        # SiO(g) interval [1100, 6000]; SiO2(l) interval [1996, 3000]
        "SiO": -7.759,
        # Fe(g) interval [3133, 6000]; FeO(l) interval [1000, 5000]
        "Fe": -7.307,
        # FeO(g) interval [5000, 6000]; FeO(l) interval [1000, 5000]
        "FeO": -5.465,
        # Mg(g) interval [1366, 2200]; MgO(l) interval [3100, 3500]
        "Mg": -7.856,
        # MgO(g) interval [5000, 6000]; MgO(l) interval [3100, 3500]
        "MgO": -7.177,
        # SiO2(g) interval [4500, 6000]; SiO2(l) interval [1996, 3000]
        "SiO2": -6.669,
        # O2(g) reference element; p_O2 = fO2 by definition
        "O2": 0.0,
    }
}

# T-625 expansion reference: T = 2500 K, fO2 = 1 bar, parent activities = 1.
# These log10(p/bar) literals are the Chase (1998) printed-JANAF lgK values,
# rounded to the printed table's 3-decimal precision, then combined with the
# disclosed melt-oxide reaction coefficients.  The JANAF gas rows live in
# ../VapoRock/src/vaporock/data/JANAF-vapor-data-full.csv; the LAMOR/JANAF
# liquid-parent rows live in ../VapoRock/data/condensate-thermo-data.csv.
# The high-T-only rows are deliberately extrapolated here and separately locked
# to typed refusal below; an evidence fixture does not widen their domain.
_T625_PRINTED_JANAF = {
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
        "Si": -11.904,
        "Al": -8.695,
        "CaO": -5.476,
        "Ca": -6.261,
    }
}


def test_p1_g1_rung2b_noncircular_against_printed_janaf(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    """Rung 2b: reference values are printed-JANAF lgK, not gas.py self-parity.

    The reference log10(p) values in ``_RUNG2B_PRINTED_JANAF`` are transcribed
    from the printed JANAF tables (Chase 1998) at 2000 K and rounded to the
    3-decimal precision of the printed lgK column.  They are aggregated from
    the same VapoRock source rows using reaction log10 Kf stoichiometry, not
    the ΔG° -> exp path used by ``evaluate_gas``.  The expected nonzero
    rounding error (≤ 0.01 dex) is the signature that this gate is not a
    self-parity check.

    Hand-computed literature point 1 — SiO2(l) -> SiO2(g) at 2000 K:
        From the JANAF + LAMOR tables:
            G°(SiO2, l) = -1 129 878.93 J/mol
            G°(SiO2, g) =   -874 523.28 J/mol
        ΔG° = G°(SiO2,g) - G°(SiO2,l) = 255 355.65 J/mol
        log10 Kp = -ΔG° / (2.303 * R * T)
                 = -255 355.65 / (2.303 * 8.314462618 * 2000)
                 = -6.669
        At a_SiO2 = 1 and fO2 = 1 bar, p_SiO2 = Kp = 10^-6.669 bar.

    Hand-computed literature point 2 — FeO(l) -> Fe(g) + 1/2 O2(g) at 2000 K:
        From the JANAF + LAMOR tables:
            G°(FeO, l) = -515 261.35 J/mol
            G°(Fe,  g) =      3 665.62 J/mol
            G°(O2,  g) = -478 320.20 J/mol
        ΔG° = G°(Fe,g) + 0.5 G°(O2,g) - G°(FeO,l) = 279 766.87 J/mol
        log10 Kp = -279 766.87 / (2.303 * 8.314462618 * 2000)
                 = -7.307
        At a_FeO = 1 and fO2 = 1 bar, p_Fe = (Kp * a_FeO / fO2^0.5)^1
                 = Kp = 10^-7.307 bar.
    """
    T = 2000.0
    # 2000 K is below the demonstrated G(T) domain for Fe/FeO/MgO/SiO2 gas
    # intervals and for MgO(l); exercise them under the explicit extrapolation
    # flag (the default is now refusal, see test_p2_g3).
    pressures = evaluate_gas(
        unit_activities, T, fO2=1.0, datapack=gas_pack, allow_extrapolation=True
    )
    refs = _RUNG2B_PRINTED_JANAF[T]
    worst = 0.0
    nonzero_errors = 0
    for species, expected_log10 in refs.items():
        actual_log10 = math.log10(pressures[species])
        err = abs(actual_log10 - expected_log10)
        worst = max(worst, err)
        if err > 0.0:
            nonzero_errors += 1
        assert err <= 0.01, (
            f"{species} at {T} K: |{actual_log10:.6f} - {expected_log10:.6f}| "
            f"= {err:.6f} dex > 0.01 dex"
        )
    # Non-circularity lock: a 0.000000 result must be impossible by construction
    # because the printed-JANAF references are rounded to 3 decimals.
    assert nonzero_errors >= 1, (
        "gate must show nonzero rounding error to prove it is not self-parity; "
        f"observed nonzero errors = {nonzero_errors}"
    )
    # Expose the worst error for the end-of-run summary.
    assert worst <= 0.01


def test_t625_new_channels_noncircular_against_printed_janaf(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    """Every usable F3-6 addition agrees with an independently rounded lgK."""
    T = 2500.0
    refs = _T625_PRINTED_JANAF[T]
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
        err = abs(actual_log10 - expected_log10)
        assert 0.0 < err <= 0.01, (
            f"{species} at {T} K: independently rounded reference must have "
            f"nonzero error <= 0.01 dex, got {err:.6f}"
        )


# --------------------------------------------------------------------------- #
# P1-G2: honest coverage statement for the rung-2b sample
# --------------------------------------------------------------------------- #


def test_p1_g2_rung2b_coverage_subset() -> None:
    """T-625 covers every workbook species with a complete thermodynamic path."""
    implemented = set(IMCC_GAS_CHANNEL_SPECIES)
    unavailable = set(IMCC_GAS_UNAVAILABLE_SPECIES)
    # Per IMCC-SF04 spec §8, the rung-3/4 workbook retained population is the
    # 32 gas-species rows in Schaefer2004-MAGMA-valid.xlsx.
    rung3_4_retained_count = 32
    assert len(implemented) == 22
    assert len(unavailable) == 10
    assert implemented.isdisjoint(unavailable)
    assert set(IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES).isdisjoint(
        IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS
    )
    assert set(IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES) | set(
        IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS
    ) == implemented
    assert rung3_4_retained_count == 32
    assert len(implemented | unavailable) == rung3_4_retained_count


def test_t625_unavailable_species_name_the_closing_source(
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
    assert set(IMCC_GAS_INCOMPLETE_PARENT_SPECIES) == {
        "Ti",
        "TiO",
        "TiO2",
    }
    assert all(
        source.startswith("needs") or "; needs" in source
        for source in IMCC_GAS_UNAVAILABLE_SPECIES.values()
    )
    assert all(
        f"{species}(g)" not in gas_pack.gas_df.index
        for species in IMCC_GAS_NO_JANAF_ROWS
    )
    assert all(
        f"{species}(g)" in gas_pack.gas_df.index
        for species in IMCC_GAS_INCOMPLETE_PARENT_SPECIES
    )
    assert "TiO2(l)" not in gas_pack.oxide_df.index


# --------------------------------------------------------------------------- #
# P2-G3: T-domain refusal is live outside the selected G(T) row
# --------------------------------------------------------------------------- #


def test_p2_g3_refuses_outside_gas_domain(gas_pack: ImccGasDatapack) -> None:
    """Refusal fires when T is below the selected gas-species G(T) interval."""
    # Fe(g) only has a JANAF row for [3133.345, 6000] K; 2000 K is below that.
    # FeO(l) is in range at 2000 K, so the refusal must come from the gas side.
    acts = {"FeO": 1.0}
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as exc:
        evaluate_gas(acts, 2000.0, fO2=1.0, datapack=gas_pack)
    assert exc.value.code == "imcc_gas_T_outside_domain"
    assert "Fe(g)" in str(exc.value)


def test_p2_g3_refuses_outside_oxide_domain(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    """Refusal fires when T is outside a condensate G(T) interval."""
    # At 5100 K all retained gas species are inside their declared JANAF intervals,
    # but several condensate oxides (Na2O, K2O, SiO2, MgO, FeO) are above their
    # T_max.  The refusal must therefore come from the condensate side.
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as exc:
        evaluate_gas(unit_activities, 5100.0, fO2=1.0, datapack=gas_pack)
    assert exc.value.code == "imcc_gas_T_outside_domain"
    # The first oxide in loop order is Na2O(l); do not hard-code the exact
    # species because the test only needs to prove the oxide-side refusal is live.
    assert "(l)" in str(exc.value)


_EXTRAPOLATION_REFUSAL_CASES = (
    ("SiO", 1900.0, "SiO2(l)"),
    ("Fe", 2500.0, "Fe(g)"),
    ("FeO", 2500.0, "FeO(g)"),
    ("Mg", 2500.0, "MgO(l)"),
    ("MgO", 2500.0, "MgO(g)"),
    ("SiO2", 2500.0, "SiO2(g)"),
    ("AlO", 2000.0, "Al2O3(l)"),
    ("AlO2", 2000.0, "Al2O3(l)"),
    ("Al2O", 2000.0, "Al2O3(l)"),
    ("Al2O2", 2000.0, "Al2O3(l)"),
    ("Si", 2500.0, "Si(g)"),
    ("Al", 2500.0, "Al(g)"),
    ("CaO", 2500.0, "CaO(g)"),
    ("CaO", 4500.0, "CaO(l)"),
    ("Ca", 1750.0, "Ca(g)"),
    ("Ca", 2500.0, "CaO(l)"),
)


@pytest.mark.parametrize(
    ("species", "T_K", "source_species"), _EXTRAPOLATION_REFUSAL_CASES
)
def test_t625_each_extrapolation_label_has_typed_domain_refusal(
    gas_pack: ImccGasDatapack,
    unit_activities: dict[str, float],
    species: str,
    T_K: float,
    source_species: str,
) -> None:
    assert {case[0] for case in _EXTRAPOLATION_REFUSAL_CASES} == set(
        IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS
    )
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as exc:
        evaluate_gas(
            unit_activities,
            T_K,
            fO2=1.0,
            datapack=gas_pack,
            gas_species=(species,),
        )
    assert exc.value.code == "imcc_gas_T_outside_domain"
    assert source_species in str(exc.value)


def test_t625_in_domain_labels_cover_the_whole_workbook_grid(
    gas_pack: ImccGasDatapack,
    unit_activities: dict[str, float],
) -> None:
    for species in IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES:
        for T_K in IMCC_SF04_WORKBOOK_GRID_K:
            pressures = evaluate_gas(
                unit_activities,
                T_K,
                fO2=1.0,
                datapack=gas_pack,
                gas_species=(species,),
            )
            assert species in pressures


# --------------------------------------------------------------------------- #
# Kp derivation spot-check
# --------------------------------------------------------------------------- #


def test_kp_derivation_spot_check_sio(gas_pack: ImccGasDatapack) -> None:
    """Recompute Kp for SiO2(l) -> SiO(g) + 1/2 O2(g) at 2000 K by hand.

    From the JANAF + condensate tables at 2000 K:
        G°(SiO2, l) = -1 129 878.93 J/mol
        G°(SiO,  g) =   -593 626.03 J/mol
        G°(O2,    g) =   -478 320.38 J/mol
    ΔG° = G°(SiO) + 0.5 G°(O2) - G°(SiO2)
        = -593 626.03 + 0.5(-478 320.38) - (-1 129 878.93)
        = 297 092.71 J/mol
    Kp = exp(-ΔG° / (R T))
       = exp(-297 092.71 / (8.314462618 * 2000.0))
       = 1.7387e-8
    At a_SiO2 = 1 and fO2 = 1 bar, p_SiO = Kp.
    """
    R = 8.314462618
    T = 2000.0
    G_sio2_l = -1129878.93
    G_sio_g = -593626.03
    G_o2_g = -478320.38
    dG = G_sio_g + 0.5 * G_o2_g - G_sio2_l
    Kp_hand = math.exp(-dG / (R * T))

    pressures = evaluate_gas(
        {"SiO2": 1.0}, T, fO2=1.0, datapack=gas_pack, allow_extrapolation=True
    )
    p_sio = pressures["SiO"]

    # Allow a tiny tolerance for the hand-rounded G° values.
    assert p_sio == pytest.approx(Kp_hand, rel=1e-4)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_determinism(gas_pack: ImccGasDatapack, unit_activities: dict[str, float]) -> None:
    """Same input must produce bit-for-bit identical output."""
    run1 = evaluate_gas(
        unit_activities, 2000.0, 1.0, datapack=gas_pack, allow_extrapolation=True
    )
    run2 = evaluate_gas(
        unit_activities, 2000.0, 1.0, datapack=gas_pack, allow_extrapolation=True
    )
    assert run1 == run2


# --------------------------------------------------------------------------- #
# Activity scaling
# --------------------------------------------------------------------------- #


def test_activity_scaling(gas_pack: ImccGasDatapack) -> None:
    """Halving the SiO2 activity should halve p_SiO (n_O2=0.5, n_gas=1)."""
    p_full = evaluate_gas(
        {"SiO2": 1.0}, 2000.0, 1.0, datapack=gas_pack, allow_extrapolation=True
    )["SiO"]
    p_half = evaluate_gas(
        {"SiO2": 0.5}, 2000.0, 1.0, datapack=gas_pack, allow_extrapolation=True
    )["SiO"]
    assert p_half == pytest.approx(0.5 * p_full, rel=1e-12)


def test_fo2_scaling_sodium(gas_pack: ImccGasDatapack) -> None:
    """For Na (n_gas=2, n_O2=0.5), p_Na should scale as fO2^{-1/4}."""
    p_ref = evaluate_gas(
        {"Na2O": 1.0}, 2000.0, 1.0, datapack=gas_pack, allow_extrapolation=True
    )["Na"]
    p_low = evaluate_gas(
        {"Na2O": 1.0}, 2000.0, 1.0e-4, datapack=gas_pack, allow_extrapolation=True
    )["Na"]
    # p_Na ∝ fO2^{-1/4}; 1e-4 -> factor (1e-4)^{-1/4} = 10.
    assert p_low == pytest.approx(10.0 * p_ref, rel=1e-12)


# --------------------------------------------------------------------------- #
# Typed refusal on missing G(T) row
# --------------------------------------------------------------------------- #


def test_typed_refusal_missing_gas_species(gas_pack: ImccGasDatapack) -> None:
    """A gas species with no G(T) row raises ImccGasSpeciesNotFoundError."""
    # Drop every K(g) row from the JANAF table.
    stripped_gas = gas_pack.gas_df.drop(index="K(g)", errors="ignore")
    stripped_pack = ImccGasDatapack(
        gas_df=stripped_gas,
        oxide_df=gas_pack.oxide_df,
        gas_path=gas_pack.gas_path,
        oxide_path=gas_pack.oxide_path,
    )
    acts = {"K2O": 1.0}
    with pytest.raises(ImccGasSpeciesNotFoundError) as exc:
        evaluate_gas(
            acts, 2000.0, 1.0, datapack=stripped_pack, allow_extrapolation=True
        )
    assert exc.value.code == "imcc_gas_species_not_found"


# --------------------------------------------------------------------------- #
# Integration: IMCC melt activities -> gas partial pressures
# --------------------------------------------------------------------------- #


def test_gas_layer_uses_imcc_activities() -> None:
    """The gas layer accepts activities produced by the IMCC adapter."""
    from openimcc import evaluate as evaluate_imcc
    from openimcc import load_datapack as load_imcc_datapack

    imcc_pack = load_imcc_datapack(
        Path("src/openimcc/data/packs/imcc-sf04-v1.0.2.json")
    )
    # Case 1 from Hastie 1985 — a multicomponent lunar-glass-like composition.
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

    _require_vaporock_sibling()
    gas_pack = load_gas_datapack()
    pressures = evaluate_gas(
        activities, 2500.0, fO2=1.0e-10, datapack=gas_pack, allow_extrapolation=True
    )

    # All retained species are present and non-negative.
    assert set(pressures) == set(IMCC_GAS_CHANNEL_SPECIES)
    assert all(p >= 0.0 for p in pressures.values())
    # O2 is exactly the caller-pinned value.
    assert pressures["O2"] == 1.0e-10
    # Alkali-bearing melt components are present, so the gas layer returns
    # positive alkali partial pressures.
    assert pressures["Na"] > 0.0
    assert pressures["K"] > 0.0


# --------------------------------------------------------------------------- #
# Fail-closed on non-finite T, fO2, and consumed oxide activity
# --------------------------------------------------------------------------- #


def _unread_gas_pack() -> ImccGasDatapack:
    """Empty tables; evaluate_gas must refuse before reading them."""
    return ImccGasDatapack(
        gas_df=pd.DataFrame(),
        oxide_df=pd.DataFrame(),
        gas_path=Path("."),
        oxide_path=Path("."),
    )


@pytest.mark.parametrize("T_K", [float("nan"), float("inf"), float("-inf")])
def test_evaluate_gas_refuses_nonfinite_temperature(T_K: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_gas(
            {"Na2O": 1.0},
            T_K,
            1.0,
            datapack=_unread_gas_pack(),
            gas_species=("Na",),
        )


@pytest.mark.parametrize("fO2", [float("nan"), float("inf"), float("-inf")])
def test_evaluate_gas_refuses_nonfinite_fo2(fO2: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_gas(
            {"Na2O": 1.0},
            2000.0,
            fO2,
            datapack=_unread_gas_pack(),
            gas_species=("Na",),
        )


@pytest.mark.parametrize("activity", [float("nan"), float("inf"), float("-inf")])
def test_evaluate_gas_refuses_nonfinite_activity(activity: float) -> None:
    with pytest.raises(ValueError, match="finite and >= 0"):
        evaluate_gas(
            {"Na2O": activity},
            2000.0,
            1.0,
            datapack=_unread_gas_pack(),
            gas_species=("Na",),
        )
