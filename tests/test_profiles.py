"""Matched pack-profile selection tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import struct
from types import SimpleNamespace

import pytest

from openimcc import (
    ImccLiquidStandardMismatchError,
    ImccTraceLimitExceededError,
    PackStandardNotice,
    ProfileConflictError,
    ProfileRedoxUnavailableError,
    evaluate,
    evaluate_gas,
    evaluate_redox,
    load_datapack,
    load_profile,
)
from openimcc.gas import (
    IMCC_PARENT_OXIDES,
    evaluate_gas_oxygen_balance,
    evaluate_gas_saturation,
    gas_boiling_temperature,
)


_BASALT = {
    "SiO2": 51.85068,
    "MgO": 4.78527,
    "FeO": 13.77307,
    "CaO": 9.02862,
    "Al2O3": 14.80572,
    "TiO2": 1.73824,
    "Na2O": 3.23108,
    "K2O": 0.78732,
}

_PUBLISHED_GAS_STANDARD_NOTICES = [
    ("SiO2", "O-038", "LAM1987", None, "mismatch"),
    ("MgO", "Mg-009", "LAM1987", None, "mismatch"),
    ("CaO", "Ca-028", "LAM1987", None, "mismatch"),
    ("Al2O3", "Al-100", "LAM1987", None, "mismatch"),
    ("K2O", "FC87-Glushko-K2O", "LAM1984", None, "mismatch"),
    ("K2O", "reference-unknown", "LAM1984", "K2SiO3", "reference-unknown"),
    ("K2O", "reference-unknown", "LAM1984", "K2Si2O5", "reference-unknown"),
    ("K2O", "reference-unknown", "LAM1984", "K2Si4O9", "reference-unknown"),
    ("K2O", "reference-unknown", "LAM1984", "KCaAlSi2O7", "reference-unknown"),
]


@pytest.mark.parametrize(
    ("name", "melt_version", "has_redox"),
    [
        ("openimcc", "1.0.6-d066-v4", True),
        ("sr-candidate", "1.0.0-d066-sr-candidate-v2", True),
        ("sr-candidate-v1", "1.0.0-d066-sr-candidate-v1", True),
        ("p2o5-candidate", "1.0.0-d066-p2o5-candidate-v1", True),
        ("cao-aluminate-candidate", "1.0.0-d066-cao-aluminate-candidate-v1", True),
        ("silicate-candidate", "1.0.0-d066-silicate-candidate-v2", True),
        ("mgsio-candidate", "1.0.0-d066-mgsio-candidate-v1", True),
        ("published", "1.0.2", False),
    ],
)
def test_named_profiles_load(name: str, melt_version: str, has_redox: bool) -> None:
    profile = load_profile(name)

    assert profile.name == name
    assert profile.melt_pack.version == melt_version
    assert profile.provenance
    assert (profile.redox_pack_path is not None) is has_redox
    if name == "published":
        assert profile.gas_datapack.gas_path.name == "gas-shomate.csv"
        assert profile.gas_datapack.oxide_path.name == "condensate.csv"
        assert "sf04-published" in str(profile.gas_datapack.oxide_path)
    else:
        assert profile.gas_datapack.gas_path.name == "gas-shomate-default-v2.csv"
        assert profile.gas_datapack.oxide_path.name == "condensate-default-v1.csv"
        assert profile.redox_pack_path.name == "openimcc-redox-v6.json"
    with pytest.raises(FrozenInstanceError):
        profile.name = "published"


def test_default_and_published_profiles_use_imcc_parent_basis() -> None:
    expected = tuple(IMCC_PARENT_OXIDES)

    assert tuple(load_datapack().parent_oxides) == expected
    assert tuple(load_profile("openimcc").melt_pack.parent_oxides) == expected
    assert tuple(load_profile("published").melt_pack.parent_oxides) == expected


def test_sr_candidate_profile_evaluates_full_parent_basis_and_sr_gas() -> None:
    profile = load_profile("sr-candidate")
    assert profile.melt_pack.parent_oxides[-1] == "SrO"
    assert len(profile.melt_pack.kernel_datapack.reactions) == 44
    for temperature in (1700.0, 2400.0):
        for composition in (
            {"SrO": 1.0},
            {"SrO": 0.5, "Al2O3": 0.5},
            {"SrO": 0.3, "Al2O3": 0.3, "SiO2": 0.4},
        ):
            melt = evaluate(composition, temperature, profile=profile)
            activities = {
                name: melt.activity(name) for name in melt.parent_oxides
            }
            gas = evaluate_gas(
                activities,
                temperature,
                1e-10,
                profile=profile,
            )
            assert "Sr" in gas
            assert gas["Sr"] > 0.0


def test_sr_candidate_v2_flags_rows_outside_their_paper_windows() -> None:
    profile = load_profile("sr-candidate")
    melt = evaluate(
        {"SrO": 0.3, "Al2O3": 0.3, "SiO2": 0.4},
        3000.0,
        profile=profile,
    )

    paper_flag, = tuple(
        flag
        for flag in melt.labels.flags
        if flag.startswith("paper-demonstrated-window:")
    )
    for complex_name in (
        "SrAl2O4",
        "SrAl4O7",
        "SrAl12O19",
        "Sr3Al2O6",
        "Sr4Al2O7",
        "SrSiO3",
        "Sr2SiO4",
    ):
        assert complex_name in paper_flag


def test_sr_candidate_v1_remains_selectable() -> None:
    profile = load_profile("sr-candidate-v1")
    assert profile.melt_pack.version == "1.0.0-d066-sr-candidate-v1"


def test_p2o5_candidate_solves_kreep_majors_and_gates_unvalidated_p_gas() -> None:
    from openimcc._constants import MOLAR_MASS_G_MOL
    from openimcc.gas import ImccP2O5AssociatesIncompleteError, evaluate_gas

    profile = load_profile("p2o5-candidate")
    composition = {
        "SiO2": 50.8,
        "MgO": 9.34,
        "FeO": 10.4,
        "CaO": 9.6,
        "Al2O3": 15.0,
        "TiO2": 2.06,
        "Na2O": 0.77,
        "K2O": 0.61,
        "MnO": 0.16,
        "P2O5": 1.5,
    }
    melts = {
        temperature: evaluate(
            composition, temperature, profile=profile, basis_type="wt"
        )
        for temperature in (1700.0, 2000.0, 2300.0)
    }
    melt = melts[2000.0]
    assert profile.melt_pack.parent_oxides[-1] == "P2O5"
    assert "P2O5" in melt.parent_oxides
    assert "MnO" in melt.parent_oxides
    assert all(melt.activity(parent) >= 0.0 for parent in melt.parent_oxides)
    assert profile.melt_pack._p2o5_channel_status == "associates_incomplete"
    mole_amounts = {
        oxide: mass / MOLAR_MASS_G_MOL[oxide]
        for oxide, mass in composition.items()
    }
    p2o5_mole_fraction = mole_amounts["P2O5"] / sum(mole_amounts.values())
    assert melt.activity("P2O5") / p2o5_mole_fraction < 1.0e-3
    p2o5_activities = [melts[t].activity("P2O5") for t in (1700.0, 2000.0, 2300.0)]
    assert p2o5_activities == pytest.approx(
        [2.8802036e-11, 6.8340384e-10, 6.0259858e-9], rel=1.0e-5
    )
    assert p2o5_activities[0] < p2o5_activities[1] < p2o5_activities[2]

    gas = evaluate_gas(
        {}, 2000.0, 1.0e-10, profile=profile, composition_wt_pct=composition
    )
    assert gas["SiO"] > 0.0
    assert gas["Na"] > 0.0
    assert "SiO" not in gas.omitted_channels
    assert "Na" not in gas.omitted_channels
    assert len(gas) == 34
    p_channels = {"P", "P2", "P4", "PO", "PO2", "P4O6", "P4O10"}
    assert p_channels.issubset(gas.omitted_channels)
    assert all(
        gas.omitted_channels[species].code == "p2o5_associates_incomplete"
        for species in p_channels
    )
    assert all("held-out" in gas.omitted_channels[species].reason for species in p_channels)
    assert all(species not in gas for species in p_channels)
    with pytest.raises(ImccP2O5AssociatesIncompleteError, match="associates_incomplete"):
        evaluate_gas(
            {},
            2000.0,
            1.0e-10,
            profile=profile,
            composition_wt_pct=composition,
            gas_species=("P2",),
        )


def test_p2o5_candidate_keeps_d066_prefix_and_associate_windows() -> None:
    import json
    from pathlib import Path

    profile = load_profile("p2o5-candidate")
    pack_dir = Path(__file__).resolve().parents[1] / "src/openimcc/data/packs"
    candidate = json.loads(
        (pack_dir / "imcc-sf04-d066-p2o5-candidate-v1.json").read_text()
    )
    d066 = json.loads((pack_dir / "imcc-sf04-d066-v4.json").read_text())
    assert candidate["rows"][:38] == d066["rows"]
    assert candidate["parents"] == [*d066["parents"], "P2O5"]
    assert all(
        row["T_domain_K"] == [1700.0, 2300.0]
        for row in candidate["rows"][38:]
    )
    assert [row["T_domain_paper_demonstrated_K"] for row in candidate["rows"][38:]] == [
        [1621.0, 3000.0],
        [298.15, 1600.0],
    ]
    assert "printed_janaf_liquid" in candidate["rows"][38]["flags"]
    assert "class_c_fusion_entropy_provisional" in candidate["rows"][39]["flags"]


def test_silicate_candidate_v2_refits_only_the_two_mg_associate_intercepts() -> None:
    import json
    from pathlib import Path

    profile = load_profile("silicate-candidate")
    default = load_profile("openimcc")
    pack_dir = Path(__file__).resolve().parents[1] / "src/openimcc/data/packs"
    candidate = json.loads(
        (pack_dir / "imcc-sf04-d066-silicate-candidate-v2.json").read_text(encoding="utf-8")
    )
    base = json.loads((pack_dir / "imcc-sf04-d066-v4.json").read_text(encoding="utf-8"))
    target = {
        "Mg2SiO4": -0.8648532598800563,
        "MgSiO3": 0.3506691491283032,
    }

    assert profile.melt_pack.parent_oxides == default.melt_pack.parent_oxides
    assert len(candidate["rows"]) == len(base["rows"]) == 38
    for base_row, candidate_row in zip(base["rows"], candidate["rows"]):
        if base_row["complex"] in target:
            assert candidate_row["A"] == target[base_row["complex"]]
            assert candidate_row["B"] == base_row["B"]
            assert candidate_row["candidate_activity_fit"]["provisional"] is True
            assert len(candidate_row["candidate_activity_fit"]["fit_source_refs"]) == 4
        else:
            assert candidate_row == base_row
    assert candidate["candidate_provenance"]["fit_set"]["n"] == 118
    assert profile.melt_pack.version == "1.0.0-d066-silicate-candidate-v2"
    assert profile.melt_pack.binding_digest != default.melt_pack.binding_digest
    manganese = evaluate_redox(
        {"Fe": 1.0, "Mn": 0.1, "Si": 0.2, "Al": 0.1, "O": 1.65},
        2200.0,
        "closed",
        profile=profile,
    )
    assert manganese.parent_activity["MnO"] > 0.0


@pytest.mark.parametrize(
    "composition",
    (
        {"SrO": 50.0, "Al2O3": 50.0},
        {"SrO": 100.0},
    ),
)
@pytest.mark.parametrize("profile_name", ("sr-candidate", "renamed-candidate"))
def test_sr_candidate_gas_composition_matches_candidate_melt_activities(
    composition: dict[str, float],
    profile_name: str,
) -> None:
    profile = load_profile("sr-candidate")
    if profile_name != profile.name:
        profile = replace(profile, name=profile_name)
    melt = evaluate(
        composition,
        2000.0,
        profile=profile,
        basis_type="wt",
        allow_out_of_envelope=True,
    )
    activities = {
        name: melt.activity(name) for name in melt.parent_oxides
    }
    by_activity = evaluate_gas(
        activities, 2000.0, 1e-10, profile=profile
    )
    by_composition = evaluate_gas(
        {},
        2000.0,
        1e-10,
        profile=profile,
        composition_wt_pct=composition,
    )

    assert by_composition["Sr"] > 0.0
    assert by_composition["Sr"] == pytest.approx(
        by_activity["Sr"], rel=1e-12, abs=0.0
    )
    if composition == {"SrO": 50.0, "Al2O3": 50.0}:
        assert by_composition["Sr"] == pytest.approx(7.488e-6, rel=1e-3)


@pytest.mark.parametrize("profile", (None, "openimcc"))
def test_default_profile_keeps_sro_trace_limit(profile: str | None) -> None:
    with pytest.raises(ImccTraceLimitExceededError):
        evaluate(
            {"SrO": 50.0, "Al2O3": 50.0},
            2000.0,
            profile=profile,
            basis_type="wt",
        )


def test_openimcc_profile_matches_readme_quickstart_values_bit_for_bit(
    packaged_gas_pack,
) -> None:
    profile = load_profile("openimcc")
    default_melt = evaluate(_BASALT, 1800.0, basis_type="wt")
    profile_melt = evaluate(
        _BASALT, 1800.0, basis_type="wt", profile=profile
    )

    for field in (
        "parent_mol",
        "parent_x",
        "parent_x_star",
        "parent_activity",
        "parent_gamma",
        "complex_x",
        "species_x",
    ):
        assert getattr(profile_melt, field).tobytes() == getattr(
            default_melt, field
        ).tobytes()
    assert struct.pack("!d", profile_melt.D) == struct.pack("!d", default_melt.D)
    assert {
        key: value
        for key, value in profile_melt.labels.identity.items()
        if key != "profile"
    } == dict(default_melt.labels.identity)
    assert profile_melt.labels.identity["profile"] == "openimcc"

    activities = {
        name: profile_melt.activity(name) for name in profile_melt.parent_oxides
    }
    default_gas = evaluate_gas(
        activities, 1800.0, 1e-10, packaged_gas_pack
    )
    profile_gas = evaluate_gas(
        activities, 1800.0, 1e-10, profile=profile
    )
    assert dict(profile_gas) == dict(default_gas)
    for result_field in fields(default_gas):
        if result_field.name == "receipt":
            continue
        assert getattr(profile_gas, result_field.name) == getattr(
            default_gas, result_field.name
        )
    assert profile_gas.gas_table_digest == default_gas.gas_table_digest
    assert profile_gas.condensate_table_digest == default_gas.condensate_table_digest
    assert profile_gas.profile == "openimcc"


def test_explicit_packs_and_profiles_are_mutually_exclusive(packaged_gas_pack) -> None:
    with pytest.raises(ProfileConflictError):
        evaluate(_BASALT, 1800.0, load_datapack(), profile="published")
    with pytest.raises(ProfileConflictError):
        evaluate_gas(
            {"SiO2": 0.5}, 1800.0, 1e-10, packaged_gas_pack,
            profile="published",
        )
    with pytest.raises(ProfileConflictError):
        evaluate_redox(
            {"Fe": 1.0, "O": 1.25}, 2200.0, "closed",
            redox_pack_path="openimcc-redox-cr-candidate-v4.json", profile="openimcc",
        )


def test_profile_selects_gas_oxygen_balance_and_saturation_packs() -> None:
    profile = load_profile("openimcc")
    melt = evaluate(_BASALT, 1800.0, basis_type="wt", profile=profile)
    activities = {name: melt.activity(name) for name in melt.parent_oxides}

    default_root, _, _ = evaluate_gas_oxygen_balance(
        activities, 1800.0, profile.gas_datapack
    )
    profile_root, profile_pressures, _ = evaluate_gas_oxygen_balance(
        activities, 1800.0, profile=profile
    )
    assert profile_root.hex() == default_root.hex()
    assert profile_pressures.profile == "openimcc"
    assert (
        profile_pressures.gas_table_digest
        == profile.gas_datapack.gas_table_digest
    )

    saturation = evaluate_gas_saturation(
        {"CaO": 1.0}, 1800.0, parent_oxides=("CaO",),
        allow_extrapolation=True, profile=profile,
    )
    assert saturation.profile == "openimcc"
    assert saturation.condensate_table_digest == (
        profile.gas_datapack.condensate_table_digest
    )


def test_gas_boiling_temperature_accepts_profile_without_datapack(monkeypatch) -> None:
    import openimcc.gas as gas_module

    def fake_saturation(_activities, temperature, _datapack, **_kwargs):
        return SimpleNamespace(
            P_sat_bar=temperature,
            diagnostics={},
            pO2_bar=1.0,
            species_pressures_bar={},
            channel_pressures_bar={},
            mole_fractions={},
            mode="synthetic",
            domain_flags={},
            provenance_class={},
        )

    monkeypatch.setattr(gas_module, "evaluate_gas_saturation", fake_saturation)
    profile = load_profile("openimcc")
    temperature, result = gas_boiling_temperature(
        {}, ambient_pressure_bar=1.5, T_bracket=(1.0, 2.0), profile=profile
    )

    assert temperature == pytest.approx(1.5)
    assert result.profile == "openimcc"
    assert result.gas_table_digest == profile.gas_datapack.gas_table_digest


def test_published_profile_refuses_redox_because_sf04_has_no_redox_model() -> None:
    with pytest.raises(ProfileRedoxUnavailableError, match="no redox model"):
        evaluate_redox(
            {"Fe": 1.0, "O": 1.25}, 2200.0, "closed", profile="published"
        )


def test_published_gas_profile_does_not_borrow_other_liquid_authority() -> None:
    profile = load_profile("published")
    feo = evaluate_gas(
        {"FeO": 1.0}, 1700.0, 1e-8,
        parent_oxides=("FeO",), gas_species=("Fe",), profile=profile,
    )
    assert "Fe" not in feo.liquid_reference_status
    assert "D&G/Masson" not in (feo.domain_flags["Fe"] or "")

    def notice_records(result):
        return [
            (
                notice.oxide,
                notice.declared_standard_id,
                notice.active_standard_id,
                notice.row_key,
                notice.kind,
            )
            for notice in result.notices
            if isinstance(notice, PackStandardNotice)
        ]

    assert notice_records(feo) == _PUBLISHED_GAS_STANDARD_NOTICES
    with pytest.raises(ImccLiquidStandardMismatchError):
        evaluate_gas(
            {"FeO": 1.0}, 1700.0, 1e-8,
            parent_oxides=("FeO",), gas_species=("Fe",), profile=profile,
            strict_pack_standards=True,
        )

    alumina_row = profile.gas_datapack.oxide_df.loc["Al2O3(l)"]
    assert str(alumina_row["Ref"]) == "LAM1987"
    alumina = evaluate_gas(
        {"Al2O3": 1.0}, 2500.0, 1e-8,
        parent_oxides=("Al2O3",), gas_species=("Al",), profile=profile,
    )
    assert all(
        status.species != "Al2O3(l)"
        for status in alumina.liquid_reference_status.values()
    )
    assert "band" not in (alumina.domain_flags["Al"] or "")
    assert notice_records(alumina) == _PUBLISHED_GAS_STANDARD_NOTICES
    with pytest.raises(ImccLiquidStandardMismatchError):
        evaluate_gas(
            {"Al2O3": 1.0}, 2500.0, 1e-8,
            parent_oxides=("Al2O3",), gas_species=("Al",), profile=profile,
            strict_pack_standards=True,
        )


def test_openimcc_profile_selects_the_default_redox_pack() -> None:
    default = evaluate_redox({"Fe": 1.0, "O": 1.25}, 2200.0, "closed")
    profiled = evaluate_redox(
        {"Fe": 1.0, "O": 1.25}, 2200.0, "closed", profile="openimcc"
    )

    assert profiled.lambda_ln_f_o2 == default.lambda_ln_f_o2
    assert profiled.redox_pack_digest == default.redox_pack_digest
    assert profiled.melt_pack_binding_digest == default.melt_pack_binding_digest
    assert profiled.profile == "openimcc"
    assert default.profile is None
    for result_field in fields(default):
        if result_field.name not in {"profile", "receipt"}:
            assert getattr(profiled, result_field.name) == getattr(
                default, result_field.name
            )
