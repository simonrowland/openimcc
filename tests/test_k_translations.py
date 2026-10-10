"""d-066 K-reference translation and E25 arm acceptance tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from openimcc import ImccMalformedDatapackError, evaluate, load_datapack
import openimcc.kernel as kernel_module
import openimcc.model as model
from openimcc.gas import (
    R_J_MOL_K,
    evaluate_gas,
    load_gas_datapack,
    species_thermo,
)


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "src/openimcc/data/packs"
LEGACY_PATH = PACK_DIR / "imcc-sf04-ext-v4.json"
PUBLISHED_PATH = PACK_DIR / "imcc-sf04-v1.0.2.json"
DEFAULT_PATH = PACK_DIR / "imcc-sf04-d066-v1.json"
SENSITIVITY_PATH = PACK_DIR / "imcc-sf04-d066-v1-kcaalsi2o7.json"
EXTENSION_PATH = PACK_DIR / "imcc-sf04-d066-ext-v1.json"
FC87_GRID_K = (1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2500, 2800, 3000)
FC87_GRID_G_KJ_MOL = (
    -550.392,
    -578.213,
    -606.515,
    -635.263,
    -664.428,
    -693.983,
    -723.907,
    -754.178,
    -784.779,
    -846.909,
    -942.232,
    -1039.868,
    -1106.141,
)
TRANSLATED_COMPLEXES = ("KAlSiO4", "KAlSi3O8", "KAlO2", "KAlSi2O6")
REFERENCE_UNKNOWN_COMPLEXES = ("K2SiO3", "K2Si2O5", "K2Si4O9")
K2O_BAND_TOLERANCE_DEX_PER_K = 0.4
COMPOSITIONS = (
    {"SiO2": 0.45, "MgO": 0.20, "FeO": 0.10, "CaO": 0.08, "Al2O3": 0.08, "TiO2": 0.02, "Na2O": 0.05, "K2O": 0.02},
    {"SiO2": 0.52, "MgO": 0.12, "FeO": 0.08, "CaO": 0.07, "Al2O3": 0.10, "TiO2": 0.02, "Na2O": 0.03, "K2O": 0.06},
    {"SiO2": 0.38, "MgO": 0.15, "FeO": 0.09, "CaO": 0.08, "Al2O3": 0.12, "TiO2": 0.03, "Na2O": 0.04, "K2O": 0.09},
    {"SiO2": 0.25, "MgO": 0.10, "FeO": 0.10, "CaO": 0.10, "Al2O3": 0.12, "TiO2": 0.03, "Na2O": 0.08, "K2O": 0.20},
)


def _rows(path: Path) -> dict[str, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["complex"]: row for row in data["rows"]}


def test_loader_filters_inactive_row_and_preserves_full_manifest_digest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))
    if hasattr(model, "_D066_ROW_COVERAGE"):
        published_core = False
        data["model_id"] = "IMCC-SF04-D066"
        data["imcc_sf04_datapack_version"] = "1.0.3-d066-loader-test"
        for source_row in data["rows"]:
            source_row["coverage"] = model._D066_ROW_COVERAGE[source_row["complex"]]
            source_row["d066_source_row"] = {
                "pack": "imcc-sf04-v1.0.2.json",
                "row": source_row["row"],
            }
            if source_row["complex"] in model._D066_REFERENCE_UNKNOWN_ROWS:
                source_row["flags"] = ["reference_unknown"]
        reason = model._D066_E25_INACTIVE_REASON
    else:
        published_core = True
        data["model_id"] = "IMCC-SF04"
        reason = "E25: out of liquid domain; owner-confirmed"
    row = next(row for row in data["rows"] if row["complex"] == "KCaAlSi2O7")
    row["active"] = False
    row["inactive_reason"] = reason
    if published_core:
        expected_published_hash = model._published_datapack_manifest_hash(
            model._published_core_manifest_payload(
                data,
                model_id="IMCC-SF04",
                version=data["imcc_sf04_datapack_version"],
            )
        )
        monkeypatch.setattr(model, "_PUBLISHED_DATAPACK_SHA256", expected_published_hash)
        monkeypatch.setattr(
            kernel_module,
            "_PUBLISHED_DATAPACK_SHA256",
            expected_published_hash,
        )

    path = tmp_path / "inactive-d066-core-fixture.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_datapack(path)
    kernel = loaded.kernel_datapack

    assert loaded.inactive_rows == (("KCaAlSi2O7", reason),)
    assert "KCaAlSi2O7" not in kernel.reactions
    assert set(kernel.coverage) == set(kernel.parent_oxides) | set(kernel.reactions)
    assert loaded.binding_digest == model._published_datapack_manifest_hash(data)


def test_loader_defaults_unmarked_published_rows_to_active() -> None:
    loaded = load_datapack(PUBLISHED_PATH)

    assert loaded.inactive_rows == ()
    assert "KCaAlSi2O7" in loaded.kernel_datapack.reactions


def test_default_is_d066_and_legacy_packs_remain_selectable() -> None:
    default_pack = load_datapack()
    assert default_pack.model_id == "IMCC-SF04-D066"
    assert default_pack.version == "1.0.3-d066-v1"
    assert "A-published-imcc" not in default_pack.kernel_datapack.coverage.values()
    assert load_datapack(LEGACY_PATH).version == "1.0.2-ext-sp-2"
    assert load_datapack(PUBLISHED_PATH).version == "1.0.2"
    assert load_datapack(SENSITIVITY_PATH).version == "1.0.3-d066-v1-kcaalsi2o7"
    assert load_datapack(EXTENSION_PATH).version == "1.0.3-d066-ext-v1"


def test_d066_row_set_coverage_and_v1_lineage_are_validated(tmp_path: Path) -> None:
    data = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    assert all(
        row["d066_source_row"] == {
            "pack": "imcc-sf04-v1.0.2.json",
            "row": row["row"],
        }
        for row in data["rows"]
    )
    assert {row["coverage"] for row in data["rows"]} == {
        "D-carried-published",
        "D-translated-d066",
        "D-reference-unknown",
        "D-e25-kcaalsi2o7",
    }

    data["rows"][0]["coverage"] = "A-published-imcc"
    bad_path = tmp_path / "d066-published-coverage.json"
    bad_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ImccMalformedDatapackError, match="invalid coverage label"):
        load_datapack(bad_path)


def test_d066_pack_cannot_claim_the_published_hash(tmp_path: Path) -> None:
    data = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    data["model_id"] = "IMCC-SF04"
    data["imcc_sf04_datapack_version"] = "1.0.2"
    bad_path = tmp_path / "d066-claiming-published.json"
    bad_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ImccMalformedDatapackError):
        load_datapack(bad_path)


def test_d066_extension_carries_ext_v4_overlay_under_its_own_identity() -> None:
    ext_source = json.loads(LEGACY_PATH.read_text(encoding="utf-8"))
    ext_data = json.loads(EXTENSION_PATH.read_text(encoding="utf-8"))
    assert ext_data["sp_extension"] == ext_source["sp_extension"]
    pack = load_datapack(EXTENSION_PATH)
    assert pack.model_id == "IMCC-SF04-D066"
    assert pack.extension_species == tuple(
        row["complex"] for row in ext_source["sp_extension"]["rows"]
    )
    result = evaluate(
        COMPOSITIONS[0],
        2000.0,
        pack,
        enable_sp_extension=True,
        allow_out_of_envelope=True,
    )
    assert result.labels.identity["model_id"] == "IMCC-SF04-D066"
    assert {"S", "P2O5"} <= set(result.parent_oxides)


def test_d066_extension_k_rows_match_the_core_arm() -> None:
    core = _rows(DEFAULT_PATH)
    extension = _rows(EXTENSION_PATH)
    for name in (*TRANSLATED_COMPLEXES, *REFERENCE_UNKNOWN_COMPLEXES, "KCaAlSi2O7"):
        assert extension[name]["A"] == core[name]["A"]
        assert extension[name]["B"] == core[name]["B"]
        assert extension[name]["coverage"] == core[name]["coverage"]


@pytest.mark.parametrize(
    ("source_path", "arm", "output_path"),
    [
        (PUBLISHED_PATH, "primary", DEFAULT_PATH),
        (PUBLISHED_PATH, "sensitivity", SENSITIVITY_PATH),
        (LEGACY_PATH, "primary", EXTENSION_PATH),
    ],
)
def test_translation_tool_reproduces_each_shipped_d066_pack(
    source_path: Path, arm: str, output_path: Path,
) -> None:
    from tools.translate_k_constants import translate_k_rows

    new_reference = load_gas_datapack()
    lam_reference = load_gas_datapack(
        oxide_path=PACK_DIR / "sf04-published/condensate.csv"
    )
    generated = translate_k_rows(
        json.loads(source_path.read_text(encoding="utf-8")),
        new_reference,
        lam_reference,
        arm=arm,
    )
    expected = json.loads(output_path.read_text(encoding="utf-8"))

    def assert_pack_matches(actual: object, wanted: object) -> None:
        if isinstance(wanted, bool) or isinstance(actual, bool):
            assert actual is wanted
        elif isinstance(wanted, (int, float)) and isinstance(actual, (int, float)):
            assert actual == pytest.approx(wanted, rel=1e-12, abs=1e-9)
        elif isinstance(wanted, dict) and isinstance(actual, dict):
            assert actual.keys() == wanted.keys()
            for key in wanted:
                assert_pack_matches(actual[key], wanted[key])
        elif isinstance(wanted, list) and isinstance(actual, list):
            assert len(actual) == len(wanted)
            for actual_item, wanted_item in zip(actual, wanted):
                assert_pack_matches(actual_item, wanted_item)
        else:
            assert actual == wanted

    assert_pack_matches(generated, expected)


def test_fc87_reference_reproduces_the_species_grid() -> None:
    from tools.translate_k_constants import fc87_k2o_gibbs_j_mol

    gas = load_gas_datapack()
    actual = [fc87_k2o_gibbs_j_mol(T, gas) / 1000.0 for T in FC87_GRID_K]
    assert actual == pytest.approx(FC87_GRID_G_KJ_MOL, abs=0.01)


def test_translation_preserves_absolute_complex_g_on_the_domain_grid() -> None:
    from tools.translate_k_constants import (
        _fit_reference_shift,
        fc87_k2o_gibbs_j_mol,
    )

    old = _rows(PUBLISHED_PATH)
    new = _rows(DEFAULT_PATH)
    gas = load_gas_datapack()
    fit_row = old[TRANSLATED_COMPLEXES[0]]
    fc87_fit = _fit_reference_shift(fit_row, gas, gas, "FC87")
    max_residual = 0.0
    for name in TRANSLATED_COMPLEXES:
        old_row, new_row = old[name], new[name]
        assert old_row["T_domain_K"] == fit_row["T_domain_K"]
        assert old_row["nu"]["K2O"] == fit_row["nu"]["K2O"]
        assert float(new_row["A"]) - float(old_row["A"]) == pytest.approx(
            fc87_fit["delta_A"], abs=1e-12
        )
        assert float(new_row["B"]) - float(old_row["B"]) == pytest.approx(
            fc87_fit["delta_B_K"], abs=1e-9
        )
        nu_k2o = float(old_row["nu"]["K2O"])
        for temperature in (1700.0, 2000.0, 2500.0, 3000.0):
            reference_shift = (
                species_thermo("K2O", "l", temperature, gas).G_J_mol
                - fc87_k2o_gibbs_j_mol(temperature, gas)
            )
            old_log_k = float(old_row["A"]) + float(old_row["B"]) / temperature
            new_log_k = float(new_row["A"]) + float(new_row["B"]) / temperature
            residual_dex = new_log_k - old_log_k - nu_k2o * reference_shift / (
                R_J_MOL_K * temperature * math.log(10.0)
            )
            max_residual = max(max_residual, abs(residual_dex))
            # The allowance is set by the stated ±0.4 dex/K K2O(l) source band,
            # not by the observed least-squares residual.
            assert abs(residual_dex) <= 2 * nu_k2o * K2O_BAND_TOLERANCE_DEX_PER_K
    for name in TRANSLATED_COMPLEXES:
        record = new[name]["k_translation"]
        assert record["delta_A"] == pytest.approx(-0.268096866344, abs=1e-12)
        assert record["delta_B_K"] == pytest.approx(448.875854425, abs=1e-9)
        assert record["delta_A"] < 0.0
        assert record["delta_B_K"] > 0.0
        assert record["max_abs_residual_dex"] == pytest.approx(
            0.02709743389, abs=1e-10
        )
        assert record["max_abs_complex_G_error_kJ_mol"] == pytest.approx(
            1.21395, abs=0.001
        )
        assert record["fit_error_fraction_of_shift_at_T_min"] == pytest.approx(
            0.870, abs=0.001
        )
        assert "source-uncertainty budget" in record["tolerance_source"]
        assert "not a fit-precision claim" in record["tolerance_source"]
    assert max_residual < K2O_BAND_TOLERANCE_DEX_PER_K


def test_unknown_reference_rows_are_unchanged_and_carry_candidates() -> None:
    old = _rows(PUBLISHED_PATH)
    new = _rows(DEFAULT_PATH)
    for name in REFERENCE_UNKNOWN_COMPLEXES:
        assert new[name]["A"] == old[name]["A"]
        assert new[name]["B"] == old[name]["B"]
        assert "reference_unknown" in new[name]["flags"]
        candidates = new[name]["translation_candidates"]
        assert candidates["provenance_only"] is True
        assert {"FC87", "LAM"} <= set(candidates)


def test_not_closable_and_out_of_domain_flags_are_recorded() -> None:
    primary = _rows(DEFAULT_PATH)
    sensitivity = _rows(SENSITIVITY_PATH)
    assert primary["KAlSiO4"]["enthalpy_flag"] == "not_closable"
    assert primary["KAlSi2O6"]["enthalpy_flag"] == "not_closable"
    assert primary["KCaAlSi2O7"]["active"] is False
    assert primary["KCaAlSi2O7"]["inactive_reason"] == (
        "E25: out of liquid domain; owner-confirmed"
    )
    assert sensitivity["KCaAlSi2O7"]["active"] is True
    assert "inactive_reason" not in sensitivity["KCaAlSi2O7"]
    assert "kcaalsi2o7_out_of_liquid_domain" in sensitivity["KCaAlSi2O7"]["flags"]


def test_kcaalsi2o7_is_absent_from_primary_and_flagged_in_sensitivity() -> None:
    primary_pack_data = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    assert len(primary_pack_data["rows"]) == 38
    assert sum(row.get("active", True) is True for row in primary_pack_data["rows"]) == 37
    primary_pack = load_datapack()
    inactive_rows = (
        ("KCaAlSi2O7", "E25: out of liquid domain; owner-confirmed"),
    )
    assert primary_pack.inactive_rows == inactive_rows
    assert "KCaAlSi2O7" not in primary_pack.kernel_datapack.reactions
    primary = evaluate(
        COMPOSITIONS[0],
        2000.0,
        primary_pack,
        allow_out_of_envelope=True,
    )
    sensitivity_pack = load_datapack(SENSITIVITY_PATH)
    assert sensitivity_pack.inactive_rows == ()
    assert "KCaAlSi2O7" in sensitivity_pack.kernel_datapack.reactions
    sensitivity = evaluate(
        COMPOSITIONS[0],
        2000.0,
        sensitivity_pack,
        allow_out_of_envelope=True,
    )
    assert "KCaAlSi2O7" not in primary.species_names
    assert primary.labels.inactive_rows == inactive_rows
    assert "KCaAlSi2O7" in sensitivity.species_names
    assert sensitivity.labels.inactive_rows == ()
    assert "kcaalsi2o7_out_of_liquid_domain" in _rows(SENSITIVITY_PATH)[
        "KCaAlSi2O7"
    ]["flags"]


def test_loaded_coverage_matches_active_species_for_every_shipped_pack() -> None:
    manifest = json.loads((PACK_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["packs"]:
        if entry["file"] == "openimcc-redox-v1.json":
            continue
        loaded = load_datapack(PACK_DIR / entry["file"])
        kernel = loaded.kernel_datapack
        active_species = set(kernel.parent_oxides) | set(kernel.reactions)
        assert set(kernel.coverage) == active_species, entry["file"]


def test_loader_still_requires_38_published_core_rows(
    tmp_path: Path,
) -> None:
    data = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))
    data["rows"] = [
        row for row in data["rows"] if row["complex"] != "KCaAlSi2O7"
    ]
    path = tmp_path / "37-published-core-rows.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(
        ImccMalformedDatapackError,
        match="exactly 38 published-core rows",
    ):
        load_datapack(path)


@pytest.mark.parametrize(
    ("active", "inactive_reason", "expected_message"),
    [
        (False, None, "inactive_reason must be a non-empty string"),
        (False, "", "inactive_reason must be a non-empty string"),
        (
            True,
            "E25: out of liquid domain; owner-confirmed",
            "inactive_reason is only allowed when active is false",
        ),
        (
            "false",
            "E25: out of liquid domain; owner-confirmed",
            "active must be a boolean",
        ),
        (
            None,
            "E25: out of liquid domain; owner-confirmed",
            "inactive_reason is only allowed when active is false",
        ),
    ],
)
def test_loader_rejects_invalid_inactive_row_metadata(
    tmp_path: Path,
    active: object,
    inactive_reason: object,
    expected_message: str,
) -> None:
    data = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))
    row = next(row for row in data["rows"] if row["complex"] == "KCaAlSi2O7")
    if active is None:
        row.pop("active", None)
    else:
        row["active"] = active
    if inactive_reason is None:
        row.pop("inactive_reason", None)
    else:
        row["inactive_reason"] = inactive_reason
    path = tmp_path / "invalid-inactive-row.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ImccMalformedDatapackError, match=expected_message):
        load_datapack(path)


def test_free_fraction_pressure_bookkeeping_and_pure_k2o_endpoint() -> None:
    from tools.translate_k_constants import fc87_k2o_gibbs_j_mol

    old_pack = load_datapack(LEGACY_PATH)
    new_pack = load_datapack(DEFAULT_PATH)
    gas = load_gas_datapack()
    temperature = 1900.0
    oxygen_fugacity = 1.0e-6
    gk = species_thermo("K", "g", temperature, gas).G_J_mol
    go2 = species_thermo("O2", "g", temperature, gas).G_J_mol
    g_fc87 = fc87_k2o_gibbs_j_mol(temperature, gas)
    g_new = species_thermo("K2O", "l", temperature, gas).G_J_mol
    reference_shift = g_new - g_fc87

    def free_k2o_fraction(result, pack) -> float:
        kernel = pack.kernel_datapack
        k2o_index = kernel.parent_oxides.index("K2O")
        bound = sum(
            float(nu) * float(complex_x)
            for nu, complex_x in zip(
                kernel.nu[k2o_index], result.complex_x
            )
        )
        free = float(result.activity("K2O"))
        return free / (free + bound)

    for composition in COMPOSITIONS:
        before = evaluate(
            composition,
            temperature,
            old_pack,
            enable_sp_extension=True,
            allow_out_of_envelope=True,
        )
        after = evaluate(
            composition,
            temperature,
            new_pack,
            allow_out_of_envelope=True,
        )
        before_fraction = free_k2o_fraction(before, old_pack)
        after_fraction = free_k2o_fraction(after, new_pack)
        gas_after = evaluate_gas(
            dict(zip(after.parent_oxides, after.parent_activity)),
            temperature,
            oxygen_fugacity,
            gas,
            gas_species=("K",),
        )["K"]
        log_pk_before = 0.5 * (
            -(2 * gk + 0.5 * go2 - g_fc87)
            / (R_J_MOL_K * temperature * math.log(10.0))
            + math.log10(before.activity("K2O"))
            - 0.5 * math.log10(oxygen_fugacity)
        )
        observed_shift = math.log10(gas_after) - log_pk_before
        predicted_shift = reference_shift / (
            2 * R_J_MOL_K * temperature * math.log(10.0)
        ) + 0.5 * math.log10(after_fraction / before_fraction) + 0.5 * math.log10(
            after.D / before.D
        )
        assert observed_shift == pytest.approx(predicted_shift, abs=1e-10)

    pure_k2o = {name: 0.0 for name in new_pack.parent_oxides}
    pure_k2o["K2O"] = 1.0
    before = evaluate(
        pure_k2o,
        temperature,
        old_pack,
        enable_sp_extension=True,
        allow_out_of_envelope=True,
    )
    after = evaluate(
        pure_k2o,
        temperature,
        new_pack,
        allow_out_of_envelope=True,
    )
    log_pk_before = 0.5 * (
        -(2 * gk + 0.5 * go2 - g_fc87)
        / (R_J_MOL_K * temperature * math.log(10.0))
        + math.log10(before.activity("K2O"))
        - 0.5 * math.log10(oxygen_fugacity)
    )
    gas_after = evaluate_gas(
        dict(zip(after.parent_oxides, after.parent_activity)),
        temperature,
        oxygen_fugacity,
        gas,
        gas_species=("K",),
    )["K"]
    observed_shift = math.log10(gas_after) - log_pk_before
    full_reference_shift = reference_shift / (
        2 * R_J_MOL_K * temperature * math.log(10.0)
    ) + 0.5 * math.log10(
        free_k2o_fraction(after, new_pack) / free_k2o_fraction(before, old_pack)
    ) + 0.5 * math.log10(after.D / before.D)
    assert before.activity("K2O") == after.activity("K2O") == 1.0
    assert observed_shift == pytest.approx(full_reference_shift, abs=1e-10)


def test_non_k_composition_is_bit_identical_to_the_published_core() -> None:
    composition = {**COMPOSITIONS[0], "K2O": 0.0}
    published = evaluate(
        composition,
        2000.0,
        load_datapack(PUBLISHED_PATH),
        allow_out_of_envelope=True,
    )
    default = evaluate(
        composition,
        2000.0,
        load_datapack(),
        allow_out_of_envelope=True,
    )
    for field in (
        "parent_mol",
        "parent_x",
        "parent_x_star",
        "parent_activity",
        "parent_gamma",
    ):
        assert getattr(default, field).tolist() == getattr(published, field).tolist()
    published_species = dict(zip(published.species_names, published.species_x))
    default_species = dict(zip(default.species_names, default.species_x))
    assert default_species.keys() <= published_species.keys()
    assert all(
        default_species[name] == published_species[name]
        for name in default_species
    )
    assert default.D == published.D


def test_published_melt_and_lam_gas_pair_reproduces_its_k_pressure_pin() -> None:
    gas_data = ROOT / "src/openimcc/data"
    published_gas = load_gas_datapack(
        gas_path=gas_data / "gas/gas-shomate.csv",
        oxide_path=PACK_DIR / "sf04-published/condensate.csv",
    )
    melt = load_datapack(PUBLISHED_PATH)
    melt_result = evaluate(
        COMPOSITIONS[0],
        2000.0,
        melt,
        allow_out_of_envelope=True,
    )
    pressure = evaluate_gas(
        dict(zip(melt_result.parent_oxides, melt_result.parent_activity)),
        2000.0,
        1.0e-6,
        published_gas,
        gas_species=("K",),
    )["K"]
    assert float(pressure).hex() == "0x1.2786df7af2be4p-18"
