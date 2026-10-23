from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from openimcc import load_datapack
from openimcc.gas import R_J_MOL_K, load_gas_datapack, species_thermo
from openimcc.redox_pack import load_redox_pack, species_gibbs_j_mol
from tools.liquid_from_solid import liquid_from_solid


ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "src/openimcc/data/packs"
GAS = ROOT / "src/openimcc/data/gas"


def _candidate_gas():
    return load_gas_datapack(
        gas_path=GAS / "gas-shomate.csv",
        oxide_path=GAS / "condensate-sio2-na2o-liquid-candidate.csv",
    )


def test_candidate_pack_identity_and_default_remain_separate() -> None:
    candidate_path = PACKS / "imcc-sf04-d066-sio2-na2o-candidate.json"
    candidate = load_datapack(candidate_path)
    default = load_datapack()
    manifest = json.loads((PACKS / "MANIFEST.json").read_text())
    entries = {entry["file"]: entry for entry in manifest["packs"]}

    assert candidate.version == "1.0.6-d066-sio2-na2o-candidate"
    assert candidate.kernel_datapack.binding_digest == "51179602bfd1bbc8be071bead0f40f9d8ff407227e5a4a904ab8a158fe471efd"
    assert entries[candidate_path.name]["sha256"] == hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    assert default.version == "1.0.7-d066-v5"
    assert default.kernel_datapack.binding_digest == "13a185563870ec1a9c20d51bd25fb58df08c9bbbbdf7b2212b0f43ead64738cf"
    assert default.kernel_datapack.binding_digest != candidate.kernel_datapack.binding_digest


def test_previous_defaults_and_condensate_remain_loadable() -> None:
    previous_melt = load_datapack(PACKS / "imcc-sf04-d066-v3.json")
    previous_redox = load_redox_pack(PACKS / "openimcc-redox-v3.json")
    assert previous_melt.version == "1.0.5-d066-v3"
    assert previous_redox.version == "1.6.0"

    legacy = load_gas_datapack(
        oxide_path=GAS / "condensate-sio2-na2o-janaf-legacy.csv"
    )
    current = load_gas_datapack()
    assert legacy.condensate_table_digest != current.condensate_table_digest
    for species, table_id in (
        ("SiO2(l)", "SiO2-RBT84-CANDIDATE"),
        ("Na2O(l)", "Na2O-RBT84-CANDIDATE"),
    ):
        current_rows = current.oxide_df.loc[species]
        if getattr(current_rows, "ndim", 1) == 1:
            current_rows = [current_rows]
        else:
            current_rows = [row for _, row in current_rows.iterrows()]
        assert {row["table_id"] for row in current_rows} == {table_id}
    with (GAS / "condensate-sio2-na2o-janaf-legacy.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    for species in ("SiO2(l)", "Na2O(l)"):
        assert all(row["table_id"] not in {"SiO2-RBT84-CANDIDATE", "Na2O-RBT84-CANDIDATE"}
                   for row in rows if row["species_name"] == species)


def test_candidate_liquids_match_shared_construction_and_join_in_g_and_s() -> None:
    gas = _candidate_gas()
    cases = {
        "SiO2": {
            "table": "O-035",
            "tm": 1996.0,
            "hfus": 9.581,
            "cp": 81.37,
            "joins": (1500.0, 1800.0, 1996.0),
            "cation": "Si",
            "cat_num": 1,
            "oxy_num": 2,
        },
        "Na2O": {
            "table": "Na-012",
            "tm": 1405.2,
            "hfus": 47.698,
            "cp": 100.6,
            "joins": (1405.2, 1500.0, 1750.0, 2000.0, 2250.0, 2500.0, 2750.0),
            "cation": "Na",
            "cat_num": 2,
            "oxy_num": 1,
        },
    }
    for species, values in cases.items():
        crystal = json.loads(
            (ROOT / f"data-src/janaf/{values['table']}.yaml").read_text()
        )
        inputs = {
            "fusion_temperature": {"value": values["tm"]},
            "fusion_entropy": {"value": values["hfus"] * 1000 / values["tm"]},
            "liquid_heat_capacity": {"value": values["cp"]},
        }
        construction = liquid_from_solid(species, crystal, inputs)
        Tm = values["tm"]
        closed_form = json.loads(
            (PACKS / "imcc-sf04-d066-sio2-na2o-candidate.json").read_text()
        )["parent_standard_candidates"]["construction"]
        assert "−S_s(T_m)(T−T_m)" in closed_form
        test_temperature = 2200.0
        crystal_g_at_fusion = (
            construction.dfh298_kj_mol * 1000
            + construction.crystal_enthalpy_increment_at_fusion_kj_mol * 1000
            - Tm * construction.crystal_entropy_at_fusion_j_mol_k
        )
        formula_g = (
            crystal_g_at_fusion
            - construction.crystal_entropy_at_fusion_j_mol_k * (test_temperature - Tm)
            + values["hfus"] * 1000 * (1 - test_temperature / Tm)
            + values["cp"]
            * (
                (test_temperature - Tm)
                - test_temperature * math.log(test_temperature / Tm)
            )
        )
        stored = species_thermo(species, "l", test_temperature, gas)
        assert abs(formula_g - construction.at(test_temperature).gibbs_kj_mol * 1000) < 1e-8
        assert abs(stored.G_J_mol - formula_g) < 1.0
        candidate_at_fusion = species_thermo(species, "l", Tm, gas)
        target_entropy = (
            construction.crystal_entropy_at_fusion_j_mol_k
            + inputs["fusion_entropy"]["value"]
        )
        assert abs(candidate_at_fusion.G_J_mol - construction.at(Tm).gibbs_kj_mol * 1000) < 0.05
        assert abs(candidate_at_fusion.S_J_molK - target_entropy) < 1e-4

        for join in values["joins"]:
            left = species_thermo(species, "l", join - 1e-6, gas)
            right = species_thermo(species, "l", join + 1e-6, gas)
            assert abs(right.G_J_mol - left.G_J_mol) < 0.002
            assert abs(right.S_J_molK - left.S_J_molK) < 1e-5

    na2o_gas = gas
    na2o_construction = liquid_from_solid(
        "Na2O",
        json.loads((ROOT / "data-src/janaf/Na-012.yaml").read_text()),
        {
            "fusion_temperature": {"value": 1405.2},
            "fusion_entropy": {"value": 47.698 * 1000 / 1405.2},
            "liquid_heat_capacity": {"value": 100.6},
        },
    )
    for temperature in range(1500, 3001, 25):
        stored = species_thermo("Na2O", "l", float(temperature), na2o_gas)
        analytic = na2o_construction.at(float(temperature))
        assert abs(stored.Cp_J_molK - 100.6) < 0.2
        assert abs(stored.G_J_mol - analytic.gibbs_kj_mol * 1000) < 1.0


def test_all_affected_associations_record_fit_residuals_and_flags() -> None:
    path = PACKS / "imcc-sf04-d066-sio2-na2o-candidate.json"
    content = json.loads(path.read_text())
    changed = [row for row in content["rows"] if "standard_state_refit" in row]
    assert len(changed) == 26
    for row in changed:
        fit = row["standard_state_refit"]
        assert fit["max_abs_residual_dex"] >= 0
        assert fit["flag_over_0_01_dex"] is (fit["max_abs_residual_dex"] > 0.01)
        assert fit["temperature_range_K"] == row["T_domain_K"]
    assert any(row["standard_state_refit"]["flag_over_0_01_dex"] for row in changed)


def test_association_refit_uses_the_stored_candidate_parent_functions() -> None:
    candidate_path = PACKS / "imcc-sf04-d066-sio2-na2o-candidate.json"
    candidate_data = json.loads(candidate_path.read_text())
    default_data = json.loads((PACKS / "imcc-sf04-d066-v3.json").read_text())
    candidate_row = next(row for row in candidate_data["rows"] if row["complex"] == "Na2SiO3")
    default_row = next(row for row in default_data["rows"] if row["complex"] == "Na2SiO3")
    candidate_gas = _candidate_gas()
    default_gas = load_gas_datapack(oxide_path=GAS / "condensate-sio2-na2o-janaf-legacy.csv")
    residuals = []
    for index in range(1001):
        temperature = 1700 + (3000 - 1700) * index / 1000
        delta_g = 0.0
        for parent in ("SiO2", "Na2O"):
            nu = float(candidate_row["nu"].get(parent, 0))
            if nu:
                candidate_g = species_thermo(parent, "l", temperature, candidate_gas).G_J_mol
                default_parent_g = species_thermo(parent, "l", temperature, default_gas).G_J_mol
                delta_g += nu * (candidate_g - default_parent_g)
        target = (
            float(default_row["A"])
            + float(default_row["B"]) / temperature
            + delta_g / (R_J_MOL_K * temperature * math.log(10))
        )
        fitted = float(candidate_row["A"]) + float(candidate_row["B"]) / temperature
        residuals.append(abs(target - fitted))
    assert math.isclose(
        max(residuals),
        candidate_row["standard_state_refit"]["max_abs_residual_dex"],
        rel_tol=0,
        abs_tol=1e-10,
    )


def test_redox_candidate_uses_the_same_silica_standard() -> None:
    candidate_path = PACKS / "openimcc-redox-sio2-na2o-candidate.json"
    candidate = load_redox_pack(candidate_path)
    default = load_redox_pack()
    gas = _candidate_gas()
    assert candidate.version == "1.7.0-sio2-na2o-candidate"
    assert candidate.canonical_digest == "78de6c0591af964d39d2f1282534e23eb78a7c2334e74d4ad6510f91f0e20fe1"
    assert default.version == "1.7.0"
    assert default.canonical_digest == "9dd2422e618d74ae0a5761f9b4343e38d5b88f5986a6ffd801cdabd8002334c4"
    wrapper = candidate.content["thermo_species"]["SiO2(l)"]
    assert "Candidate SiO2(l) standard" in wrapper["range_flags"][0]
    assert "Richet, Bottinga & Tequi" in wrapper["provenance"]["source"]
    assert wrapper["band_basis"]["provenance"] == wrapper["provenance"]["source"]
    assert wrapper["fit_residual"]["value"] < 0.001
    manifest = json.loads((PACKS / "MANIFEST.json").read_text())
    entries = {entry["file"]: entry for entry in manifest["packs"]}
    assert entries[candidate_path.name]["sha256"] == hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    for temperature in (1700.0, 1996.0, 2200.0, 2473.0):
        redox_g = species_gibbs_j_mol(
            candidate, "SiO2(l)", temperature, allow_extrapolation=True
        )
        gas_g = species_thermo("SiO2", "l", temperature, gas).G_J_mol
        assert math.isclose(redox_g, gas_g, rel_tol=0, abs_tol=0.05)
