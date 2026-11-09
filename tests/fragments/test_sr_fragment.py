from __future__ import annotations

import json
import math
from pathlib import Path

from openimcc.gas import evaluate_gas, load_gas_datapack
from openimcc.gas_data import IMCC_PARENT_OXIDES
from openimcc.redox import evaluate_redox
from openimcc.redox_pack import load_redox_pack, species_gibbs_j_mol

ROOT = Path(__file__).resolve().parents[2]
FRAGMENT_PATH = ROOT / "src/openimcc/data/fragments/sr/rows.json"
R_J_MOL_K = 8.31446261815324
FUSION_ENTROPY_PER_ATOM = 8.297530569005671


def _fragment() -> dict:
    return json.loads(FRAGMENT_PATH.read_text(encoding="utf-8"))


def _liquid_from_printed_anchor(
    *,
    hf_cr_kj_mol: float,
    hinc_cr_fusion_kj_mol: float,
    s_cr_fusion_j_mol_k: float,
    fusion_temperature_k: float,
    fusion_entropy_j_mol_k: float,
    liquid_cp_j_mol_k: float,
    temperature_k: float,
) -> float:
    h_kj_mol = (
        hf_cr_kj_mol
        + hinc_cr_fusion_kj_mol
        + fusion_temperature_k * fusion_entropy_j_mol_k / 1000.0
        + liquid_cp_j_mol_k * (temperature_k - fusion_temperature_k) / 1000.0
    )
    entropy_j_mol_k = (
        s_cr_fusion_j_mol_k
        + fusion_entropy_j_mol_k
        + liquid_cp_j_mol_k * math.log(temperature_k / fusion_temperature_k)
    )
    return h_kj_mol * 1000.0 - temperature_k * entropy_j_mol_k


def _state_from_printed_cp_interval(
    temperature_k: float,
    *,
    source_temperature_k: float,
    enthalpy_increment_kj_mol: float,
    entropy_j_mol_k: float,
    cp_j_mol_k: float,
    next_temperature_k: float,
    next_cp_j_mol_k: float,
) -> tuple[float, float]:
    slope = (next_cp_j_mol_k - cp_j_mol_k) / (
        next_temperature_k - source_temperature_k
    )
    delta = temperature_k - source_temperature_k
    log_ratio = math.log(temperature_k / source_temperature_k)
    hinc = enthalpy_increment_kj_mol + (
        cp_j_mol_k * delta + 0.5 * slope * delta**2
    ) / 1000.0
    entropy = entropy_j_mol_k + cp_j_mol_k * log_ratio + slope * (
        delta - source_temperature_k * log_ratio
    )
    return hinc, entropy


def test_sr_liquid_standards_match_printed_janaf_anchors() -> None:
    pack = load_redox_pack()
    sro_standard = _fragment()["standards"]["SrO(l)"]
    sro_g_expected = (
        (-517.292 + 156.579) * 1000.0 - 2900.0 * 208.677
    )
    assert abs(species_gibbs_j_mol(pack, "SrO(l)", 2900.0) - sro_g_expected) < 1.0

    fusion_temperature_k = sro_standard["fusion_temperature_K"]
    crystal_entropy = 183.013 + 69.174 * math.log(fusion_temperature_k / 2900.0)
    liquid_entropy = 208.677 + 66.944 * math.log(fusion_temperature_k / 2900.0)
    printed_delta_s = liquid_entropy - crystal_entropy
    crystal_enthalpy = (
        -592.036 + 155.921 + 69.174 * (fusion_temperature_k - 2900.0) / 1000.0
    )
    liquid_enthalpy = (
        -517.292 + 156.579 + 66.944 * (fusion_temperature_k - 2900.0) / 1000.0
    )
    printed_delta_h = liquid_enthalpy - crystal_enthalpy
    assert abs(printed_delta_s - sro_standard["fusion_entropy_J_mol_K"]) < 0.002
    assert abs(printed_delta_h - sro_standard["fusion_enthalpy_kJ_mol"]) < 0.01

    sr_g_expected = (
        (0.911 + 59.267) * 1000.0 - 1800.0 * 121.853
    )
    assert abs(species_gibbs_j_mol(pack, "Sr(l)", 1800.0) - sr_g_expected) < 1.0


def test_silicate_liquid_rows_follow_printed_crystal_anchors() -> None:
    pack = load_redox_pack()
    temperature_k = 1800.0

    srsi_tfus = 1580.0 + 273.15
    srsi_hinc_fusion = 79.976 + 129.226 * (srsi_tfus - 1000.0) / 1000.0
    srsi_s_fusion = 229.082 + 129.226 * math.log(srsi_tfus / 1000.0)
    srsi_expected = _liquid_from_printed_anchor(
        hf_cr_kj_mol=-1635.3,
        hinc_cr_fusion_kj_mol=srsi_hinc_fusion,
        s_cr_fusion_j_mol_k=srsi_s_fusion,
        fusion_temperature_k=srsi_tfus,
        fusion_entropy_j_mol_k=5.0 * FUSION_ENTROPY_PER_ATOM,
        liquid_cp_j_mol_k=66.944 + 85.772,
        temperature_k=temperature_k,
    )
    assert abs(
        species_gibbs_j_mol(pack, "SrSiO3(l)", temperature_k) - srsi_expected
    ) < 1.0

    sr2_tfus = 2325.0 + 273.15
    sr2_hinc_fusion = 118.607 + 194.694 * (sr2_tfus - 1000.0) / 1000.0
    sr2_s_fusion = 353.753 + 194.694 * math.log(sr2_tfus / 1000.0)
    sr2_expected = _liquid_from_printed_anchor(
        hf_cr_kj_mol=-2305.7,
        hinc_cr_fusion_kj_mol=sr2_hinc_fusion,
        s_cr_fusion_j_mol_k=sr2_s_fusion,
        fusion_temperature_k=sr2_tfus,
        fusion_entropy_j_mol_k=7.0 * FUSION_ENTROPY_PER_ATOM,
        liquid_cp_j_mol_k=2.0 * 66.944 + 85.772,
        temperature_k=temperature_k,
    )
    assert abs(
        species_gibbs_j_mol(pack, "Sr2SiO4(l)", temperature_k) - sr2_expected
    ) < 1.0


def test_aluminate_liquid_row_follows_printed_janaf_and_assessment_anchors() -> None:
    pack = load_redox_pack()
    fusion_temperature_k = 2233.0
    sr_hinc, sr_s = _state_from_printed_cp_interval(
        fusion_temperature_k,
        source_temperature_k=2200.0,
        enthalpy_increment_kj_mol=109.109,
        entropy_j_mol_k=164.568,
        cp_j_mol_k=64.563,
        next_temperature_k=2300.0,
        next_cp_j_mol_k=65.229,
    )
    al_hinc, al_s = _state_from_printed_cp_interval(
        fusion_temperature_k,
        source_temperature_k=2200.0,
        enthalpy_increment_kj_mol=237.174,
        entropy_j_mol_k=284.143,
        cp_j_mol_k=138.030,
        next_temperature_k=2300.0,
        next_cp_j_mol_k=138.741,
    )
    expected = _liquid_from_printed_anchor(
        hf_cr_kj_mol=-592.036 - 1675.692 - 79.2,
        hinc_cr_fusion_kj_mol=sr_hinc + al_hinc,
        s_cr_fusion_j_mol_k=sr_s + al_s,
        fusion_temperature_k=fusion_temperature_k,
        fusion_entropy_j_mol_k=7.0 * FUSION_ENTROPY_PER_ATOM,
        liquid_cp_j_mol_k=66.944 + 192.464,
        temperature_k=1800.0,
    )
    assert abs(species_gibbs_j_mol(pack, "SrAl2O4(l)", 1800.0) - expected) < 1.0


def test_fragment_redox_and_gas_channels_are_packaged_and_preserve_other_pressures() -> None:
    pack = load_redox_pack()
    assert {
        row["complex"] for row in pack.rows
        if row.get("complex") in {"SrO(l)", "SrSiO3(l)", "Sr2SiO4(l)", "SrAl2O4(l)"}
    } == {
        "SrO(l)",
        "SrSiO3(l)",
        "Sr2SiO4(l)",
        "SrAl2O4(l)",
    }

    inventory = {
        "Si": 0.45,
        "Al": 0.14,
        "Mg": 0.38,
        "Ca": 0.13,
        "Na": 0.025,
        "K": 0.006,
        "Ti": 0.04,
        "Sr": 0.001,
    }
    temperature_k = 1800.0
    f_o2 = 1.0e-8
    melt = evaluate_redox(
        inventory,
        temperature_k,
        mode="imposed",
        lambda_imposed=math.log(f_o2),
    )
    assert "FeO" not in melt.parent_activity
    assert melt.parent_activity["SrO"] > 0.0

    gas_pack = load_gas_datapack()
    core_parents = tuple(IMCC_PARENT_OXIDES)
    extended_parents = (*core_parents, "SrO")
    all_activities = dict(melt.parent_activity)
    gas_with_sr = dict(
        evaluate_gas(
            all_activities,
            temperature_k,
            f_o2,
            datapack=gas_pack,
            parent_oxides=extended_parents,
        )
    )
    gas_without_sr = dict(
        evaluate_gas(
            {key: value for key, value in all_activities.items() if key != "SrO"},
            temperature_k,
            f_o2,
            datapack=gas_pack,
            parent_oxides=core_parents,
        )
    )
    sr_pressures = {
        name: value for name, value in gas_with_sr.items() if name.startswith("Sr")
    }
    assert sr_pressures
    assert all(value > 0.0 and math.isfinite(value) for value in sr_pressures.values())
    for species, pressure in gas_without_sr.items():
        assert gas_with_sr[species] == pressure
