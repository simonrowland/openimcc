import json
import math
from pathlib import Path

import pytest

from openimcc import evaluate_redox
from openimcc.gas import evaluate_gas, load_gas_datapack
from openimcc.redox_pack import load_redox_pack, species_gibbs_j_mol


LI_FRAGMENT_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "openimcc"
    / "data"
    / "fragments"
    / "li"
    / "rows.json"
)
LEGACY_REDOX_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "openimcc"
    / "data"
    / "packs"
    / "openimcc-redox-cr-candidate-v4.json"
)


def test_li_gas_reference_matches_printed_janaf_node() -> None:
    pack = load_redox_pack()

    # JANAF Li-005, 2000 K row: Hf(298)=159.300 kJ/mol,
    # H-H(298)=35.386 kJ/mol, and S=178.349 J/(mol K).
    expected_g_j_mol = (159.300 + 35.386) * 1000.0 - 2000.0 * 178.349
    assert species_gibbs_j_mol(pack, "Li(g)", 2000.0) == pytest.approx(
        expected_g_j_mol, abs=2.0
    )


def test_fragment_rows_are_folded_into_the_packaged_default() -> None:
    previous_default = load_redox_pack(LEGACY_REDOX_PATH)
    fragment = json.loads(LI_FRAGMENT_PATH.read_text(encoding="utf-8"))
    default = load_redox_pack()

    assert default.canonical_digest != previous_default.canonical_digest
    assert any(parent["element"] == "Li" for parent in default.parents)
    assert "Li2O(l)" in default.content["thermo_species"]
    packaged_reactions = {row["reaction"] for row in default.rows}
    assert {row["reaction"] for row in fragment["rows"]} <= packaged_reactions


def test_li2o_liquid_row_matches_printed_janaf_fusion_anchor() -> None:
    pack = load_redox_pack()

    # JANAF Li-014/Li-015, 1843 K rows; enthalpy is kJ/mol and entropy is J/mol/K.
    tm = 1843.0
    h_cr = -598.730 + 131.649
    s_cr = 181.370
    h_liq = -553.247 + 144.742
    s_liq = 213.153
    delta_h_fus = h_liq - h_cr
    delta_s_fus = s_liq - s_cr

    assert delta_h_fus == pytest.approx(58.576, abs=1e-12)
    assert tm * delta_s_fus / 1000.0 == pytest.approx(58.576069, abs=1e-9)
    g_crystal_j_mol = (h_cr - tm * s_cr / 1000.0) * 1000.0
    g_liquid_source_j_mol = (h_liq - tm * s_liq / 1000.0) * 1000.0
    assert g_liquid_source_j_mol == pytest.approx(g_crystal_j_mol, abs=0.1)
    assert species_gibbs_j_mol(pack, "Li2O(l)", tm) == pytest.approx(
        g_liquid_source_j_mol, abs=8.0
    )


def test_li2sio3_fusion_uses_printed_crystal_and_calorimetry_anchors() -> None:
    pack = load_redox_pack()

    # Table 5 p. 315 formation/entropy anchors, Li-020 crystal row at 1474 K,
    # and Téqui et al. (1992) fusion calorimetry.
    tm = 1474.0
    hf_cr = -1646.1
    hinc_cr = 169.896
    s298_selected = 79.8
    s_janaf_298 = 80.295
    s_janaf_tm = 298.915
    delta_h_fus = 71.3
    delta_s_fus = delta_h_fus * 1000.0 / tm
    s_cr_tm = s298_selected + s_janaf_tm - s_janaf_298
    g_cr_tm_j_mol = (hf_cr + hinc_cr) * 1000.0 - tm * s_cr_tm

    assert delta_s_fus == pytest.approx(48.3717774763, abs=1e-9)
    assert tm * delta_s_fus / 1000.0 == pytest.approx(delta_h_fus, abs=1e-12)
    assert species_gibbs_j_mol(pack, "Li2SiO3(l)", tm) == pytest.approx(
        g_cr_tm_j_mol, abs=8.0
    )

    # The older Li-021 branch implies a different fusion enthalpy; keep this
    # cross-source conflict visible instead of folding it into a fit.
    h_janaf_liquid = -1625.231 + 173.660
    h_janaf_crystal = -1649.500 + hinc_cr
    assert h_janaf_liquid - h_janaf_crystal == pytest.approx(28.033, abs=1e-9)
    assert delta_h_fus - (h_janaf_liquid - h_janaf_crystal) == pytest.approx(
        43.267, abs=1e-9
    )


def test_li4sio4_fusion_and_kems_hess_targets_use_printed_anchors() -> None:
    pack = load_redox_pack()

    # Kleykamp's printed H(T)-H(298.15) relation and its derivative Cp(T).
    t0, tm = 298.15, 1531.0
    hf_cr = -2328.9  # de Abreu and Fabrichnaya 2024, Table 5 p. 315, KEM.
    s298_cr = 126.0  # Table 5 assessed value, explicitly provisional.
    def h_printed(temperature: float) -> float:
        return (
            -17156.0
            + 73.694 * temperature
            + 0.103210 * temperature**2
            - 4163115.0 / temperature
        )

    hinc_tm = (h_printed(tm) - h_printed(t0)) / 1000.0
    s_increment = (
        73.694 * math.log(tm / t0)
        + 0.206420 * (tm - t0)
        + 4163115.0 / 2.0 * (1.0 / t0**2 - 1.0 / tm**2)
    )
    s_cr_tm = s298_cr + s_increment
    delta_h_fus = 53.0
    delta_s_fus = delta_h_fus * 1000.0 / tm
    g_cr_tm_j_mol = (hf_cr + hinc_tm) * 1000.0 - tm * s_cr_tm

    assert hinc_tm == pytest.approx(334.8431157705, abs=1e-9)
    assert delta_s_fus == pytest.approx(34.6178967995, abs=1e-9)
    assert tm * delta_s_fus / 1000.0 == pytest.approx(delta_h_fus, abs=1e-12)
    assert species_gibbs_j_mol(pack, "Li4SiO4(l)", tm) == pytest.approx(
        g_cr_tm_j_mol, abs=8.0
    )

    li2sio3_h298 = -1646.1 + 169.896 + 71.3 + 167.36 * (t0 - 1474.0) / 1000.0
    required_li4_h298 = 2.0 * 159.3 + li2sio3_h298 - 946.34
    cp_liquid4 = (
        required_li4_h298 - (hf_cr + hinc_tm + delta_h_fus)
    ) * 1000.0 / (t0 - tm)
    stored_cp = pack.content["thermo_species"]["Li4SiO4(l)"]["construction"][
        "liquid_heat_capacity_J_mol_K"
    ]
    li4sio4_h298 = hf_cr + hinc_tm + delta_h_fus + stored_cp * (t0 - tm) / 1000.0
    liquid_kems_predicted = 2.0 * 159.3 + li2sio3_h298 - li4sio4_h298
    solid_kems_predicted = 2.0 * 159.3 - 1646.1 - hf_cr

    assert stored_cp == pytest.approx(cp_liquid4, abs=0.001)
    assert li2sio3_h298 == pytest.approx(-1601.694256, abs=1e-9)
    assert li4sio4_h298 == pytest.approx(-2229.43388022, abs=1e-6)
    assert liquid_kems_predicted == pytest.approx(946.34, abs=0.001)
    # Ihle et al. Table 9 p. 60: this separate solid reaction exposes the
    # unresolved source mismatch and was not used to fit the liquid Cp.
    assert solid_kems_predicted == pytest.approx(1001.4, abs=1e-9)
    assert solid_kems_predicted - 960.70 == pytest.approx(40.70, abs=1e-9)
    assert solid_kems_predicted - 957.26 == pytest.approx(44.14, abs=1e-9)
    li4_row = next(row for row in pack.rows if row.get("complex") == "Li4SiO4(l)")
    assert li4_row["independent_check"]["status"] == "tested"


def test_li_hess_rows_use_the_declared_standard_species() -> None:
    pack = load_redox_pack()

    fragment_rows = json.loads(LI_FRAGMENT_PATH.read_text(encoding="utf-8"))["rows"]
    packaged_reactions = {row["reaction"] for row in pack.rows}
    assert {row["reaction"] for row in fragment_rows} <= packaged_reactions
    for row in fragment_rows:
        temperature = 2000.0
        delta_g = math.fsum(
            term["coefficient"]
            * species_gibbs_j_mol(pack, term["species"], temperature)
            for term in row["standard_gibbs_terms"]
        )
        assert math.isfinite(delta_g)
        assert tuple(row["T_domain_K"]) == (1323, 2473)


def test_li_gas_channels_are_enabled_without_changing_non_li_outputs() -> None:
    feed = {
        "Si": 0.50,
        "Mg": 0.10,
        "Fe": 0.10,
        "Ca": 0.10,
        "Al": 0.15,
        "Ti": 0.02,
        "Na": 0.02,
        "K": 0.01,
        "Li": 0.01,
        "O": 1.60,
    }
    temperature = 2000.0
    non_li_feed = {name: amount for name, amount in feed.items() if name != "Li"}
    default_redox = evaluate_redox(non_li_feed, temperature, "closed")
    legacy_redox = evaluate_redox(
        non_li_feed, temperature, "closed", redox_pack_path=LEGACY_REDOX_PATH
    )
    assert default_redox.lambda_ln_f_o2.hex() == legacy_redox.lambda_ln_f_o2.hex()
    assert all(
        default_redox.parent_activity[name].hex()
        == legacy_redox.parent_activity[name].hex()
        for name in legacy_redox.parent_activity
    )

    redox = evaluate_redox(feed, temperature, "closed")
    lithium_activity = redox.parent_activity["Li2O"]
    oxygen_fugacity = math.exp(redox.lambda_ln_f_o2)
    assert lithium_activity > 0.0 and math.isfinite(lithium_activity)

    gas_pack = load_gas_datapack()
    li_gas = evaluate_gas(
        {"Li2O": lithium_activity},
        temperature,
        oxygen_fugacity,
        gas_pack,
        parent_oxides=("Li2O",),
        gas_species=("Li", "LiO", "Li2O", "Li2O2"),
    )
    assert set(li_gas) == {"Li", "LiO", "Li2O", "Li2O2"}
    assert all(value > 0.0 and math.isfinite(value) for value in li_gas.values())

    non_li_activities = {
        name: value for name, value in redox.parent_activity.items() if name != "Li2O"
    }
    default_gas = evaluate_gas(
        non_li_activities, temperature, oxygen_fugacity, gas_pack
    )
    extended_gas = evaluate_gas(
        {**non_li_activities, "Li2O": lithium_activity},
        temperature,
        oxygen_fugacity,
        gas_pack,
        parent_oxides=(
            "SiO2",
            "MgO",
            "FeO",
            "CaO",
            "Al2O3",
            "TiO2",
            "Na2O",
            "K2O",
            "Li2O",
        ),
    )
    assert set(default_gas) <= set(extended_gas)
    assert all(default_gas[name].hex() == extended_gas[name].hex() for name in default_gas)
