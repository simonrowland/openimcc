"""Opt-in species thermochemistry continuation follows the gas runtime path."""

from __future__ import annotations

import json
import math
import re
from dataclasses import FrozenInstanceError
from functools import lru_cache
from pathlib import Path

import pytest

from openimcc.gas import (
    ImccGasCrystalPolymorphError,
    ImccGasTemperatureOutsideDomainError,
    SpeciesThermo,
    _janaf_gibbs,
    _lamor_gibbs,
    _nearest_interval_row,
    _oxide_row_for_T,
    evaluate_gas,
    load_gas_datapack,
    species_thermo,
)
from openimcc.crystal_species import (
    _barin_phases,
    _berman_phases,
    _cp_primitive,
    _berman_properties,
    evaluate_crystal_species,
)
from openimcc.janaf_crystals import evaluate_janaf_crystal


def _value_hexes(value: SpeciesThermo) -> dict[str, str]:
    return {
        name: getattr(value, name).hex()
        for name in (
            "Cp_J_molK",
            "S_J_molK",
            "H_app_kJ_mol",
            "G_J_mol",
        )
    }


@lru_cache(maxsize=1)
def _crystal_dataset() -> dict:
    path = Path(__file__).parents[1] / "src/openimcc/data/diagnostics/crystal-phases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _element_reference_sum_kJ_mol(formula: str, temperature_K: float) -> float | None:
    references = _crystal_dataset()["element_reference_g_app_nodes_kJ_mol"]
    atoms = {
        element: int(count or "1")
        for element, count in re.findall(r"([A-Z][a-z]?)([0-9]*)", formula)
    }
    total = 0.0
    node = str(int(temperature_K))
    for element, count in atoms.items():
        if element == "O":
            element, count = "O2", count / 2.0
        value = references.get(element, {}).get(node)
        if value is None:
            return None
        total += count * float(value)
    return total


def test_species_thermo_in_range_values_remain_byte_identical() -> None:
    legacy = SpeciesThermo(1.0, 2.0, 3.0, 4.0, "row", None, None, 10.0, 20.0)
    assert legacy.extrapolated is False
    assert legacy.continuation is None
    assert legacy.authority is None
    assert legacy.phase_identity is None
    assert legacy.formation_enthalpy_298_kJ_mol is None
    assert legacy.formation_enthalpy_source_id is None

    pack = load_gas_datapack()
    gas = species_thermo("O2", "g", 2000.0, pack)
    liquid = species_thermo("SiO2", "l", 2000.0, pack)
    crystal = species_thermo("SiO2", "cr", 1500.0, pack)

    assert _value_hexes(gas) == {
        "Cp_J_molK": "0x1.2def84315b766p+5",
        "S_J_molK": "0x1.0cbf4b7f5a86ep+8",
        "H_app_kJ_mol": "0x1.d9652d0829742p+5",
        "G_J_mol": "-0x1.d31c05b57dbffp+18",
    }
    assert _value_hexes(liquid) == {
        "Cp_J_molK": "0x1.4574931af766bp+6",
        "S_J_molK": "0x1.5994ed1502996p+7",
        "H_app_kJ_mol": "-0x1.86aa470158b5ep+9",
        "G_J_mol": "-0x1.1320008d49734p+20",
    }
    assert _value_hexes(crystal) == {
        "Cp_J_molK": "0x1.240610b27dc5cp+6",
        "S_J_molK": "0x1.25a4de21ba0e6p+7",
        "H_app_kJ_mol": "-0x1.9ddd493cd318dp+9",
        "G_J_mol": "-0x1.ffb339de40090p+19",
    }
    assert gas.extrapolated is False
    assert gas.declared_interval_K == (1500.0, 3000.0)
    assert gas.continuation is None
    assert gas.authority
    assert liquid.extrapolated is False
    assert liquid.declared_interval_K == (1996.0, 3000.0)
    assert liquid.continuation is None
    assert liquid.authority
    assert crystal.extrapolated is False
    assert crystal.declared_interval_K == (1000.0, 1996.0)
    assert crystal.continuation is None
    assert crystal.authority

    with pytest.raises(FrozenInstanceError):
        gas.extrapolated = True


def test_existing_condensate_crystal_queries_remain_byte_identical() -> None:
    pack = load_gas_datapack()
    expected = {
        ("C", 1500.0): {
            "Cp_J_molK": "0x1.7e2bd59b3662dp+4",
            "S_J_molK": "0x1.0da84dd7860b3p+5",
            "H_app_kJ_mol": "0x1.73c8106d45f13p+4",
            "G_J_mol": "-0x1.aaf1b402b4672p+14",
        },
        ("Os", 3000.0): {
            "Cp_J_molK": "0x1.3b5e920517bc2p+5",
            "S_J_molK": "0x1.8d744eb6b7757p+6",
            "H_app_kJ_mol": "0x1.535d002836dc8p+6",
            "G_J_mol": "-0x1.a080f63a03f39p+17",
        },
    }
    for (formula, temperature), values in expected.items():
        assert _value_hexes(
            species_thermo(formula, "cr", temperature, pack)
        ) == values


def test_shipped_crystal_dataset_fallback_and_polymorph_selection() -> None:
    janaf = species_thermo("TiO2", "cr", 1500.0)
    assert janaf.source_table_id == "O-043"
    assert janaf.source_row_id == "O-043"
    assert janaf.phase_identity is not None
    assert janaf.phase_identity["dataset"] == "JANAF pure-oxide crystals"
    assert janaf.G_J_mol == evaluate_janaf_crystal("TiO2", 1500.0).G_J_mol

    barin = species_thermo("CaAl2O4", "cr", 1800.0)
    assert barin.source_table_id == "Barin-1995"
    assert barin.source_row_id == "CA/SOL"
    assert barin.Cp_J_molK == 193.101
    assert barin.phase_identity is not None
    assert barin.phase_identity["polymorph"] == "SOL"
    assert barin.phase_identity["declared_interval_K"] == (298.15, 1800.0)
    assert barin.phase_identity["source_solid_limit_kind"] == "decomposition"

    berman = species_thermo("CaSiO3", "cr", 1500.0)
    wollastonite = species_thermo(
        "CaSiO3", "cr", 1500.0, polymorph="wollastonite"
    )
    assert berman.source_table_id == "Berman-1988"
    assert berman.phase_identity is not None
    assert berman.phase_identity["polymorph"] == "pseudowollastonite"
    assert "lowest apparent G" in berman.phase_identity["selection_rule"]
    assert wollastonite.phase_identity is not None
    assert wollastonite.phase_identity["polymorph"] == "wollastonite"
    assert berman.G_J_mol < wollastonite.G_J_mol

    robie_hemingway = species_thermo("CaTiO3", "cr", 1500.0)
    assert robie_hemingway.source_table_id == "Robie-Hemingway-1995"
    assert robie_hemingway.phase_identity is not None
    assert robie_hemingway.phase_identity["polymorph"] == "perovskite"

    with pytest.raises(ImccGasCrystalPolymorphError) as refusal:
        species_thermo("CaSiO3", "cr", 1500.0, polymorph="quartz")
    assert "wollastonite" in str(refusal.value)
    assert "pseudowollastonite" in str(refusal.value)


def test_additional_janaf_crystals_keep_titles_transitions_and_caps() -> None:
    lead_oxide = species_thermo("PbO", "cr", 800.0)
    assert lead_oxide.source_table_id == "O-006"
    assert lead_oxide.formation_enthalpy_298_kJ_mol == pytest.approx(-218.062)
    assert lead_oxide.phase_identity is not None
    assert lead_oxide.phase_identity["source_title"].startswith(
        "Lead Oxide, Yellow"
    )
    assert lead_oxide.phase_identity["polymorph"] == "II"
    assert lead_oxide.phase_identity["temperature_segment_K"] == (762.0, 1159.0)

    alpha_tio = species_thermo("TiO", "cr", 1000.0)
    beta_tio = species_thermo("TiO", "cr", 1500.0)
    assert alpha_tio.source_table_id == beta_tio.source_table_id == "O-019"
    assert alpha_tio.phase_identity is not None
    assert beta_tio.phase_identity is not None
    assert alpha_tio.phase_identity["polymorph"] == "ALPHA"
    assert beta_tio.phase_identity["polymorph"] == "BETA"

    forsterite = species_thermo("Mg2SiO4", "cr", 1500.0)
    enstatite = species_thermo("MgSiO3", "cr", 1000.0)
    assert forsterite.source_table_id == "Mg-028"
    assert enstatite.source_table_id == "Mg-012"
    assert enstatite.phase_identity is not None
    assert enstatite.phase_identity["polymorph"] == "II"

    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("PbO", "cr", 1160.0)


def test_node_coverage_bounds_perovskite_interval_and_transition_note() -> None:
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
        species_thermo("CaTiO3", "cr", 1400.0)

    assert refusal.value.declared_interval_K == (1500.0, 1800.0)

    perovskite = species_thermo("CaTiO3", "cr", 1600.0)
    assert perovskite.T_min == 1500.0
    assert perovskite.T_max == 1800.0
    assert perovskite.phase_identity is not None
    assert perovskite.phase_identity["note"] == (
        "One row spans a transition: Orthorhombic crystals 1530 K; "
        "cubic crystals 1530 to 1800 K."
    )


def test_near_equal_polymorph_g_values_are_reported_as_a_tie() -> None:
    result = species_thermo("CaSiO3", "cr", 1400.083524)

    assert result.phase_identity is not None
    assert result.phase_identity["polymorph"] == "wollastonite"
    tie_note = result.phase_identity["tie_note"]
    assert "wollastonite" in tie_note
    assert "pseudowollastonite" in tie_note
    assert "1e-6 relative tolerance" in tie_note


def test_crystal_source_caps_and_declared_intervals_are_refused() -> None:
    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match="above its source solid limit of 1875 K",
    ):
        species_thermo(
            "CaAl2O4", "cr", 1876.0, allow_extrapolation=True
        )

    continued = species_thermo(
        "CaAl2O4", "cr", 1850.0, allow_extrapolation=True
    )
    assert continued.extrapolated is True
    assert continued.Cp_J_molK == 193.101
    assert continued.continuation is not None

    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match=r"outside the declared G\(T\) interval.*wollastonite.*pseudowollastonite",
    ):
        species_thermo("CaSiO3", "cr", 1200.0)


def test_berman_and_robie_hemingway_element_reference_anchors() -> None:
    phases = _berman_phases()
    forsterite = next(row for row in phases if row["id"] == "forsterite")
    _cp, entropy, enthalpy, _gibbs = _berman_properties(forsterite, 298.15)
    assert enthalpy * 1000.0 == forsterite["berman_parameters"]["HfPrTr_J_mol"]
    assert entropy == forsterite["berman_parameters"]["SPrTr_J_mol_K"]

    anorthite = next(row for row in phases if row["id"] == "anorthite")
    _cp, _entropy, _enthalpy, gibbs = _berman_properties(anorthite, 1800.0)
    element_g = _element_reference_sum_kJ_mol("CaAl2Si2O8", 1800.0)
    assert element_g is not None
    assert gibbs / 1000.0 - element_g == pytest.approx(
        anorthite["rh_crosscheck"]["G_crystal_1800_kJ_mol"], abs=2.5
    )

    wollastonite = next(row for row in phases if row["id"] == "wollastonite")
    for temperature in (1500.0, 1600.0, 1700.0, 1800.0):
        parent_gibbs = math.fsum(
            float(coefficient)
            * species_thermo(oxide, "l", temperature).G_J_mol
            for oxide, coefficient in wollastonite["oxide_stoichiometry"].items()
        )
        crystal_gibbs = _berman_properties(wollastonite, temperature)[3]
        source_fusion = float(
            wollastonite["fusion_gibbs_nodes_kJ_mol"][str(int(temperature))]
        )
        assert (parent_gibbs - crystal_gibbs) / 1000.0 == pytest.approx(
            source_fusion, abs=2.5
        )

    perovskite = next(
        row for row in _crystal_dataset()["phases"]
        if row["id"] == "perovskite"
    )
    residuals = []
    for temperature, source_g in perovskite[
        "source_crystal_gibbs_nodes_kJ_mol"
    ].items():
        t = float(temperature)
        result = species_thermo("CaTiO3", "cr", t, polymorph="perovskite")
        element_g = _element_reference_sum_kJ_mol("CaTiO3", t)
        assert element_g is not None
        residuals.append(
            abs(result.G_J_mol / 1000.0 - element_g - float(source_g))
        )
    assert max(residuals) == pytest.approx(2.3387, abs=1.0e-3)
    assert max(residuals) <= 2.5


def test_berman_lambda_transition_and_disorder_terms_are_retained() -> None:
    phases = _berman_phases()
    akermanite = next(row for row in phases if row["id"] == "akermanite")
    params = akermanite["berman_parameters"]
    coefficients = tuple(float(params.get(f"k{i}", 0.0)) for i in range(7))
    base_h, base_s = _cp_primitive(coefficients, 400.0, 298.15)
    _cp, entropy, enthalpy, _g = _berman_properties(akermanite, 400.0)
    assert enthalpy == pytest.approx(
        (float(params["HfPrTr_J_mol"]) + base_h + params["DtH_J_mol"]) / 1000.0
    )
    assert entropy == pytest.approx(
        params["SPrTr_J_mol_K"] + base_s
        + params["DtH_J_mol"] / params["Tlambda_K"]
    )

    gehlenite = next(row for row in phases if row["id"] == "gehlenite")
    params = gehlenite["berman_parameters"]
    disorder = gehlenite["disorder_parameters"]
    coefficients = tuple(float(params.get(f"k{i}", 0.0)) for i in range(7))
    disorder_coefficients = (
        disorder["d0"], disorder["d1"], disorder["d2"], 0.0, 0.0,
        disorder["d3"], disorder["d4"],
    )
    base_h, base_s = _cp_primitive(coefficients, 1700.0, 298.15)
    disorder_h, disorder_s = _cp_primitive(
        disorder_coefficients, params["Tmax_K"], params["Tmin_K"]
    )
    cp, entropy, enthalpy, _g = _berman_properties(gehlenite, 1700.0)
    assert enthalpy == pytest.approx(
        (
            float(params["HfPrTr_J_mol"])
            + base_h
            + disorder_h
        ) / 1000.0
    )
    assert entropy == pytest.approx(
        params["SPrTr_J_mol_K"] + base_s + disorder_s
    )
    assert cp == pytest.approx(
        math.fsum((
            coefficients[0],
            coefficients[1] / math.sqrt(1700.0),
            coefficients[2] / 1700.0**2,
            coefficients[3] / 1700.0**3,
            coefficients[4] / 1700.0,
            coefficients[5] * 1700.0,
            coefficients[6] * 1700.0**2,
        ))
    )


def test_barin_apparent_g_reconstructs_printed_delta_f_g_nodes() -> None:
    residuals = []
    for phase in _barin_phases():
        for polymorph in phase["polymorphs"]:
            for node in polymorph["source_series"]["delta_fG"]["nodes"]:
                temperature = float(node["T_K"])
                element_g = _element_reference_sum_kJ_mol(
                    phase["formula"], temperature
                )
                if element_g is None:
                    continue
                properties = evaluate_crystal_species(
                    phase["formula"], temperature,
                    polymorph=polymorph["label"],
                )
                assert properties is not None
                reconstructed = properties.G_J_mol / 1000.0 - element_g
                residuals.append(abs(reconstructed - float(node["value"])))

    # JANAF elemental references and Barin's printed formation values differ
    # by at most 19.6294 kJ/mol at the shared C12A7/1700 K node.
    assert max(residuals) == pytest.approx(19.6294, abs=1.0e-3)
    assert max(residuals) <= 20.0


def test_condensate_precedence_and_crystal_source_spread() -> None:
    spreads_J_mol = []
    for temperature in (1500.0, 1550.0, 1750.0):
        condensate = species_thermo("SiO2", "cr", temperature)
        dataset = evaluate_crystal_species("SiO2", temperature)
        assert dataset is not None
        assert condensate.source_table_id == "O-035"
        assert condensate.G_J_mol == species_thermo(
            "SiO2", "cr", temperature
        ).G_J_mol
        assert dataset.G_J_mol == evaluate_janaf_crystal(
            "SiO2", temperature
        ).G_J_mol
        spreads_J_mol.append(abs(condensate.G_J_mol - dataset.G_J_mol))
    assert max(spreads_J_mol) == pytest.approx(59.0223639, abs=1.0e-4)
    assert max(spreads_J_mol) <= 60.0


def test_readme_basalt_evaluate_gas_probe_is_unchanged() -> None:
    from openimcc import evaluate as evaluate_imcc

    basalt = {
        "SiO2": 51.85068,
        "MgO": 4.78527,
        "FeO": 13.77307,
        "CaO": 9.02862,
        "Al2O3": 14.80572,
        "TiO2": 1.73824,
        "Na2O": 3.23108,
        "K2O": 0.78732,
    }
    melt = evaluate_imcc(basalt, 1800.0, basis_type="wt")
    activities = {name: melt.activity(name) for name in melt.parent_oxides}
    assert {name: f"{value:.6g}" for name, value in activities.items()} == {
        "SiO2": "0.315634",
        "MgO": "0.00434375",
        "FeO": "0.0759",
        "CaO": "7.42238e-05",
        "Al2O3": "0.0128061",
        "TiO2": "0.00252514",
        "Na2O": "4.28163e-10",
        "K2O": "8.22663e-15",
    }
    gas = evaluate_gas(activities, 1800.0, 1.0e-10, load_gas_datapack())
    assert gas.unit == "bar"
    assert f"{gas['Na']:.6g}" == "0.000266815"
    assert f"{gas['K']:.6g}" == "6.99669e-06"
    assert gas.domain_flags["Mg"] == (
        "MgO(l) uses a labelled constant-Cp supercooled-liquid continuation "
        "from JANAF Mg-009; this is a hypothetical reference, not a prediction "
        "of a liquid condensate; band ±0.76 dex pO2"
    )


def test_species_thermo_extrapolation_reuses_gas_runtime_rows() -> None:
    pack = load_gas_datapack()
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("O2", "g", 499.0, pack)
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("SiO2", "l", 1199.0, pack)

    gas = species_thermo("O2", "g", 499.0, pack, allow_extrapolation=True)
    gas_row = _nearest_interval_row(
        pack.gas_df, "O2(g)", 499.0, allow_extrapolation=True
    )
    assert gas.G_J_mol == _janaf_gibbs(499.0, gas_row)
    assert gas.source_row_id == str(gas_row["Ref"])
    assert gas.extrapolated is True
    assert gas.declared_interval_K == (
        float(gas_row["T_min"]), float(gas_row["T_max"])
    )
    assert gas.continuation == "Shomate fit continued"
    assert gas.authority

    liquid = species_thermo(
        "SiO2", "l", 1199.0, pack, allow_extrapolation=True
    )
    liquid_row = _oxide_row_for_T(
        pack.oxide_df, "SiO2(l)", 1199.0, allow_extrapolation=True
    )
    assert liquid.G_J_mol == _lamor_gibbs(1199.0, liquid_row)
    assert liquid.source_row_id == str(liquid_row["Ref"])
    assert liquid.extrapolated is True
    assert liquid.declared_interval_K == (
        float(liquid_row["T_min"]), float(liquid_row["T_max"])
    )
    assert liquid.continuation is not None
    assert liquid.authority


def test_species_thermo_crystal_phase_identity_and_formation_anchors() -> None:
    pack = load_gas_datapack()
    high_cristobalite = species_thermo("SiO2", "cr", 1500.0, pack)
    assert high_cristobalite.phase_identity is not None
    assert high_cristobalite.phase_identity["kind"] == "single_polymorph"
    assert high_cristobalite.phase_identity["polymorph"] == "high cristobalite"
    assert high_cristobalite.phase_identity["transition_temperatures_K"][0][
        "temperature_K"
    ] == 1079
    assert "metastable below 1079 K" in high_cristobalite.phase_identity[
        "phase_notes"
    ]
    assert high_cristobalite.formation_enthalpy_298_kJ_mol == pytest.approx(
        -905.489
    )
    assert high_cristobalite.formation_enthalpy_source_id == "O-035"

    legacy_table = (
        Path(__file__).parents[1]
        / "src/openimcc/data/gas/condensate-feo-janaf-legacy.csv"
    )
    lam_pack = load_gas_datapack(oxide_path=legacy_table)
    lam_cristobalite = species_thermo("SiO2", "cr", 1500.0, lam_pack)
    assert lam_cristobalite.source_table_id == "LH87 Table 2"
    assert lam_cristobalite.phase_identity is not None
    assert lam_cristobalite.phase_identity["polymorph"] == (
        "cristobalite (low/high subtype not specified)"
    )
    assert lam_cristobalite.formation_enthalpy_298_kJ_mol == pytest.approx(
        -910.683
    )
    assert lam_cristobalite.phase_identity[
        "formation_enthalpy_reference_phase"
    ] == "quartz (SiO2(QU))"
    assert lam_cristobalite.formation_enthalpy_source_id == "LH87 Table 3"

    carbon = species_thermo("C", "cr", 1500.0, pack)
    assert carbon.phase_identity is not None
    assert carbon.phase_identity["polymorph"] == "C1(ref)"
    assert "graphite" not in carbon.phase_identity["phase_notes"]

    gas = species_thermo("H", "g", 2000.0, pack)
    assert gas.formation_enthalpy_298_kJ_mol == pytest.approx(217.999)
    assert gas.formation_enthalpy_source_id == "H-001"


def test_gas_formation_enthalpy_uses_source_anchor_across_segments() -> None:
    pack = load_gas_datapack()
    expected = (
        ("O2", 0.0, "O-029"),
        ("SiO", -100.416, "O-012"),
        ("Na", 107.3, "Na-005"),
    )
    for species, value, source_id in expected:
        low = species_thermo(species, "g", 1000.0, pack)
        high = species_thermo(species, "g", 2000.0, pack)
        assert low.formation_enthalpy_298_kJ_mol == value
        assert high.formation_enthalpy_298_kJ_mol == value
        assert low.formation_enthalpy_source_id == source_id
        assert high.formation_enthalpy_source_id == source_id


def test_gas_formation_enthalpy_uses_vendored_text_table_anchors() -> None:
    pack = load_gas_datapack()
    for species, value, source_id in (
        ("B", 560.0, "B-005"),
        ("Li", 159.3, "Li-005"),
    ):
        result = species_thermo(species, "g", 2000.0, pack)
        assert result.formation_enthalpy_298_kJ_mol == value
        assert result.formation_enthalpy_source_id == source_id


def test_species_thermo_crystal_constant_cp_continuation_is_continuous() -> None:
    pack = load_gas_datapack()
    edge = species_thermo("Os", "cr", 3000.0, pack)
    continued = species_thermo(
        "Os", "cr", 3100.0, pack, allow_extrapolation=True
    )

    assert continued.extrapolated is True
    assert continued.declared_interval_K == (1200.0, 3000.0)
    assert continued.phase_identity is not None
    polymorph = continued.phase_identity["polymorph"]
    assert continued.continuation == (
        f"constant-Cp crystal continuation within {polymorph} stability range"
    )
    assert continued.Cp_J_molK == edge.Cp_J_molK
    assert continued.H_app_kJ_mol == pytest.approx(
        edge.H_app_kJ_mol + edge.Cp_J_molK * 100.0 / 1000.0
    )
    assert continued.S_J_molK == pytest.approx(
        edge.S_J_molK + edge.Cp_J_molK * math.log(3100.0 / 3000.0)
    )
    assert continued.G_J_mol == pytest.approx(
        1000.0 * continued.H_app_kJ_mol - 3100.0 * continued.S_J_molK
    )


def test_species_thermo_refuses_transition_and_melting_crossings() -> None:
    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match="outside its source-printed stability range.*transition at 1079 K",
    ):
        species_thermo(
            "SiO2", "cr", 999.0, allow_extrapolation=True
        )

    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match="above its source-printed melting point of 1996 K",
    ):
        species_thermo(
            "SiO2", "cr", 1997.0, allow_extrapolation=True
        )

    re = species_thermo("Re", "cr", 3452.0, allow_extrapolation=True)
    assert re.extrapolated is True
    assert re.phase_identity is not None
    assert re.phase_identity["melting_temperature_K"] == 3453
    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match="above its source-printed melting point of 3453 K",
    ):
        species_thermo("Re", "cr", 3453.0, allow_extrapolation=True)


def test_species_thermo_domain_error_reports_selected_interval() -> None:
    pack = load_gas_datapack()
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
        species_thermo("SiO2", "cr", 999.0, pack)

    assert str(refusal.value) == (
        "T=999.0 K outside declared G(T) interval for 'SiO2(cr)' "
        "[1000.0, 1996.0] K"
    )
    assert refusal.value.declared_interval_K == (1000.0, 1996.0)
    assert refusal.value.species == "SiO2(cr)"
    assert refusal.value.source_row_id == "O-035"

    without_row = ImccGasTemperatureOutsideDomainError("outside")
    assert without_row.declared_interval_K is None
    assert without_row.species is None
    assert without_row.source_row_id is None


@pytest.mark.parametrize(
    ("species", "phase", "temperature", "channel", "activities", "expected"),
    (
        (
            "SiO2", "l", 1199.0, "Si", {"SiO2": 1.0e-3},
            "SiO2(l) uses a tier-1 liquid_from_solid construction from "
            "Richet-Bottinga-Tequi-1984",
        ),
        (
            "Al2O3", "l", 999.0, "Al", {"Al2O3": 1.0e-3},
            "Al2O3(l) uses a labelled constant-Cp supercooled-liquid "
            "continuation from JANAF Al-100; this is a hypothetical reference, "
            "not a prediction of a liquid condensate",
        ),
        (
            "UO2", "l", 3001.0, "U", {"UO2": 1.0e-3},
            "UO2(l) uses a labelled constant-Cp supercooled-liquid "
            "continuation from NASA Glenn card NG-1977",
        ),
        (
            "ZnO", "l", 3001.0, "Zn", {"ZnO": 1.0e-3},
            "ZnO(l) uses a tier-4 liquid_from_solid construction from "
            "TSIV-1603; the central liquid uses the printed 2250 K node, and "
            "the quadratic crystal-Cp extension is its uncertainty bound",
        ),
        (
            "Re2O7", "g", 499.0, "Re2O7", {"Re": 1.0},
            "constructed Re2O7(g) from Smith 1952 measured vapor pressure "
            "and B677 condensed functions over 503.15-598.15 K; maximum "
            "log-pressure fit scatter propagated as ±1.3 kJ/mol at 499 K",
        ),
    ),
)
def test_continuation_label_matches_evaluate_gas_flag(
    packaged_gas_pack,
    species: str,
    phase: str,
    temperature: float,
    channel: str,
    activities: dict[str, float],
    expected: str,
) -> None:
    thermo = species_thermo(
        species,
        phase,
        temperature,
        packaged_gas_pack,
        allow_extrapolation=True,
    )
    result = evaluate_gas(
        activities,
        temperature,
        1.0e-8,
        packaged_gas_pack,
        gas_species=(channel,),
        allow_extrapolation=True,
    )

    assert thermo.continuation == expected
    runtime_flag = result.domain_flags[channel]
    assert runtime_flag is not None
    assert expected in runtime_flag
