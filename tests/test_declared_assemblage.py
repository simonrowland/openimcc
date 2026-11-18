import json
import math
from pathlib import Path

import pytest

from openimcc.declared_assemblage import (
    _least_authority,
    evaluate_declared_assemblage,
)
from openimcc.gas import evaluate_gas, load_gas_datapack, species_thermo
from openimcc.gas_data import _PROVENANCE_AUTHORITY_RANK
from openimcc._constants import R_J_MOL_K
from openimcc.janaf_crystals import evaluate_janaf_crystal


@pytest.fixture(scope="module")
def gas_pack():
    return load_gas_datapack()


_REPO_ROOT = Path(__file__).resolve().parents[1]
_JANAF_CRYSTALS = {
    "SiO2": "O-035", "MgO": "Mg-008", "CaO": "Ca-027",
    "Al2O3": "Al-096", "FeO": "Fe-018", "Na2O": "Na-012",
    "TiO2": "O-043", "K2O": "K-012",
}


def _independent_janaf_apparent_gibbs(oxide, temperature_K):
    """Rebuild G_app from JANAF's separate Gibbs-energy-function column."""
    table_id = _JANAF_CRYSTALS[oxide]
    source = json.loads(
        (_REPO_ROOT / "data-src" / "janaf" / f"{table_id}.yaml")
        .read_text(encoding="utf-8")
    )["table"]
    rows = sorted(
        (
            float(row["temperature"]["value"]),
            float(row["negative_gibbs_enthalpy_function"]["value"]),
        )
        for row in source["values"]
        if row["negative_gibbs_enthalpy_function"]["value"] is not None
    )
    reference = next(
        row for row in source["values"]
        if float(row["temperature"]["value"]) == 298.15
    )

    def interpolate(column):
        upper = next(i for i, row in enumerate(rows) if row[0] >= temperature_K)
        if rows[upper][0] == temperature_K:
            while upper + 1 < len(rows) and rows[upper + 1][0] == temperature_K:
                upper += 1
            return rows[upper][column]
        low, high = rows[upper - 1], rows[upper]
        return low[column] + (high[column] - low[column]) * (
            (temperature_K - low[0]) / (high[0] - low[0])
        )

    h298_kj_mol = float(reference["formation_enthalpy"]["value"])
    negative_gibbs_enthalpy_function_j_mol_k = interpolate(1)
    return (
        h298_kj_mol * 1000.0
        - temperature_K * negative_gibbs_enthalpy_function_j_mol_k
    )


@pytest.mark.parametrize(
    ("oxide", "temperatures_K"),
    [
        ("SiO2", (1200.0, 1500.0, 1800.0, 1900.0)),
        ("MgO", (1000.0, 1800.0, 2500.0, 3000.0)),
        ("CaO", (1200.0, 1800.0, 2500.0, 3000.0)),
        ("Al2O3", (1000.0, 1500.0, 1800.0, 2000.0, 2300.0)),
        ("FeO", (1000.0, 1200.0, 1500.0, 1600.0)),
        ("Na2O", (1000.0, 1200.0, 1400.0)),
        ("TiO2", (1200.0, 1500.0, 2000.0, 2100.0)),
    ],
)
def test_janaf_pure_oxide_crystal_to_liquid_closure(
    oxide, temperatures_K, gas_pack
):
    for temperature_K in temperatures_K:
        result = evaluate_declared_assemblage(
            (oxide,), temperature_K, gas_pack=gas_pack
        )
        crystal_gibbs = _independent_janaf_apparent_gibbs(oxide, temperature_K)
        liquid_gibbs = species_thermo(oxide, "l", temperature_K, gas_pack).G_J_mol
        assert result.labels.validity == "IN"
        assert result.oxide_chemical_potentials_J_mol[oxide] == pytest.approx(
            crystal_gibbs, abs=4.0
        )
        assert result.oxide_activities[oxide] == pytest.approx(
            math.exp((crystal_gibbs - liquid_gibbs) / (R_J_MOL_K * temperature_K)),
            rel=3e-4,
        )


@pytest.mark.parametrize(
    ("oxide", "temperature_K", "expected_delta_kJ_mol", "expected_activity"),
    [
        ("SiO2", 1900.0, 0.444451, 0.972258),
        ("Al2O3", 1800.0, 21.605686, 0.236065),
    ],
)
def test_reference_apparent_gibbs_residuals_and_activities(
    oxide, temperature_K, expected_delta_kJ_mol, expected_activity, gas_pack
):
    result = evaluate_declared_assemblage((oxide,), temperature_K, gas_pack=gas_pack)
    crystal_gibbs = _independent_janaf_apparent_gibbs(oxide, temperature_K)
    packaged_liquid_gibbs = species_thermo(
        oxide, "l", temperature_K, gas_pack
    ).G_J_mol
    delta_kJ_mol = (packaged_liquid_gibbs - crystal_gibbs) / 1000.0

    assert delta_kJ_mol == pytest.approx(expected_delta_kJ_mol, abs=0.001)
    assert result.oxide_activities[oxide] == pytest.approx(
        expected_activity, abs=0.00003
    )
    assert "apparent G" in result.receipt.solver["pure_crystal_gibbs_conversion"]


@pytest.mark.parametrize(
    ("oxide", "temperature_K", "production_minus_independent_J_mol", "activity"),
    [
        ("MgO", 1800.0, -0.6, 0.136715),
        ("MgO", 2500.0, -1.5, 0.494179),
        ("CaO", 1800.0, -1.6, 0.101337),
        ("CaO", 2500.0, 0.0, 0.433703),
        ("FeO", 1200.0, 0.0, 0.123873),
        ("FeO", 1500.0, 1.0, 0.339179),
    ],
)
def test_runtime_crystal_matches_separate_janaf_gibbs_function(
    oxide, temperature_K, production_minus_independent_J_mol, activity, gas_pack
):
    result = evaluate_declared_assemblage((oxide,), temperature_K, gas_pack=gas_pack)
    independent = _independent_janaf_apparent_gibbs(oxide, temperature_K)

    assert result.oxide_chemical_potentials_J_mol[oxide] - independent == pytest.approx(
        production_minus_independent_J_mol, abs=0.05
    )
    assert result.oxide_activities[oxide] == pytest.approx(activity, abs=0.00005)


@pytest.mark.parametrize(
    ("oxide", "temperature_K"),
    [
        ("MgO", 3105.0), ("CaO", 3200.0), ("Al2O3", 2327.0),
        ("FeO", 1650.0), ("Na2O", 1405.2), ("TiO2", 2130.0),
    ],
)
def test_crystal_transition_boundary_uses_printed_janaf_row(oxide, temperature_K):
    table_id = _JANAF_CRYSTALS[oxide]
    source = json.loads(
        (_REPO_ROOT / "data-src" / "janaf" / f"{table_id}.yaml")
        .read_text(encoding="utf-8")
    )["table"]
    reference = next(
        row for row in source["values"]
        if float(row["temperature"]["value"]) == 298.15
    )
    transition = next(
        item for item in source["parse_ambiguities"]
        if float(item["raw_line"].split("\t")[0]) == temperature_K
        and "<-->" in item["raw_line"]
        and "LIQUID" in item["raw_line"].upper()
    )
    fields = transition["raw_line"].split("\t")
    independent = (
        float(reference["formation_enthalpy"]["value"]) * 1000.0
        - temperature_K * float(fields[3])
    )
    crystal = evaluate_janaf_crystal(oxide, temperature_K)

    # JANAF's separately printed Gibbs function is rounded independently of
    # the transition-row H increment and entropy columns used by production.
    assert crystal.G_J_mol == pytest.approx(independent, abs=2.0)
    assert crystal.domain_K[1] == temperature_K
    assert crystal.provenance["transition_row_used"] is True


@pytest.mark.parametrize(
    ("oxide", "temperature_K"),
    [
        ("SiO2", 1500.0), ("MgO", 1800.0), ("CaO", 1800.0),
        ("Al2O3", 1800.0), ("FeO", 1500.0), ("Na2O", 1200.0),
        ("TiO2", 1500.0), ("K2O", 1005.0),
    ],
)
def test_pure_oxide_gas_pressures_equal_evaluate_gas(
    oxide, temperature_K, gas_pack
):
    result = evaluate_declared_assemblage(
        (oxide,), temperature_K, fO2=1e-8, gas_pack=gas_pack,
        allow_extrapolation=True,
    )
    expected = evaluate_gas(
        result.oxide_activities, temperature_K, 1e-8, gas_pack,
        parent_oxides=(oxide,), allow_extrapolation=True,
    )
    assert dict(result.gas) == dict(expected)


def test_mgo_and_spinel_identify_both_parent_activities(gas_pack):
    result = evaluate_declared_assemblage(
        ("MgO", "spinel"), 1800.0, gas_pack=gas_pack
    )

    assert result.refusal is None
    assert set(result.oxide_activities) == {"MgO", "Al2O3"}
    assert result.non_identifiable_components == ()
    assert max(abs(value) for value in result.phase_affinities_ln.values()) < 1e-10
    assert 0.0 < result.oxide_activities["MgO"] <= 1.0
    assert result.oxide_activities["MgO"] == pytest.approx(0.14, abs=0.03)


def test_single_sourced_compound_does_not_identify_parent_activities(gas_pack):
    result = evaluate_declared_assemblage(
        ("forsterite",), 1800.0, fO2=1e-8,
        gas_species=("MgO",), gas_pack=gas_pack,
    )

    assert result.refusal.code == "activity_not_identifiable"
    assert result.oxide_activities == {}
    assert set(result.non_identifiable_components) == {"MgO", "SiO2"}
    assert "MgO" in result.gas_depends


def test_mullite_missing_data_withholds_requested_parent_activities(gas_pack):
    result = evaluate_declared_assemblage(
        ("mullite",), 1800.0, fO2=1e-8, gas_species=("SiO2",),
        parent_oxides=("Al2O3", "SiO2"), gas_pack=gas_pack,
    )

    assert result.refusal.code == "missing_phase_data"
    assert set(result.non_identifiable_components) == {"Al2O3", "SiO2"}
    assert result.oxide_activities == {}
    assert "SiO2" in result.gas_depends


def test_underdetermined_spinel_does_not_guess_gas_activity(gas_pack):
    result = evaluate_declared_assemblage(
        ("spinel",), 1800.0, fO2=1e-8, gas_species=("MgO",), gas_pack=gas_pack
    )

    assert result.labels.validity == "DEPENDS"
    assert result.refusal.code == "activity_not_identifiable"
    assert result.oxide_activities == {}
    assert set(result.non_identifiable_components) == {"MgO", "Al2O3"}
    assert "MgO" in result.gas_depends
    assert result.gas is None


def test_inconsistent_overdetermined_phases_return_residual(gas_pack):
    result = evaluate_declared_assemblage(
        ("MgO", "Al2O3", "spinel"), 1800.0, gas_pack=gas_pack
    )

    assert result.labels.validity == "DEPENDS"
    assert result.refusal.code == "inconsistent_phase_set"
    assert set(result.refusal.residuals) == {"MgO", "Al2O3", "spinel"}
    assert max(abs(value) for value in result.refusal.residuals.values()) > 1.0
    assert result.non_identifiable_components == ("Al2O3", "MgO")


@pytest.mark.parametrize("phase", ("Fe3O4", "magnetite", "hematite"))
def test_redox_sensitive_phase_is_refused(phase):
    result = evaluate_declared_assemblage((phase,), 1800.0)

    assert result.refusal.code == "redox_sensitive_phase"
    assert result.labels.validity == "DEPENDS"
    assert result.labels.evidence_class == "unresolved"
    assert f"crystal.{phase}.redox" in result.refusal.axes
    assert phase in result.refusal.detail


def test_missing_phase_refusal_has_unresolved_authority():
    result = evaluate_declared_assemblage(("not-a-phase",), 1800.0)

    assert result.refusal.code == "missing_phase_data"
    assert result.labels.evidence_class == "unresolved"


def test_unknown_source_authority_is_reported_as_unresolved():
    assert _least_authority(("janaf_fitted", "unknown-source")) == "unresolved"


def test_declared_barin_ca_keeps_its_pre_rank_authority():
    result = evaluate_declared_assemblage(("CA",), 1800.0)

    assert result.labels.evidence_class == "compilation_assessed"


def test_shipped_diagnostic_authorities_are_ranked():
    diagnostic_dir = _REPO_ROOT / "src" / "openimcc" / "data" / "diagnostics"
    authorities = set()
    for path in diagnostic_dir.glob("*.json"):
        pending = [json.loads(path.read_text(encoding="utf-8"))]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                authority = value.get("source_authority")
                if isinstance(authority, str):
                    authorities.add(authority)
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)

    assert authorities
    assert authorities <= _PROVENANCE_AUTHORITY_RANK.keys()


def test_phase_temperature_outside_source_range_is_refused():
    result = evaluate_declared_assemblage(("spinel",), 2500.0)

    assert result.labels.validity == "OUT_OF_DOMAIN"
    assert result.refusal.code == "phase_temperature_out_of_range"
    assert result.labels.out_of_domain_axes == ("crystal.spinel.temperature_K",)


def test_out_of_domain_declared_set_preserves_continued_phase_notice():
    result = evaluate_declared_assemblage(("CA", "K2O"), 1850.0)
    notice = "crystal_function_continued:CA:1800-1875K"

    assert result.labels.validity == "OUT_OF_DOMAIN"
    assert result.refusal.code == "phase_temperature_out_of_range"
    assert result.labels.out_of_domain_axes == ("crystal.K2O.temperature_K",)
    assert notice in result.labels.notices
    assert result.receipt.solver["crystal_function_continued"] == (notice,)


def test_k2o_estimated_reference_controls_result_authority(gas_pack):
    result = evaluate_declared_assemblage(("K2O",), 1005.0, gas_pack=gas_pack)

    assert result.labels.evidence_class == "estimated_systematic"
    assert result.standard_pressure_bar == 1.0
    assert result.receipt.request["standard_pressure_bar"] == 1.0


def test_k2o_fusion_input_band_is_not_evaluation_domain():
    inside_selected_transition = evaluate_declared_assemblage(("K2O",), 1013.0)
    above_selected_transition = evaluate_declared_assemblage(("K2O",), 1013.01)

    assert inside_selected_transition.refusal is None
    assert above_selected_transition.refusal.code == "phase_temperature_out_of_range"


def test_declared_assemblage_refuses_temperature_above_janaf_phase_cap():
    result = evaluate_declared_assemblage(("TiO2",), 2130.001)

    assert result.labels.validity == "OUT_OF_DOMAIN"
    assert result.refusal.code == "phase_temperature_out_of_range"
    assert result.labels.out_of_domain_axes == ("crystal.TiO2.temperature_K",)


def test_oxygen_balance_uses_determined_solid_buffer_activities(gas_pack):
    result = evaluate_declared_assemblage(
        ("MgO",), 1800.0, gas_pack=gas_pack, oxygen_balance=True,
        allow_extrapolation=True,
    )

    assert result.oxygen_balance_pO2_bar is not None
    assert result.fO2 == result.oxygen_balance_pO2_bar
    assert result.gas is not None
