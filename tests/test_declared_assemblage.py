import json
import math
from importlib import resources

import pytest

from openimcc.declared_assemblage import evaluate_declared_assemblage
from openimcc.gas import evaluate_gas, load_gas_datapack, species_thermo
from openimcc._constants import R_J_MOL_K


@pytest.fixture(scope="module")
def gas_pack():
    return load_gas_datapack()


@pytest.fixture(scope="module")
def pure_crystals():
    resource = resources.files("openimcc").joinpath(
        "data/diagnostics/janaf-pure-oxide-crystals.json"
    )
    return {
        phase["id"]: phase
        for phase in json.loads(resource.read_text(encoding="utf-8"))["phases"]
    }


def _interpolate(nodes, temperature_K):
    points = sorted((float(t), float(g)) for t, g in nodes.items())
    for temperature, gibbs in points:
        if temperature == temperature_K:
            return gibbs
    for (lo_t, lo_g), (hi_t, hi_g) in zip(points, points[1:]):
        if lo_t <= temperature_K <= hi_t:
            return lo_g + (hi_g - lo_g) * (temperature_K - lo_t) / (hi_t - lo_t)
    raise AssertionError(f"no source interval at {temperature_K:g} K")


@pytest.mark.parametrize(
    ("oxide", "temperatures_K"),
    [
        ("SiO2", (1200.0, 1500.0, 1800.0)),
        ("MgO", (1000.0, 1800.0, 2500.0, 3000.0)),
        ("CaO", (1200.0, 1800.0, 2500.0, 3000.0)),
        ("Al2O3", (1000.0, 1500.0, 2000.0, 2300.0)),
        ("FeO", (1000.0, 1200.0, 1500.0, 1600.0)),
        ("Na2O", (1000.0, 1200.0, 1400.0)),
        ("TiO2", (1200.0, 1500.0, 2000.0, 2100.0)),
    ],
)
def test_janaf_pure_oxide_crystal_to_liquid_closure(
    oxide, temperatures_K, pure_crystals, gas_pack
):
    phase = pure_crystals[oxide]
    for temperature_K in temperatures_K:
        result = evaluate_declared_assemblage(
            (oxide,), temperature_K, gas_pack=gas_pack
        )
        crystal_gibbs = _interpolate(
            phase["crystal_gibbs_nodes_kJ_mol"], temperature_K
        ) * 1000.0
        liquid_gibbs = species_thermo(oxide, "l", temperature_K, gas_pack).G_J_mol
        expected_activity = math.exp(
            (crystal_gibbs - liquid_gibbs) / (R_J_MOL_K * temperature_K)
        )
        assert result.labels.validity == "IN"
        assert result.oxide_chemical_potentials_J_mol[oxide] == pytest.approx(
            crystal_gibbs, abs=1e-8
        )
        assert result.oxide_activities[oxide] == pytest.approx(
            expected_activity, rel=1e-13
        )


@pytest.mark.parametrize(
    ("oxide", "temperature_K"),
    [
        ("SiO2", 1500.0), ("MgO", 1800.0), ("CaO", 1800.0),
        ("Al2O3", 1800.0), ("FeO", 1500.0), ("Na2O", 1200.0),
        ("TiO2", 1500.0), ("K2O", 1100.0),
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


def test_phase_temperature_outside_source_range_is_refused():
    result = evaluate_declared_assemblage(("spinel",), 2500.0)

    assert result.labels.validity == "OUT_OF_DOMAIN"
    assert result.refusal.code == "phase_temperature_out_of_range"
    assert result.labels.out_of_domain_axes == ("crystal.spinel.temperature_K",)


def test_k2o_estimated_reference_controls_result_authority(gas_pack):
    result = evaluate_declared_assemblage(("K2O",), 1100.0, gas_pack=gas_pack)

    assert result.labels.evidence_class == "estimated_systematic"
    assert result.standard_pressure_bar == 1.0
    assert result.receipt.request["standard_pressure_bar"] == 1.0


def test_oxygen_balance_uses_determined_solid_buffer_activities(gas_pack):
    result = evaluate_declared_assemblage(
        ("MgO",), 1800.0, gas_pack=gas_pack, oxygen_balance=True,
        allow_extrapolation=True,
    )

    assert result.oxygen_balance_pO2_bar is not None
    assert result.fO2 == result.oxygen_balance_pO2_bar
    assert result.gas is not None
