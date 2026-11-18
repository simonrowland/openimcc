from __future__ import annotations

import numpy as np
import pytest

from openimcc import (
    ImccCompositionIncompleteError,
    ImccTOutsideDatapackDomainError,
    evaluate,
)
from openimcc.diagnostics import liquidus_bracket, saturation_indices
from openimcc.gas import (
    ImccGasOxygenBalanceError,
    ImccGasSaturationError,
    evaluate_gas,
    evaluate_gas_oxygen_balance,
    evaluate_gas_saturation,
    gas_boiling_temperature,
    load_gas_datapack,
    species_thermo,
    weighted_oxygen_balance_from_pressure_model,
)
from openimcc.mqm_mgo_sio2 import evaluate_mgo_sio2_mqm


_BASALT = {
    "SiO2": 48.5,
    "Al2O3": 15.5,
    "FeO": 10.0,
    "MgO": 8.5,
    "CaO": 10.5,
    "Na2O": 3.5,
    "K2O": 1.0,
    "TiO2": 2.5,
}


@pytest.fixture(scope="module")
def gas_context():
    melt = evaluate(_BASALT, 1800.0, basis_type="wt")
    activities = {name: melt.activity(name) for name in melt.parent_oxides}
    return activities, load_gas_datapack()


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_evaluate_rejects_boolean_temperature_at_kernel_boundary(value):
    with pytest.raises(
        ImccTOutsideDatapackDomainError, match="temperature must be a real number"
    ):
        evaluate({"SiO2": 1.0, "MgO": 1.0}, value)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_evaluate_rejects_boolean_solver_tolerance_at_kernel_boundary(value):
    with pytest.raises(ValueError, match="tol must be a positive finite"):
        evaluate({"SiO2": 1.0, "MgO": 1.0}, 1800.0, tol=value)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_mqm_rejects_boolean_temperature(value):
    with pytest.raises(
        ImccTOutsideDatapackDomainError, match="temperature must be a real number"
    ):
        evaluate_mgo_sio2_mqm({"MgO": 1.0, "SiO2": 1.0}, value)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_mqm_rejects_boolean_mole_amount(value):
    with pytest.raises(ImccCompositionIncompleteError, match="moles for 'SiO2'"):
        evaluate_mgo_sio2_mqm({"MgO": 1.0, "SiO2": value}, 1800.0)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_saturation_indices_rejects_boolean_temperature(value):
    with pytest.raises(ValueError, match="temperature must be a real number"):
        saturation_indices(_BASALT, value)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_liquidus_bracket_rejects_boolean_temperature_endpoint(value):
    with pytest.raises(ValueError, match="T_low and T_high must be real numbers"):
        liquidus_bracket(_BASALT, value, 2.0)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_species_thermo_rejects_boolean_temperature(gas_context, value):
    _, datapack = gas_context
    with pytest.raises(ValueError, match="temperature must be a real number"):
        species_thermo("O2", "g", value, datapack)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_gas_oxygen_balance_rejects_boolean_temperature(gas_context, value):
    activities, datapack = gas_context
    with pytest.raises(ValueError, match="temperature must be a real number"):
        evaluate_gas_oxygen_balance(
            activities, value, datapack, parent_oxides=tuple(activities)
        )


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_gas_saturation_rejects_boolean_temperature(gas_context, value):
    activities, datapack = gas_context
    with pytest.raises(ValueError, match="temperature must be a real number"):
        evaluate_gas_saturation(
            activities, value, datapack, parent_oxides=tuple(activities)
        )


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_boiling_temperature_rejects_boolean_pressure(gas_context, value):
    activities, datapack = gas_context
    with pytest.raises(ValueError, match="ambient_pressure_bar must be finite"):
        gas_boiling_temperature(
            activities,
            datapack,
            ambient_pressure_bar=value,
            T_bracket=(1800.0, 2000.0),
            parent_oxides=tuple(activities),
        )


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_boiling_temperature_rejects_boolean_bracket_endpoint(gas_context, value):
    activities, datapack = gas_context
    with pytest.raises(ImccGasSaturationError, match="invalid boiling temperature bracket"):
        gas_boiling_temperature(
            activities,
            datapack,
            ambient_pressure_bar=0.1,
            T_bracket=(value, 2.0),
            parent_oxides=tuple(activities),
        )


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_weighted_oxygen_balance_rejects_boolean_oxygen_weight(gas_context, value):
    activities, datapack = gas_context
    pressure_output = evaluate_gas(
        activities, 1800.0, 1.0, datapack, parent_oxides=tuple(activities)
    )
    species = tuple(pressure_output)
    oxygen_weights = {name: 1.0 for name in species}
    oxygen_weights[species[0]] = value
    conductance = {name: 1.0 for name in species}
    bulk_pressures = {name: 0.0 for name in species}
    with pytest.raises(ImccGasOxygenBalanceError, match=f"invalid frozen input for '{species[0]}'"):
        weighted_oxygen_balance_from_pressure_model(
            lambda _logp: pressure_output,
            oxygen_weights,
            conductance_mol_m2_s_Pa=conductance,
            bulk_pressures_bar=bulk_pressures,
            log10_pO2_bracket=(-2.0, 0.0),
        )


def test_numeric_strings_and_numpy_real_scalars_remain_admitted(gas_context):
    _, datapack = gas_context
    assert np.isfinite(species_thermo("O2", "g", np.int64(1800), datapack).G_J_mol)
    assert np.isfinite(species_thermo("O2", "g", "1800", datapack).G_J_mol)
    result = evaluate(
        {"SiO2": 1.0, "MgO": 1.0}, np.float64(1800.0), tol=np.float64(1.0e-12)
    )
    assert np.isfinite(result.D)
    mq_m = evaluate_mgo_sio2_mqm(
        {"MgO": np.int64(1), "SiO2": np.float64(1.0)}, np.float64(1800.0)
    )
    assert np.isfinite(mq_m.a_sio2)
