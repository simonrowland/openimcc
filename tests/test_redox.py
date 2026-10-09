from __future__ import annotations

import numpy as np
import pytest

from openimcc import (
    RedoxEndpointError,
    RedoxInputError,
    RedoxNoLiveCoupleError,
    evaluate_redox,
    load_datapack,
)
from openimcc.kernel import active_residual_jacobian
from openimcc.redox_basis import oxide_inventory_to_elements
from openimcc.redox_pack import load_redox_pack, ln_k_prime


def _formula_atoms() -> dict[str, dict[str, float]]:
    pack = load_redox_pack()
    formulas = {
        str(parent["parent_oxide"]): dict(parent["parent_formula_atoms"])
        for parent in pack.parents
    }
    formulas.update(
        {
            name: dict(record["formula_atoms"])
            for name, record in pack.content["thermo_species"].items()
        }
    )
    return formulas


def _fe_only_inventory() -> dict[str, float]:
    return {"Fe": 1.0, "O": 1.25}


def test_fe_valence_splits_give_the_same_closed_solution() -> None:
    formulas = _formula_atoms()
    oxide_splits = [
        {"FeO(l)": 1.0, "O2(g)": 0.125},
        {"FeO(l)": 0.5, "Fe2O3(l)": 0.25},
    ]
    inventories = [oxide_inventory_to_elements(split, formulas) for split in oxide_splits]
    assert inventories[0] == inventories[1] == {"Fe": 1.0, "O": 1.25}

    results = [evaluate_redox(inventory, 2200.0, "closed") for inventory in inventories]
    assert results[0].lambda_ln_f_o2 == pytest.approx(
        results[1].lambda_ln_f_o2, abs=1.0e-12
    )
    assert results[0].metal_moles == pytest.approx(results[1].metal_moles, abs=1.0e-12)
    assert results[0].species_moles == pytest.approx(results[1].species_moles, abs=1.0e-12)


def test_elemental_and_oxygen_closure_are_independently_reconstructed() -> None:
    redox_pack = load_redox_pack()
    base = load_datapack().kernel_datapack
    inventory = {
        "Si": 0.5,
        "Mg": 0.15,
        "Fe": 0.15,
        "Ca": 0.10,
        "Al": 0.16,
        "Ti": 0.01,
        "Na": 0.016,
        "K": 0.004,
    }
    parent_records = {
        str(record["element"]): record for record in redox_pack.parents
    }
    oxygen_fixed = sum(
        amount
        * float(parent_records[element]["parent_formula_atoms"].get("O", 0.0))
        / float(parent_records[element]["parent_formula_atoms"][element])
        for element, amount in inventory.items()
    )
    inventory["O"] = oxygen_fixed + 0.002
    result = evaluate_redox(inventory, 2500.0, "closed")

    for element, amount in inventory.items():
        if element == "O":
            continue
        parent = parent_records[element]
        oxide = str(parent["parent_oxide"]).removesuffix("(l)")
        reconstructed = result.parent_moles[oxide] * float(
            parent["parent_formula_atoms"][element]
        )
        if element == "Fe":
            reconstructed += result.metal_moles
        assert reconstructed == pytest.approx(amount, rel=1.0e-12, abs=1.0e-13)

    parent_names = tuple(base.parent_oxides)
    oxygen_parent = np.asarray(
        [
            float(parent_records[next(
                element
                for element, record in parent_records.items()
                if str(record["parent_oxide"]).removesuffix("(l)") == oxide
            )]["parent_formula_atoms"].get("O", 0.0))
            for oxide in parent_names
        ],
        dtype=np.float64,
    )
    nu = np.column_stack(
        (base.nu, np.eye(len(parent_names), dtype=np.float64)[parent_names.index("FeO")])
    )
    oxygen_per_complex = oxygen_parent @ nu
    oxygen_per_complex[-1] -= 2.0 * float(
        redox_pack.rows[1]["external_oxygen_stoich_product_positive"]
    )
    oxygen_from_species = sum(
        result.species_moles[name] * oxygen_parent[index]
        for index, name in enumerate(parent_names)
    )
    oxygen_from_species += sum(
        result.species_moles[name] * oxygen_per_complex[index]
        for index, name in enumerate(tuple(base.reactions) + ("FeO1.5(l)",))
    )
    assert oxygen_from_species == pytest.approx(inventory["O"], rel=1.0e-12, abs=1.0e-13)
    assert result.oxygen_residual_mol == pytest.approx(0.0, abs=2.0e-12)


def test_all_registered_feedstock_elements_share_the_redox_parent_solve() -> None:
    redox_pack = load_redox_pack()
    inventory: dict[str, float] = {}
    parent_units: dict[str, float] = {}
    for parent in redox_pack.parents:
        element = str(parent["element"])
        units = 1.0 if element == "Fe" else 0.001
        parent_name = str(parent["parent_oxide"]).removesuffix("(l)")
        parent_units[parent_name] = units
        for atom, count in parent["parent_formula_atoms"].items():
            if atom != "O":
                inventory[atom] = inventory.get(atom, 0.0) + units * float(count)

    at_zero = evaluate_redox(
        inventory, 2200.0, "imposed", lambda_imposed=0.0
    )
    at_saturation = evaluate_redox(
        inventory,
        2200.0,
        "imposed",
        lambda_imposed=at_zero.lambda_sat,
    )
    inventory["O"] = at_saturation.oxygen_mol + 0.001
    result = evaluate_redox(inventory, 2200.0, "closed")

    assert not result.metal_buffered
    for parent_name, amount in parent_units.items():
        assert result.parent_moles[parent_name] == pytest.approx(amount, abs=1.0e-12)
    assert any("unassociated liquid parents" in flag for flag in result.flags)
    assert result.oxygen_residual_mol == pytest.approx(0.0, abs=2.0e-12)


def test_imposed_capacity_matches_finite_difference_and_is_monotone() -> None:
    inventory = _fe_only_inventory()
    temperature = 2200.0
    center_lambda = -4.0
    center = evaluate_redox(
        inventory, temperature, "imposed", lambda_imposed=center_lambda
    )
    h = 1.0e-4
    plus = evaluate_redox(
        inventory, temperature, "imposed", lambda_imposed=center_lambda + h
    )
    minus = evaluate_redox(
        inventory, temperature, "imposed", lambda_imposed=center_lambda - h
    )
    finite_difference = (plus.oxygen_mol - minus.oxygen_mol) / (2.0 * h)
    assert center.oxygen_capacity_mol_per_ln_f_o2 == pytest.approx(
        finite_difference, rel=2.0e-8, abs=1.0e-12
    )

    lambdas = np.linspace(center.lambda_sat, 0.0, 7)
    oxygen = [
        evaluate_redox(
            inventory, temperature, "imposed", lambda_imposed=float(value)
        ).oxygen_mol
        for value in lambdas
    ]
    assert np.all(np.diff(oxygen) > 0.0)


def test_oxygen_response_is_continuous_across_pure_fe_saturation() -> None:
    temperature = 2200.0
    imposed = evaluate_redox(
        {"Fe": 1.0}, temperature, "imposed", lambda_imposed=0.0
    )
    at_saturation = evaluate_redox(
        {"Fe": 1.0},
        temperature,
        "imposed",
        lambda_imposed=imposed.lambda_sat,
    )
    oxygen_delta = 1.0e-7
    reduced = evaluate_redox(
        {"Fe": 1.0, "O": at_saturation.oxygen_mol - oxygen_delta},
        temperature,
        "closed",
    )
    oxidised = evaluate_redox(
        {"Fe": 1.0, "O": at_saturation.oxygen_mol + oxygen_delta},
        temperature,
        "closed",
    )
    assert reduced.metal_buffered
    assert reduced.lambda_ln_f_o2 == pytest.approx(reduced.lambda_sat, abs=1.0e-12)
    assert reduced.metal_moles < 1.0e-4
    assert oxidised.metal_moles == 0.0
    assert oxidised.lambda_ln_f_o2 > oxidised.lambda_sat
    assert abs(oxidised.lambda_ln_f_o2 - at_saturation.lambda_sat) < 1.0e-4


def test_imposed_state_matches_closed_state_at_its_solved_lambda() -> None:
    inventory = _fe_only_inventory()
    closed = evaluate_redox(inventory, 2200.0, "closed")
    imposed = evaluate_redox(
        inventory,
        2200.0,
        "imposed",
        lambda_imposed=closed.lambda_ln_f_o2,
    )
    assert not closed.metal_buffered
    assert imposed.lambda_ln_f_o2 == closed.lambda_ln_f_o2
    assert imposed.species_moles == pytest.approx(closed.species_moles, rel=1.0e-12)
    assert imposed.oxygen_mol == pytest.approx(closed.oxygen_mol, abs=1.0e-12)


def test_metal_amount_matches_independent_dense_thermodynamic_scan() -> None:
    temperature = 2200.0
    oxygen_inventory = 1.0
    result = evaluate_redox(
        {"Fe": 1.0, "O": oxygen_inventory}, temperature, "closed"
    )
    base = load_datapack().kernel_datapack
    fe_index = base.parent_oxides.index("FeO")
    assert not any(
        np.count_nonzero(base.nu[:, index]) == 1
        and base.nu[fe_index, index] > 0.0
        for index in range(base.n_complexes)
    )
    redox_pack = load_redox_pack()
    ln_k_fe = ln_k_prime(redox_pack, redox_pack.rows[0], temperature)
    ln_k_ferric = ln_k_prime(redox_pack, redox_pack.rows[1], temperature)
    lambdas = np.linspace(-30.0, 0.0, 120_001, dtype=np.float64)
    ferric_ratio = np.exp(ln_k_ferric + lambdas / 4.0)
    log_a_feo = -np.log1p(ferric_ratio)
    saturation_residual = lambdas - 2.0 * (log_a_feo - ln_k_fe)
    scan_index = int(np.argmin(np.abs(saturation_residual)))
    ferric_fraction = ferric_ratio[scan_index] / (1.0 + ferric_ratio[scan_index])
    scan_metal = 1.0 - oxygen_inventory / (1.0 + 0.5 * ferric_fraction)

    assert abs(saturation_residual[scan_index]) < 2.0e-4
    assert result.metal_moles == pytest.approx(scan_metal, abs=2.0e-6)
    assert result.lambda_ln_f_o2 == pytest.approx(lambdas[scan_index], abs=2.0e-4)


@pytest.mark.parametrize(
    ("inventory", "endpoint"),
    [
        ({"Fe": 1.0, "Si": 0.5, "O": 0.5}, "reductant_exceeds_reducible_oxygen"),
        ({"Na": 0.1, "O": 1.2}, "alkali_couple_incomplete"),
        ({"Fe": 1.0, "O": 2.0}, "oxidising_fugacity_limit"),
    ],
)
def test_prescribed_inventory_endpoints_are_typed(
    inventory: dict[str, float], endpoint: str
) -> None:
    with pytest.raises(RedoxEndpointError) as exc_info:
        evaluate_redox(inventory, 2200.0, "closed")
    assert exc_info.value.endpoint == endpoint
    assert endpoint in exc_info.value.flags


def test_missing_fe_and_bad_mode_refuse_without_partial_results() -> None:
    with pytest.raises(RedoxNoLiveCoupleError):
        evaluate_redox({"O": 1.0}, 2200.0, "closed")
    with pytest.raises(RedoxInputError):
        evaluate_redox(_fe_only_inventory(), 2200.0, "unknown")  # type: ignore[arg-type]


def test_imposed_mode_refuses_below_the_pure_fe_bound() -> None:
    result = evaluate_redox(
        {"Fe": 1.0}, 2200.0, "imposed", lambda_imposed=0.0
    )
    with pytest.raises(RedoxEndpointError) as exc_info:
        evaluate_redox(
            {"Fe": 1.0},
            2200.0,
            "imposed",
            lambda_imposed=result.lambda_sat - 1.0,
        )
    assert exc_info.value.endpoint == "lambda_below_pure_fe_saturation"
    assert exc_info.value.lambda_bound == pytest.approx(result.lambda_sat)


def test_array_residual_and_jacobian_are_float64_and_consistent() -> None:
    y = np.asarray([-0.7, -1.1], dtype=np.float64)
    x_target = np.asarray([0.4, 0.6], dtype=np.float64)
    nu = np.asarray([[1.0, 2.0], [1.0, 0.0]], dtype=np.float64)
    lnK = np.asarray([-0.4, -1.3], dtype=np.float64)
    S = np.asarray([2.0, 2.0], dtype=np.float64)
    residual, _g, _D, jacobian = active_residual_jacobian(
        y, x_target, nu, lnK, S, np
    )
    step = 1.0e-6
    finite_difference = np.column_stack(
        [
            (
                active_residual_jacobian(
                    y + np.eye(2, dtype=np.float64)[index] * step,
                    x_target,
                    nu,
                    lnK,
                    S,
                    np,
                )[0]
                - active_residual_jacobian(
                    y - np.eye(2, dtype=np.float64)[index] * step,
                    x_target,
                    nu,
                    lnK,
                    S,
                    np,
                )[0]
            )
            / (2.0 * step)
            for index in range(2)
        ]
    )
    assert residual.dtype == np.float64
    assert jacobian.dtype == np.float64
    assert np.allclose(jacobian, finite_difference, rtol=2.0e-8, atol=1.0e-10)
