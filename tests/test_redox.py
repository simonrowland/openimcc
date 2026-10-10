from __future__ import annotations

import math

import numpy as np
import pytest

from openimcc import (
    ImccRefusal,
    ImccTOutsideDatapackDomainError,
    RedoxEndpointError,
    RedoxInputError,
    RedoxNoLiveCoupleError,
    RedoxNumericalError,
    RedoxResult,
    evaluate_redox,
    load_datapack,
)
import openimcc.kernel as kernel_module
from openimcc.kernel import active_residual_jacobian, solve_active_warm
from openimcc.redox_basis import oxide_inventory_to_elements
from openimcc.redox_pack import RedoxPack, load_redox_pack, ln_k_prime, species_gibbs_j_mol


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


def test_default_redox_pack_cache_is_immutable_and_reloads_changed_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import openimcc.redox as redox_module

    loaded = redox_module._load_default_datapack()
    assert loaded is redox_module._load_default_datapack()
    with pytest.raises(ValueError):
        loaded.kernel_datapack.nu[0, 0] = 0.0

    melt_source = redox_module._DEFAULT_DATAPACK_PATH
    melt_path = tmp_path / "melt.json"
    melt_path.write_bytes(melt_source.read_bytes())
    monkeypatch.setattr(redox_module, "_DEFAULT_DATAPACK_PATH", melt_path)
    melt_pack = redox_module._load_default_datapack()
    assert melt_pack is redox_module._load_default_datapack()
    melt_path.write_bytes(melt_path.read_bytes() + b"\n")
    assert melt_pack is not redox_module._load_default_datapack()

    source = redox_module._DEFAULT_REDOX_PACK_PATH
    changed_path = tmp_path / "redox.json"
    changed_path.write_bytes(source.read_bytes())
    monkeypatch.setattr(redox_module, "_DEFAULT_REDOX_PACK_PATH", changed_path)
    pack = redox_module._load_default_redox_pack()
    version = pack.version
    with pytest.raises(TypeError):
        pack.content["version"] = "changed"
    assert pack is redox_module._load_default_redox_pack()
    assert redox_module._load_default_redox_pack().version == version

    changed_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        redox_module._load_default_redox_pack()


def test_shared_parent_lp_skips_rows_inactive_for_the_current_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openimcc.redox as redox_module

    pack = load_redox_pack()
    content = dict(pack.content)
    species = dict(pack.content["thermo_species"])
    species["TiO3(inactive-sharing-test)"] = {
        "formula_atoms": {"Ti": 1, "O": 3}
    }
    species["TiO4(inactive-sharing-test)"] = {
        "formula_atoms": {"Ti": 1, "O": 4}
    }
    content["thermo_species"] = species
    content["rows"] = [
        {
            "complex": "TiO3(inactive-sharing-test)",
            "nu": {"TiO2(l)": 1.0},
        },
        {
            "complex": "TiO4(inactive-sharing-test)",
            "nu": {"TiO2(l)": 1.0},
        },
    ]
    shared_pack = RedoxPack(content, pack.canonical_digest)
    parent_names = redox_module._redox_parent_names(shared_pack)
    fe_index = parent_names.index("FeO")
    parent_moles = np.zeros(len(parent_names), dtype=np.float64)
    parent_moles[fe_index] = 1.0

    def unexpected_linprog(*args, **kwargs):
        pytest.fail("inactive shared-parent rows must not run the LP")

    monkeypatch.setattr(redox_module, "linprog", unexpected_linprog)
    surplus, deficit, shared_positive_rows = redox_module._oxygen_inventory_targets(
        parent_moles,
        redox_module._parent_oxygen_counts(parent_names, shared_pack),
        fe_index,
        1.0,
        shared_pack,
    )
    assert (surplus, deficit) == (1.0, 0.0)
    assert shared_positive_rows


def test_redox_reference_outputs_before_generalized_rows() -> None:
    fe_closed = evaluate_redox({"Fe": 1.0, "O": 1.25}, 2200.0, "closed")
    assert fe_closed.lambda_ln_f_o2 == pytest.approx(-0.7193904586464315, rel=1.0e-12)
    assert fe_closed.metal_moles == 0.0
    assert fe_closed.oxygen_capacity_mol_per_ln_f_o2 == pytest.approx(
        0.03125, rel=1.0e-12
    )
    assert {
        name: amount for name, amount in fe_closed.species_moles.items() if amount
    } == pytest.approx(
        {"FeO": 0.49999999999999967, "FeO1.5(l)": 0.5000000000000003},
        rel=1.0e-12,
        abs=0.0,
    )

    fe_buffered = evaluate_redox({"Fe": 1.0, "O": 1.0}, 2200.0, "closed")
    assert fe_buffered.lambda_ln_f_o2 == pytest.approx(
        -15.127740855106792, rel=1.0e-12
    )
    assert fe_buffered.metal_moles == pytest.approx(
        0.013097674149791579, rel=1.0e-12, abs=0.0
    )
    assert fe_buffered.oxygen_residual_mol is not None
    assert abs(fe_buffered.oxygen_residual_mol) <= 5.0e-13
    assert math.isinf(fe_buffered.oxygen_capacity_mol_per_ln_f_o2)
    assert {
        name: amount for name, amount in fe_buffered.species_moles.items() if amount
    } == pytest.approx(
        {"FeO": 0.9607069775478426, "FeO1.5(l)": 0.02619534830143822},
        rel=1.0e-12,
        abs=0.0,
    )

    fe_imposed = evaluate_redox(
        {"Fe": 1.0}, 2200.0, "imposed", lambda_imposed=-4.0
    )
    assert fe_imposed.oxygen_capacity_mol_per_ln_f_o2 == pytest.approx(
        0.026532459689033592, rel=1.0e-12
    )
    assert {
        name: amount for name, amount in fe_imposed.species_moles.items() if amount
    } == pytest.approx(
        {"FeO": 0.6942686863283202, "FeO1.5(l)": 0.3057313136716798},
        rel=1.0e-12,
        abs=0.0,
    )

    mixed = {
        "Si": 0.5,
        "Mg": 0.15,
        "Fe": 0.15,
        "Ca": 0.10,
        "Al": 0.16,
        "Ti": 0.01,
        "Na": 0.016,
        "K": 0.004,
    }
    redox_pack = load_redox_pack()
    parents = {str(parent["element"]): parent for parent in redox_pack.parents}
    mixed["O"] = sum(
        amount
        * float(parents[element]["parent_formula_atoms"].get("O", 0.0))
        / float(parents[element]["parent_formula_atoms"][element])
        for element, amount in mixed.items()
    ) + 0.002
    mixed_closed = evaluate_redox(mixed, 2500.0, "closed")
    # Reference values from the chunk-2 solver on the default (IMCC-SF04-D066)
    # pack. This K-bearing case moved when D066 became the default (translated
    # K rows; KCaAlSi2O7 inactive); the generalized solver must still match the
    # chunk-2 solver on the same pack.
    assert mixed_closed.lambda_ln_f_o2 == pytest.approx(
        -11.136182920527485, rel=1.0e-12
    )
    assert mixed_closed.metal_moles == 0.0
    assert mixed_closed.oxygen_capacity_mol_per_ln_f_o2 == pytest.approx(
        0.0004897171807855382, rel=1.0e-12
    )
    assert mixed_closed.fallback_fired is True
    assert {
        name: amount for name, amount in mixed_closed.species_moles.items() if amount
    } == pytest.approx(
        {
            "Al2O3": 0.012005629428046326,
            "Al6Si2O13": 3.732013567813373e-06,
            "Ca12Al14O33": 1.1304283457685428e-30,
            "Ca2Al2SiO7": 0.001156231774271667,
            "Ca2MgSi2O7": 0.0009086796850032818,
            "Ca2SiO4": 0.0002559256436732968,
            "CaAl12O19": 5.292932721529893e-09,
            "CaAl2O4": 0.0006549993891005081,
            "CaAl2Si2O8": 0.025735759137443383,
            "CaAl4O7": 0.00021615833710816336,
            "CaMgSi2O6": 0.028103140714583134,
            "CaO": 0.0002113050124195111,
            "CaSiO3": 0.03767539752394051,
            "CaTiO3": 0.000564324224899159,
            "CaTiSiO5": 0.0021972361616764014,
            "Fe2SiO4": 0.020746606444737678,
            "FeAl2O4": 0.007015482201161543,
            "FeO": 0.0937403199222792,
            "FeO1.5(l)": 0.004000000000014626,
            "FeTiO3": 0.0037509849870697794,
            "K2O": 6.526605153153919e-13,
            "K2Si2O5": 1.0152743819103149e-07,
            "K2Si4O9": 6.5959828274873394e-09,
            "K2SiO3": 4.5987032272748054e-08,
            "KAlO2": 2.8189645357467103e-05,
            "KAlSi2O6": 0.0025490931187974394,
            "KAlSi3O8": 0.0008164578044988179,
            "KAlSiO4": 0.0006059512091343741,
            "Mg2Al4Si5O18": 0.008716741993893213,
            "Mg2SiO4": 0.006974287130394176,
            "Mg2TiO4": 0.00016046143568658694,
            "MgAl2O4": 0.005583852072611253,
            "MgO": 0.010568243439252369,
            "MgSiO3": 0.07236871360751457,
            "MgTi2O5": 1.0996148539826194e-05,
            "MgTiO3": 0.0007533932125475661,
            "Na2O": 2.7294567270139288e-09,
            "Na2Si2O5": 1.425516674701696e-05,
            "Na2SiO3": 1.3741187243914143e-05,
            "Na2TiO3": 8.292683835821185e-07,
            "NaAlO2": 0.0003395709844588099,
            "NaAlSi2O6": 0.0003535303165441381,
            "NaAlSi3O8": 0.010346053177363736,
            "NaAlSiO4": 0.004903188817970845,
            "SiO2": 0.16069482185317643,
            "TiO2": 0.0025507784126572756,
        },
        rel=1.0e-12,
        abs=0.0,
    )


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
    loaded = load_datapack()
    base = loaded.kernel_datapack
    assert loaded.inactive_rows == (
        ("KCaAlSi2O7", "E25: out of liquid domain; owner-confirmed"),
    )
    assert "KCaAlSi2O7" not in base.reactions
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
    assert "KCaAlSi2O7" not in result.species_moles

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
        inventory, 2000.0, "imposed", lambda_imposed=0.0
    )
    at_saturation = evaluate_redox(
        inventory,
        2000.0,
        "imposed",
        lambda_imposed=at_zero.lambda_sat,
    )
    inventory["O"] = at_saturation.oxygen_mol + 0.001
    result = evaluate_redox(inventory, 2000.0, "closed")

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

    lambdas = np.linspace(center.lambda_sat + 1.0e-8, 0.0, 7)
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
        # This is the oxygen surplus at zero metal and the exact Fe saturation fugacity.
        {"Fe": 1.0, "O": 1.0132714999313717},
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
    assert reduced.metal_moles > at_saturation.metal_moles
    assert reduced.metal_moles - at_saturation.metal_moles < 1.0e-6
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
        ({"Fe": 1.0, "O": 1.5}, "fully_oxidised_inventory_limit"),
        ({"Fe": 1.0, "O": 2.0}, "fully_oxidised_inventory_limit"),
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


def test_imposed_mode_resolves_the_metal_amount_below_saturation() -> None:
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
    assert exc_info.value.endpoint == "lambda_below_dissolved_fe_saturation_range"


def test_imposed_exact_invariant_saturation_requires_oxygen_inventory() -> None:
    saturated = evaluate_redox(
        {"Fe": 1.0}, 2200.0, "imposed", lambda_imposed=0.0
    )
    with pytest.raises(RedoxEndpointError) as exc_info:
        evaluate_redox(
            {"Fe": 1.0},
            2200.0,
            "imposed",
            lambda_imposed=saturated.lambda_sat,
        )
    assert exc_info.value.endpoint == "imposed_invariant_requires_oxygen"


def test_imposed_mixed_melt_predicts_buffered_metal_without_oxygen() -> None:
    result = evaluate_redox(
        {"Fe": 1.0, "Si": 0.8, "Mg": 0.3, "Al": 0.2, "Ca": 0.1},
        2200.0,
        "imposed",
        lambda_imposed=-17.931129807709393,
    )
    assert result.metal_moles == pytest.approx(0.501694421294, abs=2.0e-9)
    assert result.oxygen_inventory_mol is None
    assert result.oxygen_residual_mol is None
    assert result.lambda_sat == pytest.approx(-17.931129807709393, abs=1.0e-9)


def test_finite_closed_root_above_one_bar_standard_state() -> None:
    result = evaluate_redox({"Fe": 1.0, "O": 1.4}, 2200.0, "closed")
    assert result.lambda_ln_f_o2 == pytest.approx(4.825786986, abs=2.0e-8)
    assert result.oxygen_residual_mol == pytest.approx(0.0, abs=2.0e-12)


@pytest.mark.parametrize(
    ("oxygen", "expected_lambda"),
    [
        (1.4999999999999998, 140.6826343756),
        (1.499999999999, 107.0317496963),
    ],
)
def test_closed_mode_solves_representable_states_near_oxidised_endpoint(
    oxygen: float, expected_lambda: float
) -> None:
    # Independent Fe-only mass action for FeO + 1/4 O2 = FeO1.5, evaluated
    # at high precision, gives these roots for O = 1 + 0.5*x_FeO1.5.
    near_fully_oxidised = evaluate_redox(
        {"Fe": 1.0, "O": oxygen}, 2200.0, "closed"
    )
    assert near_fully_oxidised.lambda_ln_f_o2 == pytest.approx(
        expected_lambda, abs=2.0e-2
    )


def test_imposed_and_closed_match_at_nextafter_oxidised_fe_endpoint() -> None:
    inventory = {"Fe": 1.0, "O": 1.4999999999999998}
    # Independent Fe-only mass action at 2200 K, with
    # O = 1 + 0.5*x_FeO1.5, gives lambda = 140.6826343756.
    closed = evaluate_redox(inventory, 2200.0, "closed")
    imposed = evaluate_redox(
        inventory,
        2200.0,
        "imposed",
        lambda_imposed=closed.lambda_ln_f_o2,
    )

    assert closed.lambda_ln_f_o2 == pytest.approx(140.6826343756, abs=2.0e-8)
    assert imposed.lambda_ln_f_o2 == closed.lambda_ln_f_o2
    assert imposed.species_moles == pytest.approx(closed.species_moles, rel=1.0e-12)
    assert imposed.oxygen_mol == pytest.approx(closed.oxygen_mol, abs=1.0e-12)


def test_mixed_inventory_signed_oxygen_cancellation_uses_unrounded_terms() -> None:
    inventory = {
        "Fe": 1.0,
        "Si": 0.8,
        "Mg": 0.3,
        "Al": 0.2,
        "Ca": 0.1,
        "O": 3.7999999999999994,
    }
    # An independent 80-digit solve of the parent and ferric mass-action
    # equations, retaining each input's exact binary value and oxide oxygen
    # product, gives lambda = 136.5922522699.
    result = evaluate_redox(inventory, 2200.0, "closed")

    assert result.lambda_ln_f_o2 == pytest.approx(136.5922522699, abs=1.0e-8)


def test_closed_residual_reconstructs_compensated_species_oxygen_ledger() -> None:
    inventory = {"Fe": 1.0, "O": 1.4999999999999998}
    result = evaluate_redox(inventory, 2200.0, "closed")
    redox_pack = load_redox_pack()
    base = load_datapack().kernel_datapack
    parent_names = tuple(base.parent_oxides)
    parent_records = {
        str(record["parent_oxide"]).removesuffix("(l)"): record
        for record in redox_pack.parents
    }
    oxygen_parent = np.asarray(
        [
            float(parent_records[name]["parent_formula_atoms"].get("O", 0.0))
            for name in parent_names
        ],
        dtype=np.float64,
    )
    fe_index = parent_names.index("FeO")
    all_nu = np.column_stack(
        (base.nu, np.eye(len(parent_names), dtype=np.float64)[fe_index])
    )
    oxygen_complex = oxygen_parent @ all_nu
    oxygen_complex[-1] -= 2.0 * float(
        redox_pack.rows[1]["external_oxygen_stoich_product_positive"]
    )
    complex_names = tuple(base.reactions) + ("FeO1.5(l)",)
    # Independent ledger: sum each returned species' oxygen atoms before
    # subtracting the prescribed O inventory. This exposes the inner-balance
    # error omitted by the former reduced-Fe-only residual.
    oxygen_terms = [
        result.species_moles[name] * oxygen_parent[index]
        for index, name in enumerate(parent_names)
    ]
    oxygen_terms.extend(
        result.species_moles[name] * oxygen_complex[index]
        for index, name in enumerate(complex_names)
    )
    reconstructed_residual = math.fsum((*oxygen_terms, -inventory["O"]))

    assert reconstructed_residual == pytest.approx(6.661338147750939e-16, abs=2.0e-30)
    assert result.oxygen_residual_mol == reconstructed_residual


def test_oxygen_sweep_is_typed_monotone_and_modes_agree_near_both_limits() -> None:
    cases = (
        ("Fe-only", {"Fe": 1.0}, 0.0, 1.5),
        (
            "mixed",
            {"Fe": 1.0, "Si": 0.8, "Mg": 0.3, "Al": 0.2, "Ca": 0.1},
            2.3,
            3.8,
        ),
    )
    for label, base_inventory, reduced_limit, oxidised_limit in cases:
        oxygen_values = [math.nextafter(reduced_limit, math.inf)]
        oxygen_values.extend(
            float(value)
            for value in np.linspace(reduced_limit, oxidised_limit, 9)[1:-1]
        )
        oxygen_values.append(math.nextafter(oxidised_limit, -math.inf))
        closed_lambdas: list[float] = []
        imposed_lambdas: list[float] = []

        for oxygen in oxygen_values:
            inventory = dict(base_inventory, O=oxygen)
            closed = evaluate_redox(inventory, 2200.0, "closed")
            assert isinstance(closed, RedoxResult)
            closed_lambdas.append(closed.lambda_ln_f_o2)
            try:
                imposed = evaluate_redox(
                    inventory,
                    2200.0,
                    "imposed",
                    lambda_imposed=closed.lambda_ln_f_o2,
                )
            except ImccRefusal as exc:
                # At the smallest positive Fe-only O float, the metal extent
                # is below float resolution; the imposed endpoint is refused
                # through the typed public error hierarchy.
                assert label == "Fe-only"
                assert oxygen == oxygen_values[0]
                assert isinstance(exc, RedoxEndpointError)
                assert exc.endpoint == "imposed_inventory_inconsistent"
                continue
            assert isinstance(imposed, RedoxResult)
            imposed_lambdas.append(imposed.lambda_ln_f_o2)
            assert imposed.lambda_ln_f_o2 == closed.lambda_ln_f_o2
            assert imposed.metal_moles == pytest.approx(closed.metal_moles, abs=2.0e-10)
            assert imposed.species_moles == pytest.approx(
                closed.species_moles, rel=2.0e-9, abs=1.0e-14
            )
            assert imposed.oxygen_mol == pytest.approx(closed.oxygen_mol, abs=2.0e-10)

        assert all(
            following >= previous
            for previous, following in zip(closed_lambdas, closed_lambdas[1:])
        )
        assert all(
            following >= previous
            for previous, following in zip(imposed_lambdas, imposed_lambdas[1:])
        )


def test_trace_oxygen_metal_extent_matches_in_closed_and_imposed_modes() -> None:
    trace_oxygen = evaluate_redox({"Fe": 1.0, "O": 1.0e-13}, 2200.0, "closed")
    imposed = evaluate_redox(
        {"Fe": 1.0, "O": 1.0e-13},
        2200.0,
        "imposed",
        lambda_imposed=trace_oxygen.lambda_ln_f_o2,
    )
    assert trace_oxygen.metal_moles == pytest.approx(
        0.9999999999999013, abs=2.0e-15
    )
    assert imposed.metal_moles == pytest.approx(
        0.9999999999999013, abs=2.0e-15
    )
    assert imposed.oxygen_residual_mol == pytest.approx(0.0, abs=2.0e-15)
    assert math.isfinite(trace_oxygen.lambda_ln_f_o2)


def test_mixed_inventory_near_reduced_endpoint_matches_imposed_mode() -> None:
    inventory = {
        "Fe": 1.0,
        "Si": 0.8,
        "Mg": 0.3,
        "Al": 0.2,
        "Ca": 0.1,
        "O": 2.300000000001,
    }
    closed = evaluate_redox(inventory, 2200.0, "closed")
    imposed = evaluate_redox(
        inventory, 2200.0, "imposed", lambda_imposed=closed.lambda_ln_f_o2
    )
    assert imposed.metal_moles == pytest.approx(closed.metal_moles, abs=2.0e-12)
    assert imposed.oxygen_mol == pytest.approx(inventory["O"], abs=2.0e-12)


@pytest.mark.parametrize(
    ("inventory", "buffered"),
    [({"Fe": 1.0, "O": 1.25}, False), ({"Fe": 1.0, "O": 0.5}, True)],
)
def test_imposed_and_closed_match_on_both_branches(
    inventory: dict[str, float], buffered: bool
) -> None:
    closed = evaluate_redox(inventory, 2200.0, "closed")
    imposed = evaluate_redox(
        inventory,
        2200.0,
        "imposed",
        lambda_imposed=closed.lambda_ln_f_o2,
    )
    assert closed.metal_buffered is buffered
    assert imposed.metal_moles == pytest.approx(closed.metal_moles, abs=2.0e-10)
    assert imposed.species_moles == pytest.approx(closed.species_moles, rel=2.0e-10)
    assert imposed.oxygen_mol == pytest.approx(closed.oxygen_mol, abs=2.0e-10)


@pytest.mark.parametrize("warm_start", [["a"], [float("nan")], [float("inf")]])
def test_invalid_warm_starts_raise_typed_input_error(warm_start: list[object]) -> None:
    with pytest.raises(RedoxInputError):
        evaluate_redox(
            {"Fe": 1.0},
            2200.0,
            "imposed",
            lambda_imposed=0.0,
            warm_start=warm_start,  # type: ignore[arg-type]
        )


def test_result_publishes_domain_flags_and_active_band_intersection() -> None:
    result = evaluate_redox({"Fe": 1.0, "O": 1.25}, 2200.0, "closed")
    assert "liquid_only_model" in result.flags
    low, high = result.active_row_band_intersection_K
    assert low <= 2200.0 <= high
    assert result.extrapolated_redox_rows == ()


@pytest.mark.parametrize("temperature", [1500.0, 2200.0, 3000.0])
def test_manganese_redox_rows_extrapolate_with_typed_flags(
    temperature: float,
) -> None:
    inventory = {
        "Fe": 1.0,
        "Mn": 0.1,
        "Si": 0.2,
        "Al": 0.1,
        "O": 1.65,
    }
    closed = evaluate_redox(inventory, temperature, "closed")
    imposed = evaluate_redox(
        inventory,
        temperature,
        "imposed",
        lambda_imposed=closed.lambda_ln_f_o2,
    )
    expected = {"MnO(l)", "MnSiO3(l)", "Mn2SiO4(l)", "MnAl2O4(l)"}
    if temperature == 1500.0:
        expected.add("FeO(l)")
    for result in (closed, imposed):
        assert set(result.extrapolated_redox_rows) == expected
        assert result.active_row_band_intersection_K == (1700.0, 2000.0)
        row_flags = [
            flag
            for flag in result.flags
            if flag.startswith("RED: redox_row_temperature_extrapolation ")
        ]
        assert len(row_flags) == len(expected)
        assert all(f"temperature_K={temperature:g}" in flag for flag in row_flags)
        flagged_rows = {
            flag.split(" row=", 1)[1].split(" temperature_K=", 1)[0]
            for flag in row_flags
        }
        assert flagged_rows == expected


@pytest.mark.parametrize("mode", ["closed", "imposed"])
def test_manganese_rows_inside_evidence_domain_have_no_extrapolation_flag(
    mode: str,
) -> None:
    inventory = {"Fe": 1.0, "Mn": 0.1, "Si": 0.2, "Al": 0.1, "O": 1.65}
    if mode == "closed":
        result = evaluate_redox(inventory, 1800.0, "closed")
    else:
        closed = evaluate_redox(inventory, 1800.0, "closed")
        result = evaluate_redox(
            inventory,
            1800.0,
            "imposed",
            lambda_imposed=closed.lambda_ln_f_o2,
        )
    assert result.extrapolated_redox_rows == ()
    assert not any(
        flag.startswith("RED: redox_row_temperature_extrapolation ")
        for flag in result.flags
    )


def test_species_gibbs_domain_still_refuses_redox_extrapolation() -> None:
    with pytest.raises(
        ImccTOutsideDatapackDomainError,
        match=r"species 'O2\(g\)' G\(T\) is outside its fitted table domain",
    ):
        evaluate_redox(
            {"Fe": 1.0, "Mn": 0.1, "Si": 0.2, "Al": 0.1, "O": 1.65},
            1400.0,
            "closed",
        )


def test_buffered_capacity_includes_metal_response() -> None:
    inventory = {"Fe": 1.0, "Si": 0.8, "Mg": 0.3, "Al": 0.2, "Ca": 0.1, "O": 2.8}
    center = evaluate_redox(inventory, 2200.0, "closed")
    assert center.metal_buffered
    h_oxygen = 1.0e-5
    plus_inventory = dict(inventory, O=inventory["O"] + h_oxygen)
    minus_inventory = dict(inventory, O=inventory["O"] - h_oxygen)
    plus = evaluate_redox(plus_inventory, 2200.0, "closed")
    minus = evaluate_redox(minus_inventory, 2200.0, "closed")
    finite_difference = (2.0 * h_oxygen) / (
        plus.lambda_ln_f_o2 - minus.lambda_ln_f_o2
    )
    assert center.oxygen_capacity_mol_per_ln_f_o2 == pytest.approx(
        finite_difference, rel=3.0e-4
    )


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


@pytest.mark.parametrize("trace_fraction", [1.0e-6, 1.0e-9])
def test_trace_parent_converges_relatively_and_reports_conditioning(
    trace_fraction: float,
) -> None:
    redox_pack = load_redox_pack()
    base = load_datapack().kernel_datapack
    parent_names = tuple(
        str(parent["parent_oxide"]).removesuffix("(l)")
        for parent in redox_pack.parents
    )
    major = {"Si": 0.5, "Mg": 0.2, "Fe": 0.2, "Ca": 0.1, "Al": 0.1}
    major_parent_total = 0.5 + 0.2 + 0.2 + 0.1 + 0.05
    inventory = dict(major)
    inventory["Ti"] = trace_fraction * major_parent_total / (1.0 - trace_fraction)
    records = {str(parent["element"]): parent for parent in redox_pack.parents}
    fixed_oxygen = math.fsum(
        amount
        * float(records[element]["parent_formula_atoms"].get("O", 0.0))
        / float(records[element]["parent_formula_atoms"][element])
        for element, amount in inventory.items()
    )
    inventory["O"] = fixed_oxygen + 0.05

    result = evaluate_redox(inventory, 2500.0, "closed")
    assert result.metal_moles == 0.0
    oxygen_scale = max(abs(inventory["O"]), math.ulp(inventory["O"]))
    assert abs(result.oxygen_residual_mol or 0.0) <= 1.0e-12 * oxygen_scale

    parent_equivalents = {
        name: float(result.species_moles[name]) for name in parent_names
    }
    for base_index, name in enumerate(base.reactions):
        for parent_index, oxide in enumerate(base.parent_oxides):
            parent_equivalents[oxide] += (
                float(base.nu[parent_index, base_index])
                * float(result.species_moles[name])
            )
    pseudo_rows = [
        row
        for row in redox_pack.rows
        if str(row["complex"]).removesuffix("(l)") not in set(parent_names)
    ]
    for row in pseudo_rows:
        amount = float(result.species_moles[str(row["complex"])])
        for parent, coefficient in row["nu"].items():
            parent_equivalents[str(parent).removesuffix("(l)")] += (
                float(coefficient) * amount
            )
    element_relative_errors = [
        abs(parent_equivalents[name] - float(result.parent_moles[name]))
        / float(result.parent_moles[name])
        for name in parent_names
        if result.parent_moles[name] > 0.0
    ]
    max_element_relative_error = max(element_relative_errors)
    assert max_element_relative_error <= 1.0e-12

    active_activity = result.parent_activity
    log_activity = {name: math.log(value) for name, value in active_activity.items()}
    weights: dict[str, float] = dict(active_activity)
    normalization_terms: list[float] = []
    for index, name in enumerate(base.reactions):
        if any(
            base.nu[parent_index, index] > 0.0
            and base.parent_oxides[parent_index] not in active_activity
            for parent_index in range(len(base.parent_oxides))
        ):
            continue
        ln_weight = math.log(10.0) * (
            float(base.A[index]) + float(base.B[index]) / 2500.0
        ) + math.fsum(
            float(base.nu[parent_index, index])
            * log_activity[oxide]
            for parent_index, oxide in enumerate(base.parent_oxides)
            if base.nu[parent_index, index] > 0.0
        )
        weights[name] = math.exp(ln_weight)
        normalization_terms.append(
            (float(np.sum(base.nu[:, index])) - 1.0) * weights[name]
        )
    for row in pseudo_rows:
        if any(
            float(amount) > 0.0
            and str(parent).removesuffix("(l)") not in active_activity
            for parent, amount in row["nu"].items()
        ):
            continue
        ln_weight = ln_k_prime(
            redox_pack,
            row,
            2500.0,
            ln_f_o2=result.lambda_ln_f_o2,
        ) + math.fsum(
            float(amount)
            * log_activity[str(parent).removesuffix("(l)")]
            for parent, amount in row["nu"].items()
            if float(amount) > 0.0
        )
        weights[str(row["complex"])] = math.exp(ln_weight)
        normalization_terms.append(
            (math.fsum(float(value) for value in row["nu"].values()) - 1.0)
            * weights[str(row["complex"])]
        )
    denominator = 1.0 + math.fsum(normalization_terms)
    basis = math.fsum(float(value) for value in result.parent_moles.values())
    expected_species = {
        name: basis * weight / denominator for name, weight in weights.items()
    }
    species_relative_errors = [
        abs(float(result.species_moles[name]) - expected) / expected
        for name, expected in expected_species.items()
        if expected > 0.0
    ]
    max_species_relative_error = max(species_relative_errors)
    assert max_species_relative_error <= 1.0e-12

    parent_array = np.asarray(
        [float(result.parent_moles[name]) for name in parent_names], dtype=np.float64
    )
    from openimcc.redox import _active_inner

    state = _active_inner(
        parent_array,
        2500.0,
        result.lambda_ln_f_o2,
        base,
        redox_pack,
        tol=1.0e-12,
        max_iter=100,
    )
    condition_number = float(np.linalg.cond(state.J))
    assert math.isfinite(condition_number)
    print(
        f"trace parent fraction={trace_fraction:.0e} "
        f"condition={condition_number:.6g} "
        f"fallback_fired={result.fallback_fired} "
        f"max_species_relative_error={max_species_relative_error:.3g} "
        f"max_element_relative_error={max_element_relative_error:.3g} "
        f"oxygen_residual={result.oxygen_residual_mol:.3g}"
    )


def test_unrowed_positive_parent_is_reported_as_red_flagged_last_resort() -> None:
    result = evaluate_redox(
        {"Fe": 1.0, "Cr": 0.2, "O": 1.4}, 2200.0, "closed"
    )
    assert result.parent_moles["Cr2O3"] == pytest.approx(0.1)
    assert result.last_resort_ideal_parents == ("Cr",)
    assert any(
        flag.startswith("RED:")
        and "last_resort_ideal_parents" in flag
        and "Cr" in flag
        for flag in result.flags
    )

    pack = load_redox_pack()
    every_parent = {
        str(parent["element"]): 0.01 for parent in pack.parents
    }
    all_active = evaluate_redox(
        every_parent, 2200.0, "imposed", lambda_imposed=0.0
    )
    # Mn has sourced rows (parent and association rows), so it is not a
    # last-resort parent.
    assert all_active.last_resort_ideal_parents == (
        "Cr", "Ni", "Co", "P", "V", "Eu", "Ce", "S"
    )
    assert any(
        flag.startswith("RED:")
        and "last_resort_ideal_parents" in flag
        and "Cr, Ni, Co, P, V, Eu, Ce, S" in flag
        for flag in all_active.flags
    )


def test_seed_uses_the_true_positive_parent_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SeedObserved(Exception):
        pass

    captured: list[np.ndarray] = []

    def capture_seed(y: np.ndarray, *_args: object) -> None:
        captured.append(y.copy())
        raise SeedObserved

    monkeypatch.setattr(kernel_module, "_active_residual", capture_seed)
    x_target = np.asarray([1.0 - 1.0e-15, 1.0e-15], dtype=np.float64)
    with pytest.raises(SeedObserved):
        solve_active_warm(
            x_target,
            np.empty((2, 0), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            1.0e-12,
            10,
        )
    assert captured[0][1] == pytest.approx(math.log(1.0e-15), abs=0.0)


def test_positive_parent_underflow_is_not_dropped_or_logged_as_zero() -> None:
    with pytest.raises(RedoxNumericalError, match="positive active parent Cr"):
        evaluate_redox(
            {"Fe": 1.0e308, "Cr": 1.0e-308, "O": 1.0e308},
            2200.0,
            "imposed",
            lambda_imposed=-4.0,
        )


def test_populated_manganese_parent_is_not_reported_as_last_resort() -> None:
    result = evaluate_redox(
        {"Fe": 1.0, "Mn": 0.1, "Cr": 0.2, "O": 1.5}, 2000.0, "closed"
    )
    assert "Mn" not in result.last_resort_ideal_parents
    assert result.last_resort_ideal_parents == ("Cr",)


def test_synthetic_second_redox_couple_closes_and_has_complete_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TEST-ONLY SYNTHETIC COUPLE: TiO2(l) = TiO + 1/2 O2(g). This balanced,
    # made-up row exercises a second registered-parent redox couple and is
    # constructed in memory; it is never added to a shipped pack.
    pack = load_redox_pack()
    synthetic_name = "TiO(test-only)"
    content = dict(pack.content)
    thermo_species = dict(pack.content["thermo_species"])
    thermo_species[synthetic_name] = {"formula_atoms": {"Ti": 1, "O": 1}}
    content["thermo_species"] = thermo_species
    synthetic_row = {
        "complex": synthetic_name,
        "nu": {"TiO2(l)": 1.0},
        "external_oxygen_stoich_product_positive": 0.5,
        "T_domain_K": [1500.0, 3000.0],
        "range_flags": ["test-only synthetic reaction"],
        "synthetic_test_only": True,
        "reaction": "TiO2(l) = TiO(test-only) + 1/2 O2(g)",
        "reactants": {"TiO2(l)": 1.0},
        "products": {synthetic_name: 1.0, "O2(g)": 0.5},
    }
    oxidised_name = "TiO3(test-only)"
    thermo_species[oxidised_name] = {"formula_atoms": {"Ti": 1, "O": 3}}
    oxidised_row = {
        "complex": oxidised_name,
        "nu": {"TiO2(l)": 1.0},
        "external_oxygen_stoich_product_positive": -0.5,
        "T_domain_K": [1500.0, 3000.0],
        "range_flags": ["test-only synthetic reaction"],
        "synthetic_test_only": True,
        "reaction": "TiO2(l) + 1/2 O2(g) = TiO3(test-only)",
        "reactants": {"TiO2(l)": 1.0, "O2(g)": 0.5},
        "products": {oxidised_name: 1.0},
    }
    content["rows"] = list(pack.rows) + [synthetic_row, oxidised_row]
    synthetic_pack = RedoxPack(content, pack.canonical_digest)
    real_ln_k_prime = ln_k_prime
    synthetic_offsets = {
        synthetic_name: 0.4,
        oxidised_name: 0.4,
    }

    def synthetic_ln_k_prime(
        redox_pack: RedoxPack,
        row: dict[str, object],
        temperature_K: float,
        ln_f_o2: float = 0.0,
        *,
        allow_extrapolation: bool = False,
    ) -> float:
        if row.get("synthetic_test_only"):
            q = -float(row["external_oxygen_stoich_product_positive"])
            return synthetic_offsets[str(row["complex"])] + q * ln_f_o2
        return real_ln_k_prime(
            redox_pack,
            row,
            temperature_K,
            ln_f_o2,
            allow_extrapolation=allow_extrapolation,
        )

    monkeypatch.setattr("openimcc.redox.load_redox_pack", lambda: synthetic_pack)
    monkeypatch.setattr("openimcc.redox.ln_k_prime", synthetic_ln_k_prime)

    formulas = _formula_atoms()
    formulas[synthetic_name] = {"Ti": 1.0, "O": 1.0}
    formulas[oxidised_name] = {"Ti": 1.0, "O": 3.0}
    first_inventory = oxide_inventory_to_elements(
        {"FeO(l)": 1.0, "TiO2(l)": 0.25}, formulas
    )
    second_inventory = oxide_inventory_to_elements(
        {"FeO(l)": 1.0, synthetic_name: 0.25, "O2(g)": 0.125}, formulas
    )
    assert first_inventory == second_inventory == {"Fe": 1.0, "Ti": 0.25, "O": 1.5}

    first = evaluate_redox(first_inventory, 2200.0, "closed")
    second = evaluate_redox(second_inventory, 2200.0, "closed")
    assert first.lambda_ln_f_o2 == pytest.approx(second.lambda_ln_f_o2, abs=1.0e-12)
    assert first.metal_moles == pytest.approx(second.metal_moles, abs=1.0e-12)
    assert first.species_moles == pytest.approx(second.species_moles, rel=1.0e-12)
    assert first.species_moles[synthetic_name] > 0.0
    assert first.oxygen_residual_mol == pytest.approx(0.0, abs=2.0e-12)

    from openimcc.redox import _active_inner, _redox_parent_names

    parent_names = _redox_parent_names(synthetic_pack)
    parent_records_by_element = {
        str(parent["element"]): parent for parent in synthetic_pack.parents
    }
    parent_moles = np.zeros(len(parent_names), dtype=np.float64)
    for element, amount in {"Fe": 1.0, "Ti": 0.25}.items():
        record = parent_records_by_element[element]
        parent_name = str(record["parent_oxide"]).removesuffix("(l)")
        parent_index = parent_names.index(parent_name)
        element_count = float(record["parent_formula_atoms"][element])
        parent_moles[parent_index] = amount / element_count
    state = _active_inner(
        parent_moles,
        2200.0,
        first.lambda_ln_f_o2,
        load_datapack().kernel_datapack,
        synthetic_pack,
        tol=1.0e-12,
        max_iter=100,
    )
    for row_name, expected_delta, expected_q in (
        ("FeO1.5(l)", 0.5, 0.25),
        (synthetic_name, -1.0, -0.5),
        (oxidised_name, 1.0, 0.5),
    ):
        row_index = state.complex_names.index(row_name)
        delta = float(state.complex_oxygen_deltas[row_index])
        q = float(state.complex_lambda_shift[row_index])
        assert delta == pytest.approx(expected_delta, abs=0.0)
        assert q == pytest.approx(expected_q, abs=0.0)
        assert delta == pytest.approx(2.0 * q, abs=0.0)

    # These inventories lie outside the Fe-only [0.5, 2.0] oxygen window.
    # The Ti row limits extend the endpoints to 0.25 and 2.25 mol O.
    reduced_window_root = evaluate_redox(
        {"Fe": 1.0, "Ti": 0.25, "O": 0.4}, 2200.0, "closed"
    )
    oxidised_window_root = evaluate_redox(
        {"Fe": 1.0, "Ti": 0.25, "O": 2.2}, 2200.0, "closed"
    )
    assert reduced_window_root.metal_buffered
    assert reduced_window_root.metal_moles > 0.8
    assert reduced_window_root.oxygen_residual_mol is not None
    assert abs(reduced_window_root.oxygen_residual_mol) <= 1.5e-13
    assert not oxidised_window_root.metal_buffered
    assert oxidised_window_root.species_moles[oxidised_name] > 0.0
    assert oxidised_window_root.oxygen_residual_mol is not None
    assert abs(oxidised_window_root.oxygen_residual_mol) <= 5.0e-14

    with pytest.raises(RedoxEndpointError) as below_minimum:
        evaluate_redox({"Fe": 1.0, "Ti": 0.25, "O": 0.24}, 2200.0, "closed")
    assert below_minimum.value.endpoint == "reductant_exceeds_reducible_oxygen"
    assert below_minimum.value.oxygen_residual_mol == pytest.approx(0.01)
    with pytest.raises(RedoxNoLiveCoupleError):
        evaluate_redox({"Fe": 1.0, "Ti": 0.25, "O": 0.25}, 2200.0, "closed")
    with pytest.raises(RedoxEndpointError) as above_maximum:
        evaluate_redox({"Fe": 1.0, "Ti": 0.25, "O": 2.26}, 2200.0, "closed")
    assert above_maximum.value.endpoint == "fully_oxidised_inventory_limit"
    assert above_maximum.value.oxygen_residual_mol == pytest.approx(0.01)
    with pytest.raises(RedoxEndpointError) as at_maximum:
        evaluate_redox({"Fe": 1.0, "Ti": 0.25, "O": 2.25}, 2200.0, "closed")
    assert at_maximum.value.endpoint == "fully_oxidised_inventory_limit"
    assert at_maximum.value.oxygen_residual_mol == pytest.approx(0.0, abs=0.0)

    # At the all-metal endpoint, this synthetic oxidising couple stays above
    # its reduced limit when the fugacity offset is large. Its speciated
    # oxygen, rather than the stoichiometric minimum, determines the bracket.
    synthetic_offsets.update(
        {synthetic_name: -80.0, oxidised_name: 80.0}
    )
    with pytest.raises(RedoxEndpointError) as oxidising_all_metal:
        evaluate_redox({"Fe": 0.1, "Ti": 1.0, "O": 2.2}, 2200.0, "closed")
    assert oxidising_all_metal.value.endpoint == "reductant_exceeds_reducible_oxygen"
    assert oxidising_all_metal.value.oxygen_residual_mol is not None
    assert oxidising_all_metal.value.oxygen_residual_mol > 0.79

    synthetic_offsets[oxidised_name] = 20.0
    metal_root = evaluate_redox(
        {"Fe": 0.1, "Ti": 1.0, "O": 2.2}, 2200.0, "closed"
    )
    assert metal_root.metal_buffered
    assert metal_root.oxygen_residual_mol is not None
    assert abs(metal_root.oxygen_residual_mol) <= 9.5e-13

    base = load_datapack().kernel_datapack
    parent_records = {
        str(parent["parent_oxide"]).removesuffix("(l)"): parent
        for parent in synthetic_pack.parents
    }
    oxygen_parent = np.asarray(
        [
            float(parent_records[name]["parent_formula_atoms"].get("O", 0.0))
            for name in base.parent_oxides
        ],
        dtype=np.float64,
    )
    oxygen_terms = [
        float(first.species_moles[name])
        * float(parent_records[name]["parent_formula_atoms"].get("O", 0.0))
        for name in (
            str(parent["parent_oxide"]).removesuffix("(l)")
            for parent in synthetic_pack.parents
        )
    ]
    oxygen_terms.extend(
        float(first.species_moles[name])
        * float(np.dot(oxygen_parent, base.nu[:, index]))
        for index, name in enumerate(base.reactions)
    )
    for row in (synthetic_pack.rows[1], synthetic_row, oxidised_row):
        amount = float(first.species_moles[str(row["complex"])])
        oxygen_terms.append(
            amount
            * float(
                synthetic_pack.content["thermo_species"][str(row["complex"])]
                ["formula_atoms"].get("O", 0.0)
            )
        )
    reconstructed_oxygen = math.fsum(oxygen_terms)
    assert reconstructed_oxygen == pytest.approx(first_inventory["O"], rel=1.0e-12)

    center = evaluate_redox(
        first_inventory,
        2200.0,
        "imposed",
        lambda_imposed=first.lambda_ln_f_o2,
    )
    h = 1.0e-5
    minus = evaluate_redox(
        first_inventory,
        2200.0,
        "imposed",
        lambda_imposed=center.lambda_ln_f_o2 - h,
    )
    plus = evaluate_redox(
        first_inventory,
        2200.0,
        "imposed",
        lambda_imposed=center.lambda_ln_f_o2 + h,
    )
    finite_difference = (plus.oxygen_mol - minus.oxygen_mol) / (2.0 * h)
    assert center.oxygen_capacity_mol_per_ln_f_o2 == pytest.approx(
        finite_difference, rel=2.0e-6, abs=1.0e-12
    )
    left_slope = (center.oxygen_mol - minus.oxygen_mol) / h
    right_slope = (plus.oxygen_mol - center.oxygen_mol) / h
    assert left_slope == pytest.approx(right_slope, rel=2.0e-4)
    assert plus.species_moles[synthetic_name] < minus.species_moles[synthetic_name]


def test_shared_parent_rows_share_the_oxygen_window_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = load_redox_pack()
    oxidised_name = "TiO3(shared-parent-test)"
    more_oxidised_name = "TiO4(shared-parent-test)"
    content = dict(pack.content)
    thermo_species = dict(pack.content["thermo_species"])
    thermo_species[oxidised_name] = {"formula_atoms": {"Ti": 1, "O": 3}}
    thermo_species[more_oxidised_name] = {"formula_atoms": {"Ti": 1, "O": 4}}
    content["thermo_species"] = thermo_species
    shared_parent_rows = [
        {
            "complex": oxidised_name,
            "nu": {"TiO2(l)": 1.0},
            "external_oxygen_stoich_product_positive": -0.5,
            "synthetic_test_only": True,
            "T_domain_K": [1500.0, 3000.0],
            "range_flags": ["test-only synthetic reaction"],
            "reaction": "TiO2(l) + 1/2 O2(g) = TiO3(test-only)",
            "reactants": {"TiO2(l)": 1.0, "O2(g)": 0.5},
            "products": {oxidised_name: 1.0},
        },
        {
            "complex": more_oxidised_name,
            "nu": {"TiO2(l)": 1.0},
            "external_oxygen_stoich_product_positive": -1.0,
            "synthetic_test_only": True,
            "T_domain_K": [1500.0, 3000.0],
            "range_flags": ["test-only synthetic reaction"],
            "reaction": "TiO2(l) + O2(g) = TiO4(test-only)",
            "reactants": {"TiO2(l)": 1.0, "O2(g)": 1.0},
            "products": {more_oxidised_name: 1.0},
        },
    ]
    content["rows"] = list(pack.rows) + shared_parent_rows
    shared_pack = RedoxPack(content, pack.canonical_digest)
    monkeypatch.setattr("openimcc.redox.load_redox_pack", lambda: shared_pack)

    real_ln_k_prime = ln_k_prime
    synthetic_offsets = {oxidised_name: 0.4, more_oxidised_name: 0.4}

    def synthetic_ln_k_prime(
        redox_pack: RedoxPack,
        row: dict[str, object],
        temperature_K: float,
        ln_f_o2: float = 0.0,
        *,
        allow_extrapolation: bool = False,
    ) -> float:
        if row.get("synthetic_test_only"):
            q = -float(row["external_oxygen_stoich_product_positive"])
            return synthetic_offsets[str(row["complex"])] + q * ln_f_o2
        return real_ln_k_prime(
            redox_pack,
            row,
            temperature_K,
            ln_f_o2,
            allow_extrapolation=allow_extrapolation,
        )

    monkeypatch.setattr("openimcc.redox.ln_k_prime", synthetic_ln_k_prime)

    for oxygen in (1.0, 1.5, 2.0, 2.2, 2.49):
        result = evaluate_redox(
            {"Fe": 1.0, "Ti": 0.25, "O": oxygen}, 2200.0, "closed"
        )
        if oxygen >= 2.0:
            assert not result.metal_buffered
        assert result.oxygen_residual_mol is not None
        endpoint_distance = min(oxygen - 0.5, 2.5 - oxygen)
        ledger_bar = max(
            1.0e-12 * endpoint_distance,
            16.0 * math.ulp(max(abs(oxygen), 1.0)),
        )
        assert abs(result.oxygen_residual_mol) <= ledger_bar

    with pytest.raises(RedoxEndpointError) as at_limit:
        evaluate_redox({"Fe": 1.0, "Ti": 0.25, "O": 2.5}, 2200.0, "closed")
    assert at_limit.value.endpoint == "fully_oxidised_inventory_limit"
    assert at_limit.value.oxygen_residual_mol == pytest.approx(0.0, abs=0.0)

    with pytest.raises(RedoxEndpointError) as above_limit:
        evaluate_redox({"Fe": 1.0, "Ti": 0.25, "O": 2.6}, 2200.0, "closed")
    assert above_limit.value.endpoint == "fully_oxidised_inventory_limit"
    assert above_limit.value.oxygen_residual_mol == pytest.approx(0.1)


def test_shared_parent_multi_element_oxidising_root_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = load_redox_pack()
    oxidised_name = "SiTiO7(shared-parent-test)"
    more_oxidised_name = "TiO4(shared-parent-test)"
    content = dict(pack.content)
    thermo_species = dict(pack.content["thermo_species"])
    thermo_species[oxidised_name] = {
        "formula_atoms": {"Si": 1, "Ti": 1, "O": 7}
    }
    thermo_species[more_oxidised_name] = {"formula_atoms": {"Ti": 1, "O": 4}}
    content["thermo_species"] = thermo_species
    shared_parent_rows = [
        {
            "complex": oxidised_name,
            "nu": {"SiO2(l)": 1.0, "TiO2(l)": 1.0},
            "external_oxygen_stoich_product_positive": -1.5,
            "synthetic_test_only": True,
            "T_domain_K": [1500.0, 3000.0],
            "range_flags": ["test-only synthetic reaction"],
            "reaction": "SiO2(l) + TiO2(l) + 3/2 O2(g) = SiTiO7(test-only)",
            "reactants": {"SiO2(l)": 1.0, "TiO2(l)": 1.0, "O2(g)": 1.5},
            "products": {oxidised_name: 1.0},
        },
        {
            "complex": more_oxidised_name,
            "nu": {"TiO2(l)": 1.0},
            "external_oxygen_stoich_product_positive": -1.0,
            "synthetic_test_only": True,
            "T_domain_K": [1500.0, 3000.0],
            "range_flags": ["test-only synthetic reaction"],
            "reaction": "TiO2(l) + O2(g) = TiO4(test-only)",
            "reactants": {"TiO2(l)": 1.0, "O2(g)": 1.0},
            "products": {more_oxidised_name: 1.0},
        },
    ]
    content["rows"] = list(pack.rows) + shared_parent_rows
    shared_pack = RedoxPack(content, pack.canonical_digest)
    monkeypatch.setattr("openimcc.redox.load_redox_pack", lambda: shared_pack)
    real_ln_k_prime = ln_k_prime

    def synthetic_ln_k_prime(
        redox_pack: RedoxPack,
        row: dict[str, object],
        temperature_K: float,
        ln_f_o2: float = 0.0,
        *,
        allow_extrapolation: bool = False,
    ) -> float:
        if row.get("synthetic_test_only"):
            q = -float(row["external_oxygen_stoich_product_positive"])
            return 0.4 + q * ln_f_o2
        return real_ln_k_prime(
            redox_pack,
            row,
            temperature_K,
            ln_f_o2,
            allow_extrapolation=allow_extrapolation,
        )

    monkeypatch.setattr("openimcc.redox.ln_k_prime", synthetic_ln_k_prime)
    result = evaluate_redox(
        {"Fe": 1.0, "Ti": 1.0, "Si": 0.5, "O": 6.9}, 2200.0, "closed"
    )
    assert not result.metal_buffered
    assert result.oxygen_residual_mol is not None
    assert abs(result.oxygen_residual_mol) <= 1.0e-13


def test_reduced_limit_counts_fe_locked_in_minimizing_redox_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = load_redox_pack()
    synthetic_name = "FeTiO(reduced-limit-test)"
    content = dict(pack.content)
    thermo_species = dict(pack.content["thermo_species"])
    thermo_species[synthetic_name] = {"formula_atoms": {"Fe": 1, "Ti": 1, "O": 1}}
    content["thermo_species"] = thermo_species
    synthetic_row = {
        "complex": synthetic_name,
        "nu": {"FeO(l)": 1.0, "TiO2(l)": 1.0},
        "external_oxygen_stoich_product_positive": 1.0,
        "synthetic_test_only": True,
        "T_domain_K": [1500.0, 3000.0],
        "range_flags": ["test-only synthetic reaction"],
        "reaction": "FeO(l) + TiO2(l) = FeTiO(test-only) + O2(g)",
        "reactants": {"FeO(l)": 1.0, "TiO2(l)": 1.0},
        "products": {synthetic_name: 1.0, "O2(g)": 1.0},
    }
    content["rows"] = list(pack.rows) + [synthetic_row]
    synthetic_pack = RedoxPack(content, pack.canonical_digest)
    monkeypatch.setattr("openimcc.redox.load_redox_pack", lambda: synthetic_pack)
    real_ln_k_prime = ln_k_prime

    def synthetic_ln_k_prime(
        redox_pack: RedoxPack,
        row: dict[str, object],
        temperature_K: float,
        ln_f_o2: float = 0.0,
        *,
        allow_extrapolation: bool = False,
    ) -> float:
        if row.get("synthetic_test_only"):
            return 80.0 - ln_f_o2
        return real_ln_k_prime(
            redox_pack,
            row,
            temperature_K,
            ln_f_o2,
            allow_extrapolation=allow_extrapolation,
        )

    monkeypatch.setattr("openimcc.redox.ln_k_prime", synthetic_ln_k_prime)

    for oxygen, expected_residual in ((0.0, 0.25), (0.1, 0.15)):
        with pytest.raises(RedoxEndpointError) as below_minimum:
            evaluate_redox(
                {"Fe": 1.0, "Ti": 0.25, "O": oxygen}, 2200.0, "closed"
            )
        assert below_minimum.value.endpoint == "reductant_exceeds_reducible_oxygen"
        assert below_minimum.value.oxygen_residual_mol == pytest.approx(
            expected_residual
        )

    with pytest.raises(RedoxNoLiveCoupleError):
        evaluate_redox({"Fe": 1.0, "Ti": 0.25, "O": 0.25}, 2200.0, "closed")


def test_only_active_rows_are_flagged_as_extrapolated() -> None:
    # Fe-only melt at 1500 K: the FeO row's evidence domain starts at 1700 K,
    # so it is extrapolated; the Mn rows are inactive (no Mn in the inventory)
    # and must not be reported even though 1500 K is outside their domain too.
    result = evaluate_redox({"Fe": 1.0, "Si": 0.8, "O": 2.4}, 1500.0, "closed")
    assert result.extrapolated_redox_rows == ("FeO(l)",)
    assert not any("MnO(l)" in flag for flag in result.flags)


def test_pack_level_evaluators_stay_strict_by_default() -> None:
    # Only evaluate_redox opts into evidence-domain extrapolation; direct pack
    # calls keep the fail-loud default outside a row's declared domain.
    pack = load_redox_pack()
    row = next(row for row in pack.rows if row["complex"] == "MnO(l)")
    with pytest.raises(ValueError):
        ln_k_prime(pack, row, 2200.0)
    with pytest.raises(ValueError):
        species_gibbs_j_mol(pack, "MnO(l)", 2200.0)


def test_missing_default_pack_is_a_typed_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import openimcc.redox as redox_module
    from openimcc.kernel import ImccRefusal

    monkeypatch.setattr(
        redox_module, "_DEFAULT_REDOX_PACK_PATH", tmp_path / "absent.json"
    )
    with pytest.raises(ImccRefusal):
        evaluate_redox({"Fe": 1.0, "O": 1.25}, 2200.0, "closed")
