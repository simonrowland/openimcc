from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from openimcc import (
    PackRowParticipation,
    evaluate,
    evaluate_redox,
    evaluate_sulfur,
    load_datapack,
)
from openimcc._constants import MOLAR_MASS_G_MOL
from openimcc.redox_basis import oxide_inventory_to_elements
from openimcc.redox_pack import load_redox_pack


_MELT = {
    "SiO2": 1.0,
    "MgO": 0.2,
    "FeO": 0.2,
    "CaO": 0.2,
    "Al2O3": 0.2,
    "TiO2": 0.05,
    "Na2O": 0.1,
    "K2O": 0.2,
}

_MORB_WT = {
    "SiO2": 50.42,
    "TiO2": 1.53,
    "Al2O3": 15.13,
    "FeO": 9.81,
    "MgO": 7.76,
    "CaO": 11.35,
    "Na2O": 2.83,
    "K2O": 0.14,
}


def _participation_state(records: tuple[PackRowParticipation, ...]):
    return tuple(
        (row.row_key, row.active, row.gating_reason) for row in records
    )


def test_potassium_rows_report_solved_and_activation_gates() -> None:
    pack = load_datapack()
    result = evaluate(_MELT, 2200.0, pack)
    rows = {row.row_key: row for row in result.pack_row_participation}
    active_k_rows = {
        "K2SiO3",
        "K2Si2O5",
        "KAlSiO4",
        "KAlSi3O8",
        "KAlO2",
        "KAlSi2O6",
        "K2Si4O9",
    }

    assert active_k_rows <= rows.keys()
    assert all(rows[name].active for name in active_k_rows)
    assert all(rows[name].gating_reason is None for name in active_k_rows)
    assert all(
        rows[name].equilibrium_mole_fraction > 0.0 for name in active_k_rows
    )

    inactive = rows["KCaAlSi2O7"]
    assert inactive.active is False
    assert inactive.gating_reason == dict(pack.inactive_rows)["KCaAlSi2O7"]
    assert inactive.equilibrium_mole_fraction == 0.0


def test_rows_missing_a_required_oxide_are_omitted() -> None:
    composition = {name: amount for name, amount in _MELT.items() if name != "CaO"}
    result = evaluate(composition, 2200.0)
    row_keys = {row.row_key for row in result.pack_row_participation}

    assert "CaSiO3" not in row_keys
    assert "KCaAlSi2O7" not in row_keys


def test_participation_records_are_frozen() -> None:
    result = evaluate(_MELT, 2200.0)
    record = result.pack_row_participation[0]

    assert isinstance(record, PackRowParticipation)
    with pytest.raises(FrozenInstanceError):
        record.active = False  # type: ignore[misc]


def test_redox_and_sulfur_expose_the_melt_rows_for_the_same_composition() -> None:
    temperature = 2200.0
    oxide_moles = {
        oxide: amount / MOLAR_MASS_G_MOL[oxide]
        for oxide, amount in _MORB_WT.items()
    }
    oxide_total_moles = math.fsum(oxide_moles.values())
    melt = evaluate(
        {
            oxide: amount / oxide_total_moles
            for oxide, amount in oxide_moles.items()
        },
        temperature,
    )
    sulfur = evaluate_sulfur(
        _MORB_WT,
        temperature,
        log10_fO2=-8.0,
        log10_fS2=-10.0,
    )

    assert sulfur.pack_row_participation == melt.pack_row_participation

    redox_pack = load_redox_pack()
    formula_atoms = {
        str(parent["parent_oxide"]).removesuffix("(l)"): parent[
            "parent_formula_atoms"
        ]
        for parent in redox_pack.parents
    }
    inventory = oxide_inventory_to_elements(oxide_moles, formula_atoms)
    inventory["O"] += 0.002
    redox = evaluate_redox(inventory, temperature, "closed")

    assert _participation_state(redox.pack_row_participation) == _participation_state(
        melt.pack_row_participation
    )

