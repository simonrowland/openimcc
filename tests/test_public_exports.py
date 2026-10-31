from __future__ import annotations

import openimcc
from openimcc import (
    IMCC_GAS_CHANNEL_SPECIES,
    IMCC_GAS_UNAVAILABLE_SPECIES,
    IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES,
    IMCC_PARENT_OXIDES,
    R_J_MOL_K,
    subsolidus,
    weighted_oxygen_balance_from_pressure_model,
)


def test_package_exports_resolve() -> None:
    assert len(openimcc.__all__) == len(set(openimcc.__all__))
    assert all(hasattr(openimcc, name) for name in openimcc.__all__)
    assert callable(weighted_oxygen_balance_from_pressure_model)
    assert subsolidus.__name__ == "openimcc.subsolidus"
    assert IMCC_GAS_CHANNEL_SPECIES
    assert isinstance(IMCC_GAS_UNAVAILABLE_SPECIES, dict)
    assert IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES
    assert IMCC_PARENT_OXIDES
    assert R_J_MOL_K > 0.0
