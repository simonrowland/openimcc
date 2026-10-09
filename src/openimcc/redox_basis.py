"""Element-mole inventory basis used by redox calculations.

An inventory contains amounts of elements in moles of atoms. Formula-unit
amounts are inputs only: each is expanded by its declared atom counts, and
oxygen remains an ordinary element total. In particular, Fe2O3 is never
rewritten as 2 FeO; an equivalent input may use 2 FeO plus 0.5 O2.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


def oxide_inventory_to_elements(
    formula_unit_moles: Mapping[str, float],
    formula_atoms: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Convert formula-unit moles into elemental moles of atoms.

    ``formula_unit_moles`` is in mol of each oxide or oxygen-bearing feed
    species. ``formula_atoms`` gives atoms per formula unit. Returned values
    are mol atoms; oxygen is counted separately under ``"O"``. This is a
    linear stoichiometric map, so different formula splits with the same
    elemental totals produce the same inventory.
    """
    totals: dict[str, float] = {}
    for formula, amount in formula_unit_moles.items():
        if formula not in formula_atoms:
            raise ValueError(f"no atom-count record for formula {formula!r}")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValueError(f"amount for {formula!r} must be a real number")
        amount_value = float(amount)
        if not math.isfinite(amount_value) or amount_value < 0.0:
            raise ValueError(f"amount for {formula!r} must be finite and non-negative")
        atoms = formula_atoms[formula]
        if not atoms:
            raise ValueError(f"formula {formula!r} has no declared atoms")
        for element, count in atoms.items():
            if isinstance(count, bool) or not isinstance(count, (int, float)):
                raise ValueError(f"atom count for {formula!r}/{element} must be real")
            count_value = float(count)
            if not math.isfinite(count_value) or count_value <= 0.0:
                raise ValueError(
                    f"atom count for {formula!r}/{element} must be finite and positive"
                )
            totals[element] = totals.get(element, 0.0) + amount_value * count_value
    return {element: totals[element] for element in sorted(totals)}
