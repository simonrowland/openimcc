"""Versioned loader and evaluator for the redox thermochemistry pack.

The pack uses the shipped condensate φ polynomial and JANAF Shomate
evaluators. This module assembles their standard Gibbs energies into balanced
reaction rows; it does not define another Gibbs-energy function.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

_PACK_NAME = "openimcc-redox-v1.json"
_SCHEMA = "openimcc-redox-pack.v1"


def canonical_redox_digest(content: Mapping[str, Any]) -> str:
    """Return SHA-256 over canonical JSON, excluding its stored digest field."""
    projection = _json_value(content)
    projection.pop("canonical_content_digest", None)
    canonical = json.dumps(
        projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert frozen mappings and tuples back to canonical JSON containers."""
    if isinstance(value, MappingABC):
        return {str(key): _json_value(member) for key, member in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(member) for member in value]
    return value


def _freeze_json(value: Any) -> Any:
    """Prevent loaded content from drifting away from its exposed digest."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(member) for key, member in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(member) for member in value)
    return value


@dataclass(frozen=True)
class RedoxPack:
    """Loaded, validated redox component and its canonical content digest."""

    content: Mapping[str, Any]
    canonical_digest: str

    @property
    def version(self) -> str:
        return str(self.content["version"])

    @property
    def parents(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.content["parent_registry"])

    @property
    def rows(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.content["rows"])


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _atom_totals(
    terms: Mapping[str, Any], species: Mapping[str, Mapping[str, Any]], label: str
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for name, coefficient in terms.items():
        if name not in species:
            raise ValueError(f"{label} refers to unknown thermochemical species {name!r}")
        amount = _finite_number(coefficient, f"{label}/{name}")
        if amount <= 0.0:
            raise ValueError(f"{label}/{name} must be positive")
        for element, count in species[name]["formula_atoms"].items():
            totals[element] = totals.get(element, 0.0) + amount * float(count)
    return totals


def _validate_content(content: Mapping[str, Any]) -> None:
    if content.get("schema") != _SCHEMA:
        raise ValueError(f"unsupported redox pack schema {content.get('schema')!r}")
    if not isinstance(content.get("version"), str) or not content["version"]:
        raise ValueError("redox pack version must be a non-empty string")
    required = {
        "units", "parent_registry", "thermo_species", "rows",
        "source_selection_rail",
    }
    missing = required - content.keys()
    if missing:
        raise ValueError(f"redox pack is missing keys: {sorted(missing)}")
    if content["units"] != {
        "amount": "mol formula units",
        "inventory": "mol atoms by element",
        "temperature": "K",
        "gibbs_energy": "J mol^-1 reaction",
        "band": "kJ mol^-1 reaction",
        "pressure": "bar relative to 1 bar standard state",
    }:
        raise ValueError("redox pack units do not match schema v1")
    if content["source_selection_rail"] != [
        "standard_state_consistency", "primary_measurement", "domain_match",
        "systematic_estimate",
    ]:
        raise ValueError("redox pack does not follow the declared source-selection rail")

    registry = content["parent_registry"]
    if not isinstance(registry, list) or not registry:
        raise ValueError("parent_registry must be a non-empty list")
    elements: set[str] = set()
    parent_names: set[str] = set()
    for index, parent in enumerate(registry):
        if not isinstance(parent, dict):
            raise ValueError(f"parent_registry[{index}] must be an object")
        element = parent.get("element")
        parent_oxide = parent.get("parent_oxide")
        if not isinstance(element, str) or not element or element in elements:
            raise ValueError(f"parent_registry[{index}] has missing or duplicate element")
        if not isinstance(parent_oxide, str) or not parent_oxide:
            raise ValueError(f"parent_registry[{index}] has no parent oxide")
        if not isinstance(parent.get("parent_standard_state"), str):
            raise ValueError(f"parent_registry[{index}] has no parent standard state")
        atoms = parent.get("parent_formula_atoms")
        if not isinstance(atoms, dict) or not atoms or element not in atoms:
            raise ValueError(f"parent_registry[{index}] has invalid parent_formula_atoms")
        for atom, count in atoms.items():
            if not isinstance(atom, str) or _finite_number(
                count, f"parent_registry[{index}]/parent_formula_atoms/{atom}"
            ) <= 0.0:
                raise ValueError(f"parent_registry[{index}] has invalid atom count")
        elements.add(element)
        if parent_oxide in parent_names:
            raise ValueError(f"parent_registry[{index}] has duplicate parent oxide")
        parent_names.add(parent_oxide)

    species = content["thermo_species"]
    if not isinstance(species, dict) or not species:
        raise ValueError("thermo_species must be a non-empty object")
    for name, record in species.items():
        if not isinstance(record, dict) or not isinstance(record.get("formula_atoms"), dict):
            raise ValueError(f"thermo_species[{name!r}] has no formula_atoms")
        if not record["formula_atoms"]:
            raise ValueError(f"thermo_species[{name!r}] has empty formula_atoms")
        for element, count in record["formula_atoms"].items():
            if not isinstance(element, str) or _finite_number(
                count, f"thermo_species[{name!r}]/formula_atoms/{element}"
            ) <= 0.0:
                raise ValueError(f"thermo_species[{name!r}] has invalid formula atoms")
        domain = record.get("T_domain_K")
        if not isinstance(domain, list) or len(domain) != 2:
            raise ValueError(f"thermo_species[{name!r}] needs a two-point T_domain_K")
        low, high = (_finite_number(x, f"thermo_species[{name!r}]/T_domain_K") for x in domain)
        if low <= 0.0 or high <= low:
            raise ValueError(f"thermo_species[{name!r}] has invalid temperature domain")
        band = record.get("band_kj_mol")
        if _finite_number(band, f"thermo_species[{name!r}]/band_kj_mol") < 0.0:
            raise ValueError(f"thermo_species[{name!r}] has negative band")
        if not isinstance(record.get("range_flags"), list):
            raise ValueError(f"thermo_species[{name!r}] needs range_flags")
        if not isinstance(record.get("provenance"), dict):
            raise ValueError(f"thermo_species[{name!r}] has no provenance")
        provenance = record["provenance"]
        if not all(isinstance(provenance.get(key), str) and provenance[key]
                   for key in ("source", "source_url", "tier", "selection_rationale")):
            raise ValueError(f"thermo_species[{name!r}] has incomplete provenance")
        evaluator = record.get("evaluator")
        if evaluator not in {"condensate_phi", "janaf_shomate", "linear_combination", "piecewise"}:
            raise ValueError(f"thermo_species[{name!r}] has unknown evaluator")
        if evaluator in {"condensate_phi", "janaf_shomate"}:
            if not isinstance(record.get("row"), dict):
                raise ValueError(f"thermo_species[{name!r}] has no evaluator row")
        if evaluator == "linear_combination":
            if not isinstance(record.get("terms"), list) or not record["terms"]:
                raise ValueError(f"thermo_species[{name!r}] has no terms")
            for term in record["terms"]:
                if term.get("species") not in species:
                    raise ValueError(f"thermo_species[{name!r}] refers to an unknown term")
        if evaluator == "piecewise":
            if not isinstance(record.get("branches"), list) or len(record["branches"]) != 2:
                raise ValueError(f"thermo_species[{name!r}] needs two branches")
            if any(branch.get("species") not in species for branch in record["branches"]):
                raise ValueError(f"thermo_species[{name!r}] has an unknown branch")

    rows = content["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list")
    for index, row in enumerate(rows):
        label = f"rows[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{label} must be an object")
        if row.get("complex") not in species:
            raise ValueError(f"{label} names an unknown product species")
        if not isinstance(row.get("nu"), dict):
            raise ValueError(f"{label} needs parent stoichiometry nu")
        for parent, amount in row["nu"].items():
            if parent not in parent_names:
                raise ValueError(f"{label}/nu names unregistered parent {parent!r}")
            if _finite_number(amount, f"{label}/nu/{parent}") < 0.0:
                raise ValueError(f"{label}/nu/{parent} must be non-negative")
        oxygen = _finite_number(
            row.get("external_oxygen_stoich_product_positive"),
            f"{label}/external_oxygen_stoich_product_positive",
        )
        if not isinstance(row.get("reactants"), dict) or not isinstance(row.get("products"), dict):
            raise ValueError(f"{label} needs reactants and products")
        reactants = _atom_totals(row["reactants"], species, f"{label}/reactants")
        products = _atom_totals(row["products"], species, f"{label}/products")
        for element in set(reactants) | set(products):
            if not math.isclose(reactants.get(element, 0.0), products.get(element, 0.0),
                                rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(f"{label} is unbalanced for {element}")
        row_oxygen = float(row["products"].get("O2(g)", 0.0)) - float(
            row["reactants"].get("O2(g)", 0.0)
        )
        for parent_oxide in parent_names:
            if not math.isclose(
                float(row["nu"].get(parent_oxide, 0.0)),
                float(row["reactants"].get(parent_oxide, 0.0)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{label} nu does not match parent reactant stoichiometry")
        if not math.isclose(
            float(row["products"].get(row["complex"], 0.0)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label} must produce one formula unit of its named complex")
        if not math.isclose(oxygen, row_oxygen, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label} oxygen convention does not match the reaction")
        domain = row.get("T_domain_K")
        if not isinstance(domain, list) or len(domain) != 2:
            raise ValueError(f"{label} needs a two-point T_domain_K")
        low, high = (_finite_number(x, f"{label}/T_domain_K") for x in domain)
        if low <= 0.0 or high <= low:
            raise ValueError(f"{label} has invalid temperature domain")
        if _finite_number(row.get("band_kj_mol"), f"{label}/band_kj_mol") < 0.0:
            raise ValueError(f"{label} needs a non-negative Gibbs band")
        for field in ("phase_or_construction", "method", "band_method", "uncertainty_note"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{label} needs {field}")
        if not isinstance(row.get("range_flags"), list) or not row["range_flags"]:
            raise ValueError(f"{label} needs explicit range flags")
        if not isinstance(row.get("provenance"), dict):
            raise ValueError(f"{label} needs provenance")
        if not isinstance(row.get("standard_gibbs_terms"), list) or not row["standard_gibbs_terms"]:
            raise ValueError(f"{label} needs standard_gibbs_terms")
        terms = {}
        for term in row["standard_gibbs_terms"]:
            species_name = term.get("species")
            coefficient = _finite_number(term.get("coefficient"), f"{label}/standard_gibbs_terms")
            if species_name not in species:
                raise ValueError(f"{label} G term refers to unknown species {species_name!r}")
            terms[species_name] = terms.get(species_name, 0.0) + coefficient
        expected = {
            name: float(row["products"].get(name, 0.0)) - float(row["reactants"].get(name, 0.0))
            for name in set(row["products"]) | set(row["reactants"])
        }
        expected = {name: value for name, value in expected.items() if value}
        terms = {name: value for name, value in terms.items() if value}
        if terms != expected:
            raise ValueError(f"{label} standard_gibbs_terms do not match reaction stoichiometry")
        if row.get("role") not in {"parent_reference", "complex"}:
            raise ValueError(f"{label} has unsupported row role")
        provenance = row["provenance"]
        if not all(isinstance(provenance.get(key), str) and provenance[key]
                   for key in ("source", "source_url", "tier", "selection_rationale")):
            raise ValueError(f"{label} has incomplete provenance")
        if not isinstance(row.get("provenance_class"), str) or not row["provenance_class"]:
            raise ValueError(f"{label} has no provenance_class")


def load_redox_pack(path: str | Path | None = None) -> RedoxPack:
    """Load the default pack, or a supplied pack path, and verify its digest."""
    if path is None:
        text = files("openimcc.data").joinpath("packs", _PACK_NAME).read_text(
            encoding="utf-8"
        )
    else:
        text = Path(path).read_text(encoding="utf-8")
    content = json.loads(text)
    if not isinstance(content, dict):
        raise ValueError("redox pack root must be a JSON object")
    _validate_content(content)
    digest = canonical_redox_digest(content)
    if content.get("canonical_content_digest") != digest:
        raise ValueError("redox pack canonical content digest mismatch")
    return RedoxPack(content=_freeze_json(content), canonical_digest=digest)


def species_gibbs_j_mol(
    pack: RedoxPack, species_name: str, temperature_k: float
) -> float:
    """Evaluate one pack species through openimcc's existing G(T) evaluators."""
    temperature = _finite_number(temperature_k, "temperature_k")
    species = pack.content["thermo_species"]
    return _species_gibbs_j_mol(species, species_name, temperature, set())


def _species_gibbs_j_mol(
    species: Mapping[str, Mapping[str, Any]],
    species_name: str,
    temperature_k: float,
    visiting: set[str],
) -> float:
    if species_name not in species:
        raise ValueError(f"unknown thermochemical species {species_name!r}")
    if species_name in visiting:
        raise ValueError(f"cyclic thermochemical reference at {species_name!r}")
    record = species[species_name]
    low, high = map(float, record["T_domain_K"])
    if not low <= temperature_k <= high:
        raise ValueError(f"{species_name} is outside its declared temperature domain")
    evaluator = record["evaluator"]
    if evaluator in {"condensate_phi", "janaf_shomate"}:
        from openimcc.gas import _janaf_gibbs, _lamor_gibbs

        row = record["row"]
        return float(
            _lamor_gibbs(temperature_k, row)
            if evaluator == "condensate_phi"
            else _janaf_gibbs(temperature_k, row)
        )
    visiting.add(species_name)
    try:
        if evaluator == "linear_combination":
            return sum(
                float(term["coefficient"])
                * _species_gibbs_j_mol(
                    species, term["species"], temperature_k, visiting
                )
                for term in record["terms"]
            )
        if evaluator == "piecewise":
            transition = float(record["transition_temperature_K"])
            branch = "below" if temperature_k < transition else "at_or_above"
            selected = next(item for item in record["branches"] if item["branch"] == branch)
            return _species_gibbs_j_mol(
                species, selected["species"], temperature_k, visiting
            )
    finally:
        visiting.remove(species_name)
    raise ValueError(f"unsupported evaluator for {species_name!r}")


def reaction_gibbs_j_mol(
    pack: RedoxPack, row: Mapping[str, Any], temperature_k: float
) -> float:
    """Return standard reaction ΔG° from its balanced standard-state terms."""
    temperature = _finite_number(temperature_k, "temperature_k")
    low, high = map(float, row["T_domain_K"])
    if not low <= temperature <= high:
        raise ValueError("reaction is outside its declared temperature domain")
    return sum(
        float(term["coefficient"])
        * species_gibbs_j_mol(pack, str(term["species"]), temperature)
        for term in row["standard_gibbs_terms"]
    )


def ln_k_prime(
    pack: RedoxPack,
    row: Mapping[str, Any],
    temperature_k: float,
    ln_f_o2: float = 0.0,
) -> float:
    """Evaluate ln K and apply ext-v4's product-positive oxygen convention.

    Since the stored oxygen coefficient is positive for product O2, the
    fugacity term is ``-ν_O2 * ln(fO2)``. For FeO + 1/4 O2 = FeO1.5,
    ``ν_O2=-1/4`` and this gives the requested ``ln K + λ/4``.
    """
    from openimcc.gas import R_J_MOL_K

    oxygen = float(row["external_oxygen_stoich_product_positive"])
    return -reaction_gibbs_j_mol(pack, row, temperature_k) / (
        R_J_MOL_K * float(temperature_k)
    ) - oxygen * float(ln_f_o2)
