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
_SCHEMA = "openimcc-redox-pack.v2"
_CANDIDATE_VERSIONS = {
    "1.4.0-cr-candidate", "1.5.0-ti-candidate", "1.6.0-cr-ti-candidate"
}


class RedoxCompositionTermError(ValueError):
    """A candidate composition-conditioned row has an invalid tagged term."""

_BAND_BASIS = {
    "source_uncertainty", "measured_construction", "independent_measurement",
    "fit_only", "none",
}


class RedoxSchemaError(ValueError):
    """A redox pack row uses an incomplete or legacy band schema."""


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
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _validate_band_schema(record: Mapping[str, Any], label: str) -> None:
    legacy = {"band_kj_mol", "band_method", "uncertainty_band"} & record.keys()
    if legacy:
        raise RedoxSchemaError(f"{label} contains legacy band fields: {sorted(legacy)}")
    required = {"fit_residual", "accuracy_band", "band_basis", "independent_check"}
    missing = required - record.keys()
    if missing:
        raise RedoxSchemaError(f"{label} is missing band schema fields: {sorted(missing)}")
    for field in ("fit_residual", "accuracy_band"):
        value = record[field]
        if value is not None:
            if not isinstance(value, dict):
                raise RedoxSchemaError(f"{label}/{field} must be an object or null")
            shape = (
                {"value", "unit", "target_source", "comparison_method", "temperature_range", "held_out_or_in_sample"}
                if field == "fit_residual"
                else {"value", "unit", "temperature_points_or_range", "coverage"}
            )
            if shape - value.keys():
                raise RedoxSchemaError(f"{label}/{field} has an incomplete structure")
            if _finite_number(value.get("value"), f"{label}/{field}/value") < 0.0:
                raise RedoxSchemaError(f"{label}/{field}/value must be non-negative")
            if not isinstance(value.get("unit"), str) or not value["unit"]:
                raise RedoxSchemaError(f"{label}/{field}/unit must be a non-empty string")
    basis = record["band_basis"]
    if not isinstance(basis, dict) or basis.get("kind") not in _BAND_BASIS:
        raise RedoxSchemaError(f"{label}/band_basis/kind is not an allowed value")
    if "provenance" not in basis or not isinstance(basis.get("explanation"), str) or not basis["explanation"]:
        raise RedoxSchemaError(f"{label}/band_basis needs an explanation")
    check = record["independent_check"]
    check_fields = {
        "status", "equilibrium", "primary_source", "conditions", "measured_value",
        "predicted_value", "miss", "miss_to_band_ratio", "reason",
    }
    if not isinstance(check, dict) or check_fields - check.keys():
        raise RedoxSchemaError(f"{label}/independent_check has an incomplete structure")
    if check.get("status") not in {"tested", "lead", "not_found", "not_applicable"}:
        raise RedoxSchemaError(f"{label}/independent_check/status is invalid")
    if check.get("status") == "not_found" and not isinstance(check.get("reason"), str):
        raise RedoxSchemaError(f"{label}/independent_check/reason must explain not_found")


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
    candidate_pack = (
        content.get("candidate_only") is True
        and content.get("version") in _CANDIDATE_VERSIONS
    )
    if content.get("candidate_only") is True and not candidate_pack:
        raise ValueError("candidate-only redox features require a recognized candidate pack version")
    if candidate_pack:
        provenance = content.get("dataset_provenance")
        if (
            not isinstance(provenance, dict)
            or not isinstance(provenance.get("anchors"), list)
            or not provenance["anchors"]
            or not isinstance(provenance.get("held_out_independent"), list)
            or not provenance["held_out_independent"]
        ):
            raise ValueError("candidate pack needs separate anchor and held-out dataset provenance")
        for group in (provenance["anchors"], provenance["held_out_independent"]):
            for dataset in group:
                if not isinstance(dataset, dict) or not all(
                    isinstance(dataset.get(key), str) and dataset[key]
                    for key in ("citation", "role", "source_url")
                ):
                    raise ValueError("candidate dataset provenance needs citation, role, and source_url")
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
        "band": "declared per record by fit_residual or accuracy_band unit",
        "pressure": "bar relative to 1 bar standard state",
    }:
        raise ValueError("redox pack units do not match schema v2")
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
        parent_state = parent.get("parent_oxidation_state")
        if isinstance(parent_state, bool) or not isinstance(parent_state, int):
            raise ValueError(
                f"parent_registry[{index}] parent_oxidation_state must be an integer"
            )
        known_states = parent.get("known_oxidation_states")
        if (
            not isinstance(known_states, list)
            or not known_states
            or any(isinstance(state, bool) or not isinstance(state, int) for state in known_states)
            or len(set(known_states)) != len(known_states)
        ):
            raise ValueError(
                f"parent_registry[{index}] known_oxidation_states must be a non-empty list of unique integers"
            )
        atoms = parent.get("parent_formula_atoms")
        if not isinstance(atoms, dict) or not atoms or element not in atoms:
            raise ValueError(f"parent_registry[{index}] has invalid parent_formula_atoms")
        for atom, count in atoms.items():
            if not isinstance(atom, str) or _finite_number(
                count, f"parent_registry[{index}]/parent_formula_atoms/{atom}"
            ) <= 0.0:
                raise ValueError(f"parent_registry[{index}] has invalid atom count")
        oxygen_atoms = float(atoms.get("O", 0.0))
        element_atoms = float(atoms[element])
        if set(atoms) <= {element, "O"} and not math.isclose(
            parent_state * element_atoms,
            2.0 * oxygen_atoms,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"parent_registry[{index}] parent_oxidation_state does not balance its oxide formula"
            )
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
        _validate_band_schema(record, f"thermo_species[{name!r}]")
        if not record["formula_atoms"]:
            raise ValueError(f"thermo_species[{name!r}] has empty formula_atoms")
        for element, count in record["formula_atoms"].items():
            if not isinstance(element, str) or _finite_number(
                count, f"thermo_species[{name!r}]/formula_atoms/{element}"
            ) <= 0.0:
                raise ValueError(f"thermo_species[{name!r}] has invalid formula atoms")
        iron_atoms = float(record["formula_atoms"].get("Fe", 0.0))
        if iron_atoms > 0.0:
            valence_counts = []
            for field in ("ferric_fe_units", "ferrous_fe_units"):
                count = _finite_number(record.get(field), f"thermo_species[{name!r}]/{field}")
                if count < 0.0:
                    raise ValueError(f"thermo_species[{name!r}] has negative {field}")
                valence_counts.append(count)
            if sum(valence_counts) > iron_atoms + 1e-10:
                raise ValueError(
                    f"thermo_species[{name!r}] ferric and ferrous Fe exceed formula Fe"
                )
        elif "ferric_fe_units" in record or "ferrous_fe_units" in record:
            for field in ("ferric_fe_units", "ferrous_fe_units"):
                if _finite_number(record.get(field, 0.0), f"thermo_species[{name!r}]/{field}") != 0.0:
                    raise ValueError(
                        f"thermo_species[{name!r}] assigns iron valence without Fe"
                    )
        domain = record.get("T_domain_K")
        if not isinstance(domain, list) or len(domain) != 2:
            raise ValueError(f"thermo_species[{name!r}] needs a two-point T_domain_K")
        low, high = (_finite_number(x, f"thermo_species[{name!r}]/T_domain_K") for x in domain)
        if low <= 0.0 or high <= low:
            raise ValueError(f"thermo_species[{name!r}] has invalid temperature domain")
        if not isinstance(record.get("range_flags"), list):
            raise ValueError(f"thermo_species[{name!r}] needs range_flags")
        if not isinstance(record.get("provenance"), dict):
            raise ValueError(f"thermo_species[{name!r}] has no provenance")
        provenance = record["provenance"]
        if not all(isinstance(provenance.get(key), str) and provenance[key]
                   for key in ("source", "source_url", "tier", "selection_rationale")):
            raise ValueError(f"thermo_species[{name!r}] has incomplete provenance")
        evaluator = record.get("evaluator")
        if evaluator == "composition_conditioned_effective_k" and not candidate_pack:
            raise ValueError("composition-conditioned effective K is candidate-pack only")
        if evaluator not in {"condensate_phi", "janaf_shomate", "linear_combination", "piecewise", "composition_conditioned_effective_k"}:
            raise ValueError(f"thermo_species[{name!r}] has unknown evaluator")
        if evaluator not in {"condensate_phi", "janaf_shomate"} and "intervals" in record:
            raise ValueError(
                f"thermo_species[{name!r}] intervals require a direct fitted evaluator"
            )
        if evaluator in {"condensate_phi", "janaf_shomate"}:
            if "intervals" not in record:
                if not isinstance(record.get("row"), dict):
                    raise ValueError(f"thermo_species[{name!r}] has no evaluator row")
            else:
                intervals = record["intervals"]
                if "row" in record:
                    raise ValueError(
                        f"thermo_species[{name!r}] cannot have both row and intervals"
                    )
                if not isinstance(intervals, list) or len(intervals) < 2:
                    raise ValueError(
                        f"thermo_species[{name!r}] intervals must contain at least two rows"
                    )
                previous_high = low
                for interval_index, interval in enumerate(intervals):
                    interval_label = (
                        f"thermo_species[{name!r}]/intervals[{interval_index}]"
                    )
                    if not isinstance(interval, dict):
                        raise ValueError(f"{interval_label} must be an object")
                    interval_domain = interval.get("T_domain_K")
                    if not isinstance(interval_domain, list) or len(interval_domain) != 2:
                        raise ValueError(f"{interval_label} needs a two-point T_domain_K")
                    interval_low, interval_high = (
                        _finite_number(value, f"{interval_label}/T_domain_K")
                        for value in interval_domain
                    )
                    if interval_low <= 0.0 or interval_high <= interval_low:
                        raise ValueError(f"{interval_label} has invalid temperature domain")
                    if interval_low != previous_high:
                        raise ValueError(
                            f"{interval_label} does not continue the preceding interval"
                        )
                    if not isinstance(interval.get("row"), dict):
                        raise ValueError(f"{interval_label} has no evaluator row")
                    previous_high = interval_high
                if previous_high != high:
                    raise ValueError(
                        f"thermo_species[{name!r}] intervals do not cover T_domain_K"
                    )
        if evaluator == "linear_combination":
            if not isinstance(record.get("terms"), list) or not record["terms"]:
                raise ValueError(f"thermo_species[{name!r}] has no terms")
            for term in record["terms"]:
                if not isinstance(term, dict) or term.get("species") not in species:
                    raise ValueError(f"thermo_species[{name!r}] refers to an unknown term")
                _finite_number(
                    term.get("coefficient"),
                    f"thermo_species[{name!r}]/terms/{term['species']}",
                )
        if evaluator == "piecewise":
            if not isinstance(record.get("branches"), list) or len(record["branches"]) != 2:
                raise ValueError(f"thermo_species[{name!r}] needs two branches")
            if any(branch.get("species") not in species for branch in record["branches"]):
                raise ValueError(f"thermo_species[{name!r}] has an unknown branch")

    for name, record in species.items():
        if record["evaluator"] == "linear_combination":
            totals: dict[str, float] = {}
            for term in record["terms"]:
                coefficient = float(term["coefficient"])
                for element, count in species[term["species"]]["formula_atoms"].items():
                    totals[element] = totals.get(element, 0.0) + coefficient * float(count)
            declared = record["formula_atoms"]
            for element in set(totals) | set(declared):
                if not math.isclose(
                    totals.get(element, 0.0), float(declared.get(element, 0.0)),
                    rel_tol=0.0, abs_tol=1e-10,
                ):
                    raise ValueError(
                        f"thermo_species[{name!r}] combination formula_atoms "
                        f"do not match constituents for {element}"
                    )
            if "Fe" in record["formula_atoms"]:
                for field in ("ferric_fe_units", "ferrous_fe_units"):
                    expected = sum(
                        float(term["coefficient"])
                        * float(species[term["species"]].get(field, 0.0))
                        for term in record["terms"]
                    )
                    if not math.isclose(
                        float(record[field]), expected, rel_tol=0.0, abs_tol=1e-10
                    ):
                        raise ValueError(
                            f"thermo_species[{name!r}] {field} does not match constituents"
                        )

    rows = content["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list")
    for index, row in enumerate(rows):
        label = f"rows[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{label} must be an object")
        _validate_band_schema(row, label)
        if row.get("complex") not in species:
            raise ValueError(f"{label} names an unknown product species")
        product_species = species[str(row["complex"])]
        product_iron = float(product_species["formula_atoms"].get("Fe", 0.0))
        if product_iron > 0.0 and not math.isclose(
            float(product_species["ferric_fe_units"])
            + float(product_species["ferrous_fe_units"]),
            product_iron,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError(
                f"{label} Fe-bearing product does not assign every Fe atom a valence"
            )
        if not isinstance(row.get("nu"), dict):
            raise ValueError(f"{label} needs parent stoichiometry nu")
        composition_coefficients = row.get("composition_coefficients")
        composition_term = row.get("composition_term")
        if composition_coefficients is not None and composition_term is not None:
            raise RedoxCompositionTermError(f"{label} cannot declare both composition term forms")
        if composition_term is not None:
            if not candidate_pack:
                raise ValueError("composition_term is candidate-pack only")
            if not isinstance(composition_term, dict):
                raise RedoxCompositionTermError(f"{label} composition_term must be an object")
            kind = composition_term.get("kind")
            if kind == "optical_basicity":
                expected_fields = {"kind", "A", "B_lambda", "C_over_T", "lambda_range"}
                if set(composition_term) != expected_fields:
                    raise RedoxCompositionTermError(f"{label} optical_basicity composition_term has unknown or missing fields")
                for field in ("A", "B_lambda", "C_over_T"):
                    _finite_number(composition_term[field], f"{label}/composition_term/{field}")
                bounds = composition_term["lambda_range"]
                if not isinstance(bounds, list) or len(bounds) != 2:
                    raise RedoxCompositionTermError(f"{label} composition_term/lambda_range must contain two numbers")
                low_lambda, high_lambda = (
                    _finite_number(value, f"{label}/composition_term/lambda_range")
                    for value in bounds
                )
                if high_lambda <= low_lambda:
                    raise RedoxCompositionTermError(f"{label} composition_term/lambda_range is invalid")
            elif kind == "linear_cation_fractions":
                if set(composition_term) != {"kind", "coefficients"}:
                    raise RedoxCompositionTermError(f"{label} linear_cation_fractions composition_term has unknown or missing fields")
                coefficients = composition_term["coefficients"]
                if not isinstance(coefficients, dict) or not coefficients:
                    raise RedoxCompositionTermError(f"{label} composition_term/coefficients must be a non-empty object")
                for oxide, coefficient in coefficients.items():
                    if oxide not in {"NaO0.5", "MgO", "Fe2+O", "AlO1.5", "Fe3+O1.5", "KO0.5", "CaO", "TiO2"}:
                        raise RedoxCompositionTermError(f"{label} has an unsupported composition coefficient key {oxide!r}")
                    _finite_number(coefficient, f"{label}/composition_term/coefficients/{oxide}")
            else:
                raise RedoxCompositionTermError(f"{label} composition_term has unknown kind {kind!r}")
        if composition_coefficients is not None:
            if not candidate_pack:
                raise ValueError("composition_coefficients are candidate-pack only")
            if not isinstance(composition_coefficients, dict) or not composition_coefficients:
                raise ValueError(f"{label} composition_coefficients must be a non-empty object")
            supported_oxides = {
                "NaO0.5", "MgO", "Fe2+O", "AlO1.5", "Fe3+O1.5",
                "KO0.5", "CaO", "TiO2",
            }
            for oxide, coefficient in composition_coefficients.items():
                if oxide not in supported_oxides:
                    raise ValueError(f"{label} has an unsupported composition coefficient key {oxide!r}")
                _finite_number(coefficient, f"{label}/composition_coefficients/{oxide}")
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
        for field in ("phase_or_construction", "method", "uncertainty_note"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{label} needs {field}")
        if not isinstance(row.get("range_flags"), list) or not row["range_flags"]:
            raise ValueError(f"{label} needs explicit range flags")
        if not isinstance(row.get("provenance"), dict):
            raise ValueError(f"{label} needs provenance")
        composition_conditioned = composition_coefficients is not None or composition_term is not None
        if not isinstance(row.get("standard_gibbs_terms"), list) or (
            not row["standard_gibbs_terms"] and not composition_conditioned
        ):
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
        if composition_conditioned:
            if row["standard_gibbs_terms"]:
                raise ValueError(f"{label} empirical effective-K row cannot declare Gibbs terms")
            if product_species.get("evaluator") != "composition_conditioned_effective_k":
                raise ValueError(f"{label} needs a composition-conditioned product record")
        elif terms != expected:
            raise ValueError(f"{label} standard_gibbs_terms do not match reaction stoichiometry")
        if row.get("role") not in {"parent_reference", "complex"}:
            raise ValueError(f"{label} has unsupported row role")
        provenance = row["provenance"]
        if not all(isinstance(provenance.get(key), str) and provenance[key]
                   for key in ("source", "source_url", "tier", "selection_rationale")):
            raise ValueError(f"{label} has incomplete provenance")
        if not isinstance(row.get("provenance_class"), str) or not row["provenance_class"]:
            raise ValueError(f"{label} has no provenance_class")

    # Complex rows may consume one registered pseudo-complex. Expand it to
    # registered-parent extents before validating the row's parent basis.
    row_by_complex = {str(row["complex"]): row for row in rows}

    def parent_equivalents(species_name: str, amount: float) -> dict[str, float]:
        if species_name in parent_names:
            return {species_name: amount}
        nested = row_by_complex.get(species_name)
        if nested is None or nested.get("role") != "complex":
            return {}
        return {
            str(parent): amount * float(coefficient)
            for parent, coefficient in nested["nu"].items()
        }

    for index, row in enumerate(rows):
        expected: dict[str, float] = {}
        for species_name, coefficient in row["reactants"].items():
            for parent, amount in parent_equivalents(
                str(species_name), float(coefficient)
            ).items():
                expected[parent] = expected.get(parent, 0.0) + amount
        for parent in parent_names:
            if not math.isclose(
                float(row["nu"].get(parent, 0.0)),
                expected.get(parent, 0.0),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"rows[{index}] nu does not match parent-equivalent reactant stoichiometry"
                )
        if float(row["external_oxygen_stoich_product_positive"]) == 0.0:
            valence_delta = {"ferric_fe_units": 0.0, "ferrous_fe_units": 0.0}
            for species_name, coefficient in row["products"].items():
                for field in valence_delta:
                    valence_delta[field] += float(coefficient) * float(
                        species[species_name].get(field, 0.0)
                    )
            for species_name, coefficient in row["reactants"].items():
                for field in valence_delta:
                    valence_delta[field] -= float(coefficient) * float(
                        species[species_name].get(field, 0.0)
                    )
            if any(abs(value) > 1e-10 for value in valence_delta.values()):
                raise ValueError(f"rows[{index}] changes Fe valence without external O2")


def load_redox_pack(path: str | Path | None = None) -> RedoxPack:
    """Load the default pack, or a supplied pack path, and verify its digest."""
    if path is None:
        pack_dir = files("openimcc.data").joinpath("packs")
        pack_bytes = pack_dir.joinpath(_PACK_NAME).read_bytes()
        manifest = json.loads(pack_dir.joinpath("MANIFEST.json").read_bytes())
        manifest_entries = (
            manifest.get("packs")
            if isinstance(manifest, dict) and isinstance(manifest.get("packs"), list)
            else []
        )
        entries = [
            entry for entry in manifest_entries
            if isinstance(entry, dict) and entry.get("file") == _PACK_NAME
        ]
        if len(entries) != 1 or not isinstance(entries[0].get("sha256"), str):
            raise ValueError("default redox pack has no unique manifest SHA-256 pin")
        if hashlib.sha256(pack_bytes).hexdigest() != entries[0]["sha256"]:
            raise ValueError("default redox pack SHA-256 does not match MANIFEST.json")
        text = pack_bytes.decode("utf-8")
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
    pack: RedoxPack,
    species_name: str,
    temperature_k: float,
    *,
    allow_extrapolation: bool = False,
) -> float:
    """Evaluate one pack species through openimcc's existing G(T) evaluators.

    ``allow_extrapolation`` permits analytic condensate expressions beyond
    their declared evidence intervals. JANAF Shomate fitted-table bounds
    remain enforced.
    """
    temperature = _finite_number(temperature_k, "temperature_k")
    species = pack.content["thermo_species"]
    return _species_gibbs_j_mol(
        species, species_name, temperature, set(), allow_extrapolation
    )


def _species_gibbs_j_mol(
    species: Mapping[str, Mapping[str, Any]],
    species_name: str,
    temperature_k: float,
    visiting: set[str],
    allow_extrapolation: bool = False,
) -> float:
    if species_name not in species:
        raise ValueError(f"unknown thermochemical species {species_name!r}")
    if species_name in visiting:
        raise ValueError(f"cyclic thermochemical reference at {species_name!r}")
    record = species[species_name]
    low, high = map(float, record["T_domain_K"])
    evaluator = record["evaluator"]
    # Redox row T_domain_K values are evidence intervals. Analytic condensate
    # expressions and their composites can be evaluated beyond those intervals,
    # while a JANAF Shomate species is bounded by its fitted table interval.
    if not low <= temperature_k <= high and (
        not allow_extrapolation or evaluator == "janaf_shomate"
    ):
        bound = (
            "fitted table domain"
            if evaluator == "janaf_shomate"
            else "declared species domain"
        )
        raise ValueError(
            f"species {species_name!r} G(T) is outside its {bound} "
            f"[{low:g}, {high:g}] K"
        )
    if evaluator in {"condensate_phi", "janaf_shomate"}:
        from openimcc.gas import _janaf_gibbs, _lamor_gibbs

        intervals = record.get("intervals")
        if intervals is None:
            row = record["row"]
        else:
            # Select the highest lower bound at a shared endpoint, matching
            # the gas-table interval convention; the last interval includes
            # its upper endpoint. This leaves legacy single-row records alone.
            first_low = float(intervals[0]["T_domain_K"][0])
            last_high = float(intervals[-1]["T_domain_K"][1])
            if temperature_k < first_low:
                selected = intervals[0]
            elif temperature_k > last_high:
                selected = intervals[-1]
            else:
                selected = intervals[-1]
                for interval in intervals:
                    interval_low, interval_high = map(
                        float, interval["T_domain_K"]
                    )
                    if interval_low <= temperature_k <= interval_high:
                        selected = interval
                        if temperature_k < interval_high:
                            break
            row = selected["row"]
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
                    species,
                    term["species"],
                    temperature_k,
                    visiting,
                    allow_extrapolation,
                )
                for term in record["terms"]
            )
        if evaluator == "piecewise":
            transition = float(record["transition_temperature_K"])
            branch = "below" if temperature_k < transition else "at_or_above"
            selected = next(item for item in record["branches"] if item["branch"] == branch)
            return _species_gibbs_j_mol(
                species,
                selected["species"],
                temperature_k,
                visiting,
                allow_extrapolation,
            )
    finally:
        visiting.remove(species_name)
    raise ValueError(f"unsupported evaluator for {species_name!r}")


def reaction_gibbs_j_mol(
    pack: RedoxPack,
    row: Mapping[str, Any],
    temperature_k: float,
    *,
    allow_extrapolation: bool = False,
) -> float:
    """Return standard reaction ΔG° from its balanced standard-state terms.

    ``allow_extrapolation`` bypasses the reaction's evidence interval while
    keeping species evaluator bounds in force.
    """
    if row.get("composition_coefficients") is not None or row.get("composition_term") is not None:
        raise ValueError("composition-conditioned effective K has no Gibbs function")
    temperature = _finite_number(temperature_k, "temperature_k")
    low, high = map(float, row["T_domain_K"])
    if not low <= temperature <= high and not allow_extrapolation:
        raise ValueError(
            f"reaction evidence domain [{low:g}, {high:g}] K does not include "
            f"{temperature:g} K"
        )
    return sum(
        float(term["coefficient"])
        * species_gibbs_j_mol(
            pack,
            str(term["species"]),
            temperature,
            allow_extrapolation=allow_extrapolation,
        )
        for term in row["standard_gibbs_terms"]
    )


def ln_k_prime(
    pack: RedoxPack,
    row: Mapping[str, Any],
    temperature_k: float,
    ln_f_o2: float = 0.0,
    *,
    allow_extrapolation: bool = False,
) -> float:
    """Evaluate ln K and apply ext-v4's product-positive oxygen convention.

    Since the stored oxygen coefficient is positive for product O2, the
    fugacity term is ``-ν_O2 * ln(fO2)``. For FeO + 1/4 O2 = FeO1.5,
    ``ν_O2=-1/4`` and this gives the requested ``ln K + λ/4``.

    ``allow_extrapolation`` bypasses the reaction evidence interval while
    keeping species evaluator bounds in force.

    Complex equilibria use registered-parent activities in the solver. When
    a balanced reaction consumes another complex, this evaluates its
    equivalent parent-basis Gibbs function; ``reaction_gibbs_j_mol`` continues
    to return the Gibbs energy of the displayed reaction itself.
    """
    if row.get("composition_coefficients") is not None or row.get("composition_term") is not None:
        raise ValueError("composition-conditioned effective K requires a melt composition")
    from openimcc.gas import R_J_MOL_K

    complex_products = {
        str(candidate["complex"])
        for candidate in pack.rows
        if candidate.get("role") == "complex"
    }
    consumes_complex = any(
        str(species) in complex_products for species in row["reactants"]
    )
    if row.get("role") == "complex" and consumes_complex:
        parent_oxygen = {
            str(parent["parent_oxide"]): float(
                parent["parent_formula_atoms"].get("O", 0.0)
            )
            for parent in pack.parents
        }
        product_oxygen = float(
            pack.content["thermo_species"][str(row["complex"])]
            ["formula_atoms"].get("O", 0.0)
        )
        consumed_oxygen = sum(
            float(amount) * parent_oxygen[str(parent)]
            for parent, amount in row["nu"].items()
        )
        # Premise: solver mass-action rows use registered-parent activities.
        # Algebra: ΔG on that basis is G(product) - ν·G(parent) - (ΔO/2)G(O2).
        # Units are J/mol. Sanity: consuming two FeO1.5 units adds one O
        # beyond three FeO parents, so the shift is lambda/2.
        lambda_shift = 0.5 * (product_oxygen - consumed_oxygen)
        reaction_gibbs = (
            species_gibbs_j_mol(
                pack,
                str(row["complex"]),
                float(temperature_k),
                allow_extrapolation=allow_extrapolation,
            )
            - math.fsum(
                float(amount)
                * species_gibbs_j_mol(
                    pack,
                    str(parent),
                    float(temperature_k),
                    allow_extrapolation=allow_extrapolation,
                )
                for parent, amount in row["nu"].items()
            )
            - lambda_shift
            * species_gibbs_j_mol(
                pack,
                "O2(g)",
                float(temperature_k),
                allow_extrapolation=allow_extrapolation,
            )
        )
    else:
        lambda_shift = -float(row["external_oxygen_stoich_product_positive"])
        reaction_gibbs = reaction_gibbs_j_mol(
            pack,
            row,
            temperature_k,
            allow_extrapolation=allow_extrapolation,
        )
    return -reaction_gibbs / (R_J_MOL_K * float(temperature_k)) + (
        lambda_shift * float(ln_f_o2)
    )
