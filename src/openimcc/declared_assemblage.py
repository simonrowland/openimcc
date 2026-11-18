"""Chemical potentials and vapour over a caller-stated crystal assemblage."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from openimcc._constants import R_J_MOL_K
from openimcc import diagnostics
from openimcc.gas_data import (
    _OXIDE_PROVENANCE_AUTHORITY,
    _PROVENANCE_AUTHORITY_RANK,
)
from openimcc.receipt import CalculationReceipt, build_calculation_receipt
from openimcc.janaf_crystals import (
    JanafCrystalTemperatureError,
    evaluate_janaf_crystal,
)

if TYPE_CHECKING:
    from openimcc.gas import ImccGasDatapack, ImccGasResult


_PURE_OXIDE_RESOURCE = "data/diagnostics/janaf-pure-oxide-crystals.json"
_CRYSTAL_RESOURCE = "data/diagnostics/crystal-phases.json"
_BARIN_RESOURCE = "data/diagnostics/barin-aluminate-crystals.json"
_PRESSURE_BAR_STANDARD = 1.0
_PHASE_RESIDUAL_TOLERANCE_J_MOL = 1.0
_REDOX_PHASE_IDENTITIES = frozenset(
    ("fe2o3", "fe3o4", "hematite", "magnetite", "ferric")
)


def _is_redox_phase_identity(identity: str) -> bool:
    normalized = "".join(
        character for character in identity.casefold() if character.isalnum()
    )
    return normalized in _REDOX_PHASE_IDENTITIES


@dataclass(frozen=True)
class PhaseEquilibriumLabels:
    identity: Mapping[str, str]
    coverage: Mapping[str, str]
    validity: Literal["IN", "OUT_OF_DOMAIN", "DEPENDS"]
    out_of_domain_axes: tuple[str, ...]
    evidence_class: str
    flags: tuple[str, ...] = ()
    notices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.validity not in ("IN", "OUT_OF_DOMAIN", "DEPENDS"):
            raise ValueError(
                "validity must be one of 'IN', 'OUT_OF_DOMAIN', or 'DEPENDS'"
            )
        object.__setattr__(self, "identity", MappingProxyType(dict(self.identity)))
        object.__setattr__(self, "coverage", MappingProxyType(dict(self.coverage)))


@dataclass(frozen=True)
class EquilibriumRefusal:
    code: str
    detail: str
    axes: tuple[str, ...] = ()
    residuals: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "residuals", MappingProxyType(dict(self.residuals)))


@dataclass(frozen=True)
class DeclaredAssemblageResult:
    temperature_K: float
    declared_phases: tuple[str, ...]
    phase_standard_gibbs_J_mol: Mapping[str, float]
    phase_affinities_ln: Mapping[str, float]
    oxide_chemical_potentials_J_mol: Mapping[str, float]
    oxide_activities: Mapping[str, float]
    non_identifiable_components: tuple[str, ...]
    gas: ImccGasResult | None
    gas_depends: Mapping[str, str]
    fO2: float | None
    oxygen_balance_pO2_bar: float | None
    oxygen_balance_diagnostics: Mapping[str, Any]
    standard_pressure_bar: float
    labels: PhaseEquilibriumLabels
    receipt: CalculationReceipt
    refusal: EquilibriumRefusal | None

    def __post_init__(self) -> None:
        for name in (
            "phase_standard_gibbs_J_mol", "phase_affinities_ln",
            "oxide_chemical_potentials_J_mol", "oxide_activities", "gas_depends",
            "oxygen_balance_diagnostics",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def _resource_bytes(relative_path: str) -> bytes:
    return resources.files("openimcc").joinpath(relative_path).read_bytes()


def _load_json(relative_path: str) -> tuple[dict[str, Any], str]:
    raw = _resource_bytes(relative_path)
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _phase_authority(phase: Mapping[str, Any]) -> str:
    declared = phase.get("source_authority")
    if isinstance(declared, str):
        return declared
    source = phase.get("source", {})
    if source.get("authors") in {"Berman", "Robie and Hemingway"}:
        return "published_fitted"
    return "external_datapack"


def _least_authority(authorities: Sequence[str]) -> str:
    if not authorities or any(
        authority not in _PROVENANCE_AUTHORITY_RANK for authority in authorities
    ):
        return "unresolved"
    return min(
        authorities,
        key=_PROVENANCE_AUTHORITY_RANK.__getitem__,
    )


def _make_receipt(
    *,
    phases: tuple[str, ...],
    temperature_K: float,
    fO2: float | None,
    gas_species: tuple[str, ...] | None,
    parent_oxides: tuple[str, ...] | None,
    include_ions: bool,
    oxygen_balance: bool,
    allow_extrapolation: bool,
    crystal_digest: str,
    pure_digest: str,
    gas_pack: Any | None,
    crystal_function_continued: tuple[str, ...] = (),
) -> CalculationReceipt:
    _barin_data, barin_digest = _load_json(_BARIN_RESOURCE)
    return build_calculation_receipt(
        request={
            "declared_phases": phases,
            "temperature_K": temperature_K,
            "fO2": fO2,
            "gas_species": gas_species,
            "parent_oxides": parent_oxides,
            "include_ions": include_ions,
            "oxygen_balance": oxygen_balance,
            "allow_extrapolation": allow_extrapolation,
            "standard_pressure_bar": _PRESSURE_BAR_STANDARD,
        },
        data_digests={
            "crystal_phases": crystal_digest,
            "barin_aluminate_crystals": barin_digest,
            "janaf_pure_oxide_crystals": pure_digest,
            "gas": getattr(gas_pack, "gas_table_digest", None),
            "condensate": getattr(gas_pack, "condensate_table_digest", None),
        },
        solver={
            "method": "float64_svd",
            "identifiability": "componentwise_nullspace_projection",
            "phase_residual_tolerance_J_mol": _PHASE_RESIDUAL_TOLERANCE_J_MOL,
            "standard_pressure_bar": _PRESSURE_BAR_STANDARD,
            "crystal_function_continued": crystal_function_continued,
            "pure_crystal_gibbs_conversion": (
                "JANAF apparent G = Hf(298.15 K) + [H(T)-H(298.15 K)] - T*S(T); "
                "kJ/mol enthalpies and J/(mol K) entropy converted to J/mol"
            ),
        },
    )


def _empty_result(
    *,
    phases: tuple[str, ...],
    temperature_K: float,
    fO2: float | None,
    gas_species: tuple[str, ...] | None,
    parent_oxides: tuple[str, ...] | None,
    include_ions: bool,
    oxygen_balance: bool,
    allow_extrapolation: bool,
    crystal_digest: str,
    pure_digest: str,
    gas_pack: Any | None,
    validity: Literal["IN", "OUT_OF_DOMAIN", "DEPENDS"],
    evidence_class: str,
    flags: tuple[str, ...],
    notices: tuple[str, ...],
    refusal: EquilibriumRefusal | None,
    axes: tuple[str, ...] = (),
    non_identifiable_components: tuple[str, ...] = (),
    phase_standard_gibbs_J_mol: Mapping[str, float] | None = None,
    phase_affinities_ln: Mapping[str, float] | None = None,
    gas_depends: Mapping[str, str] | None = None,
    crystal_function_continued: tuple[str, ...] = (),
) -> DeclaredAssemblageResult:
    labels = PhaseEquilibriumLabels(
        identity={
            "crystal_phases": "openimcc-crystal-phases-v2",
            "barin_aluminate_crystals": "openimcc-barin-aluminate-crystals-v3",
            "janaf_pure_oxide_crystals": "openimcc-janaf-apparent-crystals-v2",
            "gas_table_digest": getattr(gas_pack, "gas_table_digest", "unresolved"),
            "condensate_table_digest": getattr(
                gas_pack, "condensate_table_digest", "unresolved"
            ),
            "standard_pressure_bar": "1.0",
        },
        coverage={}, validity=validity, out_of_domain_axes=axes,
        evidence_class=evidence_class, flags=flags,
        notices=tuple(dict.fromkeys((*notices, *crystal_function_continued))),
    )
    receipt = _make_receipt(
        phases=phases, temperature_K=temperature_K, fO2=fO2,
        gas_species=gas_species, parent_oxides=parent_oxides,
        include_ions=include_ions, oxygen_balance=oxygen_balance,
        allow_extrapolation=allow_extrapolation, crystal_digest=crystal_digest,
        pure_digest=pure_digest, gas_pack=gas_pack,
        crystal_function_continued=crystal_function_continued,
    )
    return DeclaredAssemblageResult(
        temperature_K=temperature_K, declared_phases=phases,
        phase_standard_gibbs_J_mol=phase_standard_gibbs_J_mol or {},
        phase_affinities_ln=phase_affinities_ln or {},
        oxide_chemical_potentials_J_mol={}, oxide_activities={},
        non_identifiable_components=non_identifiable_components,
        gas=None, gas_depends=gas_depends or {}, fO2=fO2,
        oxygen_balance_pO2_bar=None, oxygen_balance_diagnostics={},
        standard_pressure_bar=_PRESSURE_BAR_STANDARD, labels=labels,
        receipt=receipt, refusal=refusal,
    )


def _gas_dependencies(
    gas_module: Any,
    phase_components: set[str],
    available_activities: Mapping[str, float],
    requested_species: tuple[str, ...] | None,
    include_ions: bool,
    gas_pack: Any,
) -> tuple[tuple[str, ...] | None, dict[str, str]]:
    ion_to_neutral = {
        ion: neutral for neutral, ion in gas_module._GAS_IONIZATION_PAIRS
    }
    if requested_species is None:
        candidates, _omitted = gas_module.default_gas_channels(
            tuple(sorted(phase_components)), gas_pack
        )
        species = [name for name, _reaction in candidates]
        if include_ions:
            species.extend(
                ion for neutral, ion in gas_module._GAS_IONIZATION_PAIRS
                if neutral in species
            )
    else:
        species = list(requested_species)

    depends: dict[str, str] = {}
    eligible: list[str] = []
    for name in species:
        neutral = ion_to_neutral.get(name, name)
        if name in ion_to_neutral and not include_ions:
            raise ValueError(f"gas species {name!r} requires include_ions=True")
        reaction = gas_module._GAS_REACTIONS.get(neutral)
        if reaction is None:
            eligible.append(name)
            continue
        parent = reaction[0]
        missing = []
        if parent and parent not in available_activities:
            missing.append(parent)
        side_parent = gas_module._side_parent(neutral)
        if side_parent and side_parent not in available_activities:
            missing.append(side_parent)
        if missing:
            depends[name] = "activity_not_identifiable: " + ", ".join(missing)
        else:
            eligible.append(name)
    return (tuple(eligible) if requested_species is not None else None), depends


def _evaluate_gas_attachment(
    activities: Mapping[str, float],
    temperature_K: float,
    fO2: float,
    gas_pack: Any,
    *,
    dependency_components: Sequence[str],
    parent_oxides: Sequence[str],
    gas_species: tuple[str, ...] | None = None,
    include_ions: bool = False,
    allow_extrapolation: bool = False,
) -> tuple[Any | None, dict[str, str]]:
    """Evaluate only gas channels supported by the supplied activities."""
    gas_module = __import__("openimcc.gas", fromlist=["gas"])
    explicit_species, depends = _gas_dependencies(
        gas_module,
        set(dependency_components),
        activities,
        gas_species,
        include_ions,
        gas_pack,
    )
    if gas_species is not None and not explicit_species:
        return None, depends
    result = gas_module.evaluate_gas(
        activities,
        temperature_K,
        fO2,
        gas_pack,
        parent_oxides=tuple(parent_oxides),
        allow_extrapolation=allow_extrapolation,
        gas_species=explicit_species,
        include_ions=include_ions,
    )
    return result, depends


def evaluate_declared_assemblage(
    phases: Sequence[str],
    T_K: float,
    *,
    fO2: float | None = None,
    gas_species: Sequence[str] | str | None = None,
    include_ions: bool = False,
    parent_oxides: Sequence[str] | None = None,
    condensate: str | Path | None = None,
    gas_pack: ImccGasDatapack | None = None,
    oxygen_balance: bool = False,
    allow_extrapolation: bool = False,
) -> DeclaredAssemblageResult:
    """Evaluate chemical potentials and optional vapour for stated solids.

    No bulk minimization or phase-stability selection is performed. Every
    named crystal is treated as present, and only oxide potentials fixed by
    the resulting equations are reported.
    """
    if isinstance(phases, str) or not isinstance(phases, Sequence) or not phases:
        raise ValueError("phases must be a non-empty sequence of phase identifiers")
    declared = tuple(str(phase) for phase in phases)
    if len(set(declared)) != len(declared):
        raise ValueError("phases must not contain duplicates")
    temperature = float(T_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("T_K must be finite and positive")
    if gas_species is None:
        requested_gas = None
    elif isinstance(gas_species, str):
        requested_gas = (gas_species,)
    else:
        requested_gas = tuple(str(name) for name in gas_species)
    requested_oxides = (
        None if parent_oxides is None
        else tuple(str(name) for name in parent_oxides)
    )
    if requested_gas is not None and len(set(requested_gas)) != len(requested_gas):
        raise ValueError("gas_species must not contain duplicates")
    if requested_oxides is not None and len(set(requested_oxides)) != len(requested_oxides):
        raise ValueError("parent_oxides must not contain duplicates")
    if oxygen_balance and fO2 is not None:
        raise ValueError("fO2 cannot be supplied when oxygen_balance=True")
    if oxygen_balance and requested_gas is not None:
        raise ValueError("gas_species cannot be restricted for oxygen_balance")

    pure_data, pure_digest = _load_json(_PURE_OXIDE_RESOURCE)
    crystal_data, crystal_digest = _load_json(_CRYSTAL_RESOURCE)
    barin_data, _barin_digest = _load_json(_BARIN_RESOURCE)
    pure_by_id = {row["id"]: row for row in pure_data["phases"]}
    crystal_by_id = {
        row["id"]: row for row in (*crystal_data["phases"], *barin_data["phases"])
    }
    redox_phases = tuple(
        phase for phase in declared
        if _is_redox_phase_identity(phase)
    )
    if redox_phases:
        refusal = EquilibriumRefusal(
            "redox_sensitive_phase",
            "Fe3+-bearing phases are unsupported: " + ", ".join(redox_phases),
            tuple(f"crystal.{phase}.redox" for phase in redox_phases),
        )
        return _empty_result(
            phases=declared, temperature_K=temperature, fO2=fO2,
            gas_species=requested_gas, parent_oxides=requested_oxides,
            include_ions=include_ions, oxygen_balance=oxygen_balance,
            allow_extrapolation=allow_extrapolation, crystal_digest=crystal_digest,
            pure_digest=pure_digest, gas_pack=gas_pack, validity="DEPENDS",
            evidence_class="unresolved", flags=("redox_phase_refused",),
            notices=("declared_assemblage_not_minimized",), refusal=refusal,
        )

    unknown = tuple(
        phase for phase in declared
        if phase not in pure_by_id and phase not in crystal_by_id
    )
    if unknown:
        refusal = EquilibriumRefusal(
            "missing_phase_data", f"No crystal Gibbs function is available for {unknown}.",
            tuple(f"crystal.{phase}" for phase in unknown),
        )
        return _empty_result(
            phases=declared, temperature_K=temperature, fO2=fO2,
            gas_species=requested_gas, parent_oxides=requested_oxides,
            include_ions=include_ions, oxygen_balance=oxygen_balance,
            allow_extrapolation=allow_extrapolation, crystal_digest=crystal_digest,
            pure_digest=pure_digest, gas_pack=gas_pack, validity="DEPENDS",
            evidence_class="unresolved", flags=(),
            notices=("declared_assemblage_not_minimized",), refusal=refusal,
            non_identifiable_components=(
                tuple(sorted(set(requested_oxides))) if requested_oxides else ()
            ),
            gas_depends=(
                {name: "missing_phase_data: " + ", ".join(unknown)
                 for name in requested_gas}
                if requested_gas else {}
            ),
        )

    gas_module = __import__("openimcc.gas", fromlist=["gas"])
    if condensate is not None and gas_pack is not None:
        raise ValueError("condensate and gas_pack cannot both be supplied")
    _default_pack, reference_pack = diagnostics._crystal_thermo_context(condensate)
    active_pack = gas_pack if gas_pack is not None else _default_pack

    phase_ranges: dict[str, str] = {}
    continued_notices: list[str] = []
    pure_crystal_values = {}
    phase_authorities: list[str] = []
    out_of_domain_axes: list[str] = []
    for phase_id in declared:
        row = pure_by_id.get(phase_id)
        if row is None:
            row = crystal_by_id[phase_id]
            lo, hi = diagnostics._crystal_fusion_domain(row)
            if not lo <= temperature <= hi:
                out_of_domain_axes.append(f"crystal.{phase_id}.temperature_K")
                continue
            phase_ranges[phase_id] = f"{lo:g}-{hi:g} K"
            phase_authorities.append(_phase_authority(row))
            continuation = diagnostics._crystal_continuation_range(row)
            if continuation is not None and temperature > continuation[0]:
                continued_notices.append(
                    f"crystal_function_continued:{phase_id}:"
                    f"{continuation[0]:g}-{continuation[1]:g}K"
                )
        else:
            try:
                crystal_value = evaluate_janaf_crystal(phase_id, temperature)
            except JanafCrystalTemperatureError as exc:
                lo, hi = exc.domain_K
                out_of_domain_axes.append(f"crystal.{phase_id}.temperature_K")
            else:
                lo, hi = crystal_value.domain_K
                pure_crystal_values[phase_id] = crystal_value
                phase_authorities.append(
                    str(crystal_value.provenance["source_authority"])
                )
            phase_ranges[phase_id] = f"{lo:g}-{hi:g} K"

    if out_of_domain_axes:
        unavailable_components = tuple(sorted({
            oxide for phase_id in declared
            for oxide in (pure_by_id.get(phase_id) or crystal_by_id[phase_id])[
                "oxide_stoichiometry"
            ]
        }))
        refusal = EquilibriumRefusal(
            "phase_temperature_out_of_range",
            "A declared crystal phase has no source Gibbs value at the requested temperature.",
            tuple(out_of_domain_axes),
        )
        return _empty_result(
            phases=declared, temperature_K=temperature, fO2=fO2,
            gas_species=requested_gas, parent_oxides=requested_oxides,
            include_ions=include_ions, oxygen_balance=oxygen_balance,
            allow_extrapolation=allow_extrapolation, crystal_digest=crystal_digest,
            pure_digest=pure_digest, gas_pack=active_pack, validity="OUT_OF_DOMAIN",
            evidence_class=_least_authority(phase_authorities),
            flags=("crystal_phase_out_of_range",),
            notices=("declared_assemblage_not_minimized",), refusal=refusal,
            axes=tuple(out_of_domain_axes),
            non_identifiable_components=unavailable_components,
            crystal_function_continued=tuple(continued_notices),
        )

    phase_components = sorted({
        oxide for phase_id in declared
        for oxide in (pure_by_id.get(phase_id) or crystal_by_id[phase_id])[
            "oxide_stoichiometry"
        ]
    })
    component_column = {oxide: i for i, oxide in enumerate(phase_components)}
    matrix = np.zeros((len(declared), len(phase_components)), dtype=np.float64)
    crystal_gibbs: dict[str, float] = {}
    liquid_gibbs: dict[str, float] = {}
    coverage = dict(phase_ranges)
    active_authorities = []
    try:
        for oxide in phase_components:
            liquid = gas_module.species_thermo(oxide, "l", temperature, active_pack)
            liquid_gibbs[oxide] = liquid.G_J_mol
            authority = _OXIDE_PROVENANCE_AUTHORITY.get(oxide, "unresolved")
            if active_pack.condensate_table_digest != reference_pack.condensate_table_digest:
                authority = "external_datapack"
            active_authorities.append(authority)
            coverage[f"parent_liquid.{oxide}"] = (
                f"{liquid.source_table_id or liquid.source_row_id}; "
                f"{liquid.T_min:g}-{liquid.T_max:g} K"
            )
        reference_liquid_gibbs = (
            liquid_gibbs
            if active_pack.condensate_table_digest == reference_pack.condensate_table_digest
            else {
                oxide: gas_module.species_thermo(
                    oxide, "l", temperature, reference_pack,
                ).G_J_mol
                for oxide in phase_components
            }
        )

        for phase_index, phase_id in enumerate(declared):
            row = pure_by_id.get(phase_id)
            if row is not None:
                g_crystal = pure_crystal_values[phase_id].G_J_mol
                if phase_id == "K2O":
                    active_authorities.append("estimated_systematic")
            else:
                row = crystal_by_id[phase_id]
                fusion_kj = diagnostics._crystal_fusion_gibbs_kj_mol(
                    row, temperature,
                )
                if (
                    active_pack.condensate_table_digest
                    != reference_pack.condensate_table_digest
                ):
                    fusion_kj += math.fsum(
                        float(nu) * (
                            liquid_gibbs[oxide] - reference_liquid_gibbs[oxide]
                        ) / 1000.0
                        for oxide, nu in row["oxide_stoichiometry"].items()
                    )
                # Premise: dissolution is crystal -> Σν parent(l), so
                # ΔG_fus° = ΣνG_l° − G_cr°. Rearranging gives G_cr° in
                # J/mol formula; ν is mol oxide/mol formula and the stored
                # fusion nodes are kJ/mol formula, hence the factor 1000.
                # The same conversion gives a=exp((G_cr°−G_l°)/(RT)); for a
                # pure JANAF pair below Tm, ΔG_fus>0 and therefore a<1.
                g_crystal = (
                    math.fsum(
                        float(nu) * liquid_gibbs[oxide]
                        for oxide, nu in row["oxide_stoichiometry"].items()
                    ) - fusion_kj * 1000.0
                )
            crystal_gibbs[phase_id] = g_crystal
            stoichiometry = row["oxide_stoichiometry"]
            for oxide, nu in stoichiometry.items():
                matrix[phase_index, component_column[oxide]] = float(nu)
    except gas_module.ImccGasTemperatureOutsideDomainError as exc:
        axes = ("parent_liquid.temperature_K",)
        refusal = EquilibriumRefusal(
            "phase_temperature_out_of_range", str(exc), axes,
        )
        return _empty_result(
            phases=declared, temperature_K=temperature, fO2=fO2,
            gas_species=requested_gas, parent_oxides=requested_oxides,
            include_ions=include_ions, oxygen_balance=oxygen_balance,
            allow_extrapolation=allow_extrapolation, crystal_digest=crystal_digest,
            pure_digest=pure_digest, gas_pack=active_pack, validity="OUT_OF_DOMAIN",
            evidence_class=_least_authority((*phase_authorities, *active_authorities)),
            flags=("parent_liquid_standard_out_of_range",),
            notices=("declared_assemblage_not_minimized",), refusal=refusal, axes=axes,
        )

    right_hand_side = np.asarray([crystal_gibbs[phase] for phase in declared])
    # SVD gives the least-squares minimum-norm solution of A μ = G while
    # retaining an explicit right-nullspace for component identifiability.
    # Singular values below eps*max(shape)*s_max are numerical zero at float64
    # precision; this rank threshold is independent of the later component test.
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        matrix, full_matrices=True
    )
    rank_threshold = (
        np.finfo(np.float64).eps * max(matrix.shape)
        * (float(singular_values[0]) if singular_values.size else 1.0)
    )
    rank = int(np.count_nonzero(singular_values > rank_threshold))
    nullspace = right_vectors[rank:, :]
    coefficients = (
        (left_vectors[:, :rank].T @ right_hand_side) / singular_values[:rank]
        if rank else np.zeros(0, dtype=np.float64)
    )
    minimum_norm_mu = (
        right_vectors[:rank, :].T @ coefficients
        if rank else np.zeros(len(phase_components), dtype=np.float64)
    )
    # Residual is predicted phase G minus supplied crystal G, in J/mol formula.
    residual_vector = matrix @ minimum_norm_mu - right_hand_side
    residuals = {
        phase: float(residual)
        for phase, residual in zip(declared, residual_vector)
    }
    if any(abs(value) > _PHASE_RESIDUAL_TOLERANCE_J_MOL for value in residuals.values()):
        refusal = EquilibriumRefusal(
            "inconsistent_phase_set",
            "No oxide chemical-potential vector satisfies every stated phase equation.",
            tuple(f"phase.{phase}" for phase in declared), residuals,
        )
        return _empty_result(
            phases=declared, temperature_K=temperature, fO2=fO2,
            gas_species=requested_gas, parent_oxides=requested_oxides,
            include_ions=include_ions, oxygen_balance=oxygen_balance,
            allow_extrapolation=allow_extrapolation, crystal_digest=crystal_digest,
            pure_digest=pure_digest, gas_pack=active_pack, validity="DEPENDS",
            evidence_class=_least_authority((*phase_authorities, *active_authorities)),
            flags=("phase_equations_inconsistent",),
            notices=("declared_assemblage_not_minimized",), refusal=refusal,
            non_identifiable_components=tuple(phase_components),
            phase_standard_gibbs_J_mol=crystal_gibbs,
            phase_affinities_ln={
                phase: float(residual_vector[index] / (R_J_MOL_K * temperature))
                for index, phase in enumerate(declared)
            },
            crystal_function_continued=tuple(continued_notices),
        )

    identifiable = []
    for column, oxide in enumerate(phase_components):
        # Adding any vector in null(A) preserves every phase equation. A
        # component μ_i is unique iff its coordinate projection onto null(A)
        # vanishes; use the 1e-10 norm threshold to tolerate SVD roundoff.
        null_component = float(np.linalg.norm(nullspace[:, column])) if nullspace.size else 0.0
        if null_component <= 1.0e-10:
            identifiable.append(oxide)
    identifiable_set = set(identifiable)
    selected = (
        tuple(phase_components) if requested_oxides is None else requested_oxides
    )
    reported = tuple(oxide for oxide in selected if oxide in identifiable_set)
    non_identifiable = tuple(sorted(
        (set(phase_components) - identifiable_set)
        | (set(selected) - identifiable_set)
    ))
    chemical_potentials = {
        oxide: float(minimum_norm_mu[component_column[oxide]]) for oxide in reported
    }
    activities = {
        oxide: math.exp(
            (chemical_potentials[oxide] - liquid_gibbs[oxide])
            / (R_J_MOL_K * temperature)
        )
        for oxide in reported
    }
    affinities = {
        # Residuals are J/mol formula; dividing by RT (J/mol) gives a
        # dimensionless log-affinity with the same predicted-minus-supplied sign.
        phase: float(residual_vector[index] / (R_J_MOL_K * temperature))
        for index, phase in enumerate(declared)
    }
    gas_depends: dict[str, str] = {}
    gas_result = None
    balance_pO2 = None
    balance_diagnostics: Mapping[str, Any] = {}
    available_activities = activities
    gas_parent_oxides = tuple(activities)

    if oxygen_balance:
        missing_components = tuple(
            oxide for oxide in phase_components if oxide not in activities
        )
        if missing_components:
            gas_depends["oxygen_balance"] = (
                "activity_not_identifiable: " + ", ".join(missing_components)
            )
        else:
            try:
                balance_pO2, gas_result, diagnostics_map = (
                    gas_module.evaluate_gas_oxygen_balance(
                        activities, temperature, active_pack,
                        parent_oxides=gas_parent_oxides,
                        allow_extrapolation=allow_extrapolation,
                        include_ions=include_ions,
                    )
                )
                balance_diagnostics = diagnostics_map
            except gas_module.ImccGasTemperatureOutsideDomainError as exc:
                axes = ("gas_species.temperature_K",)
                refusal = EquilibriumRefusal("missing_phase_data", str(exc), axes)
                return _empty_result(
                    phases=declared, temperature_K=temperature, fO2=None,
                    gas_species=requested_gas, parent_oxides=requested_oxides,
                    include_ions=include_ions, oxygen_balance=oxygen_balance,
                    allow_extrapolation=allow_extrapolation,
                    crystal_digest=crystal_digest, pure_digest=pure_digest,
                    gas_pack=active_pack, validity="OUT_OF_DOMAIN",
                    evidence_class=_least_authority(
                        (*phase_authorities, *active_authorities)
                    ), flags=("gas_species_out_of_range",),
                    notices=("declared_assemblage_not_minimized",),
                    refusal=refusal, axes=axes,
                    crystal_function_continued=tuple(continued_notices),
                )
    elif fO2 is not None:
        try:
            gas_result, gas_depends = _evaluate_gas_attachment(
                available_activities, temperature, float(fO2), active_pack,
                dependency_components=phase_components,
                parent_oxides=gas_parent_oxides,
                gas_species=requested_gas,
                include_ions=include_ions,
                allow_extrapolation=allow_extrapolation,
            )
        except gas_module.ImccGasTemperatureOutsideDomainError as exc:
            axes = ("gas_species.temperature_K",)
            refusal = EquilibriumRefusal("missing_phase_data", str(exc), axes)
            return _empty_result(
                phases=declared, temperature_K=temperature, fO2=fO2,
                gas_species=requested_gas, parent_oxides=requested_oxides,
                include_ions=include_ions, oxygen_balance=oxygen_balance,
                allow_extrapolation=allow_extrapolation,
                crystal_digest=crystal_digest, pure_digest=pure_digest,
                gas_pack=active_pack, validity="OUT_OF_DOMAIN",
                evidence_class=_least_authority(
                    (*phase_authorities, *active_authorities)
                ), flags=("gas_species_out_of_range",),
                notices=("declared_assemblage_not_minimized",),
                refusal=refusal, axes=axes,
                crystal_function_continued=tuple(continued_notices),
            )

    flags = ["fixed_pressure_1_bar", "gas_is_post_processing"]
    notices = [
        "declared_assemblage_assumed",
        "declared_assemblage_not_minimized",
        "gas_not_in_condensed_mass_balance",
        *continued_notices,
    ]
    if "K2O" in declared:
        flags.append("k2o_liquid_standard_estimated")
        notices.append("K2O uses the selected liquid-from-solid reference construction.")
    if non_identifiable:
        flags.append("activity_not_identifiable")
        notices.append("Some oxide activities are not uniquely fixed by the stated phases.")
    if gas_depends:
        flags.append("gas_channel_depends_on_nonidentifiable_activity")
    if fO2 is None and not oxygen_balance:
        notices.append("gas_not_evaluated_without_fO2")
    if gas_result is not None:
        flags.extend(
            f"gas_domain:{name}" for name, detail in gas_result.domain_flags.items()
            if detail is not None
        )

    authorities = [*phase_authorities, *active_authorities]
    if gas_result is not None:
        authorities.extend(gas_result.provenance_class.values())
    evidence_class = _least_authority(authorities)
    if gas_result is not None and gas_result.domain_flags and any(
        detail is not None for detail in gas_result.domain_flags.values()
    ):
        validity: Literal["IN", "OUT_OF_DOMAIN", "DEPENDS"] = "OUT_OF_DOMAIN"
        domain_axes = tuple(
            f"gas_species.{name}.temperature_K"
            for name, detail in gas_result.domain_flags.items() if detail is not None
        )
    elif non_identifiable or gas_depends:
        validity = "DEPENDS"
        domain_axes = ()
    else:
        validity = "IN"
        domain_axes = ()

    refusal = None
    if non_identifiable:
        refusal = EquilibriumRefusal(
            "activity_not_identifiable",
            "The stated phases do not uniquely determine every requested parent-oxide activity.",
            tuple(f"activity.{oxide}" for oxide in non_identifiable),
        )
    elif gas_depends:
        refusal = EquilibriumRefusal(
            "gas_activity_not_identifiable",
            "Some requested gas channels need parent activities not fixed by the stated phases.",
            tuple(f"gas_species.{name}" for name in gas_depends),
        )

    labels = PhaseEquilibriumLabels(
        identity={
            "crystal_phases": "openimcc-crystal-phases-v2",
            "barin_aluminate_crystals": str(barin_data["schema"]),
            "janaf_pure_oxide_crystals": str(pure_data["schema"]),
            "gas_table_digest": active_pack.gas_table_digest,
            "condensate_table_digest": active_pack.condensate_table_digest,
            "standard_pressure_bar": "1.0",
        },
        coverage=coverage, validity=validity, out_of_domain_axes=domain_axes,
        evidence_class=evidence_class, flags=tuple(dict.fromkeys(flags)),
        notices=tuple(dict.fromkeys(notices)),
    )
    receipt = _make_receipt(
        phases=declared, temperature_K=temperature, fO2=fO2,
        gas_species=requested_gas, parent_oxides=requested_oxides,
        include_ions=include_ions, oxygen_balance=oxygen_balance,
        allow_extrapolation=allow_extrapolation, crystal_digest=crystal_digest,
        pure_digest=pure_digest, gas_pack=active_pack,
        crystal_function_continued=tuple(continued_notices),
    )
    return DeclaredAssemblageResult(
        temperature_K=temperature, declared_phases=declared,
        phase_standard_gibbs_J_mol=crystal_gibbs,
        phase_affinities_ln=affinities,
        oxide_chemical_potentials_J_mol=chemical_potentials,
        oxide_activities=activities,
        non_identifiable_components=non_identifiable,
        gas=gas_result, gas_depends=gas_depends,
        fO2=(balance_pO2 if balance_pO2 is not None else fO2),
        oxygen_balance_pO2_bar=balance_pO2,
        oxygen_balance_diagnostics=balance_diagnostics,
        standard_pressure_bar=_PRESSURE_BAR_STANDARD, labels=labels,
        receipt=receipt, refusal=refusal,
    )
