"""IMCC-SF04 domain adapter: JSON datapack loader and caller-facing API.

Wraps the kernel from ``openimcc.kernel`` with a JSON datapack loader, a
wt-to-mol basis converter, and a caller-facing label block.

This module is the MODEL half and is deliberately free of simulator policy:
its only non-model dependency is ``openimcc.scalar_boundary`` (a stdlib-only
leaf). The ``MeltBackend`` surface used by ``resolve_backend`` lives in
the simulator's backend glue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from fractions import Fraction
from importlib import resources
from pathlib import Path
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Dict, Optional

import numpy as np

from openimcc.kernel import (
    ImccComponentOutsideDomainError,
    ImccCompositionIncompleteError,
    ImccDatapack,
    ImccFerricInputUnsupportedError,
    ImccNonconvergenceError,
    ImccRefusal,
    ImccResult,
    ImccTOutsideDatapackDomainError,
    _PUBLISHED_DATAPACK_SHA256,
    _PUBLISHED_MODEL_ID,
    _label_loaded_datapack,
    _published_datapack_manifest_hash,
    label_research_datapack,
    solve_imcc_sf04,
)
from openimcc.scalar_boundary import is_declared_real_scalar



# Stable molar masses (g/mol) for the IMCC-SF04 parent components.
# Values from the project physical-constants table (CIAAW/NIST derived).
_OXIDE_MOLAR_MASS_G_MOL = MappingProxyType(
    {
        "SiO2": 60.0843,
        "MgO": 40.3044,
        "FeO": 71.844,
        "CaO": 56.0774,
        "Al2O3": 101.9613,
        "TiO2": 79.866,
        "Na2O": 61.9789,
        "K2O": 94.196,
        "S": 32.06,
        "P2O5": 141.9445,
    }
)

_EXPECTED_PARENT_OXIDES = (
    "SiO2",
    "MgO",
    "FeO",
    "CaO",
    "Al2O3",
    "TiO2",
    "Na2O",
    "K2O",
)

_PUBLISHED_CORE_ROWS = 38

_SP_EXTENSION_MODEL_ID = "IMCC-SF04-EXT"
_SP_EXTENSION_PARENTS = ("S", "P2O5")
_SP_EXTENSION_TIER = "EXT-SP"
_SP_EXTENSION_FLAG = "enable_sp_extension"
_SP_EXTENSION_PROVENANCE_CLASS = "extension-compound-thermo"
_ENVELOPE_RELATIVE_SLACK = 1.0e-5
_SILICA_SINK_FAMILY_SHARE = 0.20
_PARENT_CATION_SYMBOL = MappingProxyType(
    {
        "SiO2": "Si",
        "MgO": "Mg",
        "FeO": "Fe",
        "CaO": "Ca",
        "Al2O3": "Al",
        "TiO2": "Ti",
        "Na2O": "Na",
        "K2O": "K",
        "S": "S",
        "P2O5": "P",
    }
)
# Data derivation (2026-09-25; no solver retune):
# (a) The smallest strict-path validated ratio is 1.95684635346e-3
#     (kume2000_s145, 1823 K), and every SF04 Table 5 rock also remains
#     unflagged.
# (b) At 1473 K, evaluate both binaries with both overrides and scan
#     X_Me2O=.450,.451,...,.500. For each point, residual is
#     log10(a_model(Me2O)) minus the linear bridge between the published
#     anchors. The relevant boundary rows are:
#
#       binary       X       ratio             bridge residual   below floor  flag
#       K2O-SiO2     .493    2.183642e-3       +1.33              no           no
#       K2O-SiO2     .494    1.864382e-3       +1.40              yes          yes
#       K2O-SiO2     .495    1.547655e-3       +1.47              yes          yes
#       K2O-SiO2     .496    1.233415e-3       +1.57              yes          yes
#       Na2O-SiO2    .497    2.472783e-3       +0.47              no           no
#       Na2O-SiO2    .498    1.644304e-3       +0.64              yes          yes
#       Na2O-SiO2    .499    8.225617e-4       +0.93              yes          yes
#       Na2O-SiO2    .500    5.783685e-5       +2.07              yes          yes
#
# The K bridge is -7.46 at X=.457 to -7.27 at X=.500; the Na bridge is
# -6.220 at X=.450 to -5.622 at X=.500. The largest (b) ratio below the
# floor is therefore 1.86438232260e-3 at K2O-SiO2 X=.494. The same 0.001
# scan at the other temperatures gives this compact stability table (the
# first sub-floor point is also the largest sub-floor point in each scan):
#
#       T (K)   K2O-SiO2 X / ratio       Na2O-SiO2 X / ratio
#       1373    .493 / 1.762926e-3       .498 / 1.410644e-3
#       1473    .494 / 1.864382e-3       .498 / 1.644304e-3
#       1573    .495 / 1.863152e-3       .498 / 1.879865e-3
#
# Every stability-table ratio stays below the final cut; the 1473 K
# published-anchor evidence sets the lower bound.
# The admissible window is (1.86438232260e-3, 1.95684635346e-3), so use
# their geometric mean, sqrt(product) = 1.91005490744e-3. This is a
# predict-and-flag threshold, not a coefficient change.
_SPECIES_COVERAGE_EDGE_RATIO = 1.9100549074388355e-3
_ALKALI_BIAS_NOTICE = (
    "Na and K activities from IMCC-SF04 are biased low against published "
    "anchors (SF04 Table 9 Na −1.4 dex; Hastie 1981 K −0.9 dex); see "
    "docs/ROADMAP.md"
)

class ImccMalformedDatapackError(ImccRefusal):
    """Raised when a datapack JSON file is malformed or schema-invalid."""

    code = "imcc_malformed_datapack"


class ImccUnprovenDatapackError(ImccRefusal):
    """Raised when adapter evaluation receives an unlabelled raw datapack."""

    code = "imcc_unproven_datapack"


class ImccCompositionOutsideValidatedEnvelopeError(ImccRefusal):
    """Raised when the melt lies outside the validated Tier-A envelope."""

    code = "imcc_composition_outside_validated_envelope"


class ImccSPComponentRequiresExtensionError(ImccComponentOutsideDomainError):
    """Raised when S/P input lacks the EXT model plus explicit enable flag."""

    code = "imcc_sp_extension_required"


@dataclass(frozen=True)
class ImccLoadedDatapack:
    """Adapter-level loaded datapack: kernel datapack plus provenance metadata."""

    kernel_datapack: ImccDatapack
    version: str
    parent_oxides: Sequence[str]
    domain_basis: Sequence[str]
    extension_parents: Sequence[str] = ()
    extension_species: Sequence[str] = ()

    @property
    def model_id(self) -> str:
        return self.kernel_datapack.model_id

    @property
    def paper_domains(self) -> Sequence[tuple[float, float] | None]:
        return self.kernel_datapack.paper_domains


@dataclass(frozen=True)
class ImccAdapterLabels:
    """Caller-facing identity, coverage, edge metric, and flags."""

    identity: Mapping[str, str]
    coverage: Mapping[str, str]
    envelope_status: str
    flags: tuple[str, ...] = ()
    notices: tuple[str, ...] = ()
    acid_sink_ratio: float | None = None


def _as_fraction(value: Any) -> Fraction:
    """Parse a JSON numeric value as an exact rational."""
    if is_declared_real_scalar(value, allow_numeric_str=True):
        return Fraction(str(value))
    raise TypeError(f"cannot parse {value!r} as a rational")


def _published_core_manifest_payload(
    data: Mapping[str, Any],
    *,
    model_id: str,
    version: str,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in data.items()
        if key not in {"model_id", "sp_extension"}
    }
    if model_id == _SP_EXTENSION_MODEL_ID:
        version = version.partition("-ext-sp-")[0]
    payload["model_id"] = _PUBLISHED_MODEL_ID
    payload["imcc_sf04_datapack_version"] = version
    return payload


def _validate_published_core(
    data: Mapping[str, Any],
    *,
    model_id: str,
    version: str,
) -> tuple[list[dict[str, Any]], str]:
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != _PUBLISHED_CORE_ROWS:
        raise ImccMalformedDatapackError(
            "datapack must contain exactly 38 published-core rows, got "
            f"{len(rows) if isinstance(rows, list) else None}"
        )
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ImccMalformedDatapackError(
                f"published core row {idx} is not an object"
            )
    try:
        content_hash = _published_datapack_manifest_hash(
            _published_core_manifest_payload(
                data,
                model_id=model_id,
                version=version,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ImccMalformedDatapackError(
            "published IMCC datapack cannot be canonically serialized"
        ) from exc
    if content_hash != _PUBLISHED_DATAPACK_SHA256:
        raise ImccMalformedDatapackError(
            "published IMCC datapack canonical hash mismatch: "
            f"expected {_PUBLISHED_DATAPACK_SHA256}, got {content_hash}"
        )
    return rows, content_hash


def _build_nu_vector(
    parent_oxides: Sequence[str],
    nu: Mapping[str, Any],
    *,
    row_label: str,
) -> list[float]:
    """Return a nu column aligned to parent_oxides, preserving exact rationals."""
    unknown_nu = sorted(set(nu) - set(parent_oxides))
    if unknown_nu:
        raise ImccMalformedDatapackError(
            f"{row_label} nu has unknown key {unknown_nu[0]!r}"
        )
    return [float(_as_fraction(nu.get(name, 0))) for name in parent_oxides]


def _wt_to_mol(vector: np.ndarray, parent_oxides: Sequence[str]) -> np.ndarray:
    """Convert a mass-fraction vector (g) to a mole vector (mol)."""
    molar_masses = np.array(
        [_OXIDE_MOLAR_MASS_G_MOL[name] for name in parent_oxides],
        dtype=float,
    )
    return vector / molar_masses


def load_datapack(path: str | Path | None = None) -> ImccLoadedDatapack:
    """Load an IMCC-SF04 datapack JSON into the kernel datapack object.

    Validates the complete canonical published datapack hash. ``IMCC-SF04-EXT``
    packs may add the separately labelled ``sp_extension`` section; their base
    datapack projects to the same frozen published identity. With no path, the
    packaged ``imcc-sf04-v1.0.2`` resource is loaded.
    """
    source = (
        resources.files("openimcc")
        / "data"
        / "packs"
        / "imcc-sf04-v1.0.2.json"
        if path is None
        else Path(path)
    )
    try:
        with source.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ImccMalformedDatapackError(
            f"datapack JSON at {source} is not valid JSON"
        ) from exc
    except FileNotFoundError as exc:
        raise ImccMalformedDatapackError(
            f"datapack file not found: {source}"
        ) from exc

    if not isinstance(data, dict):
        raise ImccMalformedDatapackError("datapack JSON root must be an object")

    version = data.get("imcc_sf04_datapack_version")
    if not isinstance(version, str) or not version:
        raise ImccMalformedDatapackError(
            "datapack missing 'imcc_sf04_datapack_version' string"
        )

    base_parents = data.get("parents")
    if not isinstance(base_parents, list) or base_parents != list(_EXPECTED_PARENT_OXIDES):
        raise ImccMalformedDatapackError(
            f"datapack parents {base_parents!r} do not match expected "
            f"{_EXPECTED_PARENT_OXIDES!r}"
        )
    base_parents = tuple(base_parents)

    model_id = data.get("model_id", "IMCC-SF04")
    sp_extension = data.get("sp_extension")
    extension_parents: tuple[str, ...] = ()
    extension_rows: list[dict[str, Any]] = []
    if sp_extension is None:
        if model_id != "IMCC-SF04":
            raise ImccMalformedDatapackError(
                f"model_id {model_id!r} requires a recognized extension section"
            )
    else:
        if model_id != _SP_EXTENSION_MODEL_ID:
            raise ImccMalformedDatapackError(
                "sp_extension requires model_id='IMCC-SF04-EXT'"
            )

    rows, published_manifest_sha256 = _validate_published_core(
        data,
        model_id=model_id,
        version=version,
    )

    if sp_extension is not None:
        if not isinstance(sp_extension, dict):
            raise ImccMalformedDatapackError("sp_extension must be an object")
        if sp_extension.get("enable_flag") != _SP_EXTENSION_FLAG:
            raise ImccMalformedDatapackError(
                "sp_extension enable_flag must be 'enable_sp_extension'"
            )
        if sp_extension.get("tier") != _SP_EXTENSION_TIER:
            raise ImccMalformedDatapackError("sp_extension tier must be 'EXT-SP'")
        if sp_extension.get("certification") != "denied":
            raise ImccMalformedDatapackError(
                "sp_extension certification must be explicitly denied"
            )
        raw_extension_parents = sp_extension.get("parents")
        if raw_extension_parents != list(_SP_EXTENSION_PARENTS):
            raise ImccMalformedDatapackError(
                f"sp_extension parents must be {list(_SP_EXTENSION_PARENTS)!r}"
            )
        raw_extension_rows = sp_extension.get("rows")
        if not isinstance(raw_extension_rows, list) or not raw_extension_rows:
            raise ImccMalformedDatapackError(
                "sp_extension rows must be a non-empty list"
            )
        for idx, row in enumerate(raw_extension_rows):
            if not isinstance(row, dict):
                raise ImccMalformedDatapackError(
                    f"sp_extension row {idx} is not an object"
                )
            if row.get("provenance_class") != _SP_EXTENSION_PROVENANCE_CLASS:
                raise ImccMalformedDatapackError(
                    f"sp_extension row {idx} provenance_class must be "
                    f"{_SP_EXTENSION_PROVENANCE_CLASS!r}"
                )
            if row.get("tier") != _SP_EXTENSION_TIER:
                raise ImccMalformedDatapackError(
                    f"sp_extension row {idx} tier must be 'EXT-SP'"
                )
            if row.get("certification") != "denied":
                raise ImccMalformedDatapackError(
                    f"sp_extension row {idx} certification must be denied"
                )
            provenance = row.get("provenance")
            if not isinstance(provenance, dict):
                raise ImccMalformedDatapackError(
                    f"sp_extension row {idx} missing provenance object"
                )
            if not all(
                isinstance(provenance.get(field), str) and provenance[field]
                for field in ("source", "table_id")
            ):
                raise ImccMalformedDatapackError(
                    f"sp_extension row {idx} provenance requires source and table_id"
                )
            extension_nu = row.get("nu")
            if isinstance(extension_nu, dict):
                unknown_nu = set(extension_nu) - set(
                    base_parents + _SP_EXTENSION_PARENTS
                )
                if unknown_nu:
                    raise ImccMalformedDatapackError(
                        f"sp_extension row {idx} nu has unknown parents "
                        f"{sorted(unknown_nu)}"
                    )
                if not any(
                    _as_fraction(extension_nu.get(parent, 0)) > 0
                    for parent in _SP_EXTENSION_PARENTS
                ):
                    raise ImccMalformedDatapackError(
                        f"sp_extension row {idx} must consume S or P2O5"
                    )
            extension_rows.append(row)
        extension_parents = _SP_EXTENSION_PARENTS

    parents = base_parents + extension_parents
    all_rows = rows + extension_rows

    reactions: list[str] = []
    nu_cols: list[list[float]] = []
    A: list[float] = []
    B: list[float] = []
    domains: list[tuple[float, float]] = []
    paper_domains: list[tuple[float, float] | None] = []
    domain_basis: list[str] = []

    for idx, row in enumerate(all_rows):
        if not isinstance(row, dict):
            raise ImccMalformedDatapackError(f"row {idx} is not an object")

        complex_name = row.get("complex")
        if not isinstance(complex_name, str):
            raise ImccMalformedDatapackError(
                f"row {idx} missing 'complex' string"
            )
        reactions.append(complex_name)

        nu = row.get("nu")
        if not isinstance(nu, dict):
            raise ImccMalformedDatapackError(
                f"row {idx} missing 'nu' object"
            )
        row_label = (
            f"published core row {idx}"
            if idx < len(rows)
            else f"sp_extension row {idx - len(rows)}"
        )
        nu_cols.append(_build_nu_vector(parents, nu, row_label=row_label))

        A_val = row.get("A")
        B_val = row.get("B")
        if (
            not is_declared_real_scalar(A_val)
            or not isinstance(A_val, (int, float))
            or not is_declared_real_scalar(B_val)
            or not isinstance(B_val, (int, float))
        ):
            raise ImccMalformedDatapackError(
                f"row {idx} A/B must be numeric"
            )
        A_float = float(A_val)
        B_float = float(B_val)
        if idx >= len(rows) and (
            not np.isfinite(A_float) or not np.isfinite(B_float)
        ):
            raise ImccMalformedDatapackError(
                f"{row_label} A and B must be finite"
            )
        A.append(A_float)
        B.append(B_float)

        t_domain = row.get("T_domain_K")
        if not isinstance(t_domain, list) or len(t_domain) != 2:
            raise ImccMalformedDatapackError(
                f"row {idx} missing valid T_domain_K [low, high]"
            )
        if not all(
            is_declared_real_scalar(value, allow_numeric_str=True)
            for value in t_domain
        ):
            raise ImccMalformedDatapackError(
                f"row {idx} T_domain_K values must be numeric"
            )
        domain_values = (float(t_domain[0]), float(t_domain[1]))
        if idx >= len(rows) and not all(
            np.isfinite(value) for value in domain_values
        ):
            raise ImccMalformedDatapackError(
                f"{row_label} endpoints must be finite Kelvin values, "
                f"got ({t_domain[0]!r}, {t_domain[1]!r})"
            )
        domains.append(domain_values)

        paper_domain = row.get("T_domain_paper_demonstrated_K")
        if paper_domain is None:
            paper_domains.append(None)
        else:
            if not isinstance(paper_domain, list) or len(paper_domain) != 2:
                raise ImccMalformedDatapackError(
                    f"row {idx} T_domain_paper_demonstrated_K must be a pair"
                )
            if not all(
                is_declared_real_scalar(value, allow_numeric_str=True)
                for value in paper_domain
            ):
                raise ImccMalformedDatapackError(
                    f"row {idx} T_domain_paper_demonstrated_K values must be numeric"
                )
            paper_domain_values = (float(paper_domain[0]), float(paper_domain[1]))
            if not all(np.isfinite(value) for value in paper_domain_values):
                raise ImccMalformedDatapackError(
                    f"row {idx} T_domain_paper_demonstrated_K values must be finite"
                )
            paper_domains.append(paper_domain_values)

        basis = row.get("T_domain_basis")
        if not isinstance(basis, str):
            raise ImccMalformedDatapackError(
                f"row {idx} missing T_domain_basis string"
            )
        domain_basis.append(basis)

    if len(set(reactions)) != len(reactions):
        raise ImccMalformedDatapackError("datapack complex names must be unique")

    # nu_cols is (n_complexes, n_parents); transpose to kernel shape.
    nu_array = np.array(nu_cols, dtype=float).T

    kernel_datapack = ImccDatapack(
        reactions=reactions,
        nu=nu_array,
        A=np.array(A, dtype=float),
        B=np.array(B, dtype=float),
        domains=domains,
        paper_domains=paper_domains,
        version=version,
        parent_oxides=parents,
    )
    extension_species = tuple(str(row["complex"]) for row in extension_rows)
    extension_names = set(extension_parents) | set(extension_species)
    coverage = {
        name: (_SP_EXTENSION_TIER if name in extension_names else "A-published-imcc")
        for name in (*parents, *reactions)
    }
    kernel_datapack = _label_loaded_datapack(
        kernel_datapack,
        model_id=model_id,
        coverage=coverage,
        published_manifest_sha256=published_manifest_sha256,
    )

    return ImccLoadedDatapack(
        kernel_datapack=kernel_datapack,
        version=version,
        parent_oxides=parents,
        domain_basis=tuple(domain_basis),
        extension_parents=extension_parents,
        extension_species=extension_species,
    )


def _amount_as_float(value: Any, label: str) -> float:
    """Convert one supplied amount, refusing a non-numeric value as invalid input.

    A bare float() here would raise an untyped ValueError or TypeError for
    "abc" or None, so a caller would get a crash instead of a refusal that
    names the bad entry.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ImccCompositionIncompleteError(
            f"{label} is not a number: {value!r}"
        ) from None
    except OverflowError:
        # An int beyond float range (e.g. 10**400). The value is not echoed:
        # its repr can run to hundreds of digits.
        raise ImccCompositionIncompleteError(
            f"{label} is too large to represent as a float"
        ) from None


def _sp_extension_refusal(component_names: Sequence[str]) -> ImccSPComponentRequiresExtensionError:
    names = ", ".join(sorted(component_names)) or "S/P EXT component(s)"
    return ImccSPComponentRequiresExtensionError(
        f"{names} belong to the S/P EXT component class; unlock only with "
        "model_id='IMCC-SF04-EXT' and enable_sp_extension=True. Plain "
        "IMCC-SF04 intentionally excludes S/P speciation."
    )


def evaluate(
    composition: Mapping[str, float] | Sequence[float],
    T_K: float,
    pack: ImccLoadedDatapack | ImccDatapack | None = None,
    *,
    basis: float | None = None,
    basis_type: str = "mol",
    extra_mol: Mapping[str, float] | None = None,
    enable_sp_extension: bool = False,
    allow_extrapolation: bool = False,
    allow_out_of_envelope: bool = False,
    tol: float = 1.0e-12,
    max_iter: int = 100,
) -> ImccResult:
    """Caller-facing IMCC-SF04 evaluation.

    Accepts a composition on either a mol or wt basis with a declared basis.
    Converts wt-to-mol before any cache boundary, validates the FeO-equivalent
    contract, and delegates to the kernel.

    Parameters
    ----------
    composition:
        Dict keyed by parent-oxide name, or a numeric vector aligned with the
        pack's parent oxides.
    T_K:
        Temperature in Kelvin.
    pack:
        Loaded datapack, or a raw kernel datapack labelled by
        ``label_research_datapack()``. If omitted, the packaged
        ``imcc-sf04-v1.0.2`` datapack is used. Unlabelled raw packs are refused.
    basis:
        Declared normalization basis in the same units as ``basis_type``. If
        ``None``, the composition sum is used.
    basis_type:
        ``"mol"`` or ``"wt"``.
    extra_mol:
        Additional components in mol (e.g. Tier-B/C screens). Positive Fe2O3
        raises ``ImccFerricInputUnsupportedError``; other positives outside the
        parent basis raise ``ImccComponentOutsideDomainError``.
    enable_sp_extension:
        Explicitly enable S and P2O5 parents in an ``IMCC-SF04-EXT`` pack.
        The flag alone never widens a plain ``IMCC-SF04`` pack.
    allow_extrapolation:
        If ``True``, evaluate outside declared T domains and mark the result
        as extrapolated.
    allow_out_of_envelope:
        If ``True``, evaluate outside the validated Tier-A composition
        envelope and mark the result ``outside_validated``.
    tol:
        Infinity-norm residual convergence tolerance passed to the kernel.
    max_iter:
        Newton iteration ceiling passed to the kernel.

    Returns
    -------
    ImccResult
        Kernel result with the adapter's label block installed.
    """
    if pack is None:
        pack = load_datapack()

    # Resolve the kernel datapack and metadata.
    if isinstance(pack, ImccLoadedDatapack):
        kernel_pack = pack.kernel_datapack
        parent_oxides = pack.parent_oxides
        extension_parents = tuple(pack.extension_parents)
    else:
        kernel_pack = pack
        parent_oxides = pack.parent_oxides
        extension_parents = tuple(
            name for name in _SP_EXTENSION_PARENTS if name in parent_oxides
        )

    if not kernel_pack.identity_is_proven:
        raise ImccUnprovenDatapackError(
            "raw ImccDatapack has no proven identity; load a frozen JSON pack "
            "with load_datapack() or apply explicit non-published provenance "
            "with label_research_datapack()"
        )

    pack_version = kernel_pack.version
    model_id = kernel_pack.model_id

    if isinstance(composition, Mapping):
        for name, raw in composition.items():
            if not np.isfinite(_amount_as_float(raw, f"composition value for {name}")):
                raise ImccCompositionIncompleteError(
                    "composition contains non-finite values"
                )
    if extra_mol:
        for species, mol in extra_mol.items():
            value = _amount_as_float(mol, f"extra component {species}")
            if not np.isfinite(value):
                raise ImccCompositionIncompleteError(
                    f"extra component {species} has non-finite moles {value}"
                )
    supplied_sp_names: set[str] = set()
    if isinstance(composition, Mapping):
        supplied_sp_names.update(
            name for name in _SP_EXTENSION_PARENTS
            if float(composition.get(name, 0.0)) != 0.0
        )
    elif len(composition) > len(_EXPECTED_PARENT_OXIDES):
        supplied_sp_names.update(_SP_EXTENSION_PARENTS)
    if extra_mol:
        supplied_sp_names.update(
            name for name in _SP_EXTENSION_PARENTS
            if float(extra_mol.get(name, 0.0)) != 0.0
        )

    if model_id == _SP_EXTENSION_MODEL_ID:
        if not enable_sp_extension:
            raise _sp_extension_refusal(extension_parents)
    elif enable_sp_extension or supplied_sp_names:
        raise _sp_extension_refusal(supplied_sp_names)

    if basis_type not in ("mol", "wt"):
        raise ImccCompositionIncompleteError(
            f"basis_type must be 'mol' or 'wt', got {basis_type!r}"
        )

    # Normalize composition to a numeric vector aligned with parent_oxides.
    if isinstance(composition, Mapping):
        # Ferric refusal at the adapter boundary (FeO-equivalent contract).
        if "Fe2O3" in composition and composition["Fe2O3"] != 0:
            raise ImccFerricInputUnsupportedError(
                "Fe2O3 input is unsupported; convert to FeO under the "
                "caller's redox model before calling IMCC-SF04"
            )
        unknown = set(composition.keys()) - set(parent_oxides)
        if unknown:
            raise ImccComponentOutsideDomainError(
                f"component(s) outside IMCC-SF04 domain: {sorted(unknown)}"
            )
        vector = np.array(
            [float(composition.get(name, 0.0)) for name in parent_oxides],
            dtype=float,
        )
    else:
        vector = np.asarray(composition, dtype=float)
        if vector.ndim != 1:
            raise ImccCompositionIncompleteError(
                "composition vector must be 1-D"
            )
        if vector.shape[0] != len(parent_oxides):
            if vector.shape[0] > len(parent_oxides):
                raise ImccComponentOutsideDomainError(
                    f"composition vector length {vector.shape[0]} exceeds "
                    f"the {len(parent_oxides)}-oxide basis"
                )
            raise ImccCompositionIncompleteError(
                f"composition vector length {vector.shape[0]} does not match "
                f"the {len(parent_oxides)}-oxide basis"
            )

    if not np.all(np.isfinite(vector)):
        raise ImccCompositionIncompleteError(
            "composition contains non-finite values"
        )
    if np.any(vector < 0.0):
        raise ImccCompositionIncompleteError(
            "composition contains negative values"
        )

    total = float(vector.sum())
    if total <= 0.0:
        raise ImccCompositionIncompleteError("composition total is zero")

    if basis is None:
        basis = total
    else:
        basis = float(basis)
        # nan and inf both pass a bare `<= 0.0` (IEEE 754); see the matching
        # guard in kernel.solve_imcc_sf04.
        if not np.isfinite(basis) or basis <= 0.0:
            raise ImccCompositionIncompleteError(
                f"declared basis must be a positive finite number, got {basis}"
            )
        if abs(total - basis) > 1.0e-6 * basis:
            raise ImccCompositionIncompleteError(
                f"composition sum {total:.12g} does not match declared basis "
                f"{basis:.12g} within 1e-6 relative"
            )

    # Convert wt to mol before the cache boundary (E8).
    if basis_type == "wt":
        vector = _wt_to_mol(vector, parent_oxides)
        basis = float(vector.sum())

    # X_Me2O = (n_Na2O + n_K2O) / sum(n_oxide) over the canonical
    # 8-oxide mol vector. Treat the boundary as inclusive within a 1e-5
    # relative comparison slack; decimal wt% input for a printed X=0.5 can
    # arrive a few ppm above the ideal ratio after conversion. The 1e-5
    # cutoff accepts the measured 3.947e-6 rounding residual but still
    # refuses a composition at X=0.50001.
    alkali_mol = sum(
        float(vector[parent_oxides.index(name)]) for name in ("Na2O", "K2O")
    )
    canonical_oxide_mol = sum(
        float(vector[parent_oxides.index(name)])
        for name in _EXPECTED_PARENT_OXIDES
    )
    if canonical_oxide_mol <= 0.0:
        raise ImccCompositionIncompleteError(
            "canonical 8-oxide composition total is zero"
        )
    x_me2o = alkali_mol / canonical_oxide_mol
    outside_validated_envelope = x_me2o > 0.5 * (1.0 + _ENVELOPE_RELATIVE_SLACK)
    if outside_validated_envelope and not allow_out_of_envelope:
        raise ImccCompositionOutsideValidatedEnvelopeError(
            f"X_Me2O={x_me2o:.12g} exceeds validated bound 0.5"
        )

    result = solve_imcc_sf04(
        vector,
        T_K,
        kernel_pack,
        basis=basis,
        extra_mol=extra_mol,
        allow_extrapolation=allow_extrapolation,
        tol=tol,
        max_iter=max_iter,
    )

    # Build the caller-facing label block. The paper-window flag is computed
    # by the kernel from row metadata; the edge flag is computed from the
    # solved free-parent fraction, so neither condition is a hardcoded Na bound.
    identity: Mapping[str, str] = {
        "model_id": model_id,
        "datapack_version": pack_version,
    }
    coverage: Mapping[str, str] = result.labels.coverage
    flags = list(result.labels.flags)
    si_index = parent_oxides.index("SiO2") if "SiO2" in parent_oxides else None
    acid_sink_ratio = (
        float(result.parent_x_star[si_index] / result.parent_x[si_index])
        if si_index is not None and result.parent_x[si_index] > 0.0
        else None
    )
    if acid_sink_ratio is not None and acid_sink_ratio < _SPECIES_COVERAGE_EDGE_RATIO:
        # The ratio is continuous and comes from the solved free-parent
        # fraction, so it remains visible even when the flag is crossed.
        # Partition bound silica from solved complexes, not input parents.
        # A complex contributes nu(SiO2) * x(complex); only families holding
        # at least 20% of that solved bound silica are named. Cations follow
        # their order in the complex formula, so KCaAlSi2O7 is K–Ca–Al.
        family_silica: dict[tuple[str, ...], float] = {}
        bound_silica = 0.0
        for complex_index, complex_name in enumerate(kernel_pack.reactions):
            silica_amount = float(
                kernel_pack.nu[si_index, complex_index]
                * result.complex_x[complex_index]
            )
            if silica_amount <= 0.0:
                continue
            cations = tuple(
                sorted(
                    (
                        _PARENT_CATION_SYMBOL.get(name, name)
                        for name in parent_oxides
                        if name != "SiO2"
                        and kernel_pack.nu[
                            parent_oxides.index(name), complex_index
                        ]
                        > 0.0
                    ),
                    key=complex_name.find,
                )
            )
            family_silica[cations] = (
                family_silica.get(cations, 0.0) + silica_amount
            )
            bound_silica += silica_amount
        dominant_families = [
            "–".join(cations)
            for cations, silica_amount in sorted(
                family_silica.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if bound_silica > 0.0
            and silica_amount / bound_silica >= _SILICA_SINK_FAMILY_SHARE
        ]
        if dominant_families:
            family = f"{' / '.join(dominant_families)} silicate"
        else:
            family = "solution silicate"
        flags.append(
            "species-coverage-edge: free x*(SiO2) is below "
            f"{_SPECIES_COVERAGE_EDGE_RATIO:g} of nominal x(SiO2); "
            f"the {family} ladder has exhausted its acidic sink"
        )

    notices = (
        (_ALKALI_BIAS_NOTICE,)
        if any(
            vector[parent_oxides.index(name)] > 0.0
            for name in ("Na2O", "K2O")
            if name in parent_oxides
        )
        else ()
    )

    adapter_labels = ImccAdapterLabels(
        identity=identity,
        coverage=coverage,
        envelope_status=(
            "outside_validated"
            if outside_validated_envelope
            else "inside"
        ),
        acid_sink_ratio=acid_sink_ratio,
        flags=tuple(flags),
        notices=notices,
    )

    return replace(result, labels=adapter_labels)
