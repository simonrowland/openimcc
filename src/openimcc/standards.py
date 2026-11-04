"""Liquid-reference standard declarations for shipped thermochemistry packs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from openimcc.profiles import Profile


PackStandardSource = Literal["embedded", "registry", "undeclared"]
StandardBasis = Literal["source", "operational", "translation-record"]
StandardCheckKind = Literal[
    "mismatch",
    "reference-unknown",
    "undeclared",
    "redox-mismatch",
    "non-source-basis",
    "basis-insufficient",
]


@dataclass(frozen=True)
class PackLiquidStandards:
    """Immutable standard declarations associated with a loaded pack or path."""

    standards: Mapping[str, str]
    row_overrides: Mapping[str, Mapping[str, str]]
    standard_basis: Mapping[str, StandardBasis]
    standard_citations: Mapping[str, str]
    row_override_basis: Mapping[str, Mapping[str, StandardBasis]]
    row_override_citations: Mapping[str, Mapping[str, str]]
    source: PackStandardSource
    pack_digest: str | None


@dataclass(frozen=True)
class LiquidStandardMismatch:
    """One pack-wide or row-specific comparison with the active table set."""

    oxide: str
    declared: str
    active: str | None
    kind: StandardCheckKind
    row_key: str | None
    standard_id: str | None = None
    basis: StandardBasis | None = None
    citation: str | None = None


class PackRowLiquidStandards(dict[str, str]):
    """Row-specific IDs, with immutable basis and citation metadata."""

    def __init__(
        self,
        standards: Mapping[str, str],
        *,
        standard_basis: Mapping[str, StandardBasis],
        standard_citations: Mapping[str, str],
    ) -> None:
        super().__init__(standards)
        self.standard_basis = MappingProxyType(dict(standard_basis))
        self.standard_citations = MappingProxyType(dict(standard_citations))


def _read_registry() -> dict[str, Any]:
    resource = files("openimcc.data").joinpath("liquid-standards.json")
    value = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("liquid standard registry must be a JSON object")
    return value


def _immutable_standards(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        return MappingProxyType({})
    return MappingProxyType(
        {str(oxide): str(standard) for oxide, standard in value.items()}
    )


def _immutable_text(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        return MappingProxyType({})
    return MappingProxyType(
        {str(key): str(text) for key, text in value.items()}
    )


def _immutable_overrides(value: Any) -> Mapping[str, Mapping[str, str]]:
    if not isinstance(value, Mapping):
        return MappingProxyType({})
    return MappingProxyType(
        {
            str(row_key): _immutable_standards(standards)
            for row_key, standards in value.items()
            if isinstance(standards, Mapping)
        }
    )


def _from_declaration(
    declaration: Mapping[str, Any],
    *,
    source: PackStandardSource,
    digest: str | None,
) -> PackLiquidStandards:
    return PackLiquidStandards(
        standards=_immutable_standards(declaration.get("standards")),
        row_overrides=_immutable_overrides(declaration.get("row_overrides")),
        standard_basis=_immutable_text(declaration.get("standard_basis")),
        standard_citations=_immutable_text(
            declaration.get("standard_citations")
        ),
        row_override_basis=_immutable_overrides(
            declaration.get("row_override_basis")
        ),
        row_override_citations=_immutable_overrides(
            declaration.get("row_override_citations")
        ),
        source=source,
        pack_digest=digest,
    )


def _embedded_declaration(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    standards = value.get("standards", value)
    if not isinstance(standards, Mapping):
        return None
    return {
        "standards": standards,
        "row_overrides": value.get("row_overrides", {}),
        "standard_basis": value.get("standard_basis", {}),
        "standard_citations": value.get("standard_citations", {}),
        "row_override_basis": value.get("row_override_basis", {}),
        "row_override_citations": value.get("row_override_citations", {}),
    }


def _object_embedded_declaration(pack: Any) -> Mapping[str, Any] | None:
    embedded = (
        pack.get("liquid_standards")
        if isinstance(pack, Mapping)
        else getattr(pack, "liquid_standards", None)
    )
    if embedded is None:
        content = getattr(pack, "content", None)
        if isinstance(content, Mapping):
            embedded = content.get("liquid_standards")
    return _embedded_declaration(embedded)


def _path_pack(path: Path) -> tuple[dict[str, Any] | None, str, str | None]:
    raw = path.read_bytes()
    byte_digest = hashlib.sha256(raw).hexdigest()
    try:
        content = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, byte_digest, None
    if not isinstance(content, dict):
        return None, byte_digest, None

    if isinstance(content.get("schema"), str) and content["schema"].startswith(
        "openimcc-redox-pack."
    ):
        from openimcc.redox_pack import load_redox_pack

        loaded = load_redox_pack(path)
        return content, byte_digest, loaded.canonical_digest

    if isinstance(content.get("imcc_sf04_datapack_version"), str):
        from openimcc.model import load_datapack

        loaded = load_datapack(path)
        return content, loaded.binding_digest or byte_digest, None
    return content, byte_digest, None


def _registry_entry(
    registry: Mapping[str, Any], digest: str | None, canonical_digest: str | None
) -> tuple[str, Mapping[str, Any]] | None:
    entries = registry.get("packs")
    if not isinstance(entries, Mapping):
        return None
    if digest is not None and isinstance(entries.get(digest), Mapping):
        return digest, entries[digest]
    if canonical_digest is not None:
        for key, value in entries.items():
            if (
                isinstance(value, Mapping)
                and value.get("canonical_content_digest") == canonical_digest
            ):
                return str(key), value
    return None


def pack_liquid_standards(pack: Any) -> PackLiquidStandards:
    """Return embedded declarations first, then the content-bound registry entry.

    Melt paths use the loader's binding digest. Redox paths use their exact file
    SHA-256 because that loader exposes only a canonical content digest.
    """
    if isinstance(pack, (str, Path)):
        path = Path(pack)
        try:
            content, digest, canonical_digest = _path_pack(path)
        except OSError:
            return _from_declaration({}, source="undeclared", digest=None)
        embedded = _embedded_declaration(
            content.get("liquid_standards") if content is not None else None
        )
        if embedded is not None:
            return _from_declaration(embedded, source="embedded", digest=digest)
    else:
        embedded = _object_embedded_declaration(pack)
        digest_value = getattr(pack, "binding_digest", None)
        canonical_digest = getattr(pack, "canonical_digest", None)
        digest = digest_value if isinstance(digest_value, str) else None
        if embedded is not None:
            return _from_declaration(embedded, source="embedded", digest=digest)

    registry = _read_registry()
    found = _registry_entry(registry, digest, canonical_digest)
    if found is None:
        return _from_declaration({}, source="undeclared", digest=digest)
    key, declaration = found
    return _from_declaration(declaration, source="registry", digest=key)


def pack_row_liquid_standards(pack: Any, row_key: str) -> PackRowLiquidStandards:
    """Merge IDs and provenance for one exact row key.

    The returned mapping remains oxide-to-standard-id; its immutable
    ``standard_basis`` and ``standard_citations`` attributes describe those
    same merged declarations.
    """
    declared = pack_liquid_standards(pack)
    merged = dict(declared.standards)
    basis = dict(declared.standard_basis)
    citations = dict(declared.standard_citations)
    overrides = declared.row_overrides.get(str(row_key), {})
    merged.update(overrides)
    override_basis = declared.row_override_basis.get(str(row_key), {})
    override_citations = declared.row_override_citations.get(str(row_key), {})
    for oxide in overrides:
        basis.pop(oxide, None)
        citations.pop(oxide, None)
        if oxide in override_basis:
            basis[oxide] = override_basis[oxide]
        if oxide in override_citations:
            citations[oxide] = override_citations[oxide]
    return PackRowLiquidStandards(
        merged, standard_basis=basis, standard_citations=citations
    )


def active_liquid_standards(
    profile: str | Profile | None = None,
) -> dict[str, str]:
    """Return the liquid table identifiers active for a shipped profile."""
    from openimcc import gas
    from openimcc.liquid_reference import feo_liquid_standard
    from openimcc.profiles import _resolve_profile
    from openimcc.redox_pack import load_redox_pack

    selected = (
        _resolve_profile(profile)
        if profile is not None
        else _resolve_profile("openimcc")
    )
    gas_pack = selected.gas_datapack
    standards: dict[str, str] = {}
    for row_name, row in gas_pack.oxide_df.iterrows():
        name = str(row_name)
        if not name.endswith("(l)"):
            continue
        oxide = name[:-3]
        if "O" not in oxide:
            continue
        standard_id = row.get("table_id")
        if not isinstance(standard_id, str) or not standard_id or standard_id == "none":
            source_ref = str(row["Ref"])
            try:
                standard_id = gas._condensate_source_table_id(
                    oxide, "l", source_ref, gas_pack.oxide_df
                )
            except gas.ImccGasUnrecognizedSourceError:
                # The published profile names its LAM tables directly in Ref;
                # those table IDs are not aliases in the default-table map.
                if selected.name != "published":
                    raise
                standard_id = source_ref
        if not isinstance(standard_id, str) or not standard_id:
            raise ValueError(f"no source table id is available for {oxide}(l)")
        existing = standards.get(oxide)
        if existing is not None and existing != standard_id:
            raise ValueError(f"multiple liquid source tables are active for {oxide}")
        standards[oxide] = standard_id

    # FeO's selected source and authority are owned by liquid_reference.py.
    standards["FeO"] = feo_liquid_standard(selected).standard_id

    # FeO1.5(l) is half a metastable Fe2O3(l) construction. Fe-030 is its
    # hematite crystal basis, not a JANAF liquid table.
    redox_pack = (
        load_redox_pack(selected.redox_pack_path)
        if selected.redox_pack_path is not None
        else load_redox_pack()
    )
    species = redox_pack.content["thermo_species"]
    feo15 = species.get("FeO1.5(l)", {})
    fe2o3 = species.get("Fe2O3(l)", {})
    construction = fe2o3.get("construction", {})
    provenance = fe2o3.get("provenance", {})
    terms = feo15.get("terms", []) if isinstance(feo15, Mapping) else []
    if (
        isinstance(construction, Mapping)
        and construction.get("crystal_source_table") == "Fe-030"
        and construction.get("fusion_enthalpy_kj_mol") == 114.5
        and construction.get("fusion_temperature_K") == 1895
        and isinstance(provenance, Mapping)
        and "Lesher and Spera supplemental Table S5.5"
        in str(provenance.get("source", ""))
        and any(
            isinstance(term, Mapping)
            and term.get("species") == "Fe2O3(l)"
            and term.get("coefficient") == 0.5
            for term in terms
        )
    ):
        standards["FeO1.5"] = (
            "Fe2O3-l-construction(Fe-030 cr + Lesher-Spera fusion)/2"
        )

    # Extension parent references supplement oxide rows omitted from the gas
    # table without overriding profile-specific table identifiers.
    extension_path = (
        files("openimcc.data")
        .joinpath("packs")
        .joinpath("imcc-sf04-d066-ext-v1.json")
    )
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    reference_states = extension.get("sp_extension", {}).get(
        "parent_reference_states", {}
    )
    if isinstance(reference_states, Mapping):
        for parent, state in reference_states.items():
            if (
                str(parent) not in standards
                and isinstance(state, Mapping)
                and state.get("phase") == "liquid"
                and isinstance(state.get("table_id"), str)
            ):
                standards[str(parent)] = state["table_id"]

    return dict(sorted(standards.items()))


def _parent_names_from_content(content: Mapping[str, Any]) -> tuple[str, ...]:
    parent_registry = content.get("parent_registry")
    if isinstance(parent_registry, Sequence) and not isinstance(
        parent_registry, (str, bytes)
    ):
        return tuple(
            dict.fromkeys(
                str(item["parent_oxide"]).removesuffix("(l)")
                for item in parent_registry
                if isinstance(item, Mapping) and isinstance(item.get("parent_oxide"), str)
            )
        )

    parents = list(content.get("parents", ()))
    extension = content.get("sp_extension")
    if isinstance(extension, Mapping):
        parents.extend(extension.get("parents", ()))
    return tuple(dict.fromkeys(str(parent) for parent in parents))


def _parent_names(pack: Any) -> tuple[str, ...]:
    if isinstance(pack, (str, Path)):
        try:
            content = json.loads(Path(pack).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return ()
        return _parent_names_from_content(content) if isinstance(content, Mapping) else ()

    content = getattr(pack, "content", None)
    if isinstance(content, Mapping) and "parent_registry" in content:
        return _parent_names_from_content(content)
    parents = getattr(pack, "parent_oxides", None)
    if isinstance(parents, Sequence) and not isinstance(parents, (str, bytes)):
        return tuple(dict.fromkeys(str(parent) for parent in parents))
    if isinstance(content, Mapping):
        return _parent_names_from_content(content)
    return ()


def _comparison(
    oxide: str,
    declared: str,
    active: str | None,
    *,
    row_key: str | None,
    redox: bool = False,
) -> LiquidStandardMismatch | None:
    if declared == "reference-unknown":
        kind: StandardCheckKind = "reference-unknown"
    elif declared == "undeclared":
        kind = "undeclared"
    elif active is None or declared != active:
        kind = "redox-mismatch" if redox else "mismatch"
    else:
        return None
    return LiquidStandardMismatch(
        oxide=oxide,
        declared=declared,
        active=active,
        kind=kind,
        row_key=row_key,
    )


def _basis_records(
    standards: Mapping[str, str],
    bases: Mapping[str, StandardBasis],
    citations: Mapping[str, str],
    active: Mapping[str, str],
    *,
    row_key: str | None,
    require_basis: Literal["source"] | None,
) -> list[LiquidStandardMismatch]:
    records: list[LiquidStandardMismatch] = []
    for oxide, basis in bases.items():
        standard_id = standards.get(oxide)
        if standard_id is None or standard_id in {
            "undeclared",
            "reference-unknown",
        }:
            continue
        citation = citations.get(oxide)
        fields = {
            "oxide": oxide,
            "declared": standard_id,
            "active": active.get(oxide),
            "row_key": row_key,
            "standard_id": standard_id,
            "basis": basis,
            "citation": citation,
        }
        if basis != "source":
            records.append(
                LiquidStandardMismatch(kind="non-source-basis", **fields)
            )
            if require_basis == "source":
                records.append(
                    LiquidStandardMismatch(kind="basis-insufficient", **fields)
                )
    return records


def check_pack_standards(
    pack: Any,
    profile: str | Profile | None = None,
    *,
    redox_pack: Any = None,
    require_basis: Literal["source"] | None = None,
) -> list[LiquidStandardMismatch]:
    """Compare declarations with active tables and report basis provenance.

    Non-source basis records are informational. ``require_basis="source"``
    additionally reports each such declaration as basis-insufficient.
    """
    if require_basis not in {None, "source"}:
        raise ValueError("require_basis must be None or 'source'")
    active = active_liquid_standards(profile)
    declared = pack_liquid_standards(pack)
    output: list[LiquidStandardMismatch] = []

    for oxide, standard_id in declared.standards.items():
        record = _comparison(
            oxide, standard_id, active.get(oxide), row_key=None
        )
        if record is not None:
            output.append(record)

    output.extend(
        _basis_records(
            declared.standards,
            declared.standard_basis,
            declared.standard_citations,
            active,
            row_key=None,
            require_basis=require_basis,
        )
    )

    for row_key, overrides in declared.row_overrides.items():
        for oxide, standard_id in overrides.items():
            record = _comparison(
                oxide, standard_id, active.get(oxide), row_key=row_key
            )
            if record is not None:
                output.append(record)
        output.extend(
            _basis_records(
                overrides,
                declared.row_override_basis.get(row_key, {}),
                declared.row_override_citations.get(row_key, {}),
                active,
                row_key=row_key,
                require_basis=require_basis,
            )
        )

    for oxide in _parent_names(pack):
        if (
            oxide not in declared.standards
            and not any(oxide in values for values in declared.row_overrides.values())
        ):
            output.append(
                LiquidStandardMismatch(
                    oxide=oxide,
                    declared="undeclared",
                    active=active.get(oxide),
                    kind="undeclared",
                    row_key=None,
                )
            )

    if not _parent_names(pack) and declared.source == "undeclared":
        output.append(
            LiquidStandardMismatch(
                oxide="*",
                declared="undeclared",
                active=None,
                kind="undeclared",
                row_key=None,
            )
        )

    if redox_pack is not None:
        redox = pack_liquid_standards(redox_pack)
        for oxide, standard_id in redox.standards.items():
            record = _comparison(
                oxide,
                standard_id,
                active.get(oxide),
                row_key=None,
                redox=oxide == "FeO",
            )
            if record is not None:
                output.append(record)
        output.extend(
            _basis_records(
                redox.standards,
                redox.standard_basis,
                redox.standard_citations,
                active,
                row_key=None,
                require_basis=require_basis,
            )
        )
        if "FeO" not in redox.standards:
            output.append(
                LiquidStandardMismatch(
                    oxide="FeO",
                    declared="undeclared",
                    active=active.get("FeO"),
                    kind="undeclared",
                    row_key=None,
                )
            )

    return output


__all__ = [
    "LiquidStandardMismatch",
    "PackLiquidStandards",
    "active_liquid_standards",
    "check_pack_standards",
    "pack_liquid_standards",
    "pack_row_liquid_standards",
]
