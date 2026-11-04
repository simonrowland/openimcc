"""Content-bound liquid-standard declarations and public lookup API tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import openimcc.standards as standards_module
from openimcc import (
    LiquidStandardMismatch,
    active_liquid_standards,
    check_pack_standards,
    load_datapack,
    pack_liquid_standards,
    pack_row_liquid_standards,
)
from openimcc.redox_pack import load_redox_pack


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "src/openimcc/data/packs"
REGISTRY_PATH = ROOT / "src/openimcc/data/liquid-standards.json"
UNKNOWN_BAND = "not sized (Glushko basis function not available at this commit)"


def _registry() -> dict[str, object]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_registry_schema_and_every_shipped_pack_has_a_declaration() -> None:
    registry = _registry()
    assert registry["schema_version"] == 1
    assert registry["identity"] == {
        "melt_key": "ImccLoadedDatapack.binding_digest",
        "redox_key": "SHA-256 of exact pack file bytes",
        "redox_loaded_lookup": "canonical_content_digest recorded alongside the file SHA-256",
    }
    assert "complex" in registry["row_key"]
    entries = registry["packs"]
    shipped = sorted(path for path in PACK_DIR.glob("*.json") if path.name != "MANIFEST.json")
    assert len(shipped) == 9

    for path in shipped:
        pack = pack_liquid_standards(path)
        assert pack.source == "registry"
        assert pack.pack_digest in entries
        entry = entries[pack.pack_digest]
        assert entry["file"] == path.name
        assert entry["model_id"]
        assert entry["version"]
        assert dict(pack.standards) == entry["standards"]
        assert {key: dict(value) for key, value in pack.row_overrides.items()} == entry["row_overrides"]
        assert set(entry["standard_citations"]) == set(entry["standards"])
        declared_ids = {
            oxide for oxide, standard_id in entry["standards"].items()
            if standard_id != "undeclared"
        }
        assert set(entry["standard_basis"]) == declared_ids
        assert set(pack.standard_basis) == declared_ids
        assert dict(pack.standard_citations) == entry["standard_citations"]
        assert all(
            basis in {"source", "operational", "translation-record"}
            for basis in entry["standard_basis"].values()
        )
        assert all(entry["standard_citations"][oxide] for oxide in declared_ids)
        assert set(entry["row_override_basis"]) <= set(entry["row_overrides"])
        assert all(
            basis in {"source", "operational", "translation-record"}
            for rows in entry["row_override_basis"].values()
            for basis in rows.values()
        )
        assert {
            row: dict(values) for row, values in pack.row_override_basis.items()
        } == entry["row_override_basis"]
        assert {
            row: dict(values) for row, values in pack.row_override_citations.items()
        } == entry["row_override_citations"]
        assert all(
            declaration["standard_id"] == "undeclared"
            for declaration in entry["undeclared"].values()
        )
        for oxide, declaration in entry["undeclared"].items():
            assert entry["standards"][oxide] == "undeclared"
            assert entry["standard_citations"][oxide] == declaration["citation"]

        if entry.get("declaration_status") == "candidate-unresolved":
            assert entry["undeclared_reason"]
            assert entry["undeclared"]
        else:
            assert entry["standards"] or entry["row_overrides"]

        raw = json.loads(path.read_text(encoding="utf-8"))
        if entry["file"] in {
            "imcc-sf04-v1.0.2.json",
            "imcc-sf04-ext-v4.json",
            "imcc-sf04-d066-v1.json",
            "imcc-sf04-d066-v1-kcaalsi2o7.json",
            "imcc-sf04-d066-ext-v1.json",
        }:
            feo_citation = entry["standard_citations"]["FeO"]
            assert entry["standards"]["FeO"] == "Fe-019"
            assert entry["standard_basis"]["FeO"] == "operational"
            assert "FC87/SF04 derivation" in feo_citation
            assert "later openimcc D066 pack version, not present at this commit" in feo_citation
            assert "re-anchored the Fe-complex rows" in feo_citation
            assert "iron-saturated FeO(l) standard" in feo_citation
            assert "translated onto the Fe-019 reference" not in feo_citation
            assert "gas.py:1495" in feo_citation
            assert "condensate.csv:5" in feo_citation
            nonzero_feo_rows = [
                row for row in raw.get("rows", [])
                if float(row.get("nu", {}).get("FeO", 0)) != 0
            ]
            assert all(row["complex"] in feo_citation for row in nonzero_feo_rows)
            if entry["file"] in {
                "imcc-sf04-ext-v4.json",
                "imcc-sf04-d066-ext-v1.json",
            }:
                assert "Fe-024" in feo_citation
                assert "nu FeO=1" in feo_citation
                assert "FeO parent Fe-019" in feo_citation

        if entry["file"].startswith("imcc-sf04-"):
            assert ":1855" in entry["standard_citations"]["K2O"]

        if entry["pack_type"] == "redox":
            assert ":775" in entry["standard_citations"]["FeO1.5"]
            assert "redox_pack.py" not in entry["standard_citations"]["FeO1.5"]
            assert ":1881" in entry["standard_citations"]["SiO2"]
            assert ":1209" in entry["standard_citations"]["Al2O3"]

        row_keys = {row["complex"] for row in raw.get("rows", [])}
        if entry["pack_type"] == "melt":
            parents = list(raw.get("parents", []))
            parents.extend(raw.get("sp_extension", {}).get("parents", []))
            assert set(parents) <= set(entry["standards"]) | set(entry["undeclared"])
        for row_key, overrides in entry["row_overrides"].items():
            assert row_key in row_keys
            assert set(entry["row_override_citations"][row_key]) == set(overrides)
            if "reference-unknown" in overrides.values():
                diagnostic = entry["row_diagnostics"][row_key]["diagnostic_delta_g_j_mol"]
                assert diagnostic["band"] == UNKNOWN_BAND
                for part in ("Premise:", "Algebra:", "Unit check:", "Sanity check:"):
                    assert part in diagnostic["derivation_comment"]

        if path.name.startswith("imcc-"):
            loaded = load_datapack(path)
            assert loaded.binding_digest == pack.pack_digest
        else:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == pack.pack_digest
            assert load_redox_pack(path).canonical_digest == entry["canonical_content_digest"]


def test_active_standards_include_all_eight_melt_parents_and_other_active_oxides() -> None:
    active = active_liquid_standards()
    assert {
        "FeO": "Fe-019",
        "SiO2": "O-038",
        "Na2O": "Na-013",
        "K2O": "K-012",
        "CaO": "Ca-028",
        "MgO": "Mg-009",
        "Al2O3": "Al-100",
        "TiO2": "O-044",
        "FeO1.5": "Fe-030/2",
        "MnO": "MnO-liquid-construction",
        "P2O5": "Jung-Hudon-2012",
        "S": "S-004",
    }.items() <= active.items()
    assert "Cr2O3" in active
    with pytest.raises(ValueError, match="profiles are not defined"):
        active_liquid_standards("other")


def test_loaded_objects_paths_and_row_overrides_resolve_to_registry() -> None:
    default_path = PACK_DIR / "imcc-sf04-d066-v1.json"
    default_loaded = load_datapack(default_path)
    path_info = pack_liquid_standards(default_path)
    loaded_info = pack_liquid_standards(default_loaded)
    assert path_info.pack_digest == loaded_info.pack_digest
    assert dict(path_info.standards) == dict(loaded_info.standards)
    assert dict(path_info.standard_basis) == dict(loaded_info.standard_basis)
    assert dict(path_info.standard_citations) == dict(loaded_info.standard_citations)
    translated_row = pack_row_liquid_standards(default_loaded, "KAlSiO4")
    assert translated_row["K2O"] == "K-012"
    assert translated_row.standard_basis["K2O"] == "translation-record"
    unknown_row = pack_row_liquid_standards(default_loaded, "K2SiO3")
    assert unknown_row["K2O"] == "reference-unknown"
    assert "K2O" not in unknown_row.standard_basis
    assert "reference_unknown marker" in unknown_row.standard_citations["K2O"]
    with pytest.raises(TypeError):
        path_info.standards["FeO"] = "other"
    with pytest.raises(TypeError):
        path_info.row_overrides["K2SiO3"]["K2O"] = "K-012"
    with pytest.raises(TypeError):
        path_info.standard_basis["FeO"] = "source"

    redox_path = PACK_DIR / "openimcc-redox-v1.json"
    redox_loaded = load_redox_pack(redox_path)
    redox_path_info = pack_liquid_standards(redox_path)
    redox_loaded_info = pack_liquid_standards(redox_loaded)
    assert redox_path_info.pack_digest == redox_loaded_info.pack_digest
    assert redox_loaded_info.standards["FeO"] == "Fe-019"


@pytest.mark.parametrize(
    "pack_name",
    ("imcc-sf04-d066-v1.json", "imcc-sf04-d066-ext-v1.json"),
)
def test_inactive_kcaalsi2o7_rows_have_unknown_k2o_reference(pack_name: str) -> None:
    row = pack_row_liquid_standards(PACK_DIR / pack_name, "KCaAlSi2O7")
    assert row["K2O"] == "reference-unknown"
    assert "2237" in row.standard_citations["K2O"]


def test_redox_path_and_loaded_object_checks_match_for_every_shipped_pack() -> None:
    for path in sorted(PACK_DIR.glob("openimcc-redox*.json")):
        loaded = load_redox_pack(path)
        path_info = pack_liquid_standards(path)
        loaded_info = pack_liquid_standards(loaded)
        assert path_info.pack_digest == loaded_info.pack_digest, path.name
        assert check_pack_standards(path) == check_pack_standards(loaded), path.name


def test_default_melt_and_redox_tables_have_no_declared_mismatches() -> None:
    records = check_pack_standards(
        load_datapack(), redox_pack=load_redox_pack()
    )
    assert not [
        record for record in records
        if record.kind in {"mismatch", "redox-mismatch"}
    ]
    assert any(record.kind == "reference-unknown" for record in records)
    assert not any(record.kind == "undeclared" for record in records)
    assert any(
        record.oxide == "FeO"
        and record.standard_id == "Fe-019"
        and record.basis == "operational"
        and record.kind == "non-source-basis"
        for record in records
    )
    assert all(isinstance(record, LiquidStandardMismatch) for record in records)

    candidate = load_redox_pack(PACK_DIR / "openimcc-redox-cr-candidate.json")
    candidate_records = check_pack_standards(load_datapack(), redox_pack=candidate)
    assert not any(record.kind == "redox-mismatch" for record in candidate_records)
    assert any(
        record.oxide == "Cr2O3" and record.kind == "undeclared"
        for record in candidate_records
    )


def test_active_feo_table_change_is_a_melt_mismatch(monkeypatch) -> None:
    active = dict(active_liquid_standards())
    active["FeO"] = "DG46-FeO(l)"
    monkeypatch.setattr(
        standards_module, "active_liquid_standards", lambda profile=None: active
    )

    records = check_pack_standards(PACK_DIR / "imcc-sf04-d066-v1.json")
    assert LiquidStandardMismatch(
        oxide="FeO",
        declared="Fe-019",
        active="DG46-FeO(l)",
        kind="mismatch",
        row_key=None,
    ) in records


def test_source_requirement_reports_operational_basis_as_insufficient() -> None:
    pack = PACK_DIR / "imcc-sf04-d066-v1.json"
    permissive = check_pack_standards(pack)
    assert not any(record.kind == "basis-insufficient" for record in permissive)

    records = check_pack_standards(pack, require_basis="source")
    assert any(
        record.kind == "non-source-basis"
        and record.oxide == "FeO"
        and record.standard_id == "Fe-019"
        and record.basis == "operational"
        and record.citation
        and record.row_key is None
        for record in records
    )
    assert any(
        record.kind == "basis-insufficient"
        and record.oxide == "FeO"
        and record.standard_id == "Fe-019"
        and record.basis == "operational"
        and record.citation
        and record.row_key is None
        for record in records
    )


def test_synthetic_melt_and_redox_mismatches_are_reported(monkeypatch) -> None:
    registry = copy.deepcopy(_registry())
    entries = registry["packs"]
    melt_digest = load_datapack().binding_digest
    redox_digest = hashlib.sha256(
        (PACK_DIR / "openimcc-redox-v1.json").read_bytes()
    ).hexdigest()
    entries[melt_digest]["standards"]["MgO"] = "synthetic-table"
    entries[redox_digest]["standards"]["FeO"] = "synthetic-redox-table"
    monkeypatch.setattr(standards_module, "_read_registry", lambda: registry)

    records = check_pack_standards(
        load_datapack(), redox_pack=load_redox_pack()
    )
    assert LiquidStandardMismatch(
        oxide="MgO",
        declared="synthetic-table",
        active="Mg-009",
        kind="mismatch",
        row_key=None,
    ) in records
    assert LiquidStandardMismatch(
        oxide="FeO",
        declared="synthetic-redox-table",
        active="Fe-019",
        kind="redox-mismatch",
        row_key=None,
    ) in records


def test_embedded_declarations_precede_the_registry_and_unknown_packs_stay_unknown() -> None:
    loaded = load_datapack()
    embedded = SimpleNamespace(
        binding_digest=loaded.binding_digest,
        liquid_standards={
            "standards": {"FeO": "embedded-table"},
            "standard_basis": {"FeO": "operational"},
            "standard_citations": {"FeO": "embedded source text"},
            "row_overrides": {"row-x": {"FeO": "reference-unknown"}},
            "row_override_citations": {"row-x": {"FeO": "row source text"}},
        },
    )
    info = pack_liquid_standards(embedded)
    assert info.source == "embedded"
    assert info.standards == {"FeO": "embedded-table"}
    assert info.standard_basis == {"FeO": "operational"}
    assert info.standard_citations == {"FeO": "embedded source text"}
    assert dict(info.row_overrides["row-x"]) == {"FeO": "reference-unknown"}
    embedded_row = pack_row_liquid_standards(embedded, "row-x")
    assert embedded_row["FeO"] == "reference-unknown"
    assert embedded_row.standard_citations["FeO"] == "row source text"

    unknown = pack_liquid_standards(SimpleNamespace(binding_digest="unknown-digest"))
    assert unknown.source == "undeclared"
    assert not unknown.standards
    assert not unknown.row_overrides

    unknown_redox = pack_liquid_standards(
        SimpleNamespace(canonical_digest="unknown-redox-digest")
    )
    assert unknown_redox.source == "undeclared"
    assert unknown_redox.pack_digest is None
