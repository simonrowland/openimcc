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
from openimcc.profiles import ProfileError, load_profile


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
    assert {entry["file"] for entry in entries.values()} == {path.name for path in shipped}

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
        if entry["pack_type"] == "melt":
            feo_citation = entry["standard_citations"]["FeO"]
            if isinstance(raw.get("feo_liquid_candidate"), dict):
                assert entry["standards"]["FeO"] == "DG46-MASSON65"
                assert entry["standard_basis"]["FeO"] == "source"
                assert "Darken & Gurry" in feo_citation
            else:
                assert entry["standards"]["FeO"] == "Fe-019"
                assert entry["standard_basis"]["FeO"] == "operational"
                assert "Fe-019" in feo_citation

            for oxide, standard_id in {
                "SiO2": "SiO2-RBT84-CANDIDATE",
                "Na2O": "Na2O-RBT84-CANDIDATE",
            }.items():
                if oxide in raw.get("parent_standard_candidates", {}):
                    assert entry["standards"][oxide] == standard_id
                    assert entry["standard_basis"][oxide] == "source"
                    assert "10.1111/j.1151-2916.1984.tb19155.x" in entry["standard_citations"][oxide]

            if entry["standards"].get("K2O") == "K-012":
                assert entry["standard_basis"]["K2O"] == "translation-record"
                assert "JANAF K-012" in entry["standard_citations"]["K2O"]
            elif "K2O" in entry["standards"]:
                assert entry["standard_basis"]["K2O"] == "source"
                assert "Glushko/Gurvich" in entry["standard_citations"]["K2O"]

        if entry["pack_type"] == "redox":
            species = raw["thermo_species"]
            feo_url = species["FeO(l)"]["provenance"]["source_url"]
            expected_feo = "Fe-019" if "Fe-019.html" in feo_url else "DG46-MASSON65"
            assert entry["standards"]["FeO"] == expected_feo
            assert "thermo_species.FeO(l).provenance" in entry["standard_citations"]["FeO"]
            feo15_citation = entry["standard_citations"]["FeO1.5"]
            assert entry["standards"]["FeO1.5"] == (
                "Fe2O3-l-construction(Fe-030 cr + Lesher-Spera fusion)/2"
            )
            assert entry["standard_basis"]["FeO1.5"] == "source"
            assert "thermo_species.FeO1.5(l).terms" in feo15_citation
            assert "thermo_species.Fe2O3(l).construction" in feo15_citation
            assert "114.5 kJ/mol at 1895 K" in feo15_citation
            assert "Lesher and Spera supplemental Table S5.5" in feo15_citation
            assert "Fe-030.html" in feo15_citation
            assert "thermo_species.Al2O3(l).provenance" in entry["standard_citations"]["Al2O3"]
            si_url = species["SiO2(l)"]["provenance"]["source_url"]
            expected_silica = (
                "SiO2-RBT84-CANDIDATE"
                if "1151-2916.1984.tb19155" in si_url
                else "O-038"
            )
            assert entry["standards"]["SiO2"] == expected_silica

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
        "FeO": "DG46-MASSON65",
        "SiO2": "SiO2-RBT84-CANDIDATE",
        "Na2O": "Na2O-RBT84-CANDIDATE",
        "K2O": "K-012",
        "CaO": "Ca-028",
        "MgO": "Mg-009",
        "Al2O3": "Al-100",
        "TiO2": "O-044",
        "FeO1.5": "Fe2O3-l-construction(Fe-030 cr + Lesher-Spera fusion)/2",
        "MnO": "MnO-liquid-construction",
        "P2O5": "Jung-Hudon-2012",
        "S": "S-004",
    }.items() <= active.items()
    assert "Cr2O3" in active
    published = active_liquid_standards("published")
    assert published["FeO"] == "Fe-019"
    assert published["SiO2"] == "LAM1987"
    assert published["Na2O"] == "Na-013"
    assert active_liquid_standards(load_profile("published")) == published
    with pytest.raises(ProfileError):
        active_liquid_standards("other")


def test_d066_v1_mixed_standards_are_caught_and_v4_matches_default() -> None:
    v1_path = PACK_DIR / "imcc-sf04-d066-v1.json"
    v4_path = PACK_DIR / "imcc-sf04-d066-v4.json"
    v1_default = check_pack_standards(v1_path)
    v4_default = check_pack_standards(v4_path)
    v1_mismatches = {record.oxide for record in v1_default if record.kind == "mismatch"}
    assert {"FeO", "SiO2", "Na2O"} <= v1_mismatches
    assert not [record for record in v4_default if record.kind == "mismatch"]
    for name in ("imcc-sf04-d066-v5.json", "imcc-sf04-d066-feo-hess-candidate.json"):
        assert not [
            record
            for record in check_pack_standards(PACK_DIR / name)
            if record.kind == "mismatch"
        ], name

    v1_published = check_pack_standards(v1_path, "published")
    v4_published = check_pack_standards(v4_path, "published")
    assert not any(
        record.oxide == "FeO" and record.kind == "mismatch"
        for record in v1_published
    )
    assert any(
        record.oxide == "FeO"
        and record.declared == "DG46-MASSON65"
        and record.active == "Fe-019"
        and record.kind == "mismatch"
        for record in v4_published
    )


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
    assert "reference-unknown marker" in unknown_row.standard_citations["K2O"]
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
    assert redox_path_info == redox_loaded_info
    assert redox_loaded_info.standards["FeO"] == "Fe-019"


@pytest.mark.parametrize(
    "pack_name",
    ("imcc-sf04-d066-v1.json", "imcc-sf04-d066-ext-v1.json"),
)
def test_inactive_kcaalsi2o7_rows_have_unknown_k2o_reference(pack_name: str) -> None:
    row = pack_row_liquid_standards(PACK_DIR / pack_name, "KCaAlSi2O7")
    assert row["K2O"] == "reference-unknown"
    assert "KCaAlSi2O7" in row.standard_citations["K2O"]


def test_melt_path_and_loaded_object_declarations_match_for_every_pack() -> None:
    for path in sorted(PACK_DIR.glob("imcc-*.json")):
        if path.name == "MANIFEST.json":
            continue
        loaded = load_datapack(path)
        assert pack_liquid_standards(path) == pack_liquid_standards(loaded), path.name


def test_redox_path_and_loaded_object_checks_match_for_every_shipped_pack() -> None:
    for path in sorted(PACK_DIR.glob("openimcc-redox*.json")):
        loaded = load_redox_pack(path)
        path_info = pack_liquid_standards(path)
        loaded_info = pack_liquid_standards(loaded)
        assert path_info == loaded_info, path.name
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
    assert not any(
        record.oxide == "FeO" and record.kind == "non-source-basis"
        for record in records
    )
    assert any(
        record.oxide == "K2O"
        and record.standard_id == "K-012"
        and record.basis == "translation-record"
        and record.kind == "non-source-basis"
        for record in records
    )
    assert all(isinstance(record, LiquidStandardMismatch) for record in records)

    candidate = load_redox_pack(PACK_DIR / "openimcc-redox-cr-candidate.json")
    candidate_records = check_pack_standards(load_datapack(), redox_pack=candidate)
    assert any(
        record.kind == "redox-mismatch"
        and record.oxide == "FeO"
        and record.declared == "Fe-019"
        and record.active == "DG46-MASSON65"
        for record in candidate_records
    )
    assert any(
        record.kind == "mismatch"
        and record.oxide == "SiO2"
        and record.declared == "O-038"
        and record.active == "SiO2-RBT84-CANDIDATE"
        for record in candidate_records
    )
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
    redox_path = PACK_DIR / "openimcc-redox-v4.json"
    redox_digest = hashlib.sha256(
        redox_path.read_bytes()
    ).hexdigest()
    entries[melt_digest]["standards"]["MgO"] = "synthetic-table"
    entries[redox_digest]["standards"]["FeO"] = "synthetic-redox-table"
    monkeypatch.setattr(standards_module, "_read_registry", lambda: registry)

    records = check_pack_standards(
        load_datapack(), redox_pack=load_redox_pack(redox_path)
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
        active="DG46-MASSON65",
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
