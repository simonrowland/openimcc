"""Content-bound liquid-standard declarations and public lookup API tests."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import openimcc.standards as standards_module
from openimcc import (
    ImccLiquidStandardMismatchError,
    LiquidStandardMismatch,
    PackStandardNotice,
    active_liquid_standards,
    check_pack_standards,
    evaluate,
    load_datapack,
    pack_liquid_standards,
    pack_row_liquid_standards,
)
from openimcc.gas import (
    R_J_MOL_K,
    evaluate_gas,
    evaluate_gas_saturation,
    load_gas_datapack,
    species_thermo,
)
from openimcc.redox import evaluate_redox
from openimcc.redox_pack import canonical_redox_digest, load_redox_pack
from openimcc.profiles import ProfileError, load_profile


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "src/openimcc/data/packs"
REGISTRY_PATH = ROOT / "src/openimcc/data/liquid-standards.json"
K2O_REFERENCE_ROWS = ("K2SiO3", "K2Si2O5", "K2Si4O9", "KCaAlSi2O7")
K2O_DIAGNOSTIC_TEMPERATURES_K = (1323, 1500, 1700, 2000, 2400)
K2O_DIAGNOSTIC_LABEL = (
    "basis unconfirmed: ONE-SIDED FC87/Glushko candidate; JANAF-3 K2O(l) "
    "values unavailable from a verified primary table (Hastie & Bonnell 1985 "
    "anchor pending)"
)


def _registry() -> dict[str, object]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_registry_schema_and_every_shipped_pack_has_a_declaration() -> None:
    registry = _registry()
    assert registry["schema_version"] == 2
    assert registry["identity"] == {
        "melt_key": "ImccLoadedDatapack.binding_digest",
        "redox_key": "SHA-256 of exact pack file bytes",
        "redox_loaded_lookup": "canonical_content_digest recorded alongside the file SHA-256",
    }
    assert "best available" in registry["basis"]
    assert "never applied" in registry["diagnostic_policy"]
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
            basis
            in {"source", "operational", "translation-record", "best available"}
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
                diagnostic = entry["row_diagnostics"][row_key]
                delta_g = diagnostic["diagnostic_delta_g_j_mol"]
                if entry["file"].startswith(("imcc-sf04-d066-", "imcc-sf04-ext-")):
                    assert row_key in K2O_REFERENCE_ROWS
                    assert delta_g["units"] == "J/mol"
                    assert delta_g["temperatures_K"] == list(
                        K2O_DIAGNOSTIC_TEMPERATURES_K
                    )
                    assert delta_g["one_sided"] is True
                    assert set(delta_g["components"]) == {"FC87/Glushko"}
                    assert delta_g["envelope"] == pytest.approx(
                        [
                            abs(value)
                            for value in delta_g["components"]["FC87/Glushko"]
                        ]
                    )
                    assert diagnostic["diagnostic_label"] == K2O_DIAGNOSTIC_LABEL
                    assert (
                        diagnostic["equivalent_delta_log10_K"]["units"]
                        == "log10 K"
                    )
                    for part in (
                        "Premise:",
                        "Algebra:",
                        "Unit check:",
                        "Sanity check:",
                    ):
                        assert part in delta_g["derivation_comment"]
                else:
                    assert delta_g["band"] == (
                        "not sized (Glushko basis function not available at this commit)"
                    )
                    for part in ("Premise:", "Algebra:", "Unit check:", "Sanity check:"):
                        assert part in delta_g["derivation_comment"]

        if path.name.startswith("imcc-"):
            loaded = load_datapack(path)
            assert loaded.binding_digest == pack.pack_digest
        else:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == pack.pack_digest
            assert load_redox_pack(path).canonical_digest == entry["canonical_content_digest"]


def test_active_standards_include_all_eight_melt_parents_and_other_active_oxides() -> None:
    active = active_liquid_standards()
    assert active == {
        "Ag2O": "P672-Ag2O-c",
        "Al2O3": "Al-100",
        "As2O3": "P672-As2O3",
        "B2O3": "B-096",
        "BaO": "Ba-035",
        "BeO": "NG-BeO-L-1352",
        "Bi2O3": "P672-T82",
        "CaO": "Ca-028",
        "CdO": "P672-T102",
        "Ce2O3": "Konings-Table88-Tf2512",
        "CeO2": "Konings-Table88-Tf3083",
        "CoO": "TSIV-1334",
        "Cr2O3": "Cr-015",
        "Cs2O": "NG-1842",
        "Cu2O": "Cu-020",
        "Dy2O3": "Konings-Table88-Tf2680",
        "Er2O3": "Konings-Table88-Tf2690",
        "Eu2O3": "Konings-Table88-Tf2622",
        "FeO": "DG46-MASSON65",
        "FeO1.5": "Fe2O3-l-construction(Fe-030 cr + Lesher-Spera fusion)/2",
        "Ga2O3": "NG-12403",
        "Gd2O3": "Konings-Table88-Tf2693",
        "GeO2": "NG-12434",
        "HfO2": "P672-HfO2-c",
        "Ho2O3": "Konings-Table88-Tf2686",
        "In2O3": "NG-12662",
        "K2O": "K-012",
        "La2O3": "Konings-Table88-Tf2577",
        "Li2O": "Li-015",
        "Lu2O3": "Konings-Table88-Tf2762",
        "MgO": "Mg-009",
        "MnO": "MnO-liquid-construction",
        "MoO3": "Mo-015",
        "Na2O": "Na2O-RBT84-CANDIDATE",
        "NbO2": "Nb-" "013",
        "Nd2O3": "Konings-Table88-Tf2577",
        "NiO": "TSIV-2026",
        "P2O5": "Jung-Hudon-2012",
        "PbO": "O-007",
        "Pr2O3": "Konings-Table88-Tf2583",
        "Rb2O": "NG-1841",
        "S": "S-004",
        "Sb2O3": "P672-Sb2O3-c",
        "Sc2O3": "NG-1857",
        "SeO2": "P672-SeO2",
        "SiO2": "SiO2-RBT84-CANDIDATE",
        "Sm2O3": "Konings-Table88-Tf2613",
        "SnO": "NG-1848",
        "SrO": "O-014",
        "Ta2O5": "O-078",
        "Tb2O3": "Konings-Table88-Tf2682",
        "TeO2": "P672-TeO2-l",
        "ThO2": "P672-ThO2-c",
        "TiO2": "O-044",
        "Tl2O": "P672-Tl2O-l",
        "Tm2O3": "Konings-Table88-Tf2682",
        "UO2": "NG-1977",
        "V2O3": "O-063",
        "WO3": "O-066",
        "Y2O3": "P672-Y2O3-c",
        "Yb2O3": "Konings-Table88-Tf2707",
        "ZnO": "TSIV-1603",
        "ZrO2": "O-053",
    }
    published = active_liquid_standards("published")
    assert published == {
        "Al2O3": "LAM1987",
        "CaO": "LAM1987",
        "Cr2O3": "Cr-015",
        "FeO": "Fe-019",
        "FeO1.5": "Fe2O3-l-construction(Fe-030 cr + Lesher-Spera fusion)/2",
        "K2O": "LAM1984",
        "Li2O": "Li-015",
        "MgO": "LAM1987",
        "Na2O": "Na-013",
        "NbO2": "Nb-" "013",
        "PbO": "O-007",
        "Rb2O": "NG-1841",
        "S": "S-004",
        "SiO2": "LAM1987",
        "TiO2": "O-044",
        "V2O3": "O-063",
    }
    assert active_liquid_standards(load_profile("published")) == published
    with pytest.raises(ProfileError):
        active_liquid_standards("other")


@pytest.mark.parametrize(
    "published_profile_kind",
    ("named", "object"),
    ids=("published-name", "published-object"),
)
def test_custom_condensate_unknown_ref_is_reported_for_only_its_oxide(
    tmp_path: Path,
    published_profile_kind: str,
) -> None:
    gas_dir = ROOT / "src/openimcc/data/gas"
    condensate_path = tmp_path / "condensate.csv"
    shutil.copyfile(gas_dir / "condensate.csv", condensate_path)
    with condensate_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    changed = False
    for row in rows:
        if row["species_name"] == "MgO(l)":
            row["Ref"] = "NOT-A-TABLE"
            row["validity_kind"] = "none"
            row["continuation_kind"] = "none"
            row["cp_extension"] = "false"
            row["construction_method"] = "none"
            row["table_id"] = "none"
            row["source_label"] = "none"
            row["source_liquid_marker"] = "false"
            row["supercooled_continuation"] = "false"
            for column in (
                "source_fit_min_K",
                "source_fit_max_K",
                "source_limit_K",
                "fusion_temperature_K",
                "supercooled_segment_min_K",
                "supercooled_segment_max_K",
                "construction_tier",
                "glass_join_temperature_K",
            ):
                row[column] = "0"
            changed = True
    assert changed
    with condensate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    custom_gas_pack = load_gas_datapack(
        gas_path=gas_dir / "gas-shomate.csv", oxide_path=condensate_path
    )
    default = active_liquid_standards()
    active = active_liquid_standards(gas_pack=custom_gas_pack)
    marker = active["MgO"]
    assert marker.startswith("reference-unknown(Ref=")
    assert "NOT-A-TABLE" in marker
    assert {oxide: value for oxide, value in active.items() if oxide != "MgO"} == {
        oxide: value for oxide, value in default.items() if oxide != "MgO"
    }

    records = check_pack_standards(
        PACK_DIR / "imcc-sf04-d066-v4.json", gas_pack=custom_gas_pack
    )
    mg_records = [record for record in records if record.oxide == "MgO"]
    unknown_records = [
        record for record in mg_records if record.kind == "reference-unknown"
    ]
    assert len(unknown_records) == 1
    assert unknown_records[0].active == marker
    assert unknown_records[0].row_key is None
    assert not any(record.kind == "mismatch" for record in mg_records)

    if published_profile_kind == "named":
        published_profile = "published"
        published_gas_pack = custom_gas_pack
    else:
        published_profile = replace(
            load_profile("published"), gas_datapack=custom_gas_pack
        )
        published_gas_pack = None
    published_active = active_liquid_standards(
        published_profile, gas_pack=published_gas_pack
    )
    published_records = check_pack_standards(
        PACK_DIR / "imcc-sf04-d066-v4.json",
        published_profile,
        gas_pack=published_gas_pack,
    )
    published_mg_records = [
        record for record in published_records if record.oxide == "MgO"
    ]
    published_unknown_records = [
        record
        for record in published_mg_records
        if record.kind == "reference-unknown"
    ]
    assert (
        published_active["MgO"],
        [record.kind for record in published_mg_records],
        [record.active for record in published_unknown_records],
    ) == (marker, ["reference-unknown"], [marker])


def test_declared_standard_without_an_active_liquid_row_is_not_active(
    tmp_path: Path,
) -> None:
    gas_dir = ROOT / "src/openimcc/data/gas"
    condensate_path = tmp_path / "condensate.csv"
    shutil.copyfile(gas_dir / "condensate.csv", condensate_path)
    with condensate_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row["species_name"] != "TiO2(l)"]

    with condensate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    custom_gas_pack = load_gas_datapack(
        gas_path=gas_dir / "gas-shomate.csv", oxide_path=condensate_path
    )
    default_active = active_liquid_standards()
    custom_active = active_liquid_standards(gas_pack=custom_gas_pack)
    assert "TiO2" not in custom_active
    assert {oxide: value for oxide, value in custom_active.items() if oxide != "TiO2"} == {
        oxide: value for oxide, value in default_active.items() if oxide != "TiO2"
    }

    pack_path = PACK_DIR / "imcc-sf04-d066-v4.json"
    default_records = check_pack_standards(pack_path)
    custom_records = check_pack_standards(pack_path, gas_pack=custom_gas_pack)
    ti_records = [record for record in custom_records if record.oxide == "TiO2"]
    assert len(ti_records) == 1
    assert ti_records[0].declared == "O-044"
    assert ti_records[0].active is None
    assert ti_records[0].kind == "not-active"
    assert not any(record.kind == "mismatch" for record in ti_records)
    assert [record for record in custom_records if record.oxide != "TiO2"] == [
        record for record in default_records if record.oxide != "TiO2"
    ]


def test_redox_declaration_without_an_active_liquid_row_is_not_active(
    monkeypatch,
) -> None:
    active = dict(active_liquid_standards())
    active.pop("FeO")
    monkeypatch.setattr(
        standards_module,
        "active_liquid_standards",
        lambda profile=None, *, gas_pack=None: active,
    )

    redox_pack = load_redox_pack(PACK_DIR / "openimcc-redox-v4.json")
    records = check_pack_standards(
        SimpleNamespace(binding_digest="unknown-melt"), redox_pack=redox_pack
    )
    feo_records = [
        record for record in records if record.oxide == "FeO" and record.row_key is None
    ]
    assert len(feo_records) == 1
    assert feo_records[0].declared == pack_liquid_standards(redox_pack).standards["FeO"]
    assert feo_records[0].active is None
    assert feo_records[0].kind == "not-active"


def test_two_recognized_condensate_tables_for_one_oxide_still_raise(
    tmp_path: Path,
) -> None:
    gas_dir = ROOT / "src/openimcc/data/gas"
    condensate_path = tmp_path / "condensate.csv"
    shutil.copyfile(gas_dir / "condensate.csv", condensate_path)
    with condensate_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    changed = False
    for row in rows:
        if row["species_name"] == "MgO(l)" and row["Ref"] != "Mg-009":
            row["Ref"] = "Mg-008"
            row["table_id"] = "Mg-008"
            row["source_label"] = "Mg-008"
            changed = True
            break
    assert changed
    with condensate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    custom_gas_pack = load_gas_datapack(
        gas_path=gas_dir / "gas-shomate.csv", oxide_path=condensate_path
    )
    with pytest.raises(
        ValueError, match="multiple liquid source tables are active for MgO"
    ):
        active_liquid_standards(gas_pack=custom_gas_pack)


def test_d066_v1_mixed_standards_are_caught_and_v4_matches_default() -> None:
    v1_path = PACK_DIR / "imcc-sf04-d066-v1.json"
    v4_path = PACK_DIR / "imcc-sf04-d066-v4.json"
    v1_default = check_pack_standards(v1_path)
    v4_default = check_pack_standards(v4_path)
    v1_mismatches = {record.oxide for record in v1_default if record.kind == "mismatch"}
    assert v1_mismatches == {"FeO", "SiO2", "Na2O"}
    assert not [record for record in v4_default if record.kind == "mismatch"]
    for name in (
        "imcc-sf04-d066-v5.json",
        "imcc-sf04-d066-feo-hess-candidate.json",
    ):
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


def test_gas_evaluators_report_pack_standard_notices_and_strict_refusals() -> None:
    from openimcc.profiles import Profile

    default_profile = load_profile("openimcc")
    v1_profile = replace(
        default_profile,
        melt_pack=load_datapack(PACK_DIR / "imcc-sf04-d066-v1.json"),
    )
    assert isinstance(v1_profile, Profile)

    v1_gas = evaluate_gas(
        {"SiO2": 1.0},
        2000.0,
        1.0e-8,
        profile=v1_profile,
        gas_species=("SiO",),
    )
    v1_mismatch_notices = tuple(
        notice
        for notice in v1_gas.notices
        if isinstance(notice, PackStandardNotice) and notice.kind == "mismatch"
    )
    assert {notice.oxide for notice in v1_mismatch_notices} == {
        "FeO",
        "SiO2",
        "Na2O",
    }
    expected = {
        (record.oxide, record.declared, record.active, record.row_key)
        for record in check_pack_standards(
            v1_profile.melt_pack, gas_pack=default_profile.gas_datapack
        )
        if record.kind == "mismatch"
    }
    assert {
        (
            notice.oxide,
            notice.declared_standard_id,
            notice.active_standard_id,
            notice.row_key,
        )
        for notice in v1_mismatch_notices
    } == expected
    assert all(isinstance(notice, PackStandardNotice) for notice in v1_mismatch_notices)
    with pytest.raises(ImccLiquidStandardMismatchError, match="FeO") as refused:
        evaluate_gas(
            {"SiO2": 1.0},
            2000.0,
            1.0e-8,
            profile=v1_profile,
            gas_species=("SiO",),
            strict_pack_standards=True,
        )
    assert {record.oxide for record in refused.value.mismatches} == {
        "FeO",
        "SiO2",
        "Na2O",
    }
    with pytest.raises(ImccLiquidStandardMismatchError):
        evaluate_gas_saturation(
            {"CaO": 1.0},
            1800.0,
            profile=v1_profile,
            parent_oxides=("CaO",),
            strict_pack_standards=True,
        )

    v4_gas = evaluate_gas(
        {"SiO2": 1.0},
        2000.0,
        1.0e-8,
        profile=default_profile,
        gas_species=("SiO",),
        strict_pack_standards=True,
    )
    v4_pack_notices = tuple(
        notice for notice in v4_gas.notices if isinstance(notice, PackStandardNotice)
    )
    correctness_kinds = {"mismatch", "redox-mismatch", "undeclared", "not-active"}
    assert not {notice.kind for notice in v4_pack_notices} & correctness_kinds
    assert all(isinstance(notice, PackStandardNotice) for notice in v4_pack_notices)
    expected_unknown = {
        (record.oxide, record.declared, record.active, record.row_key)
        for record in check_pack_standards(
            default_profile.melt_pack, gas_pack=default_profile.gas_datapack
        )
        if record.kind == "reference-unknown"
    }
    actual_unknown = {
        (
            notice.oxide,
            notice.declared_standard_id,
            notice.active_standard_id,
            notice.row_key,
        )
        for notice in v4_pack_notices
        if notice.kind == "reference-unknown"
    }
    assert len(expected_unknown) == 4
    assert {item[0] for item in expected_unknown} == {"K2O"}
    assert actual_unknown == expected_unknown
    assert len(v4_pack_notices) == 4

    v4_saturation = evaluate_gas_saturation(
        {"CaO": 1.0},
        1800.0,
        profile=default_profile,
        parent_oxides=("CaO",),
        strict_pack_standards=True,
    )
    saturation_pack_notices = tuple(
        notice
        for notice in v4_saturation.notices
        if isinstance(notice, PackStandardNotice)
    )
    assert all(isinstance(notice, PackStandardNotice) for notice in saturation_pack_notices)
    assert {
        (
            notice.oxide,
            notice.declared_standard_id,
            notice.active_standard_id,
            notice.row_key,
        )
        for notice in saturation_pack_notices
    } == expected_unknown
    assert len(saturation_pack_notices) == 4
    assert not {notice.kind for notice in saturation_pack_notices} & correctness_kinds


def test_gas_saturation_rejects_datapack_with_profile() -> None:
    from openimcc.profiles import ProfileConflictError

    profile = load_profile("openimcc")
    with pytest.raises(ProfileConflictError, match="cannot be combined"):
        evaluate_gas_saturation(
            {}, 1800.0, profile.gas_datapack, profile=profile
        )


def test_evaluator_standard_check_cache_uses_pack_and_table_identity(monkeypatch) -> None:
    profile = load_profile("openimcc")
    original_check = standards_module.check_pack_standards
    calls = 0

    def counted_check(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_check(*args, **kwargs)

    monkeypatch.setattr(standards_module, "_STANDARD_CHECK_CACHE", {})
    monkeypatch.setattr(standards_module, "check_pack_standards", counted_check)
    activities = {"CaO": 1.0}
    for _ in range(2):
        evaluate_gas(
            activities,
            1800.0,
            1.0e-8,
            profile=profile,
            gas_species=("Ca",),
        )
    assert calls == 1

    oxide_df = profile.gas_datapack.oxide_df
    dataframe_type = type(oxide_df)
    original_iterrows = dataframe_type.iterrows
    table_iterations = 0

    def counted_iterrows(frame):
        nonlocal table_iterations
        if frame is oxide_df:
            table_iterations += 1
        return original_iterrows(frame)

    monkeypatch.setattr(dataframe_type, "iterrows", counted_iterrows)
    standards_module._cached_check_pack_standards(
        profile.melt_pack, profile, gas_pack=profile.gas_datapack
    )
    assert table_iterations == 0
    with monkeypatch.context() as path_patches:
        path_patches.setattr(
            Path,
            "resolve",
            lambda *args, **kwargs: pytest.fail("cache hit resolved a path"),
        )
        path_patches.setattr(
            Path,
            "stat",
            lambda *args, **kwargs: pytest.fail("cache hit stated a path"),
        )
        standards_module._cached_check_pack_standards(
            profile.melt_pack, profile, gas_pack=profile.gas_datapack
        )

    changed_oxide = profile.gas_datapack.oxide_df.copy()
    changed_oxide.loc["SiO2(l)", "table_id"] = "synthetic-source-table"
    changed_tables = replace(profile.gas_datapack, oxide_df=changed_oxide)
    changed_profile = replace(profile, gas_datapack=changed_tables)
    evaluate_gas(
        activities,
        1800.0,
        1.0e-8,
        profile=changed_profile,
        gas_species=("Ca",),
    )
    assert calls == 2


def test_profile_redox_pack_is_loaded_and_bound_to_content_digest(tmp_path) -> None:
    original = load_profile("openimcc")
    source = PACK_DIR / "openimcc-redox-v4.json"
    custom_path = tmp_path / "custom-redox.json"
    custom_path.write_bytes(source.read_bytes())
    old_profile = replace(original, redox_pack_path=custom_path)
    old_redox_pack = old_profile.redox_pack
    assert old_profile.redox_pack is old_redox_pack

    old_records = standards_module._cached_check_pack_standards(
        old_profile.melt_pack,
        old_profile,
        gas_pack=old_profile.gas_datapack,
        redox_pack=old_redox_pack,
    )
    old_feo15 = active_liquid_standards(old_profile)["FeO1.5"]
    inventory = {"Fe": 1.0, "O": 1.25}
    old_result = evaluate_redox(inventory, 2200.0, "closed", profile=old_profile)
    content = json.loads(custom_path.read_text(encoding="utf-8"))
    content["thermo_species"]["Fe2O3(l)"]["construction"][
        "crystal_source_table"
    ] = "Fe-031"
    content["canonical_content_digest"] = canonical_redox_digest(content)
    custom_path.write_text(json.dumps(content), encoding="utf-8")

    assert old_profile.redox_pack is old_redox_pack
    assert old_profile.redox_pack.canonical_digest != content["canonical_content_digest"]
    assert active_liquid_standards(old_profile)["FeO1.5"] == old_feo15
    new_profile = replace(original, redox_pack_path=custom_path)
    assert new_profile.redox_pack.canonical_digest == content["canonical_content_digest"]
    assert "FeO1.5" not in active_liquid_standards(new_profile)
    old_result_after_edit = evaluate_redox(
        inventory, 2200.0, "closed", profile=old_profile
    )
    new_result = evaluate_redox(inventory, 2200.0, "closed", profile=new_profile)
    assert old_result_after_edit.redox_pack_digest == old_result.redox_pack_digest
    assert old_result_after_edit.notices == old_result.notices
    assert new_result.redox_pack_digest == new_profile.redox_pack.canonical_digest
    assert new_result.notices != old_result.notices
    old_records_after_edit = standards_module._cached_check_pack_standards(
        old_profile.melt_pack,
        old_profile,
        gas_pack=old_profile.gas_datapack,
        redox_pack=old_profile.redox_pack,
    )
    new_records = standards_module._cached_check_pack_standards(
        new_profile.melt_pack,
        new_profile,
        gas_pack=new_profile.gas_datapack,
        redox_pack=new_profile.redox_pack,
    )
    assert old_records_after_edit == old_records
    assert new_records != old_records
    old_notices = standards_module._pack_standard_notices(old_records)
    assert all(isinstance(notice, PackStandardNotice) for notice in old_notices)
    new_notices = standards_module._pack_standard_notices(new_records)
    assert all(isinstance(notice, PackStandardNotice) for notice in new_notices)
    assert new_notices != old_notices


def test_standard_cache_uses_supplied_pack_digests_and_skips_unknown_identity(
    monkeypatch,
) -> None:
    calls = 0

    def counted_check(*args, **kwargs):
        nonlocal calls
        calls += 1
        return []

    monkeypatch.setattr(standards_module, "_STANDARD_CHECK_CACHE", {})
    monkeypatch.setattr(standards_module, "check_pack_standards", counted_check)
    first = SimpleNamespace(canonical_digest="custom-pack-content")
    second = SimpleNamespace(canonical_digest="custom-pack-content")
    standards_module._cached_check_pack_standards(first)
    standards_module._cached_check_pack_standards(second)
    assert calls == 1

    no_digest_first = SimpleNamespace()
    no_digest_second = SimpleNamespace()
    standards_module._cached_check_pack_standards(no_digest_first)
    standards_module._cached_check_pack_standards(no_digest_first)
    standards_module._cached_check_pack_standards(no_digest_second)
    assert calls == 4


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
    assert unknown_row.standard_basis["K2O"] == "best available"
    assert unknown_row.diagnostic_label == K2O_DIAGNOSTIC_LABEL
    assert unknown_row.diagnostic_delta_g_j_mol is not None
    assert unknown_row.equivalent_delta_log10_K is not None
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


def test_k2o_one_sided_band_is_exposed_and_does_not_change_evaluate() -> None:
    from tools.translate_k_constants import fc87_k2o_gibbs_j_mol

    gas = load_gas_datapack()
    registry = _registry()
    applicable_entries = [
        entry
        for entry in registry["packs"].values()
        if entry["pack_type"] == "melt"
        and entry["file"].startswith(("imcc-sf04-d066-", "imcc-sf04-ext-"))
        and any(
            values.get("K2O") == "reference-unknown"
            for values in entry["row_overrides"].values()
        )
    ]
    assert applicable_entries
    for entry in applicable_entries:
        pack = load_datapack(PACK_DIR / entry["file"])
        assert set(entry["row_overrides"]) >= set(K2O_REFERENCE_ROWS)
        for row_key in K2O_REFERENCE_ROWS:
            row = pack_row_liquid_standards(pack, row_key)
            assert row["K2O"] == "reference-unknown"
            assert row.standard_basis["K2O"] == "best available"
            assert row.diagnostic_label == K2O_DIAGNOSTIC_LABEL

            delta_g = row.diagnostic_delta_g_j_mol
            assert delta_g is not None
            component = delta_g["components"]["FC87/Glushko"]
            for index, temperature in enumerate(K2O_DIAGNOSTIC_TEMPERATURES_K):
                calculated = fc87_k2o_gibbs_j_mol(temperature, gas) - species_thermo(
                    "K2O", "l", temperature, gas
                ).G_J_mol
                assert component[index] == pytest.approx(calculated, abs=1.0)

            envelope = delta_g["envelope"]
            expected_envelope = [
                max(abs(values[index]) for values in delta_g["components"].values())
                for index in range(len(K2O_DIAGNOSTIC_TEMPERATURES_K))
            ]
            assert envelope == pytest.approx(expected_envelope, abs=1e-6)
            nu_k2o = 0.5 if row_key == "KCaAlSi2O7" else 1.0
            equivalent = row.equivalent_delta_log10_K
            assert equivalent is not None
            assert equivalent["nu_K2O"] == nu_k2o
            expected_delta_log = [
                nu_k2o
                * value
                / (R_J_MOL_K * temperature * math.log(10.0))
                for temperature, value in zip(
                    K2O_DIAGNOSTIC_TEMPERATURES_K, envelope, strict=True
                )
            ]
            assert equivalent["envelope"] == pytest.approx(
                expected_delta_log, abs=1e-12
            )

    records = check_pack_standards(PACK_DIR / "imcc-sf04-d066-v4.json")
    for row_key in K2O_REFERENCE_ROWS:
        record = next(
            record
            for record in records
            if record.oxide == "K2O" and record.row_key == row_key
        )
        row = pack_row_liquid_standards(
            PACK_DIR / "imcc-sf04-d066-v4.json", row_key
        )
        assert record.kind == "reference-unknown"
        assert record.diagnostic_delta_g_j_mol == row.diagnostic_delta_g_j_mol
        assert record.diagnostic_label == K2O_DIAGNOSTIC_LABEL
        assert record.equivalent_delta_log10_K == row.equivalent_delta_log10_K

    composition = {
        "SiO2": 0.45,
        "MgO": 0.20,
        "FeO": 0.10,
        "CaO": 0.08,
        "Al2O3": 0.08,
        "TiO2": 0.02,
        "Na2O": 0.05,
        "K2O": 0.02,
    }
    # These pre-band outputs pin the unchanged evaluation path at a fixed composition.
    result = evaluate(
        composition,
        2000.0,
        load_datapack(PACK_DIR / "imcc-sf04-d066-v5.json"),
        allow_out_of_envelope=True,
    )
    assert result.D == pytest.approx(2.0799216753260055, abs=1e-12)
    assert result.parent_activity == pytest.approx(
        [
            0.0871364710737131,
            0.060030609790782945,
            0.1270835418355807,
            0.0004061159375203468,
            0.010035636097007061,
            0.003269823103028347,
            6.080403277506625e-07,
            3.040702681203824e-10,
        ],
        abs=1e-12,
    )


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
        standards_module,
        "active_liquid_standards",
        lambda profile=None, *, gas_pack=None: active,
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
