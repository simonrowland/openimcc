from __future__ import annotations

import csv
import hashlib
import json
import re
import runpy
from importlib import resources
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from openimcc import coefficient_ancestry
from openimcc.model import load_datapack
from openimcc.janaf_crystals import (
    evaluate_janaf_apparent_gibbs,
    evaluate_janaf_crystal,
    _evaluate_janaf_liquid_species,
)
from openimcc.profiles import load_profile
from openimcc.standards import pack_liquid_standards


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROLES = {"derived", "fitted", "selected", "anchor", "diagnostic"}


def test_redox_source_role_requires_bounded_citation_match():
    redox_source_role = runpy.run_path(
        str(ROOT / "tools" / "build_coefficient_ancestry.py")
    )["_redox_source_role"]

    assert redox_source_role(
        "Li (2001)",
        {"selection_rationale": "Selected from the alkali (2001) calibration."},
        "derived",
    ) == "derived"
    assert redox_source_role(
        "Li (2001)",
        {"selection_rationale": "Li (2001) was selected for this redox row."},
        "derived",
    ) == "selected"


def test_coefficient_source_lookup_requires_bounded_citation_match():
    source_mapper = runpy.run_path(
        str(ROOT / "tools" / "build_coefficient_ancestry.py")
    )["SourceMapper"](
        {"dois": {}, "citation_keys": {"Li (2001)": "li-work"}, "patterns": []},
        {"li-work"},
    )

    assert source_mapper.resolve("alkali (2001)") == []
    assert source_mapper.resolve("Li (2001)") == ["li-work"]


def test_context_checked_citation_mappings_and_rejections():
    source_mapper = runpy.run_path(
        str(ROOT / "tools" / "build_coefficient_ancestry.py")
    )["SourceMapper"]
    mapping = json.loads(
        (ROOT / "tools" / "coefficient-ancestry-sources.json").read_text(
            encoding="utf-8"
        )
    )
    vocabulary = json.loads(
        (ROOT / "tests" / "data" / "coefficient-ancestry-vocabulary.json").read_text(
            encoding="utf-8"
        )
    )
    mapper = source_mapper(mapping, set(vocabulary["work_ids"]))

    mapped = {
        "10.1016/j.jeurceramsoc.2018.03.007": "costa-2015-yb-silicate-silica-activity",
        "corrected Costa & Jacobson Table 2": "costa-2015-yb-silicate-silica-activity",
        "corrected Costa & Jacobson thermochemical cycle, Table 2": "costa-2015-yb-silicate-silica-activity",
        "de Abreu and Fabrichnaya 2024 Table 5": "de-abreu-2024-li2o-sio2-thermochemistry",
        "de Abreu and Fabrichnaya 2024 Table 5 p. 315": "de-abreu-2024-li2o-sio2-thermochemistry",
        "Chase (1998).": "janaf-4th",
        "LH87": "lamoreaux-hildenbrand-hildenbrand-1987",
        "TSIV-1334": "ivtan-mno-coo-thermo",
        "TSIV-1334-SC-2090": "ivtan-mno-coo-thermo",
    }
    for citation, work_id in mapped.items():
        assert mapper.resolve(citation) == [work_id]

    for citation in (
        "Costa et al.",
        "Robie et al.",
        "Hemingway rhodonite crystal thermochemistry",
        "MnSiO3-Robie-1995",
        "NBS compilation",
        "IVTAN-264",
        "TSIV-1603",
        "TSIV-2026",
    ):
        assert mapper.resolve(citation) == []


def test_vocabulary_index_accepts_matching_ids_and_dois() -> None:
    check_vocabulary_index = runpy.run_path(
        str(ROOT / "tools" / "build_coefficient_ancestry.py")
    )["_check_vocabulary_index"]
    index = yaml.safe_load(
        (ROOT / "tests" / "data" / "coefficient-ancestry-index-fixture.yaml").read_text(
            encoding="utf-8"
        )
    )

    check_vocabulary_index(
        {"work_ids": ["work-a", "work-b"], "work_dois": {"work-a": "10.1000/a"}},
        index,
    )


def test_vocabulary_index_rejects_missing_work_ids() -> None:
    check_vocabulary_index = runpy.run_path(
        str(ROOT / "tools" / "build_coefficient_ancestry.py")
    )["_check_vocabulary_index"]
    index = yaml.safe_load(
        (ROOT / "tests" / "data" / "coefficient-ancestry-index-fixture.yaml").read_text(
            encoding="utf-8"
        )
    )

    with pytest.raises(ValueError, match="missing from INDEX.yaml: work-missing"):
        check_vocabulary_index(
            {"work_ids": ["work-a", "work-missing"], "work_dois": {}}, index
        )


def test_vocabulary_index_rejects_doi_mismatches() -> None:
    check_vocabulary_index = runpy.run_path(
        str(ROOT / "tools" / "build_coefficient_ancestry.py")
    )["_check_vocabulary_index"]
    index = yaml.safe_load(
        (ROOT / "tests" / "data" / "coefficient-ancestry-index-fixture.yaml").read_text(
            encoding="utf-8"
        )
    )

    with pytest.raises(ValueError, match="DOI mismatch for work ID work-a"):
        check_vocabulary_index(
            {"work_ids": ["work-a"], "work_dois": {"work-a": "10.1000/incorrect"}},
            index,
        )


def _json_resource(path: str) -> dict:
    return json.loads(
        resources.files("openimcc")
        .joinpath(*path.split("/"))
        .read_text(encoding="utf-8")
    )


def _artifact(entries: list[dict], identity: str) -> dict:
    return next(entry for entry in entries if entry["identity"] == identity)


def _expected_row_ids(artifact: dict) -> set[str]:
    file = artifact["file"].removeprefix("data/")
    path = resources.files("openimcc").joinpath("data", file)
    if file == "packs/imcc-sf04-d066-v4.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return {f"{row['row']}:{row['complex']}" for row in content["rows"]}
    if file in {
        "packs/imcc-sf04-d066-sr-candidate-v1.json",
        "packs/imcc-sf04-d066-sr-candidate-v2.json",
        "packs/imcc-sf04-d066-p2o5-candidate-v1.json",
        "packs/imcc-sf04-d066-cao-aluminate-candidate-v1.json",
        "packs/imcc-sf04-d066-silicate-candidate-v2.json",
    }:
        content = json.loads(path.read_text(encoding="utf-8"))
        return {f"{row['row']}:{row['complex']}" for row in content["rows"]}
    if file in {
        "gas/gas-shomate-default-v1.csv",
        "gas/gas-shomate-default-v2.csv",
        "gas/condensate-default-v1.csv",
    }:
        rows = csv.DictReader(path.read_text(encoding="utf-8").splitlines())
        return {
            f"{index}:{row['species_name']}"
            + (f":{row['T_interval']}" if row.get("T_interval") else "")
            for index, row in enumerate(rows, start=1)
        }
    if file == "packs/openimcc-redox-v6.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        reaction_ids = {
            f"reaction:{index}"
            for index, row in enumerate(content["rows"], start=1)
        }
        species_ids = {
            f"thermo_species:{name}" for name in content["thermo_species"]
        }
        return reaction_ids | species_ids
    if file == "liquid-standards.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        row_ids: set[str] = set()
        for digest, pack in content["packs"].items():
            row_ids.update(
                f"{digest}:standard:{standard}"
                for standard in pack.get("standards", {})
            )
            row_ids.update(
                f"{digest}:override:{complex_name}:{standard}"
                for complex_name, overrides in pack.get("row_overrides", {}).items()
                for standard in overrides
            )
            row_ids.update(
                f"{digest}:diagnostic:{complex_name}"
                for complex_name in pack.get("row_diagnostics", {})
            )
        return row_ids
    if file == "diagnostics/janaf-pure-oxide-crystals.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return {f"phases:{row['id']}" for row in content["phases"]}
    if file == "diagnostics/janaf-crystal-species.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return {f"phases:{row['id']}" for row in content["phases"]}
    if file == "diagnostics/rh-crystal-species.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return {f"phases:{row['id']}" for row in content["phases"]}
    if file == "diagnostics/janaf-pure-oxide-liquid-pairs.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return {f"rows:{row['table_id']}" for row in content["rows"]}
    if file == "diagnostics/janaf-liquid-species.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return {f"rows:{row['table_id']}" for row in content["rows"]}
    if file == "diagnostics/barin-aluminate-crystals.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return {f"phases:{row['id']}" for row in content["phases"]}
    if file == "diagnostics/crystal-phases.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        phases = {f"phases:{row['id']}" for row in content["phases"]}
        controls = {
            f"pure_oxide_controls:{row['id']}"
            for row in content["pure_oxide_controls"]
        }
        return phases | controls
    if file == "diagnostics/elemental-reference-crystal-cards.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return {f"records:{row['formula']}" for row in content["records"]}
    if file.startswith("fragments/"):
        content = json.loads(path.read_text(encoding="utf-8"))
        rows = content.get("rows", {})
        if isinstance(rows, dict):
            return {str(row_id) for row_id in rows}
        return {
            str(row.get("row", index))
            for index, row in enumerate(rows, start=1)
        }
    raise AssertionError(f"unexpected packaged ancestry artifact: {file}")


def test_default_artifacts_match_the_loaded_package_data() -> None:
    entries = coefficient_ancestry()
    manifest = _json_resource("data/coefficient-ancestry.json")
    profile = load_profile("openimcc")

    assert manifest["schema"] == "openimcc-coefficient-ancestry.v1"
    assert len(entries) == 44
    for entry in entries:
        raw = resources.files("openimcc").joinpath(*entry["file"].split("/")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["digest"]
        assert coefficient_ancestry(entry["identity"]) == entry
        assert coefficient_ancestry(entry["digest"]) == entry

    melt = _artifact(entries, f"{profile.melt_pack.model_id}@{profile.melt_pack.version}")
    assert melt["runtime_digest"] == profile.melt_pack.binding_digest

    gas = _artifact(entries, "gas-shomate-default-v2")
    assert gas["runtime_digest"] == profile.gas_datapack.gas_table_digest

    historical_gas = _artifact(entries, "gas-shomate-default-v1")
    assert historical_gas["runtime_digest"] == (
        "4d069a8951cd11e0d2edb37420e3aa04e4927c51c6eee00bbc53368b20d6c492"
    )

    condensate = _artifact(entries, "condensate-default-v1")
    assert condensate["runtime_digest"] == profile.gas_datapack.condensate_table_digest

    redox = _artifact(entries, profile.redox_pack.version)
    assert redox["runtime_digest"] == profile.redox_pack.canonical_digest

    registry = _artifact(entries, "liquid-standards-v2")
    registry_data = _json_resource("data/liquid-standards.json")
    assert registry["runtime_digest"] == registry["digest"]
    loaded_standards = pack_liquid_standards(profile.melt_pack)
    declaration = registry_data["packs"][profile.melt_pack.binding_digest]
    assert dict(loaded_standards.standards) == declaration["standards"]

    crystals = _artifact(entries, "openimcc-janaf-apparent-crystals-v2")
    crystal_data = _json_resource(crystals["file"])
    phase = crystal_data["phases"][0]
    temperature = max(298.15, phase["source_temperature_K"][0])
    result = evaluate_janaf_crystal(phase["id"], temperature)
    assert result.provenance["resource_sha256"] == crystals["runtime_digest"]

    additional_crystals = _artifact(
        entries, "openimcc-janaf-apparent-crystal-species-v1"
    )
    additional_data = _json_resource(additional_crystals["file"])
    additional_phase = additional_data["phases"][0]
    temperature = max(298.15, additional_phase["source_temperature_K"][0])
    result = evaluate_janaf_crystal(additional_phase["id"], temperature)
    assert result.provenance["resource_sha256"] == additional_crystals["runtime_digest"]

    liquid_pairs = _artifact(entries, "openimcc-janaf-apparent-liquid-pairs-v1")
    pair_data = _json_resource(liquid_pairs["file"])
    pair = pair_data["rows"][0]
    temperature = max(298.15, pair["source_temperature_K"][0])
    result = evaluate_janaf_apparent_gibbs(pair["table_id"], temperature)
    assert result.provenance["resource_sha256"] == liquid_pairs["runtime_digest"]

    liquid_species = _artifact(entries, "openimcc-janaf-liquid-species-v1")
    liquid_species_data = _json_resource(liquid_species["file"])
    liquid_species_row = liquid_species_data["rows"][0]
    temperature = sum(liquid_species_row["temperature_K"]) / 2.0
    result = _evaluate_janaf_liquid_species(
        liquid_species_row["formula"], temperature
    )
    assert result is not None
    assert result.resource_sha256 == liquid_species["runtime_digest"]

    barin = _artifact(entries, "openimcc-barin-aluminate-crystals-v3")
    assert set(barin["work_ids"]) == {"barin-1995", "janaf-4th"}
    assert all(
        source["role"] == "derived"
        for row in barin["rows"] for source in row["sources"]
    )

    crystals = _artifact(entries, "openimcc-crystal-phases-v2")
    assert set(crystals["work_ids"]) == {
        "berman-1988-internally-consistent-mineral-thermodynamic-data",
        "darken-gurry-1946-iron-oxygen-ii",
        "janaf-4th",
        "robie-hemingway-1995-usgs-b2131",
    }
    # Green 3645e2c2bb indexes Robie–Hemingway, resolving these five citations.
    assert crystals["complete"] is True
    assert crystals["counts"]["unmapped_citations"] == 0
    rh_rows = {
        f"phases:{phase}"
        for phase in ("fayalite", "hercynite", "ilmenite", "perovskite", "ulvospinel")
    }
    rh_sources = [
        (row["id"], source)
        for row in crystals["rows"]
        for source in row["sources"]
        if source["citation"] == "Robie and Hemingway (1995), USGS Bulletin 2131"
    ]
    assert {row_id for row_id, _source in rh_sources} == rh_rows
    assert {source["work_id"] for _row_id, source in rh_sources} == {
        "robie-hemingway-1995-usgs-b2131"
    }
    rh_table_artifact = _artifact(entries, "openimcc-rh-crystal-species-v1")
    new_rh_rows = {
        f"phases:{phase}"
        for phase in (
            "perovskite-table", "silver-fcc", "silver-liquid", "thorium-alpha",
            "thorium-beta", "uranium-alpha", "uranium-beta", "uranium-gamma",
            "uranium-liquid", "arsenic-rhombohedral", "cerium-alpha",
            "cerium-beta", "cerium-liquid", "antimony-rhombohedral",
            "antimony-liquid", "tin-crystals", "tin-liquid",
            "tellurium-crystals", "tellurium-liquid",
        )
    }
    assert new_rh_rows <= {row["id"] for row in rh_table_artifact["rows"]}
    assert rh_table_artifact["work_ids"] == ["robie-hemingway-1995-usgs-b2131"]
    for phase in ("fayalite", "hercynite", "ilmenite", "ulvospinel"):
        row = next(item for item in crystals["rows"] if item["id"] == f"phases:{phase}")
        assert any(
            source["work_id"] == "darken-gurry-1946-iron-oxygen-ii"
            for source in row["sources"]
        )
        assert not any("Fe-019" in source["citation"] for source in row["sources"])


def test_every_default_row_has_mapped_or_unmapped_lineage() -> None:
    entries = coefficient_ancestry()
    for entry in entries:
        expected = _expected_row_ids(entry)
        actual_rows = entry["rows"]
        actual = {row["id"] for row in actual_rows}
        assert actual == expected
        assert len(actual_rows) == len(actual)
        unmapped_row_ids = {item["row"] for item in entry["unmapped"]}
        for row in actual_rows:
            assert row["sources"] or row["id"] in unmapped_row_ids
            assert all(source["role"] in ALLOWED_ROLES for source in row["sources"])
        parent_ids = {
            f"parent_standard:{standard['standard']}"
            for standard in entry.get("parent_standards", [])
        }
        assert all(
            item["row"] in actual or item["row"] in parent_ids
            for item in entry["unmapped"]
        )
        assert entry["counts"]["rows"] == len(expected)
        assert entry["complete"] is (not entry["unmapped"])


def test_sr_candidate_artifacts_have_parameter_level_ancestry() -> None:
    entries = coefficient_ancestry()
    default = _artifact(entries, "IMCC-SF04-D066@1.0.6-d066-v4")
    default_rows = {row["id"]: row for row in default["rows"]}
    default_pack = _json_resource("data/packs/imcc-sf04-d066-v4.json")
    vocabulary = set(
        json.loads(
            (ROOT / "tests" / "data" / "coefficient-ancestry-vocabulary.json")
            .read_text(encoding="utf-8")
        )["work_ids"]
    )
    common_parameters = {
        "formation_delta_h_ox",
        "formation_delta_s_ox",
        "formation_delta_cp_ox",
        "fusion_temperature_marker",
        "fusion_entropy",
        "liquid_cp",
    }
    expected_provisional = {
        ("39:SrAl2O4", "fusion_temperature_marker"),
        ("39:SrAl2O4", "fusion_entropy"),
        ("39:SrAl2O4", "liquid_cp"),
        ("39:SrAl2O4", "parent_liquid_standard_Al2O3"),
        ("40:SrAl4O7", "formation_delta_s_ox"),
        ("40:SrAl4O7", "formation_delta_cp_ox"),
        ("40:SrAl4O7", "fusion_entropy"),
        ("40:SrAl4O7", "liquid_cp"),
        ("40:SrAl4O7", "parent_liquid_standard_Al2O3"),
        ("41:SrAl12O19", "formation_delta_s_ox"),
        ("41:SrAl12O19", "formation_delta_cp_ox"),
        ("41:SrAl12O19", "fusion_entropy"),
        ("41:SrAl12O19", "liquid_cp"),
        ("41:SrAl12O19", "parent_liquid_standard_Al2O3"),
        ("42:Sr3Al2O6", "formation_delta_s_ox"),
        ("42:Sr3Al2O6", "formation_delta_cp_ox"),
        ("42:Sr3Al2O6", "fusion_entropy"),
        ("42:Sr3Al2O6", "liquid_cp"),
        ("42:Sr3Al2O6", "parent_liquid_standard_Al2O3"),
        ("43:Sr4Al2O7", "formation_delta_s_ox"),
        ("43:Sr4Al2O7", "formation_delta_cp_ox"),
        ("43:Sr4Al2O7", "fusion_entropy"),
        ("43:Sr4Al2O7", "liquid_cp"),
        ("43:Sr4Al2O7", "parent_liquid_standard_Al2O3"),
        ("44:SrSiO3", "formation_delta_s_ox"),
        ("44:SrSiO3", "formation_delta_cp_ox"),
        ("44:SrSiO3", "fusion_entropy"),
        ("44:SrSiO3", "liquid_cp"),
        ("44:SrSiO3", "parent_liquid_standard_SiO2"),
        ("45:Sr2SiO4", "formation_delta_s_ox"),
        ("45:Sr2SiO4", "formation_delta_cp_ox"),
        ("45:Sr2SiO4", "fusion_entropy"),
        ("45:Sr2SiO4", "liquid_cp"),
        ("45:Sr2SiO4", "parent_liquid_standard_SiO2"),
    }
    for version in ("v1", "v2"):
        filename = f"imcc-sf04-d066-sr-candidate-{version}.json"
        pack_path = ROOT / "src" / "openimcc" / "data" / "packs" / filename
        loaded = load_datapack(pack_path)
        artifact = _artifact(
            entries,
            f"IMCC-SF04-D066-SR-CANDIDATE@1.0.0-d066-sr-candidate-{version}",
        )
        assert artifact["runtime_digest"] == loaded.binding_digest
        assert coefficient_ancestry(artifact["runtime_digest"]) == artifact
        assert hashlib.sha256(pack_path.read_bytes()).hexdigest() == artifact["digest"]
        assert artifact["complete"] is False

        candidate_pack = _json_resource(f"data/packs/{filename}")
        for default_row, candidate_row in zip(
            default_pack["rows"], candidate_pack["rows"][:38]
        ):
            assert (
                default_row["row"],
                default_row["complex"],
                default_row["nu"],
                default_row["A"],
                default_row["B"],
            ) == (
                candidate_row["row"],
                candidate_row["complex"],
                candidate_row["nu"],
                candidate_row["A"],
                candidate_row["B"],
            )
            row_id = f"{candidate_row['row']}:{candidate_row['complex']}"
            ancestry_row = next(row for row in artifact["rows"] if row["id"] == row_id)
            assert ancestry_row["sources"] == default_rows[row_id]["sources"]
            assert "parameters" not in ancestry_row

        rows = {
            row["id"]: row
            for row in artifact["rows"]
            if int(row["id"].split(":", 1)[0]) >= 39
        }
        assert len(rows) == 7
        provisional = {
            (row["id"], parameter["parameter"])
            for row in rows.values()
            for parameter in row["parameters"]
            if parameter["provisional"]
        }
        assert provisional == expected_provisional
        for row_number, row in enumerate(
            candidate_pack["rows"][38:], start=39
        ):
            row_id = f"{row_number}:{row['complex']}"
            ancestry_row = rows[row_id]
            parameters = ancestry_row["parameters"]
            by_name = {parameter["parameter"]: parameter for parameter in parameters}
            expected_names = set(common_parameters)
            expected_names.update(
                f"parent_liquid_standard_{parent}"
                for parent in ("SrO", "Al2O3")
                if parent in row["standard_state_refit"]["parents"]
            )
            expected_names.update(
                f"parent_liquid_standard_{parent}"
                for parent in ("SrO", "SiO2")
                if parent in row["standard_state_refit"]["parents"]
            )
            assert set(by_name) == expected_names
            assert all(parameter["sources"] for parameter in parameters)
            assert all(
                source["role"] in ALLOWED_ROLES
                and (source["work_id"] is None or source["work_id"] in vocabulary)
                for parameter in parameters
                for source in parameter["sources"]
            )

            unmapped_parameters = {
                (item.get("parameter"), item["citation"])
                for item in artifact["unmapped"]
                if item["row"] == row_id
            }
            for parameter in parameters:
                for source in parameter["sources"]:
                    if source["work_id"] is None:
                        assert (
                            parameter["parameter"], source["citation"]
                        ) in unmapped_parameters
                        if "Eskola" in source["citation"]:
                            assert "Table I, p. 335" in source["citation"]
                        else:
                            assert "DOI " in source["citation"]

        for row_id in ("42:Sr3Al2O6", "43:Sr4Al2O7"):
            for parameter_name in ("formation_delta_s_ox", "formation_delta_cp_ox"):
                parameter = next(
                    item
                    for item in rows[row_id]["parameters"]
                    if item["parameter"] == parameter_name
                )
                assert any(
                    "Jacob & Shreyas" in source["citation"]
                    and source["role"] == "diagnostic"
                    for source in parameter["sources"]
                )

        sr_silicate_marker = next(
            item
            for item in rows["44:SrSiO3"]["parameters"]
            if item["parameter"] == "fusion_temperature_marker"
        )
        eskola = next(
            source for source in sr_silicate_marker["sources"]
            if "Eskola" in source["citation"]
        )
        assert eskola["work_id"] is None
        assert eskola["role"] == "diagnostic"
        assert any(
            item["row"] == "44:SrSiO3"
            and item["parameter"] == "fusion_temperature_marker"
            and item["citation"] == eskola["citation"]
            for item in artifact["unmapped"]
        )

        parent_standards = {
            source["standard"]: source
            for source in artifact["parent_standards"]
        }
        assert set(parent_standards) == {"Na2O", "SiO2", "SrO", "Al2O3"}
        assert all(item["sources"] for item in parent_standards.values())


def test_p2o5_candidate_artifact_has_parameter_level_ancestry() -> None:
    entries = coefficient_ancestry()
    path = ROOT / "src/openimcc/data/packs/imcc-sf04-d066-p2o5-candidate-v1.json"
    loaded = load_datapack(path)
    artifact = _artifact(
        entries,
        "IMCC-SF04-D066-P2O5-CANDIDATE@1.0.0-d066-p2o5-candidate-v1",
    )
    assert artifact["runtime_digest"] == loaded.binding_digest
    assert artifact["complete"] is False
    assert artifact["counts"]["rows"] == 40
    rows = {row["id"]: row for row in artifact["rows"]}
    assert {"39:Mg3(PO4)2", "40:Ca3(PO4)2"} <= set(rows)
    assert rows["39:Mg3(PO4)2"]["parameters"]
    assert rows["40:Ca3(PO4)2"]["parameters"]
    mg_h = next(
        item
        for item in rows["39:Mg3(PO4)2"]["parameters"]
        if item["parameter"] == "formation_delta_h_ox"
    )
    ca_h = next(
        item
        for item in rows["40:Ca3(PO4)2"]["parameters"]
        if item["parameter"] == "formation_delta_h_ox"
    )
    for parameter in (mg_h, ca_h):
        assert any(
            source["work_id"] == "janaf-4th"
            and "P-012" in source["citation"]
            for source in parameter["sources"]
        )
    provisional = {
        (row_id, parameter["parameter"])
        for row_id, row in rows.items()
        for parameter in row.get("parameters", [])
        if parameter["provisional"]
    }
    assert provisional == {
        ("40:Ca3(PO4)2", "formation_delta_h_ox"),
        ("40:Ca3(PO4)2", "formation_delta_s_ox"),
        ("40:Ca3(PO4)2", "formation_delta_cp_ox"),
        ("40:Ca3(PO4)2", "fusion_entropy"),
        ("40:Ca3(PO4)2", "liquid_cp"),
        ("40:Ca3(PO4)2", "liquid_association_branch"),
    }
    assert {item["standard"] for item in artifact["parent_standards"]} == {
        "MgO", "CaO", "P2O5"
    }
    p2o5_standard = next(
        item for item in artifact["parent_standards"] if item["standard"] == "P2O5"
    )
    assert any(
        source["work_id"] == "janaf-4th"
        and "P-012" in source["citation"]
        for source in p2o5_standard["sources"]
    )
    assert {"janaf-4th", "robie-hemingway-1995-usgs-b2131"} <= set(artifact["work_ids"])
    assert any("10.1039/JR9610004442" in item["citation"] for item in artifact["unmapped"])


def test_cao_aluminate_candidate_records_fitted_intercepts() -> None:
    entries = coefficient_ancestry()
    path = ROOT / "src/openimcc/data/packs/imcc-sf04-d066-cao-aluminate-candidate-v1.json"
    loaded = load_datapack(path)
    artifact = _artifact(
        entries,
        "IMCC-SF04-D066@1.0.0-d066-cao-aluminate-candidate-v1",
    )
    assert artifact["runtime_digest"] == loaded.binding_digest
    assert artifact["counts"]["rows"] == 38
    rows = {row["id"]: row for row in artifact["rows"]}
    for row_id in ("9:CaAl4O7", "22:CaAl12O19"):
        parameters = {item["parameter"]: item for item in rows[row_id]["parameters"]}
        assert {"A", "B"} <= set(parameters)
        assert any(
            source["role"] == "fitted"
            and source["work_id"] == "kems-051-allibert-1981"
            for source in parameters["A"]["sources"]
        )
        assert any(
            source["role"] == "selected"
            and source["work_id"] == "no27-fegley-cameron-1987-epsl"
            for source in parameters["B"]["sources"]
        )


def test_silicate_candidate_v2_ancestry_marks_only_mg_intercepts_provisional() -> None:
    entries = coefficient_ancestry()
    filename = "imcc-sf04-d066-silicate-candidate-v2.json"
    path = ROOT / "src/openimcc/data/packs" / filename
    loaded = load_datapack(path)
    artifact = _artifact(
        entries,
        "IMCC-SF04-D066@1.0.0-d066-silicate-candidate-v2",
    )
    default = _artifact(entries, "IMCC-SF04-D066@1.0.6-d066-v4")
    default_rows = {row["id"]: row for row in default["rows"]}
    candidate_rows = {row["id"]: row for row in artifact["rows"]}

    assert artifact["runtime_digest"] == loaded.binding_digest
    assert len(candidate_rows) == 38
    assert artifact["complete"] is False
    for row_id, row in candidate_rows.items():
        if row["label"] not in {"Mg2SiO4", "MgSiO3"}:
            assert row == default_rows[row_id]
            continue
        parameters = {item["parameter"]: item for item in row["parameters"]}
        assert set(parameters) == {"A", "B"}
        assert parameters["A"]["provisional"] is True
        assert parameters["B"]["provisional"] is False
        assert {source["role"] for source in parameters["A"]["sources"]} == {
            "fitted"
        }
        assert len(parameters["A"]["sources"]) == 4
        citations = " ".join(
            source["citation"]
            for parameter in row["parameters"]
            for source in parameter["sources"]
        ).casefold()
        assert "zaitsev" not in citations
        assert "shornikov" not in citations


def test_elemental_reference_crystal_cards_are_registered_in_ancestry() -> None:
    entries = coefficient_ancestry()
    artifact = _artifact(
        entries, "openimcc-elemental-reference-crystal-cards-v1"
    )

    assert {row["id"] for row in artifact["rows"]} == {
        f"records:{formula}"
        for formula in ("Au", "Pd", "Pt", "Rh", "Ir", "Os", "Ru")
    }
    assert artifact["work_ids"] == ["burcat-third-millennium"]
    assert artifact["complete"] is False
    assert any("P672-Rh-cr" in item["citation"] for item in artifact["unmapped"])
    assert any("P672-Ru-cr" in item["citation"] for item in artifact["unmapped"])
    gold = next(row for row in artifact["rows"] if row["id"] == "records:Au")
    assert any(
        source["work_id"] == "burcat-third-millennium"
        and "BU-0054" in source["citation"]
        for source in gold["sources"]
    )


def test_every_shipped_fragment_row_appears_in_the_manifest() -> None:
    entries = coefficient_ancestry()
    fragment_entries = {
        entry["file"]: entry
        for entry in entries
        if entry["file"].startswith("data/fragments/")
    }
    fragment_files = sorted(
        path.relative_to(ROOT / "src" / "openimcc").as_posix()
        for path in (ROOT / "src" / "openimcc" / "data" / "fragments").glob(
            "*/rows.json"
        )
    )

    assert set(fragment_entries) == set(fragment_files)
    for file, entry in fragment_entries.items():
        assert {row["id"] for row in entry["rows"]} == _expected_row_ids(entry)


def test_moo3_fragment_cites_oneill_and_eggins_2002() -> None:
    manifest = _json_resource("data/coefficient-ancestry.json")
    molybdenum = _artifact(manifest["artifacts"], "openimcc-review-gamma-activity.v1:mo")
    row = next(
        row
        for row in molybdenum["rows"]
        if row["id"] == "MoO3-Oneill-Eggins-2002-Table7-CMAS-1400C"
    )

    assert any(
        source["work_id"] == "oneill-2002-feo-activity-coefficients-cmas"
        and "O'Neill" in source["citation"]
        for source in row["sources"]
    )


def test_sossi_fegley_table_two_is_a_coefficient_source() -> None:
    manifest = _json_resource("data/coefficient-ancestry.json")
    rows = (
        ("as", "AsO1.5-1573K", "10.1007/s11663-010-9430-0"),
        ("bi", "BiO1.5-1573K", "10.1179/1743285512Y.0000000033"),
        ("ge", "GeO2-1573K", "10.1007/s11663-016-0759-x"),
        ("cu", "CuO0.5-1573K", None),
        ("cu", "CuO0.5-1923K", None),
    )
    for element, row_id, primary_doi in rows:
        artifact = _artifact(
            manifest["artifacts"],
            f"openimcc-review-gamma-activity.v1:{element}",
        )
        row = next(row for row in artifact["rows"] if row["id"] == row_id)
        work_ids = {source["work_id"] for source in row["sources"]}

        assert "kems-041-sossi-fegley-2018" in work_ids
        if primary_doi:
            assert any(
                item["row"] == row_id
                and primary_doi in item["citation"]
                and item["reason"]
                == (
                    "No curated mapping resolves this DOI to a work ID "
                    "in the vocabulary."
                )
                for item in artifact["unmapped"]
            )

    cu = _artifact(
        manifest["artifacts"], "openimcc-review-gamma-activity.v1:cu"
    )
    cu_sources = {
        row["id"]: {source["work_id"] for source in row["sources"]}
        for row in cu["rows"]
    }
    assert "holzheid-lodders-2001-copper-solubility" in cu_sources["CuO0.5-1573K"]
    assert "wood-wade-2013-trace-activities" in cu_sources["CuO0.5-1923K"]


def test_manifest_work_ids_match_the_recorded_vocabulary() -> None:
    manifest = _json_resource("data/coefficient-ancestry.json")
    snapshot = json.loads(
        (ROOT / "tests" / "data" / "coefficient-ancestry-vocabulary.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = manifest["vocabulary_snapshot"]
    encoded_ids = json.dumps(
        snapshot["work_ids"], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    # Pin the vocabulary to the simulator green that added the cited source ID.
    assert snapshot["commit"] == "3645e2c2bb"
    assert metadata["commit"] == snapshot["commit"]
    assert metadata["sha256"] == hashlib.sha256(encoded_ids).hexdigest()

    vocabulary = set(snapshot["work_ids"])
    referenced_ids: set[str] = set()
    for entry in manifest["artifacts"]:
        used_ids = {
            source["work_id"]
            for row in entry["rows"]
            for source in row["sources"]
        }
        assert used_ids <= vocabulary
        assert entry["work_ids"] == sorted(used_ids)
        referenced_ids.update(used_ids)
    assert vocabulary == referenced_ids


def test_unmapped_citations_do_not_match_a_vocabulary_work_doi() -> None:
    manifest = _json_resource("data/coefficient-ancestry.json")
    snapshot = json.loads(
        (ROOT / "tests" / "data" / "coefficient-ancestry-vocabulary.json").read_text(
            encoding="utf-8"
        )
    )
    vocabulary_dois = {
        re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi.strip().casefold()).rstrip(
            ".,;:)"
        )
        for doi in snapshot["work_dois"].values()
    }
    assert vocabulary_dois

    doi_pattern = re.compile(r"10\.[0-9]{4,9}/[^\s<>\]\[{}\"'?#]+", re.I)
    for artifact in manifest["artifacts"]:
        for item in artifact["unmapped"]:
            citation_dois = {
                match.group(0).casefold().rstrip(".,;:)")
                for match in doi_pattern.finditer(item["citation"])
            }
            assert not (citation_dois & vocabulary_dois), (
                artifact["identity"], item["row"], item["citation"]
            )


def test_manifest_regeneration_is_byte_identical() -> None:
    result = subprocess.run(
        [sys.executable, "tools/build_coefficient_ancestry.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "byte-identical" in result.stdout


def test_provenance_citations_survive_mapping_and_role_assignment() -> None:
    entries = coefficient_ancestry()
    melt = _artifact(entries, "IMCC-SF04-D066@1.0.6-d066-v4")
    redox = _artifact(entries, "1.8.0")

    def citations(artifact: dict, row_id: str) -> list[tuple[str, str]]:
        row = next(item for item in artifact["rows"] if item["id"] == row_id)
        mapped = [(item["citation"], item["role"]) for item in row["sources"]]
        unmapped = [
            (item["citation"], "unmapped")
            for item in artifact["unmapped"]
            if item["row"] == row_id
        ]
        return mapped + unmapped

    melt_citations = citations(melt, "4:MgTiO3")
    assert any("JANAF" in citation and role == "derived" for citation, role in melt_citations)
    assert any(
        source["work_id"] == "janaf-4th"
        for source in next(row for row in melt["rows"] if row["id"] == "4:MgTiO3")["sources"]
    )

    condensate = _artifact(entries, "condensate-default-v1")
    ti_row = next(row for row in condensate["rows"] if row["id"] == "11:TiO2(l)")
    assert any(source["role"] == "fitted" for source in ti_row["sources"])

    reaction_two = citations(redox, "reaction:2")
    assert any("Lange and Navrotsky (1992)" in citation for citation, _ in reaction_two)

    reaction_seven = next(row for row in redox["rows"] if row["id"] == "reaction:7")
    mapped_anchors = [
        source for source in reaction_seven["sources"]
        if source.get("provenance_role", "").startswith("direct Cr K-edge XANES")
    ]
    assert any(
        source["work_id"] == "berry-oneill-2004-chromium-xanes"
        and source["role"] == "anchor"
        for source in mapped_anchors
    )
    assert any(
        source["work_id"] == "berry-2006-chromium-composition"
        and source["role"] == "anchor"
        for source in mapped_anchors
    )
    assert any(
        source["work_id"] == "gca-366-210-berry-2024"
        for row in _artifact(entries, "liquid-standards-v2")["rows"]
        for source in row["sources"]
    )
    held_out = [
        item for item in redox["unmapped"]
        if item["row"] == "reaction:7"
        and item.get("provenance_role") == "independent chemical redox titration"
    ]
    assert held_out and all(
        item["role"] == "diagnostic" for item in held_out
    )


def test_prose_citation_decisions_cover_detected_rows() -> None:
    melt_data = _json_resource("data/packs/imcc-sf04-d066-v4.json")
    sidecar = json.loads(
        (ROOT / "tools" / "coefficient-ancestry-prose-citations.json").read_text(
            encoding="utf-8"
        )
    )
    decisions = sidecar["artifacts"]["melt"]
    assert melt_data["rows"][3]["source_refs"] == ["FC87"]
    known_authors = (
        "kelley", "robie", "hemingway", "pankratz", "stebbins", "chan",
        "allibert", "rein", "chipman", "blander", "pelton", "stout",
        "piwinski", "proks", "richet", "bottinga", "janaf", "shornikov",
        "glushko", "beyer", "schairer", "bowen", "bell", "roseboom",
        "berry", "masson", "darken", "gurry", "lange", "navrotsky",
        "chen", "sugawara", "akaogi", "king", "barin", "jeon", "zaitseva",
        "ye", "brisi", "abbattista", "liang", "costa", "jacobson", "toropov",
        "bondar", "nasiri", "chase", "navarro", "ilatovskaya", "fabrichnaya",
    )

    def appears_to_cite(text: str) -> bool:
        folded = text.casefold()
        return bool(
            re.search(r"\b(?:19|20)\d{2}[a-z]?\b", text)
            or re.search(r"\bet\s+al\.", text, re.I)
            or any(
                re.search(
                    rf"(?<![a-z0-9_]){re.escape(author)}(?![a-z0-9_])",
                    folded,
                )
                for author in known_authors
            )
            or re.search(r"\bdata\s+from\s+[A-Z][A-Z0-9-]+", text)
        )

    flagged = {
        f"{row['row']}:{row['complex']}"
        for row in melt_data["rows"]
        if appears_to_cite(row.get("derivation", ""))
    }
    assert "4:MgTiO3" in flagged
    row_four = melt_data["rows"][3]
    structured_citations = " ".join(
        melt_data["sources"][key]["citation"]
        for key in row_four["source_refs"]
    )
    # The fc26f9d inputs only expose FC87 here; the prose-only JANAF cite needs the sidecar.
    assert "JANAF" not in structured_citations
    assert any(
        item["citation"] == "JANAF" and item["work_id"] == "janaf-4th"
        for item in decisions["4:MgTiO3"]["citations"]
    )
    assert flagged <= set(decisions)
    assert all(
        bool(decision.get("citations")) != bool(decision.get("none"))
        for decision in (decisions[row_id] for row_id in flagged)
    )
    prose_citations = {
        item["citation"]
        for decision in decisions.values()
        for item in decision.get("citations", [])
    }
    assert {
        "Kelley", "Robie et al.", "Hemingway et al.", "Pankratz & Kelley",
        "Stebbins et al.", "Chan et al.", "Allibert et al.", "Rein & Chipman",
        "Blander & Pelton", "Proks et al.",
    } <= prose_citations
    assert any("Stout & Piwinskii" in citation for citation in prose_citations)

    redox_data = _json_resource("data/packs/openimcc-redox-v6.json")
    redox_decisions = sidecar["artifacts"]["redox"]
    redox_rows = [
        (f"reaction:{index}", row)
        for index, row in enumerate(redox_data["rows"], start=1)
    ] + [
        (f"thermo_species:{name}", row)
        for name, row in redox_data["thermo_species"].items()
    ]
    def nested_text(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return " ".join(nested_text(child) for child in value.values())
        if isinstance(value, list):
            return " ".join(nested_text(child) for child in value)
        return ""

    redox_flagged = set()
    base_redox_flagged = set()
    for row_id, row in redox_rows:
        provenance = row.get("provenance", {})
        base_prose = " ".join(
            str(provenance.get(key, ""))
            for key in ("selection_rationale",)
        ) + " " + " ".join(
            str(row.get(key, ""))
            for key in ("derivation", "phase_or_construction", "method")
        )
        if appears_to_cite(base_prose):
            base_redox_flagged.add(row_id)
        prose = base_prose + " " + nested_text(row.get("construction", {}))
        if appears_to_cite(prose):
            redox_flagged.add(row_id)
    new_construction_flags = redox_flagged - base_redox_flagged
    assert new_construction_flags == {
        "thermo_species:Li2SiO3(l)", "thermo_species:Zn2SiO4(l)"
    }
    assert redox_flagged <= set(redox_decisions)
    assert all(
        bool(redox_decisions[row_id].get("citations"))
        != bool(redox_decisions[row_id].get("none"))
        for row_id in redox_flagged
    )

    condensate_decisions = sidecar["artifacts"]["condensate"]
    with (ROOT / "src/openimcc/data/gas/condensate-default-v1.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        condensate_flagged = {
            f"{index}:{row['species_name']}"
            for index, row in enumerate(csv.DictReader(handle), start=1)
            if appears_to_cite(row.get("construction_method", ""))
        }
    assert condensate_flagged <= set(condensate_decisions)

    spinel = decisions["3:MgAl2O4"]["citations"]
    assert any(
        item["citation"] == "independent Richet et al. DTA value is 53.50 ± 5.50 kJ/mol"
        and item["work_id"] is None
        for item in spinel
    )
    melt = _artifact(coefficient_ancestry(), "IMCC-SF04-D066@1.0.6-d066-v4")
    spinel_row = next(row for row in melt["rows"] if row["id"] == "3:MgAl2O4")
    assert any(
        item["row"] == "3:MgAl2O4"
        and item["citation"] == "independent Richet et al. DTA value is 53.50 ± 5.50 kJ/mol"
        for item in melt["unmapped"]
    )
    assert any(source["work_id"] == "janaf-4th" for source in spinel_row["sources"])

    redox = _artifact(coefficient_ancestry(), "1.8.0")
    for species in ("Si(l)", "Cr(l)", "Ni(l)", "Co(l)"):
        row = next(row for row in redox["rows"] if row["id"] == f"thermo_species:{species}")
        assert row["sources"] and all(source["role"] == "fitted" for source in row["sources"])
    hematite_unmapped = [item["citation"] for item in redox["unmapped"] if item["row"] == "thermo_species:Fe2O3(l)"]
    assert any("Lange and Navrotsky (1992)" in citation for citation in hematite_unmapped)
    assert any("10.1007/bf00310746" in citation.casefold() for citation in hematite_unmapped)
    reaction_seven_unmapped = [item["citation"] for item in redox["unmapped"] if item["row"] == "reaction:7"]
    assert not any("dataset anchors and independent checks" in citation for citation in reaction_seven_unmapped)
