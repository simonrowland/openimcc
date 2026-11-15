#!/usr/bin/env python3
"""Build the packaged ancestry manifest for openimcc's default data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "openimcc" / "data"
OUTPUT = DATA / "coefficient-ancestry.json"
MAPPINGS = ROOT / "tools" / "coefficient-ancestry-sources.json"
VOCABULARY = ROOT / "tests" / "data" / "coefficient-ancestry-vocabulary.json"
ROLES = {"derived", "fitted", "selected", "anchor", "diagnostic"}
DOI_RE = re.compile(r"10\.[0-9]{4,9}/[^\s<>\]\[{}\"']+", re.IGNORECASE)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _normalized_doi(value: str) -> str:
    doi = value.strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    return doi.rstrip(".,;:)")


def _segments(value: str) -> list[str]:
    return [
        segment.strip()
        for segment in value.split(";")
        if segment.strip()
        and not segment.strip().lower().startswith(("src/", "tests/"))
        and not segment.strip().casefold().startswith(
            ("thermo_species.", "parent_standard_candidates.", "rows.")
        )
    ]


class SourceMapper:
    def __init__(self, mapping: dict[str, Any], vocabulary: set[str]) -> None:
        self.dois = {
            _normalized_doi(doi): work_id
            for doi, work_id in mapping["dois"].items()
        }
        self.citation_keys = {
            key.casefold(): work_id
            for key, work_id in mapping["citation_keys"].items()
        }
        self.patterns = [
            (re.compile(item["pattern"], re.IGNORECASE), item["work_id"])
            for item in mapping["patterns"]
        ]
        self.vocabulary = vocabulary
        for work_id in [*self.dois.values(), *self.citation_keys.values()]:
            self._check_work_id(work_id)
        for _pattern, work_id in self.patterns:
            self._check_work_id(work_id)

    def _check_work_id(self, work_id: str) -> None:
        if work_id not in self.vocabulary:
            raise ValueError(f"mapped work ID is absent from the vocabulary: {work_id}")

    def resolve(self, citation: str, citation_key: str | None = None) -> list[str]:
        evidence = " ".join(value for value in (citation_key, citation) if value)
        work_ids: list[str] = []
        for match in DOI_RE.finditer(evidence):
            work_id = self.dois.get(_normalized_doi(match.group(0)))
            if work_id:
                work_ids.append(work_id)
        folded = evidence.casefold()
        for key, work_id in self.citation_keys.items():
            if key in folded:
                work_ids.append(work_id)
        candidates = [citation_key, citation]
        for pattern, work_id in self.patterns:
            if any(value and pattern.search(value.strip()) for value in candidates):
                work_ids.append(work_id)
        return list(dict.fromkeys(work_ids))

    def unmapped_reason(self, citation: str) -> str:
        dois = [_normalized_doi(m.group(0)) for m in DOI_RE.finditer(citation)]
        if dois:
            return "No curated mapping resolves this DOI to a work ID in the vocabulary."
        return "No curated citation-key mapping resolves to a work ID in the vocabulary."


def _row(row_id: str, label: str) -> dict[str, Any]:
    return {"id": row_id, "label": label, "sources": [], "unmapped": []}


def _add_citation(
    row: dict[str, Any],
    citation: str,
    role: str,
    mapper: SourceMapper,
    *,
    citation_key: str | None = None,
) -> None:
    if role not in ROLES:
        raise ValueError(f"invalid ancestry role: {role}")
    chunks = _segments(citation)
    if not chunks:
        return
    for chunk in chunks:
        work_ids = mapper.resolve(chunk, citation_key)
        if not work_ids:
            item = {
                "row": row["id"],
                "citation": chunk,
                "reason": mapper.unmapped_reason(chunk),
            }
            if item not in row["unmapped"]:
                row["unmapped"].append(item)
            continue
        for work_id in work_ids:
            item = {"work_id": work_id, "role": role, "citation": chunk}
            if item not in row["sources"]:
                row["sources"].append(item)


def _missing_provenance(row: dict[str, Any]) -> None:
    if row["sources"] or row["unmapped"]:
        return
    row["unmapped"].append(
        {
            "row": row["id"],
            "citation": "No source citation is recorded in the row provenance.",
            "reason": "The row has no source citation to resolve to a vocabulary work ID.",
        }
    )


def _finish_artifact(
    *,
    identity: str,
    file: str,
    raw: bytes,
    rows: list[dict[str, Any]],
    runtime_digest: str,
) -> dict[str, Any]:
    work_ids = sorted(
        {source["work_id"] for row in rows for source in row["sources"]}
    )
    unmapped = [item for row in rows for item in row["unmapped"]]
    row_entries = [
        {key: value for key, value in row.items() if key != "unmapped"}
        for row in rows
    ]
    return {
        "identity": identity,
        "file": file,
        "digest": _sha256(raw),
        "runtime_digest": runtime_digest,
        "rows": row_entries,
        "work_ids": work_ids,
        "unmapped": unmapped,
        "complete": not unmapped,
        "counts": {
            "rows": len(rows),
            "mapped_rows": sum(bool(row["sources"]) for row in rows),
            "unmapped_citations": len(unmapped),
            "rows_with_unmapped": sum(bool(row["unmapped"]) for row in rows),
        },
    }


def _melt_artifact(mapper: SourceMapper, runtime_digest: str) -> dict[str, Any]:
    path = DATA / "packs" / "imcc-sf04-d066-v4.json"
    raw = path.read_bytes()
    pack = json.loads(raw)
    rows: list[dict[str, Any]] = []
    for source_row in pack["rows"]:
        complex_name = source_row["complex"]
        row_id = f"{source_row['row']}:{complex_name}"
        record = _row(row_id, complex_name)
        derivation = source_row.get("derivation", "").casefold()
        for source_key in source_row.get("source_refs", []):
            source = pack.get("sources", {}).get(source_key, {})
            citation = source.get("citation", source_key)
            doi = source.get("doi")
            if doi:
                citation = f"{citation}. DOI {doi}"
            surname = citation.split(",", 1)[0].split()[0].casefold().rstrip(".,")
            role = "selected" if "select" in derivation and surname in derivation else "derived"
            _add_citation(
                record, citation, role, mapper, citation_key=source_key
            )

        refit = source_row.get("standard_state_refit")
        if isinstance(refit, dict):
            candidates = pack.get("parent_standard_candidates", {})
            for parent, coefficient in refit.get("parents", {}).items():
                if not coefficient or parent not in candidates:
                    continue
                source = candidates[parent]
                citation = source.get("source", parent)
                doi = source.get("doi")
                if doi:
                    citation = f"{citation}. DOI {doi}"
                _add_citation(record, citation, "anchor", mapper)
        _missing_provenance(record)
        rows.append(record)
    return _finish_artifact(
        identity=f"{pack['model_id']}@{pack['imcc_sf04_datapack_version']}",
        file="data/packs/" + path.name,
        raw=raw,
        rows=rows,
        runtime_digest=runtime_digest,
    )


def _table_artifact(
    *,
    relative_file: str,
    identity: str,
    role: str,
    mapper: SourceMapper,
    runtime_digest: str,
) -> dict[str, Any]:
    path = DATA / relative_file
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, source_row in enumerate(csv.DictReader(handle), start=1):
            species = source_row["species_name"]
            interval = source_row.get("T_interval")
            row_id = f"{index}:{species}" + (f":{interval}" if interval else "")
            record = _row(row_id, species)
            ref = source_row.get("Ref", "").strip()
            construction = source_row.get("construction_method", "none").strip()
            row_role = "anchor" if ref.startswith("LFS[") else role
            if construction not in ("", "none") and row_role != "anchor":
                row_role = "derived"
            _add_citation(
                record,
                ref,
                row_role,
                mapper,
                citation_key=ref,
            )
            for key in ("source_label", "table_id"):
                value = source_row.get(key, "").strip()
                if value and value.casefold() != "none" and value != ref:
                    _add_citation(record, value, row_role, mapper)
            _missing_provenance(record)
            rows.append(record)
    return _finish_artifact(
        identity=identity,
        file="data/" + relative_file,
        raw=raw,
        rows=rows,
        runtime_digest=runtime_digest,
    )


def _redox_rows(content: dict[str, Any], mapper: SourceMapper) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_rows = [
        (f"reaction:{index}", item["reaction"], item)
        for index, item in enumerate(content["rows"], start=1)
    ]
    source_rows.extend(
        (f"thermo_species:{name}", name, item)
        for name, item in content["thermo_species"].items()
    )
    for row_id, label, source_row in source_rows:
        record = _row(row_id, label)
        provenance = source_row.get("provenance", {})
        source_role = "selected" if provenance.get("selection_rationale") else "derived"
        source = provenance.get("source")
        if isinstance(source, str):
            for chunk in _segments(source):
                _add_citation(record, chunk, source_role, mapper)
        source_url = provenance.get("source_url")
        if isinstance(source_url, str) and source_url:
            _add_citation(record, source_url, source_role, mapper)

        construction = source_row.get("construction", {})
        if isinstance(construction, dict):
            for key, value in construction.items():
                if not isinstance(value, str) or not any(
                    token in key.casefold() for token in ("source", "citation", "anchor")
                ):
                    continue
                role = "anchor" if "anchor" in key.casefold() else "derived"
                _add_citation(record, value, role, mapper)

        check = source_row.get("independent_check", {})
        if isinstance(check, dict):
            primary_source = check.get("primary_source")
            if isinstance(primary_source, str) and primary_source.strip():
                _add_citation(record, primary_source, "diagnostic", mapper)
        _missing_provenance(record)
        rows.append(record)
    return rows


def _redox_artifact(mapper: SourceMapper, runtime_digest: str) -> dict[str, Any]:
    path = DATA / "packs" / "openimcc-redox-v6.json"
    raw = path.read_bytes()
    content = json.loads(raw)
    return _finish_artifact(
        identity=content["version"],
        file="data/packs/" + path.name,
        raw=raw,
        rows=_redox_rows(content, mapper),
        runtime_digest=runtime_digest,
    )


def _standards_artifact(mapper: SourceMapper) -> dict[str, Any]:
    path = DATA / "liquid-standards.json"
    raw = path.read_bytes()
    content = json.loads(raw)
    rows: list[dict[str, Any]] = []
    for pack_digest, pack in content["packs"].items():
        standards = pack.get("standards", {})
        citations = pack.get("standard_citations", {})
        bases = pack.get("standard_basis", {})
        for standard, standard_id in standards.items():
            row_id = f"{pack_digest}:standard:{standard}"
            record = _row(row_id, standard)
            citation = citations.get(standard, standard_id)
            basis = bases.get(standard, "source")
            role = "derived" if basis == "translation-record" else "selected"
            record["basis"] = basis
            _add_citation(record, citation, role, mapper, citation_key=standard_id)
            if not record["sources"] and not record["unmapped"]:
                _missing_provenance(record)
            rows.append(record)

        for complex_name, overrides in pack.get("row_overrides", {}).items():
            override_citations = pack.get("row_override_citations", {}).get(
                complex_name, {}
            )
            override_bases = pack.get("row_override_basis", {}).get(
                complex_name, {}
            )
            for standard, standard_id in overrides.items():
                row_id = f"{pack_digest}:override:{complex_name}:{standard}"
                record = _row(row_id, f"{complex_name}/{standard}")
                citation = override_citations.get(standard, standard_id)
                basis = override_bases.get(standard, "source")
                role = "derived" if basis == "translation-record" else "selected"
                record["basis"] = basis
                _add_citation(record, citation, role, mapper, citation_key=standard_id)
                _missing_provenance(record)
                rows.append(record)

        for complex_name, diagnostic in pack.get("row_diagnostics", {}).items():
            row_id = f"{pack_digest}:diagnostic:{complex_name}"
            record = _row(row_id, complex_name)
            components = diagnostic.get("component_sources", {})
            for citation in components.values():
                if isinstance(citation, str):
                    _add_citation(record, citation, "diagnostic", mapper)
            _missing_provenance(record)
            rows.append(record)

    return _finish_artifact(
        identity=f"liquid-standards-v{content['schema_version']}",
        file="data/" + path.name,
        raw=raw,
        rows=rows,
        runtime_digest=_sha256(raw),
    )


def _janaf_artifact(
    *,
    relative_file: str,
    identity: str,
    collection: str,
    mapper: SourceMapper,
) -> dict[str, Any]:
    path = DATA / relative_file
    raw = path.read_bytes()
    content = json.loads(raw)
    rows: list[dict[str, Any]] = []
    for source_row in content[collection]:
        label = source_row.get("id") or source_row.get("table_id") or source_row.get("oxide")
        table_id = source_row.get("crystal_table_id") or source_row.get("table_id")
        record = _row(f"{collection}:{label}", str(label))
        source = content.get("source", "Chase (1998), NIST-JANAF Thermochemical Tables")
        citation = f"{source}; table {table_id}" if table_id else source
        _add_citation(record, citation, "derived", mapper, citation_key=table_id)
        _missing_provenance(record)
        rows.append(record)
    return _finish_artifact(
        identity=identity,
        file="data/" + relative_file,
        raw=raw,
        rows=rows,
        runtime_digest=_sha256(raw),
    )


def build() -> dict[str, Any]:
    mapping = _json(MAPPINGS)
    vocabulary = _json(VOCABULARY)
    work_ids = vocabulary["work_ids"]
    if work_ids != sorted(set(work_ids)):
        raise ValueError("the recorded work-ID vocabulary must be sorted and unique")
    mapper = SourceMapper(mapping, set(work_ids))

    sys.path.insert(0, str(ROOT / "src"))
    from openimcc.profiles import load_profile

    profile = load_profile("openimcc")
    artifacts = [
        _melt_artifact(mapper, profile.melt_pack.binding_digest or ""),
        _table_artifact(
            relative_file="gas/gas-shomate-default-v1.csv",
            identity="gas-shomate-default-v1",
            role="fitted",
            mapper=mapper,
            runtime_digest=profile.gas_datapack.gas_table_digest,
        ),
        _table_artifact(
            relative_file="gas/condensate-default-v1.csv",
            identity="condensate-default-v1",
            role="fitted",
            mapper=mapper,
            runtime_digest=profile.gas_datapack.condensate_table_digest,
        ),
        _redox_artifact(mapper, profile.redox_pack.canonical_digest),
        _standards_artifact(mapper),
        _janaf_artifact(
            relative_file="diagnostics/janaf-pure-oxide-crystals.json",
            identity="openimcc-janaf-apparent-crystals-v2",
            collection="phases",
            mapper=mapper,
        ),
        _janaf_artifact(
            relative_file="diagnostics/janaf-pure-oxide-liquid-pairs.json",
            identity="openimcc-janaf-apparent-liquid-pairs-v1",
            collection="rows",
            mapper=mapper,
        ),
    ]
    vocabulary_bytes = _canonical_json(work_ids)
    return {
        "schema": "openimcc-coefficient-ancestry.v1",
        "vocabulary_snapshot": {
            "commit": vocabulary["commit"],
            "directories": vocabulary["directories"],
            "work_id_count": len(work_ids),
            "sha256": _sha256(vocabulary_bytes),
        },
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless the checked-in manifest matches regenerated bytes",
    )
    arguments = parser.parse_args()
    generated = (json.dumps(build(), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if arguments.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != generated:
            print(f"{OUTPUT.relative_to(ROOT)} is not regenerated; rerun this tool")
            return 1
        print("coefficient ancestry manifest is byte-identical")
        return 0
    OUTPUT.write_bytes(generated)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
