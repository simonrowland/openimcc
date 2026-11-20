#!/usr/bin/env python3
"""Build the packaged ancestry manifest for openimcc's default data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "openimcc" / "data"
OUTPUT = DATA / "coefficient-ancestry.json"
MAPPINGS = ROOT / "tools" / "coefficient-ancestry-sources.json"
VOCABULARY = ROOT / "tests" / "data" / "coefficient-ancestry-vocabulary.json"
ROLES = {"derived", "fitted", "selected", "anchor", "diagnostic"}
DOI_RE = re.compile(r"10\.[0-9]{4,9}/[^\s<>\]\[{}\"']+", re.IGNORECASE)
JANAF_TABLE_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z][a-z]?-[0-9]{3})(?![A-Za-z0-9])")
PROSE_CITATIONS = ROOT / "tools" / "coefficient-ancestry-prose-citations.json"
SOSSI_FEGLEY_2018_KEY = "sossi & fegley (2018)"


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


def _check_vocabulary_index(vocabulary: dict[str, Any], index: dict[str, Any]) -> None:
    sources = index.get("sources")
    if not isinstance(sources, list):
        raise ValueError("INDEX.yaml must contain a sources list")
    source_by_id = {
        source["source_id"]: source
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("source_id"), str)
    }
    missing_ids = sorted(set(vocabulary["work_ids"]) - source_by_id.keys())
    if missing_ids:
        raise ValueError(f"work IDs missing from INDEX.yaml: {', '.join(missing_ids)}")
    for work_id, expected_doi in vocabulary["work_dois"].items():
        source = source_by_id.get(work_id)
        actual_doi = _normalized_doi(str(source.get("doi", ""))) if source else ""
        if actual_doi != _normalized_doi(expected_doi):
            raise ValueError(f"DOI mismatch for work ID {work_id}")


def _vocabulary_index_from_repo(repo: Path, commit: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "show",
            f"{commit}:data/literature/INDEX.yaml",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(
            f"could not read data/literature/INDEX.yaml at {commit}: {result.stderr.strip()}"
        )
    import yaml

    index = yaml.safe_load(result.stdout)
    if not isinstance(index, dict):
        raise ValueError("INDEX.yaml must contain a mapping")
    return index


def _segments(value: str) -> list[str]:
    segments: list[str] = []
    for clause in value.split(";"):
        clause = re.sub(
            r"([A-Z][A-Za-z’'-]+),\s*([A-Z][A-Za-z’'-]+)\s+(?:&|and)\s+([A-Z][A-Za-z’'-]+)(\s+\((?:19|20)[0-9]{2}[a-z]?\))",
            lambda match: match.group(0).replace(",", "\u241f", 1),
            clause,
        )
        pieces = re.split(
            r",\s+and\s+(?=[A-Z])|,(?=\s*[A-Z][^,;]{0,100}\(?(?:19|20)[0-9]{2})|,(?=\s*[A-Z][A-Za-z’'-]+\s+(?:and|&)\s+[A-Z])",
            clause,
        )
        for piece in pieces:
            segment = piece.replace("\u241f", ",").strip(" ,")
            if (
                segment
                and not segment.lower().startswith(("src/", "tests/"))
                and not segment.casefold().startswith(
                    ("thermo_species.", "parent_standard_candidates.", "rows.")
                )
            ):
                segments.append(segment)
    return segments


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

    def _candidates(self, citation: str, citation_key: str | None = None) -> list[str]:
        evidence = " ".join(value for value in (citation_key, citation) if value)
        work_ids: list[str] = []
        for match in DOI_RE.finditer(evidence):
            work_id = self.dois.get(_normalized_doi(match.group(0)))
            if work_id:
                work_ids.append(work_id)
        for key, work_id in self.citation_keys.items():
            if _contains_bounded_citation(key, evidence):
                work_ids.append(work_id)
        for pattern, work_id in self.patterns:
            if any(
                value and pattern.search(value.strip())
                for value in (citation_key, citation)
            ):
                work_ids.append(work_id)
        return list(dict.fromkeys(work_ids))

    def resolve(self, citation: str, citation_key: str | None = None) -> list[str]:
        return [
            work_id
            for work_id in self._candidates(citation, citation_key)
            if work_id in self.vocabulary
        ]

    def outside_vocabulary(
        self, citation: str, citation_key: str | None = None
    ) -> list[str]:
        return [
            work_id
            for work_id in self._candidates(citation, citation_key)
            if work_id not in self.vocabulary
        ]

    def unmapped_reason(self, citation: str, citation_key: str | None = None) -> str:
        if self.outside_vocabulary(citation, citation_key):
            return "not in the public store vocabulary"
        dois = [_normalized_doi(m.group(0)) for m in DOI_RE.finditer(citation)]
        if dois:
            return "No curated mapping resolves this DOI to a work ID in the vocabulary."
        return "No curated citation-key mapping resolves to a work ID in the vocabulary."


def _citation_atoms(citation: str) -> list[str]:
    """Keep each DOI and JANAF table citation independently auditable."""
    atoms = [_normalized_doi(match.group(0)) for match in DOI_RE.finditer(citation)]
    atoms.extend(match.group(0) for match in JANAF_TABLE_RE.finditer(citation))
    return list(dict.fromkeys(atoms))


def _contains_bounded_citation(citation: str, text: str) -> bool:
    normalized_citation = re.sub(r"[^a-z0-9]+", " ", citation.casefold()).strip()
    if not normalized_citation:
        return False
    normalized_text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_citation)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def _role_from_evidence(text: str, default: str = "derived") -> str:
    folded = text.casefold()
    if re.search(r"(?<![a-z])fit(?:ted|ting)?(?![a-z])", folded):
        return "fitted"
    if re.search(r"(?<![a-z])select(?:ed|s|ion)?(?![a-z])", folded):
        return "selected"
    if re.search(r"(?<![a-z])anchor(?:ed|s)?(?![a-z])", folded):
        return "anchor"
    return default


def _redox_source_role(
    citation: str, provenance: dict[str, Any], default: str
) -> str:
    rationale = str(provenance.get("selection_rationale", "")).casefold()
    dated_reference = re.match(r"(.+?\((?:19|20)[0-9]{2}\))", citation)
    reference = dated_reference.group(1) if dated_reference else citation.split(",", 1)[0]
    if "select" in rationale and _contains_bounded_citation(reference, rationale):
        return "selected"
    return default


def _text_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _text_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _text_values(child)


def _merge_prose_citations(
    record: dict[str, Any], sidecar: dict[str, Any], artifact: str, row_id: str
) -> None:
    decision = sidecar.get("artifacts", {}).get(artifact, {}).get(row_id)
    if decision is None:
        return
    for citation in decision.get("citations", []):
        record["sources"].append(
            {
                "work_id": citation["work_id"],
                "role": citation["role"],
                "citation": citation["citation"],
            }
        ) if citation.get("work_id") else record["unmapped"].append(
            {
                "row": row_id,
                "citation": citation["citation"],
                "reason": citation.get("reason", "No public vocabulary work ID is known for this citation."),
                "role": citation["role"],
            }
        )


def _drop_resolved_unmapped(rows: list[dict[str, Any]], mapper: SourceMapper) -> None:
    for row in rows:
        resolved = {source["work_id"] for source in row["sources"]}
        retained = []
        for item in row["unmapped"]:
            if (
                item["reason"]
                == "No curated mapping resolves this DOI to a work ID in the vocabulary."
                and _contains_bounded_citation(
                    SOSSI_FEGLEY_2018_KEY, item["citation"]
                )
            ):
                retained.append(item)
                continue
            if resolved.intersection(mapper.resolve(item["citation"])):
                continue
            retained.append(item)
        row["unmapped"] = retained


def _structured_citation_fields(value: Any):
    if isinstance(value, list):
        for child in value:
            yield from _structured_citation_fields(child)
    elif isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str) and any(
                token in key.casefold()
                for token in ("source", "citation", "anchor", "doi", "url")
            ):
                yield key, child
            elif isinstance(child, (dict, list)):
                yield from _structured_citation_fields(child)


def _add_structured_citation(
    row: dict[str, Any], citation: str, role: str, mapper: SourceMapper
) -> None:
    candidates = set(mapper.resolve(citation))
    resolved = {source["work_id"] for source in row["sources"]}
    if candidates and candidates <= resolved:
        return
    _add_citation(row, citation, role, mapper)


def _row(row_id: str, label: str) -> dict[str, Any]:
    return {"id": row_id, "label": label, "sources": [], "unmapped": []}


def _add_citation_evidence(
    row: dict[str, Any],
    citation: str,
    role: str,
    mapper: SourceMapper,
    *,
    citation_key: str | None = None,
    provenance_role: str | None = None,
    preserve_full_citation: bool = False,
    resolve_citation_key_with_atoms: bool = False,
) -> None:
    atoms = _citation_atoms(citation)
    atom_work_ids: set[str] = set()
    for atom in atoms:
        work_ids = mapper.resolve(atom)
        atom_work_ids.update(work_ids)
        for work_id in work_ids:
            item = {
                "work_id": work_id,
                "role": role,
                "citation": citation if preserve_full_citation else atom,
            }
            if provenance_role:
                item["provenance_role"] = provenance_role
            if item not in row["sources"]:
                row["sources"].append(item)
        if not work_ids:
            item = {
                "row": row["id"],
                "citation": citation if preserve_full_citation else atom,
                "reason": mapper.unmapped_reason(atom),
                "role": role,
            }
            if provenance_role:
                item["provenance_role"] = provenance_role
            if item not in row["unmapped"]:
                row["unmapped"].append(item)

    text_work_ids = mapper.resolve(citation, citation_key) if not atoms else (
        mapper.resolve(citation_key)
        if resolve_citation_key_with_atoms and citation_key
        else []
    )
    for work_id in text_work_ids:
        if work_id in atom_work_ids:
            continue
        item = {"work_id": work_id, "role": role, "citation": citation}
        if provenance_role:
            item["provenance_role"] = provenance_role
        if item not in row["sources"]:
            row["sources"].append(item)

    if not atoms and not text_work_ids:
        item = {
            "row": row["id"],
            "citation": citation,
            "reason": mapper.unmapped_reason(citation, citation_key),
            "role": role,
        }
        if provenance_role:
            item["provenance_role"] = provenance_role
        if item not in row["unmapped"]:
            row["unmapped"].append(item)


def _add_citation(
    row: dict[str, Any],
    citation: str,
    role: str,
    mapper: SourceMapper,
    *,
    citation_key: str | None = None,
    provenance_role: str | None = None,
) -> None:
    if role not in ROLES:
        raise ValueError(f"invalid ancestry role: {role}")
    for chunk in _segments(citation):
        _add_citation_evidence(
            row,
            chunk,
            role,
            mapper,
            citation_key=citation_key,
            provenance_role=provenance_role,
        )


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
    mapper: SourceMapper,
) -> dict[str, Any]:
    _drop_resolved_unmapped(rows, mapper)
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


def _melt_artifact(mapper: SourceMapper, runtime_digest: str, sidecar: dict[str, Any]) -> dict[str, Any]:
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
            role = (
                "selected" if "select" in derivation and surname in derivation
                else "fitted" if "fit" in derivation and surname in derivation
                else "derived"
            )
            _add_citation(
                record, citation, role, mapper, citation_key=source_key
            )

        _merge_prose_citations(record, sidecar, "melt", row_id)

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
        mapper=mapper,
    )


def _table_artifact(
    *,
    relative_file: str,
    identity: str,
    role: str,
    mapper: SourceMapper,
    runtime_digest: str,
    sidecar: dict[str, Any],
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
            if row_role != "anchor" and construction not in ("", "none"):
                row_role = _role_from_evidence(construction)
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
            if identity == "condensate-default-v1":
                _merge_prose_citations(record, sidecar, "condensate", row_id)
            _missing_provenance(record)
            rows.append(record)
    return _finish_artifact(
        identity=identity,
        file="data/" + relative_file,
        raw=raw,
        rows=rows,
        runtime_digest=runtime_digest,
        mapper=mapper,
    )


def _redox_rows(content: dict[str, Any], mapper: SourceMapper, sidecar: dict[str, Any]) -> list[dict[str, Any]]:
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
        role_evidence = " ".join(_text_values(source_row))
        source_role = "derived"
        if re.search(r"\bfitted\s+(?:with|using|to)\b", role_evidence, re.I):
            source_role = "fitted"
        source = provenance.get("source")
        source_roles: dict[str, str] = {}
        if isinstance(source, str):
            for chunk in _segments(source):
                if chunk.casefold().strip(" .") == (
                    "dataset anchors and independent checks are listed separately in dataset_provenance"
                ):
                    continue
                evidenced_role = _role_from_evidence(chunk, source_role)
                evidenced_role = _redox_source_role(chunk, provenance, evidenced_role)
                for work_id in mapper.resolve(chunk):
                    source_roles[work_id] = evidenced_role
                _add_citation(
                    record,
                    chunk,
                    evidenced_role,
                    mapper,
                )
        source_url = provenance.get("source_url")
        if isinstance(source_url, str) and source_url:
            url_roles = {
                source_roles[work_id]
                for work_id in mapper.resolve(source_url)
                if work_id in source_roles
            }
            _add_citation(
                record,
                source_url,
                next(iter(url_roles)) if len(url_roles) == 1 else source_role,
                mapper,
            )

        for key, value in _structured_citation_fields(provenance):
            if key not in {"source", "source_url"}:
                _add_structured_citation(record, value, source_role, mapper)

        construction = source_row.get("construction", {})
        if isinstance(construction, dict):
            for key, value in _structured_citation_fields(construction):
                role = "anchor" if "anchor" in key.casefold() else "derived"
                _add_structured_citation(record, value, role, mapper)

        check = source_row.get("independent_check", {})
        if isinstance(check, dict):
            primary_source = check.get("primary_source")
            if isinstance(primary_source, str) and primary_source.strip():
                _add_citation(record, primary_source, "diagnostic", mapper)

        if row_id == "reaction:7":
            for category, role in (("anchors", "anchor"), ("held_out_independent", "diagnostic")):
                for dataset in content.get("dataset_provenance", {}).get(category, []):
                    citation = dataset.get("citation", "")
                    provenance_role = dataset.get("role", "")
                    dataset_role = (
                        _role_from_evidence(provenance_role, role)
                        if category == "anchors"
                        else "diagnostic"
                    )
                    source_url = dataset.get("source_url", "")
                    _add_citation(
                        record,
                        f"{citation}; {source_url}" if source_url else citation,
                        dataset_role,
                        mapper,
                        provenance_role=provenance_role,
                    )
        _merge_prose_citations(record, sidecar, "redox", row_id)
        _missing_provenance(record)
        rows.append(record)
    return rows


def _redox_artifact(mapper: SourceMapper, runtime_digest: str, sidecar: dict[str, Any]) -> dict[str, Any]:
    path = DATA / "packs" / "openimcc-redox-v6.json"
    raw = path.read_bytes()
    content = json.loads(raw)
    return _finish_artifact(
        identity=content["version"],
        file="data/packs/" + path.name,
        raw=raw,
        rows=_redox_rows(content, mapper, sidecar),
        runtime_digest=runtime_digest,
        mapper=mapper,
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
        mapper=mapper,
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
        mapper=mapper,
    )


def _rh_crystal_species_artifact(mapper: SourceMapper) -> dict[str, Any]:
    relative_file = "diagnostics/rh-crystal-species.json"
    path = DATA / relative_file
    raw = path.read_bytes()
    content = json.loads(raw)
    source = content["source"]
    rows: list[dict[str, Any]] = []
    for phase in content["phases"]:
        record = _row(f"phases:{phase['id']}", phase["formula"])
        phase_source = phase["source"]
        citation = (
            f"{source['citation']} DOI {source['doi']}; "
            f"{phase_source['source_locator']}"
        )
        _add_citation(
            record,
            citation,
            "derived",
            mapper,
            citation_key=source["doi"],
        )
        _missing_provenance(record)
        rows.append(record)
    return _finish_artifact(
        identity=content["schema"],
        file="data/" + relative_file,
        raw=raw,
        rows=rows,
        runtime_digest=_sha256(raw),
        mapper=mapper,
    )


def _barin_artifact(mapper: SourceMapper) -> dict[str, Any]:
    relative_file = "diagnostics/barin-aluminate-crystals.json"
    path = DATA / relative_file
    raw = path.read_bytes()
    content = json.loads(raw)
    rows: list[dict[str, Any]] = []
    for phase in content["phases"]:
        record = _row(f"phases:{phase['id']}", phase["formula"])
        source = phase["source"]
        _add_citation(
            record, source["citation"], "derived", mapper,
            citation_key=source["work_id"],
        )
        for oxide in phase["oxide_stoichiometry"]:
            reference = content["conversion"]["parent_oxide_references"][oxide]
            _add_citation(
                record, reference["citation"], "derived", mapper,
                citation_key=reference["table_id"],
            )
        _missing_provenance(record)
        rows.append(record)
    return _finish_artifact(
        identity=content["schema"],
        file="data/" + relative_file,
        raw=raw,
        rows=rows,
        runtime_digest=_sha256(raw),
        mapper=mapper,
    )


def _crystal_phase_artifact(mapper: SourceMapper) -> dict[str, Any]:
    relative_file = "diagnostics/crystal-phases.json"
    path = DATA / relative_file
    raw = path.read_bytes()
    content = json.loads(raw)
    janaf_crystal_path = DATA / "diagnostics/janaf-pure-oxide-crystals.json"
    janaf_crystal_content = json.loads(janaf_crystal_path.read_bytes())
    janaf_liquid_path = DATA / "diagnostics/janaf-pure-oxide-liquid-pairs.json"
    janaf_liquid_content = json.loads(janaf_liquid_path.read_bytes())
    crystal_rows = {
        row["formula"]: row for row in janaf_crystal_content["phases"]
    }
    liquid_rows = {
        row["oxide"]: row for row in janaf_liquid_content["rows"]
    }
    rows: list[dict[str, Any]] = []
    for phase in content["phases"]:
        record = _row(f"phases:{phase['id']}", phase["formula"])
        source = phase["source"]
        is_berman = source.get("authors") == "Berman"
        if is_berman:
            _add_citation(
                record,
                source["journal"],
                "derived",
                mapper,
                citation_key=source.get("doi"),
            )
            parent_rows = crystal_rows
        else:
            citation = "Robie and Hemingway (1995), USGS Bulletin 2131"
            _add_citation(record, citation, "derived", mapper)
            parent_rows = liquid_rows
        for oxide in phase["oxide_stoichiometry"]:
            if not is_berman and oxide == "FeO":
                _add_citation(
                    record,
                    "Darken & Gurry (1946), Tables IIIA, IIIB, VI, "
                    "selected FeO(l) standard reduced via Masson (1965), Eq. 17",
                    "derived",
                    mapper,
                    citation_key="darken & gurry (1946)",
                )
                continue
            reference = parent_rows[oxide]
            table_id = reference.get("crystal_table_id") or reference["table_id"]
            citation = f"Chase (1998), NIST-JANAF Thermochemical Tables; table {table_id}"
            _add_citation(
                record,
                citation,
                "derived",
                mapper,
                citation_key=table_id,
            )
        _missing_provenance(record)
        rows.append(record)

    for control in content["pure_oxide_controls"]:
        record = _row(
            f"pure_oxide_controls:{control['id']}", control["formula"],
        )
        _add_citation(
            record,
            control["source"]["journal"],
            "derived",
            mapper,
            citation_key=control["source"].get("doi"),
        )
        for table_rows, table_key in (
            (crystal_rows, "crystal_table_id"),
            (liquid_rows, "table_id"),
        ):
            reference = table_rows[control["id"]]
            table_id = reference[table_key]
            citation = f"Chase (1998), NIST-JANAF Thermochemical Tables; table {table_id}"
            _add_citation(
                record,
                citation,
                "derived",
                mapper,
                citation_key=table_id,
            )
        _missing_provenance(record)
        rows.append(record)
    return _finish_artifact(
        identity=content["schema"],
        file="data/" + relative_file,
        raw=raw,
        rows=rows,
        runtime_digest=_sha256(raw),
        mapper=mapper,
    )


def _elemental_reference_crystal_card_artifact(
    mapper: SourceMapper,
) -> dict[str, Any]:
    relative_file = "diagnostics/elemental-reference-crystal-cards.json"
    path = DATA / relative_file
    raw = path.read_bytes()
    content = json.loads(raw)
    rows: list[dict[str, Any]] = []
    for card in content["records"]:
        row = _row(f"records:{card['formula']}", str(card["source_card_id"]))
        if card["form"] == "nasa7":
            citation = card["source_title"]
        else:
            citation = f"Pankratz Bulletin 672 table {card['source_card_id']}"
        _add_citation(
            row,
            citation,
            "derived",
            mapper,
            citation_key=str(card["source_card_id"]),
        )
        _missing_provenance(row)
        rows.append(row)
    return _finish_artifact(
        identity=content["schema"],
        file="data/" + relative_file,
        raw=raw,
        rows=rows,
        runtime_digest=_sha256(raw),
        mapper=mapper,
    )


_FRAGMENT_SOURCE_KEYS = {
    "citation",
    "doi",
    "primary_source",
    "source",
    "source_citation",
    "source_citations",
    "source_doi",
    "source_reference",
    "source_url",
    "target_source",
    "underlying_data_doi",
    "url",
}
_FRAGMENT_LOCATOR_KEYS = {
    "locator",
    "location",
    "source_locator",
    "source_location",
}


def _fragment_source_refs(
    value: Any, role: str = "derived"
) -> tuple[list[dict[str, str]], list[str]]:
    references: list[dict[str, str]] = []
    citation_keys: list[str] = []

    def visit(item: Any, item_role: str) -> None:
        if isinstance(item, dict):
            locator = "; ".join(
                str(item[key]).strip()
                for key in _FRAGMENT_LOCATOR_KEYS
                if isinstance(item.get(key), str) and item[key].strip()
            )
            for key, child in item.items():
                if key == "citations" and isinstance(child, list):
                    citation_keys.extend(
                        entry for entry in child if isinstance(entry, str)
                    )
                    continue
                if key in _FRAGMENT_SOURCE_KEYS:
                    if (
                        key == "target_source"
                        and isinstance(child, str)
                        and not DOI_RE.search(child)
                        and not re.search(r"\b(?:19|20)[0-9]{2}\b", child)
                    ):
                        continue
                    values = child if isinstance(child, list) else [child]
                    for entry in values:
                        if isinstance(entry, str) and entry.strip():
                            chunks = (
                                _segments(entry)
                                if key in {"source", "source_citations"}
                                else [entry.strip()]
                            )
                            references.extend(
                                {"citation": chunk, "role": item_role, "locator": locator}
                                for chunk in chunks
                            )
                        elif isinstance(entry, (dict, list)):
                            visit(entry, item_role)
                elif isinstance(child, (dict, list)):
                    visit(child, "diagnostic" if key == "independent_check" else item_role)
        elif isinstance(item, list):
            for child in item:
                visit(child, item_role)

    visit(value, role)
    unique_refs: list[dict[str, str]] = []
    for reference in references:
        if reference not in unique_refs:
            unique_refs.append(reference)
    return unique_refs, list(dict.fromkeys(citation_keys))


def _fragment_citation_identity(value: str) -> tuple[str, str] | None:
    year = re.search(r"\b((?:19|20)[0-9]{2})[a-z]?\b", value, re.I)
    if not year:
        return None
    prefix = value[:year.start()].split(";", 1)[0]
    author = prefix.split(",", 1)[0]
    normalized = unicodedata.normalize("NFKD", author)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"(?<=[a-z])['’](?=[a-z])", "", normalized.casefold())
    names = re.findall(r"[a-z]+", normalized)
    names = [name for name in names if name not in {"and", "et", "al", "jr"}]
    if not names:
        return None
    surname = " ".join(names[:2]) if names[0] in {"de", "del", "van", "von"} and len(names) > 1 else names[0]
    return surname, year.group(1)


def _fragment_citation_authors(value: str) -> set[str]:
    author = value.split(",", 1)[0]
    normalized = unicodedata.normalize("NFKD", author)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"(?<=[a-z])['’](?=[a-z])", "", normalized.casefold())
    return {
        name
        for name in re.findall(r"[a-z]+", normalized)
        if name not in {"and", "et", "al", "jr"}
    }


def _fragment_citation_matches(
    citation: dict[str, Any], references: list[dict[str, str]]
) -> list[dict[str, str]]:
    citation_text = str(citation.get("citation", ""))
    citation_metadata = " ".join(str(value) for value in citation.values())
    citation_dois = {
        _normalized_doi(match.group(0)) for match in DOI_RE.finditer(citation_metadata)
    }
    citation_tables = {
        match.group(0) for match in JANAF_TABLE_RE.finditer(citation_metadata)
    }
    citation_identity = _fragment_citation_identity(citation_text)
    citation_authors = _fragment_citation_authors(citation_text)
    matched: list[dict[str, str]] = []
    for reference in references:
        reference_text = reference["citation"]
        reference_dois = {
            _normalized_doi(match.group(0))
            for match in DOI_RE.finditer(reference_text)
        }
        reference_tables = {
            match.group(0) for match in JANAF_TABLE_RE.finditer(reference_text)
        }
        reference_identity = _fragment_citation_identity(reference_text)
        same_author_year = bool(
            citation_identity
            and reference_identity
            and citation_identity == reference_identity
        )
        normalized_reference = re.sub(r"[^a-z0-9]+", " ", reference_text.casefold())
        author_match = bool(
            not reference_identity
            and len(citation_authors) > 1
            and citation_authors <= set(normalized_reference.split())
        )
        citation_code = re.search(r"\b[A-Z]{2,6}\s*[- ]?\d{3,6}\b", citation_text)
        code_match = bool(
            citation_code
            and re.search(re.escape(citation_code.group(0)), reference_text, re.I)
        )
        same_doi = bool(citation_dois.intersection(reference_dois)) and not (
            citation_tables
            and reference_tables
            and not citation_tables.intersection(reference_tables)
        )
        if same_doi or same_author_year or author_match or code_match:
            matched.append(reference)
    return matched


def _fragment_citation_text(
    citation: str,
    *,
    doi: str | None = None,
    url: str | None = None,
    locator: str | None = None,
) -> str:
    parts = [citation.strip()] if citation.strip() else []
    if doi and _normalized_doi(doi) not in {
        _normalized_doi(match.group(0)) for match in DOI_RE.finditer(citation)
    }:
        parts.append(f"DOI {doi}")
    if url and url not in citation:
        parts.append(f"URL {url}")
    if locator and locator.casefold() not in citation.casefold():
        parts.append(f"locator: {locator}")
    return "; ".join(parts)


def _add_fragment_citation(
    row: dict[str, Any],
    citation: str,
    role: str,
    mapper: SourceMapper,
    *,
    citation_key: str | None = None,
) -> None:
    if citation.strip():
        _add_citation_evidence(
            row,
            citation,
            role,
            mapper,
            citation_key=citation_key,
            preserve_full_citation=True,
            resolve_citation_key_with_atoms=True,
        )


def _fragment_citation_records(content: dict[str, Any]) -> dict[str, dict[str, Any]]:
    citations = content.get("citations", {})
    if isinstance(citations, dict):
        return {
            key: value for key, value in citations.items() if isinstance(value, dict)
        }
    if isinstance(citations, list):
        return {
            str(item.get("id") or item.get("key") or f"citation:{index}"): item
            for index, item in enumerate(citations, start=1)
            if isinstance(item, dict)
        }
    return {}


def _fragment_locator_citation_keys(locator: str, mapper: SourceMapper) -> list[str]:
    if (
        SOSSI_FEGLEY_2018_KEY in mapper.citation_keys
        and _contains_bounded_citation(SOSSI_FEGLEY_2018_KEY, locator)
    ):
        return [SOSSI_FEGLEY_2018_KEY]
    return []


def _fragment_artifact(path: Path, mapper: SourceMapper) -> dict[str, Any]:
    raw = path.read_bytes()
    content = json.loads(raw)
    element = str(content["element"])
    source_rows = content.get("rows", {})
    items = (
        [(str(row_id), row) for row_id, row in source_rows.items()]
        if isinstance(source_rows, dict)
        else [
            (str(row.get("row", index)), row)
            for index, row in enumerate(source_rows, start=1)
        ]
    )
    citation_records = _fragment_citation_records(content)
    rows: list[dict[str, Any]] = []
    for row_id, source_row in items:
        label = str(source_row.get("complex") or source_row.get("row") or row_id)
        record = _row(row_id, label)
        references, citation_keys = _fragment_source_refs(source_row)
        matched_reference_ids: set[int] = set()
        explicit_keys = set(citation_keys)
        for key, citation in citation_records.items():
            matches = _fragment_citation_matches(citation, references)
            if key not in explicit_keys and not matches:
                continue
            matched_reference_ids.update(id(reference) for reference in matches)
            display = str(citation.get("citation", ""))
            doi = citation.get("doi") or citation.get("equation_doi")
            url = citation.get("url") or citation.get("source_url")
            locator = (
                citation.get("locator")
                or citation.get("source_locator")
                or citation.get("location")
                or citation.get("source_location")
            )
            matched_locators = list(
                dict.fromkeys(ref["locator"] for ref in matches if ref["locator"])
            )
            if matched_locators:
                locator_parts = ([str(locator)] if locator else []) + matched_locators
                locator = "; ".join(dict.fromkeys(locator_parts))
            if not display:
                display = url or doi or key
            citation_text = _fragment_citation_text(
                display,
                doi=str(doi) if doi else None,
                url=str(url) if url else None,
                locator=str(locator) if locator else None,
            )
            for reference in matches:
                if (
                    reference["citation"]
                    and reference["citation"].casefold() not in citation_text.casefold()
                ):
                    citation_text += f"; fragment citation: {reference['citation']}"
            role = (
                "diagnostic"
                if any(ref["role"] == "diagnostic" for ref in matches)
                or any(term in citation_text.casefold() for term in ("held-out", "held out", "not used"))
                else "derived"
            )
            if "comparison" in str(source_row.get("property", "")).casefold():
                role = "diagnostic"
            locator_keys = [
                key
                for matched_locator in matched_locators
                for key in _fragment_locator_citation_keys(matched_locator, mapper)
            ]
            for citation_key in dict.fromkeys(locator_keys) or [None]:
                _add_fragment_citation(
                    record,
                    citation_text,
                    role,
                    mapper,
                    citation_key=citation_key,
                )
        for reference in references:
            if id(reference) not in matched_reference_ids:
                citation_text = _fragment_citation_text(
                    reference["citation"], locator=reference["locator"]
                )
                locator_keys = _fragment_locator_citation_keys(
                    reference["locator"], mapper
                )
                for citation_key in locator_keys or [None]:
                    _add_fragment_citation(
                        record,
                        citation_text,
                        reference["role"],
                        mapper,
                        citation_key=citation_key,
                    )
        for key in citation_keys:
            if key not in citation_records:
                _add_fragment_citation(record, key, "derived", mapper)
        _missing_provenance(record)
        rows.append(record)

    schema = (
        content.get("schema")
        or content.get("fragment_schema_version")
        or content.get("fragment_id")
        or "activity-fragment"
    )
    relative_file = path.relative_to(DATA).as_posix()
    return _finish_artifact(
        identity=f"{schema}:{element.casefold()}",
        file="data/" + relative_file,
        raw=raw,
        rows=rows,
        runtime_digest=_sha256(raw),
        mapper=mapper,
    )


def build() -> dict[str, Any]:
    mapping = _json(MAPPINGS)
    vocabulary = _json(VOCABULARY)
    sidecar = _json(PROSE_CITATIONS)
    work_ids = vocabulary["work_ids"]
    if work_ids != sorted(set(work_ids)):
        raise ValueError("the recorded work-ID vocabulary must be sorted and unique")
    mapper = SourceMapper(mapping, set(work_ids))
    sidecar_work_ids = {
        item["work_id"]
        for rows in sidecar.get("artifacts", {}).values()
        for decision in rows.values()
        for item in decision.get("citations", [])
        if item.get("work_id")
    }
    if not sidecar_work_ids <= set(work_ids):
        raise ValueError("the prose-citation sidecar references work IDs outside the vocabulary")

    sys.path.insert(0, str(ROOT / "src"))
    from openimcc.profiles import load_profile

    profile = load_profile("openimcc")
    artifacts = [
        _melt_artifact(mapper, profile.melt_pack.binding_digest or "", sidecar),
        _table_artifact(
            relative_file="gas/gas-shomate-default-v1.csv",
            identity="gas-shomate-default-v1",
            role="fitted",
            mapper=mapper,
            runtime_digest=profile.gas_datapack.gas_table_digest,
            sidecar=sidecar,
        ),
        _table_artifact(
            relative_file="gas/condensate-default-v1.csv",
            identity="condensate-default-v1",
            role="fitted",
            mapper=mapper,
            runtime_digest=profile.gas_datapack.condensate_table_digest,
            sidecar=sidecar,
        ),
        _redox_artifact(mapper, profile.redox_pack.canonical_digest, sidecar),
        _standards_artifact(mapper),
        _janaf_artifact(
            relative_file="diagnostics/janaf-pure-oxide-crystals.json",
            identity="openimcc-janaf-apparent-crystals-v2",
            collection="phases",
            mapper=mapper,
        ),
        _janaf_artifact(
            relative_file="diagnostics/janaf-crystal-species.json",
            identity="openimcc-janaf-apparent-crystal-species-v1",
            collection="phases",
            mapper=mapper,
        ),
        _janaf_artifact(
            relative_file="diagnostics/janaf-pure-oxide-liquid-pairs.json",
            identity="openimcc-janaf-apparent-liquid-pairs-v1",
            collection="rows",
            mapper=mapper,
        ),
        _barin_artifact(mapper),
        _rh_crystal_species_artifact(mapper),
        _crystal_phase_artifact(mapper),
        _elemental_reference_crystal_card_artifact(mapper),
    ]
    artifacts.extend(
        _fragment_artifact(path, mapper)
        for path in sorted((DATA / "fragments").glob("*/rows.json"))
    )
    referenced_work_ids = {
        work_id for artifact in artifacts for work_id in artifact["work_ids"]
    }
    if set(work_ids) != referenced_work_ids:
        raise ValueError("the recorded vocabulary must contain exactly the referenced work IDs")
    vocabulary_bytes = _canonical_json(work_ids)
    return {
        "schema": "openimcc-coefficient-ancestry.v1",
        "vocabulary_snapshot": {
            "commit": vocabulary["commit"],
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
    parser.add_argument(
        "--vocabulary-from-repo",
        type=Path,
        help="also verify snapshot IDs and DOIs against INDEX.yaml at its recorded commit",
    )
    arguments = parser.parse_args()
    if arguments.vocabulary_from_repo is not None:
        vocabulary = _json(VOCABULARY)
        index = _vocabulary_index_from_repo(
            arguments.vocabulary_from_repo, vocabulary["commit"]
        )
        _check_vocabulary_index(vocabulary, index)
        print(f"vocabulary matches INDEX.yaml at {vocabulary['commit']}")
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
