#!/usr/bin/env python3
"""Migrate published fit residuals and accuracy bands to the split schema."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def independent_check(reason: str) -> dict[str, Any]:
    return {
        "status": "not_found",
        "equilibrium": None,
        "primary_source": None,
        "conditions": None,
        "measured_value": None,
        "predicted_value": None,
        "miss": None,
        "miss_to_band_ratio": None,
        "reason": reason,
    }


def band_basis(kind: str, explanation: str, provenance: Any = None) -> dict[str, Any]:
    return {"kind": kind, "explanation": explanation, "provenance": provenance}


def _temperature(record: dict[str, Any]) -> Any:
    return record.get("T_domain_K", record.get("T_range_K"))


def _redox(root: Path) -> None:
    path = root / "src/openimcc/data/packs/openimcc-redox-v1.json"
    pack = json.loads(path.read_text())
    changed = False
    expected_band_unit = "declared per record by fit_residual or accuracy_band unit"
    if pack.get("units", {}).get("band") != expected_band_unit:
        pack["units"]["band"] = expected_band_unit
        changed = True
    for name, row in pack["thermo_species"].items():
        if "band_kj_mol" not in row:
            if "band_method" in row:
                row.pop("band_method")
                changed = True
            continue
        changed = True
        value = row.pop("band_kj_mol")
        legacy_method = row.pop("band_method", None)
        rationale = row.get("provenance", {}).get("selection_rationale", "")
        construction = row.get("construction")
        tier = row.get("provenance", {}).get("tier")
        rationale_lower = str(rationale).lower()
        source_band = bool(construction) or tier == "tier4_systematic_estimate" or (
            (tier == "tier3_flagged_estimate" and any(
                word in rationale_lower for word in ("uncertainty", "construction", "fusion", "measured")
            )) or row.get("evaluator") == "linear_combination"
        )
        if source_band:
            explanation = str(legacy_method or rationale or construction)
            row["accuracy_band"] = {
                "value": value, "unit": "kJ mol^-1 species",
                "temperature_points_or_range": "T_domain_K", "coverage": explanation,
            }
            kind = "measured_construction" if construction and row.get("provenance", {}).get("tier") == "tier4_systematic_estimate" else "source_uncertainty"
            row["band_basis"] = band_basis(kind, explanation, row.get("provenance", {}).get("source"))
            row["fit_residual"] = None
        else:
            row["fit_residual"] = {
                "value": value, "unit": "kJ mol^-1 species",
                "target_source": rationale or row.get("provenance", {}).get("source"),
                "comparison_method": legacy_method or "maximum absolute fit residual against the cited source function",
                "temperature_range": "T_domain_K",
                "held_out_or_in_sample": "in_sample",
            }
            row["accuracy_band"] = None
            row["band_basis"] = band_basis("fit_only", str(legacy_method or rationale), row.get("provenance", {}).get("source"))
        row["independent_check"] = independent_check("No row-specific independent equilibrium check was identified.")

    for row in pack["rows"]:
        value = row.pop("band_kj_mol", None)
        if value is None:
            if "band_method" in row:
                row.pop("band_method")
                changed = True
            continue
        changed = True
        method = str(row.pop("band_method", ""))
        note = str(row.get("uncertainty_note", ""))
        row_number = row.get("row")
        classification = f"{method} {note}".lower()
        construction_band = any(word in classification for word in ("construction", "constructed", "leave-one-out", "census"))
        source_band = not construction_band and any(word in classification for word in ("fusion-enthalpy", "source uncertainty", "tier-2 liquid-cp uncertainty", "propagated fe2o3"))
        if source_band or construction_band:
            row["accuracy_band"] = {
                "value": value, "unit": "kJ mol^-1 reaction",
                "temperature_points_or_range": "T_domain_K",
                "coverage": method,
            }
            row["fit_residual"] = None
            row["band_basis"] = band_basis(
                "source_uncertainty" if source_band else "measured_construction",
                note or method, row.get("provenance_class"),
            )
        else:
            row["accuracy_band"] = None
            row["fit_residual"] = {
                "value": value, "unit": "kJ mol^-1 reaction",
                "target_source": "JANAF FeO formation-G values" if row_number == 1 else method,
                "comparison_method": method,
                "temperature_range": "T_domain_K",
                "held_out_or_in_sample": "in_sample",
            }
            row["band_basis"] = band_basis("fit_only", note or method, row.get("provenance_class"))
        row["independent_check"] = independent_check("No row-specific independent equilibrium check was identified.")
    # The published Fe buffer check tests the reaction collectively; it cannot
    # be apportioned to any one component standard.
    reaction = pack["rows"][0]
    if not reaction.get("independent_check"):
        changed = True
        reaction["independent_check"] = {
        "status": "tested", "equilibrium": "Fe(metal) + 1/2 O2(g) = FeO(l)",
        "primary_source": "Distin, Whiteway & Masson (1971), DOI 10.1179/cmq.1971.10.1.13",
        "conditions": "1700–2200 K; stoichiometric FeO(l) standard",
        "measured_value": "5–9 kJ/mol miss",
        "predicted_value": "Existing fit-only reaction residual",
        "miss": "5–9 kJ/mol", "miss_to_band_ratio": "12.8–23.1",
        "reason": "Collective reaction check; the discrepancy cannot be assigned to an individual component.",
        }
    if changed:
        projection = dict(pack)
        projection.pop("canonical_content_digest", None)
        canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        pack["canonical_content_digest"] = hashlib.sha256(canonical).hexdigest()
        path.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n")


def _redox_source(root: Path) -> None:
    path = root / "data-src/redox-thermochemistry.json"
    doc = json.loads(path.read_text())
    records = list(doc.get("tables", {}).values())
    if isinstance(doc.get("hematite_crystal_table"), dict):
        records.append(doc["hematite_crystal_table"])
    changed = False
    for record in records:
        if all(key in record for key in ("fit_residual", "accuracy_band", "band_basis", "independent_check")):
            continue
        record.update({
            "fit_residual": None,
            "accuracy_band": None,
            "band_basis": band_basis(
                "none", "Source thermochemistry only; model fit residuals and accuracy bands are recorded in the redox pack.",
                record.get("url"),
            ),
            "independent_check": independent_check(
                "A source table is not an evaluated OpenIMCC row; check status is not applicable."
            ),
        })
        record["independent_check"]["status"] = "not_applicable"
        changed = True
    if changed:
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


def _gas(root: Path) -> None:
    path = root / "src/openimcc/data/gas/PROVENANCE.yaml"
    source_text = path.read_text()
    doc = yaml.safe_load(source_text)
    parent_checks = {
        "PbO(l)": ("Holman & Fulrath (1973), LBL-1462", "Modified Knudsen effusion; pure PbO vapor-pressure curve", "approximately 1050–1180 K; compared vaporization enthalpy and entropy with JANAF, not liquid Gibbs energy", "Raw Fig. 22 points and Appendix C need transcription before calculating a pressure or Gibbs residual."),
        "Na2O(l)": ("Kobertz (2019), Calphad 64, 327–333, DOI 10.1016/j.calphad.2019.01.006", "KEMS of pure Na2O(c), Na and O2 pressures", "900–1165 K; reported solid-reference reaction Gibbs at 1000 K; no liquid or OpenIMCC difference", "Near-transition vapor evidence; reconcile phase and raw points before a matched comparison."),
        "CaO(l)": ("Halwax et al. (2024), Metallurgical and Materials Transactions B 55, 821–835, DOI 10.1007/s11663-024-02995-6", "Pure-CaO KEMS in an iridium cell", "1825–2125 K; measured Ca, O and O2 pressures compared with FactSage 7.3 FactPS; below the OpenIMCC liquid interval", "Measured solid-vapor evidence; model comparator is FactPS and no OpenIMCC miss was computed."),
        "MgO(l)": ("Halwax et al. (2024), Metallurgical and Materials Transactions B 55, 821–835, DOI 10.1007/s11663-024-02995-6", "Pure-MgO KEMS in an iridium cell", "1675–2075 K; measured Mg, O and O2 pressures compared with FactSage 7.3 FactPS; below the OpenIMCC liquid interval", "Measured solid-vapor evidence; model comparator is FactPS and no OpenIMCC miss was computed."),
        "FeO(l)": ("Distin, Whiteway & Masson (1971), DOI 10.1179/cmq.1971.10.1.13", "Oxygen solubility in liquid iron in contact with liquid iron oxide", "1800–2230 K; combined Fe(l)/FeO(l)/O2 buffer equilibrium", "Coupled redox check is not attributable to the isolated gas-parent FeO Gibbs energy."),
    }
    not_found = "No same-standard independent equilibrium check was located in the 2026-10-15 research sweep."
    for row in doc["rows"]:
        legacy = any(key in row for key in ("max_residual_J_per_mol", "max_residual_log10_K", "uncertainty_band", "combined_band"))
        if not legacy and all(key in row for key in ("fit_residual", "accuracy_band", "band_basis", "independent_check")):
            continue
        old_residuals = {key: row.pop(key) for key in ("max_residual_J_per_mol", "max_residual_log10_K") if key in row}
        old_band = row.pop("uncertainty_band", None)
        if "combined_band" in row:
            old_band = {"combined_band": row.pop("combined_band"), "source_uncertainty": row.pop("source_uncertainty", None)}
        source = row.get("source_locator") or row.get("source_record") or row.get("source_path")
        temperature = row.get("T_range_K") or row.get("T_interval")
        if old_residuals:
            row["fit_residual"] = {
                "value": old_residuals,
                "unit": {"max_residual_J_per_mol": "J mol^-1", "max_residual_log10_K": "log10 K"},
                "target_source": source,
                "comparison_method": row.get("method", "source-function fit"),
                "temperature_range": "T_range_K or T_interval",
                "held_out_or_in_sample": "in_sample",
            }
        else:
            row["fit_residual"] = None
        if old_band is not None:
            kind = str(old_band.get("kind", ""))
            row["accuracy_band"] = {
                "value": old_band,
                "unit": old_band.get("units", "J mol^-1"),
                "temperature_points_or_range": "values_J_per_mol keys or T_range_K",
                "coverage": kind or "published source uncertainty or measured construction envelope",
            }
            basis_kind = "source_uncertainty" if "TSIV" in kind or "source uncertainty" in kind else "measured_construction"
            row["band_basis"] = band_basis(basis_kind, kind or "Existing source or continuation band", source)
        else:
            row["accuracy_band"] = None
            row["band_basis"] = band_basis("fit_only" if old_residuals else "none", "Source-fit residual only" if old_residuals else "No quantified row-level band", source)
        parent = row.get("species_name")
        if parent in parent_checks:
            citation, method, conditions, limitation = parent_checks[parent]
            row["independent_check"] = {
                "status": "lead", "equilibrium": parent,
                "primary_source": citation, "conditions": conditions,
                "measured_value": method, "predicted_value": None,
                "miss": None, "miss_to_band_ratio": None,
                "reason": limitation,
            }
        elif row.get("table") == "condensate" and parent and parent.endswith("(l)"):
            row["independent_check"] = independent_check(not_found)
        else:
            row["independent_check"] = independent_check("Not a liquid gas-parent row; independent check status is not applicable.")
    lines = source_text.splitlines(keepends=True)
    row_header = next(i for i, line in enumerate(lines) if line.startswith("rows:"))
    row_end = next(
        (i for i in range(row_header + 1, len(lines)) if lines[i] and not lines[i][0].isspace()),
        len(lines),
    )
    rows = "".join(
        "  - " + yaml.safe_dump(
            row, allow_unicode=True, sort_keys=False, default_flow_style=True, width=1_000_000
        ).strip() + "\n"
        for row in doc["rows"]
    )
    path.write_text("".join(lines[:row_header]) + "rows:\n" + rows + "".join(lines[row_end:]))


def _liquid_inputs(root: Path) -> None:
    path = root / "data-src/liquid-from-solid-inputs.json"
    doc = json.loads(path.read_text())
    for formula, row in doc["rows"].items():
        if not any(key in row for key in ("measured_band", "crystal_cp_extrapolation_band")) and all(
            key in row for key in ("fit_residual", "accuracy_band", "band_basis", "independent_check")
        ):
            continue
        old_measured = row.pop("measured_band", None)
        old_extension = row.pop("crystal_cp_extrapolation_band", None)
        if old_measured is not None:
            row["accuracy_band"] = {
                "value": old_measured,
                "unit": "J mol^-1",
                "temperature_points_or_range": list(old_measured.get("values_J_per_mol", {}).keys()),
                "coverage": old_measured.get("kind", "measured construction envelope"),
            }
            row["band_basis"] = band_basis("measured_construction", old_measured.get("source", old_measured.get("kind", "")), formula)
        elif old_extension is not None:
            row["accuracy_band"] = {"value": old_extension, "unit": "J mol^-1", "temperature_points_or_range": "source_limit_K", "coverage": old_extension.get("method", "Cp extrapolation comparison")}
            row["band_basis"] = band_basis("measured_construction", old_extension.get("method", "Cp extrapolation comparison"), formula)
        else:
            row["accuracy_band"] = None
            row["band_basis"] = band_basis("source_uncertainty" if row.get("tier") else "none", "Input provenance only; no composed liquid Gibbs band stored on this input row.", formula)
        row["fit_residual"] = None
        row["independent_check"] = independent_check("This row supplies an input to a liquid construction; no row-specific equilibrium check is recorded.")
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


def _melt_packs(root: Path) -> None:
    for path in sorted((root / "src/openimcc/data/packs").glob("imcc-sf04*.json")):
        doc = json.loads(path.read_text())
        for row in doc["rows"]:
            if all(key in row for key in ("fit_residual", "accuracy_band", "band_basis", "independent_check")):
                continue
            # Existing diagnostics are retained under fit_residual as their
            # own payload; their source fields remain available for audit.
            diagnostics = {key: row.pop(key) for key in list(row) if "residual" in key.lower()}
            row["fit_residual"] = None if not diagnostics else {
                "value": diagnostics,
                "unit": "source-specific",
                "target_source": row.get("source") or row.get("provenance"),
                "comparison_method": "Existing held-out or candidate-fit diagnostic; see original field names in value.",
                "temperature_range": row.get("T_domain_K"),
                "held_out_or_in_sample": "held_out_or_candidate",
            }
            row["accuracy_band"] = None
            row["band_basis"] = band_basis("none", "Complex association row has no row-specific uncertainty band.", row.get("provenance"))
            row["independent_check"] = independent_check("No row-specific pure-complex equilibrium check was identified.")
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


def migrate(root: Path = ROOT) -> None:
    _redox(root)
    _redox_source(root)
    _gas(root)
    _liquid_inputs(root)
    _melt_packs(root)


if __name__ == "__main__":
    migrate()
