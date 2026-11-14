#!/usr/bin/env python3
"""Migrate published fit residuals and accuracy bands to the split schema."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _normalized_controlled_value(value: Any) -> str:
    text = str(value or "").lower().replace("-", " ").replace("_", " ")
    return " ".join(text.split())


# Keys are the exact normalized values in the committed migration inputs.
# Unlisted values take the conservative class in each classifier below.
_SOURCE_BAND_CLASS_BY_KIND = {
    'mixed source parent scale inconsistency': 'measured_construction',
    'not separately quantified by selected source': None,
    'measured held cp continuation error envelope against published source tables': 'measured_construction',
    'tsiv dfh298 ±1.0 kj/mol uncertainty plus phi uncertainty, propagated fusion input spread, and quadratic crystal cp extrapolation band': 'source_uncertainty',
    'measured rocksalt monoxide fusion input range plus published cp versus held cp extension difference': 'measured_construction',
    'measured held cp continuation error envelope against published liquid source tables': 'measured_construction',
    'repository held cp continuation error envelope against published source tables': 'measured_construction',
    'reviewed fusion enthalpy and transition temperature spreads plus crystal cp extension model spread': 'source_uncertainty',
    'constant cp alternative continuation envelope': 'measured_construction',
    'table 1436 prints accuracy class 4 e; numeric uncertainty band is not quantified by the cited page': None,
    'shared redox liquid from solid below fusion band, including tier 3 fusion entropy and liquid cp spreads and the measured fusion temperature interval; the fitted row residual is included': 'measured_construction',
    'repository held cp continuation error envelope': 'measured_construction',
    'tier 3 leave one out liquid from solid gibbs error against 48 janaf fusion pairs; uncertainty estimate is extrapolated above +800 k': 'measured_construction',
    'tier 3 leave one out liquid from solid gibbs error against 48 janaf fusion pairs; envelope slope continues 1500 2500 k past decomposition near the 500 k construction anchor': 'measured_construction',
    'tier 3 leave one out liquid from solid gibbs envelope with its +800 k calibrated slope continued 1500 2500 k past decomposition near the 500 k anchor': 'measured_construction',
    'held cp continuation envelope': 'measured_construction',
    'printed cp rounding envelope': 'measured_construction',
    'maximum envelope of tsiv phi uncertainty (linearly interpolated between stated knots and multiplied by t) and the estimated ±5 kj/mol fusion enthalpy contribution dg = dh fus(1 − t/tm) below tm; max avoids an unsupported independence assumption; formation anchor uncertainty is separate': 'source_uncertainty',
    'tsiv phi uncertainty, linearly interpolated between stated knots and multiplied by t; formation anchor uncertainty is separate': 'source_uncertainty',
    'tsiv phi uncertainty, linearly interpolated between stated knots and multiplied by t; formation anchor uncertainty is separate; the estimated 40 kj/mol fusion enthalpy has no stated numeric uncertainty, so no fusion term is added': 'source_uncertainty',
    'measured g app difference from pankratz bulletin 672': 'measured_construction',
    'not numerically quantified by source; tsiv accuracy class 4 f': None,
    'propagated oecd/nea fusion input uncertainty plus crystal extension model spread': 'source_uncertainty',
    'propagated fusion input uncertainty plus crystal extension model spread': 'source_uncertainty',
    'propagated source reported δfh°298 and s°298 uncertainties': 'source_uncertainty',
    'propagated printed formation, transition, fusion and temperature uncertainties': 'source_uncertainty',
    'propagated printed phase chain uncertainty': 'source_uncertainty',
    'propagated printed formation': 'source_uncertainty',
    'propagated source reported formation enthalpy uncertainty': 'source_uncertainty',
    'propagated fusion input, fusion temperature, and liquid cp uncertainty': 'source_uncertainty',
}
_LIQUID_INPUT_BAND_CLASS_BY_KIND = {
    'tsiv dfh298 uncertainty, phi uncertainty, and fitted crystal cp extension band': 'source_uncertainty',
    'tier 3 liquid from solid leave one out gibbs error against the 48 janaf crystal/liquid pairs': 'measured_construction',
    'tier 3 liquid from solid leave one out gibbs error against 17 measured class c janaf pairs': 'measured_construction',
}
_REDOX_SPECIES_BAND_CLASS_BY_EVIDENCE = {
    ('tier1 direct assessed', 'direct assessed liquid function; reaction residual is at most 0.31 kj/mol against the directly fitted janaf elemental references over 1700 3000 k. the band records reaction fit residual, not thermochemical accuracy; no fe(l) fusion anchor adjustment is applied.', False, 'condensate phi'): 'fit_only',
    ('tier1 direct assessed', 'the 1200 1500 k interval uses the existing lower janaf shomate fit; the original 1500 3000 k coefficients are retained unchanged. both intervals reproduce their janaf source nodes within 1 j/mol.', False, 'janaf shomate'): 'fit_only',
    ('tier1 direct assessed', 'separate alpha, gamma, and delta fits use the phase side h h(298), s, and cp values at the 1184 and 1665 k transitions. the delta fit extends nine kelvin beyond the final 1800 k node to the transition marker. this band records source node fit residual only, not thermochemical accuracy.', False, 'condensate phi'): 'fit_only',
    ('tier1 direct assessed', 'the 1700 3000 k fe(l) fit uses the cited fe 006 nodes directly. its maximum source node fit residual is 1.507 j/mol; the 2.345 j/mol fe(cr)/fe(l) seam at the 1809 k fusion marker remains unadjusted and is within the 10 j/mol fit gate. this band records fit residual only, not thermochemical accuracy.', False, 'condensate phi'): 'fit_only',
    ('tier1 direct assessed', 'stable pure iron reference state uses the crystal below the janaf 1809 k fusion point and liquid at and above it; a fe is fixed to one for the saturation reference. the 2.345 j/mol branch seam residual is within the 10 j/mol fit gate; this band records fit and seam residual only, not thermochemical accuracy.', False, 'piecewise'): 'fit_only',
    ('tier3 flagged estimate', 'the central nominal fusion value follows the explicit lesher and spera secondary transcription; the genuine sugawara and akaogi primary estimate is retained as an uncertainty bound. the tier 2 liquid component cp is used centrally with its stated uncertainty and a high temperature extrapolation flag; the hypothetical hematite phase remains metastable.', True, 'condensate phi'): 'source_uncertainty',
    ('tier3 flagged estimate', 'stoichiometric pseudo complex reference; it adds no independent gibbs function.', False, 'linear combination'): 'source_uncertainty',
    ('tier1 direct assessed', 'elemental standard fitted with the existing condensate φ fitter.', False, 'condensate phi'): 'fit_only',
    ('tier1 direct assessed', 'the original 1600 2200 k coefficients are retained; separate condensate φ fits use the cited janaf liquid nodes over 1200 1600 k and 2200 3000 k. their maximum source residual is 0.524 j/mol.', False, 'condensate phi'): 'fit_only',
    ('tier3 flagged estimate', 'uses the existing al 100 sc cp continuation over 1200 2500 k and the direct liquid fit over 2500 3000 k; the numeric band remains based on the measured tier 1 leave one out maximum at fusion.', False, 'condensate phi'): 'measured_construction',
    ('tier4 systematic estimate', "the original 1600 2200 k coefficients are retained. the same liquid from solid construction is independently fitted below and above that interval on 100 k nodes; maximum outer interval residual is 0.271 j/mol. the construction uses the tool's leave one out mean fusion entropy per formula atom; ivtan's 40 ± 5 kj/mol fusion enthalpy is a cross check only.", True, 'condensate phi'): 'measured_construction',
    ('tier4 systematic estimate', 'the original 1500 2000 k coefficients are retained. the same liquid from solid construction is fitted on source nodes in 1200 1500 k and 2000 3000 k; the largest additional source node residual is 2.278 j/mol. no activity data were used to select coefficients.', True, 'condensate phi'): 'measured_construction',
    ('tier4 systematic estimate', 'the original 1500 2000 k coefficients are retained. the same liquid from solid construction is fitted on source nodes in 1200 1500 k and 2000 3000 k; the largest additional source node residual is 3.510 j/mol. no activity data were used to select coefficients.', True, 'condensate phi'): 'measured_construction',
    ('tier4 systematic estimate', 'the original 1600 2200 k coefficients are retained. the same liquid from solid construction is fitted below and above that interval on 100 k nodes; the maximum additional source node residual is 0.975 j/mol. chen activities were not used to select coefficients.', True, 'condensate phi'): 'measured_construction',
    ('tier3 flagged estimate', 'uses the existing supercooled branch below the direct liquid function interval.', False, 'condensate phi'): 'fit_only',
    ('tier1 direct assessed', 'uses the direct assessed liquid function from its tabulated interval.', False, 'condensate phi'): 'fit_only',
    ('tier3 flagged estimate', 'the two existing standard state branches meet within 0.01 kj/mol at their 1800 k join.', False, 'piecewise'): 'fit_only',
}
_CONSTRUCTION_RECORD_KEYSETS = frozenset(
    {
        frozenset({
            "alternate_construction", "band_scope", "crystal_source_table",
            "fit_T_K", "fit_intervals_K", "fitter", "fusion_enthalpy_kj_mol",
            "fusion_enthalpy_spread_kj_mol", "fusion_entropy_derivation",
            "fusion_entropy_j_mol_K", "fusion_entropy_spread_j_mol_K",
            "fusion_temperature_K", "fusion_temperature_status",
            "liquid_heat_capacity_flag", "liquid_heat_capacity_j_mol_K",
            "liquid_heat_capacity_method", "liquid_heat_capacity_spread_j_mol_K",
            "runtime_T_K",
        }),
        frozenset({
            "crystal_source_table", "fit_T_K", "fit_intervals_K", "fitter",
            "fusion_entropy_j_mol_K", "fusion_entropy_spread_j_mol_K",
            "fusion_temperature_K", "fusion_temperature_spread_K",
            "liquid_heat_capacity_j_mol_K", "liquid_heat_capacity_spread_j_mol_K",
            "runtime_T_K",
        }),
        frozenset({
            "crystal_formation_enthalpy_spread_kj_mol", "crystal_source_table",
            "fit_T_K", "fit_intervals_K", "fitter", "fusion_entropy_j_mol_K",
            "fusion_entropy_spread_j_mol_K", "fusion_temperature_K",
            "fusion_temperature_spread_K", "liquid_heat_capacity_j_mol_K",
            "liquid_heat_capacity_spread_j_mol_K", "runtime_T_K",
        }),
        frozenset({
            "crystal_entropy_spread_j_mol_K",
            "crystal_formation_enthalpy_spread_kj_mol", "crystal_source_table",
            "fit_T_K", "fit_intervals_K", "fitter", "fusion_entropy_j_mol_K",
            "fusion_entropy_spread_j_mol_K", "fusion_temperature_K",
            "fusion_temperature_spread_K", "liquid_heat_capacity_j_mol_K",
            "liquid_heat_capacity_spread_j_mol_K", "runtime_T_K",
        }),
    }
)
_CONSTRUCTION_NUMERIC_FIELDS = frozenset({
    "fusion_temperature_K", "fusion_temperature_spread_K",
    "fusion_enthalpy_kj_mol", "fusion_enthalpy_spread_kj_mol",
    "fusion_entropy_j_mol_K", "fusion_entropy_spread_j_mol_K",
    "liquid_heat_capacity_j_mol_K", "liquid_heat_capacity_spread_j_mol_K",
    "crystal_formation_enthalpy_spread_kj_mol",
    "crystal_entropy_spread_j_mol_K",
})
_CONSTRUCTION_STRING_FIELDS = frozenset({
    "crystal_source_table", "fusion_temperature_status",
    "fusion_entropy_derivation", "liquid_heat_capacity_method",
    "liquid_heat_capacity_flag", "band_scope", "fitter",
})
_REDOX_REACTION_BAND_CLASS_BY_EVIDENCE = {
    ('maximum absolute reaction gibbs residual against the janaf feo formation gibbs values over 1700 3000 k.', 'the 0.39 kj/mol band bounds the fitted reaction gibbs residual against janaf feo values; it is a residual band, not a thermochemical accuracy estimate. the fe(cr) endpoint extrapolation is separately flagged.'): 'fit_only',
    ('conservative sum of one half the propagated fe2o3 fusion enthalpy and tier 2 liquid cp band, the feo reaction band, and one quarter the o2 fit band.', 'the band includes the primary fusion estimate bound and stated tier 2 cp uncertainty; high temperature cp extrapolation and hypothetical hematite metastability remain flagged.'): 'source_uncertainty',
    ('conservative sum of the below fusion mno construction band using the seven rock salt monoxide maximum residuals, the mn(l) source fit band, and one half the o2 fit band.', 'the mno liquid reference is supercooled throughout the declared domain; the below fusion band uses domain matched measured monoxide residuals and the measured leave one out anchor.'): 'measured_construction',
    ('sum of the mnsio3(l) band from interpolated tier 3 leave one out columns, its independent fusion temperature interval and fit residual, the mno(l) band, and the sio2(l) band.', 'the leave one out columns are interpolated to t t fus; the same residuals are not also stacked as propagated entropy and heat capacity spreads. the chen activity data remain held out.'): 'measured_construction',
    ('sum of the mn2sio4(l) band from interpolated tier 3 leave one out columns, its independent source enthalpy and fusion temperature uncertainties and fit residual, two mno(l) bands, and the sio2(l) band.', 'the leave one out columns are interpolated to t t fus; the same residuals are not also stacked as propagated entropy and heat capacity spreads. the primary tephroite enthalpy uncertainty is counted once. the chen activity data remain held out.'): 'measured_construction',
    ('sum of the constructed galaxite band, including its crystal formation enthalpy and entropy uncertainties, the mno(l) band, and the existing al2o3(l) supercooled reference band.', 'the heat capacity fit is extrapolated above its measured range and the formation enthalpy is assessed rather than directly measured. chen activity data were held out and did not select the equilibrium constant.'): 'measured_construction',
}
_MELT_DIAGNOSTIC_SPLIT_BY_KEY = {
    'translation candidates/fc87/max abs residual dex': 'held_out_or_candidate',
    'translation candidates/lam/max abs residual dex': 'held_out_or_candidate',
    'k translation/max abs residual dex': 'held_out_or_candidate',
    'k translation/residual at t min dex': 'held_out_or_candidate',
    'k translation/residual at t max dex': 'held_out_or_candidate',
    'provenance/max abs heldout log10k residual': 'held_out',
    'standard state refit/max abs residual dex': 'held_out_or_candidate',
}


def _liquid_input_band_class(kind: Any) -> str | None:
    return _LIQUID_INPUT_BAND_CLASS_BY_KIND.get(
        _normalized_controlled_value(kind)
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _numeric_pair(value: Any) -> bool:
    return type(value) is list and len(value) == 2 and all(
        _finite_number(item) for item in value
    )


def _valid_construction_record(value: Any) -> bool:
    if type(value) is not dict or frozenset(value) not in _CONSTRUCTION_RECORD_KEYSETS:
        return False
    if any(
        not _finite_number(value[key])
        for key in _CONSTRUCTION_NUMERIC_FIELDS
        if key in value
    ):
        return False
    if any(
        type(value[key]) is not str or not value[key]
        for key in _CONSTRUCTION_STRING_FIELDS
        if key in value
    ):
        return False
    if value.get("fitter") != "tools.liquid_from_solid.LiquidConstruction.fit_condensate_row":
        return False
    if not _numeric_pair(value.get("fit_T_K")) or not _numeric_pair(
        value.get("runtime_T_K")
    ):
        return False
    intervals = value.get("fit_intervals_K")
    if type(intervals) is not list or not intervals or not all(
        _numeric_pair(interval) for interval in intervals
    ):
        return False
    if "alternate_construction" in value:
        alternate = value["alternate_construction"]
        if type(alternate) is not dict or frozenset(alternate) != {
            "fusion_enthalpy_kj_mol", "fusion_enthalpy_spread_kj_mol",
            "fusion_entropy_j_mol_K", "source",
        }:
            return False
        if any(
            not _finite_number(alternate[key])
            for key in (
                "fusion_enthalpy_kj_mol", "fusion_enthalpy_spread_kj_mol",
                "fusion_entropy_j_mol_K",
            )
        ) or type(alternate["source"]) is not str or not alternate["source"]:
            return False
    return True


def _redox_species_band_evidence(
    row: dict[str, Any],
) -> tuple[str, str, bool, str] | None:
    provenance = row.get("provenance", {})
    if type(provenance) is not dict:
        return None
    if "construction" not in row:
        construction = False
    else:
        construction_value = row["construction"]
        if type(construction_value) is bool:
            construction = construction_value
        elif _valid_construction_record(construction_value):
            construction = True
        else:
            return None
    return (
        _normalized_controlled_value(provenance.get("tier")),
        _normalized_controlled_value(provenance.get("selection_rationale")),
        construction,
        _normalized_controlled_value(row.get("evaluator")),
    )


def _redox_species_band_class(row: dict[str, Any]) -> str:
    evidence = _redox_species_band_evidence(row)
    if evidence is None:
        return "fit_only"
    return _REDOX_SPECIES_BAND_CLASS_BY_EVIDENCE.get(evidence, "fit_only")


def _redox_reaction_band_class(method: Any, note: Any) -> str:
    key = (
        _normalized_controlled_value(method),
        _normalized_controlled_value(note),
    )
    return _REDOX_REACTION_BAND_CLASS_BY_EVIDENCE.get(key, "fit_only")


def _melt_held_out_split(diagnostic_keys: Any) -> str:
    for key in diagnostic_keys:
        if (
            _MELT_DIAGNOSTIC_SPLIT_BY_KEY.get(
                _normalized_controlled_value(key)
            )
            == "held_out"
        ):
            return "held_out"
    return "held_out_or_candidate"


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
    value = record.get("T_domain_K", record.get("T_range_K", record.get("T_interval")))
    return deepcopy(value)


def _source_band_kind(value: Any) -> tuple[str | None, str]:
    """Classify a published band using its declared provenance kind."""
    explanation = str(value.get("kind", "")) if isinstance(value, dict) else ""
    kind = _SOURCE_BAND_CLASS_BY_KIND.get(
        _normalized_controlled_value(explanation)
    )
    return kind, explanation or "Unclassified source band kind."


def _redox(root: Path, path: Path | None = None) -> None:
    if path is None:
        paths = [root / "src/openimcc/data/packs/openimcc-redox-v1.json"]
        paths.extend(sorted((root / "src/openimcc/data/packs").glob("openimcc-redox-*-candidate.json")))
        for pack_path in paths:
            _redox(root, pack_path)
        return
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
            fit = row.get("fit_residual")
            if (
                isinstance(fit, dict)
                and fit.get("temperature_range") in ("T_domain_K", "T_range_K or T_interval")
            ):
                fit["temperature_range"] = _temperature(row)
                changed = True
            band_class = _redox_species_band_class(row)
            basis = row.get("band_basis")
            if (
                isinstance(basis, dict)
                and basis.get("kind") == "source_uncertainty"
                and row.get("provenance", {}).get("tier") == "tier3_flagged_estimate"
                and band_class == "measured_construction"
            ):
                basis["kind"] = "measured_construction"
                changed = True
            continue
        changed = True
        value = row.pop("band_kj_mol")
        legacy_method = row.pop("band_method", None)
        rationale = row.get("provenance", {}).get("selection_rationale", "")
        band_class = _redox_species_band_class(row)
        source_band = band_class != "fit_only"
        if source_band:
            explanation = str(
                legacy_method or rationale or row.get("construction") or ""
            )
            row["accuracy_band"] = {
                "value": value, "unit": "kJ mol^-1 species",
                "temperature_points_or_range": "T_domain_K", "coverage": explanation,
            }
            row["band_basis"] = band_basis(
                band_class, explanation, row.get("provenance", {}).get("source")
            )
            row["fit_residual"] = None
        else:
            row["fit_residual"] = {
                "value": value, "unit": "kJ mol^-1 species",
                "target_source": rationale or row.get("provenance", {}).get("source"),
                "comparison_method": legacy_method or "maximum absolute fit residual against the cited source function",
                "temperature_range": _temperature(row),
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
            fit = row.get("fit_residual")
            if (
                isinstance(fit, dict)
                and fit.get("temperature_range") in ("T_domain_K", "T_range_K or T_interval")
            ):
                fit["temperature_range"] = _temperature(row)
                changed = True
            continue
        changed = True
        method = str(row.pop("band_method", ""))
        note = str(row.get("uncertainty_note", ""))
        row_number = row.get("row")
        band_class = _redox_reaction_band_class(method, note)
        if band_class != "fit_only":
            row["accuracy_band"] = {
                "value": value, "unit": "kJ mol^-1 reaction",
                "temperature_points_or_range": "T_domain_K",
                "coverage": method,
            }
            row["fit_residual"] = None
            row["band_basis"] = band_basis(
                band_class,
                note or method, row.get("provenance_class"),
            )
        else:
            row["accuracy_band"] = None
            row["fit_residual"] = {
                "value": value, "unit": "kJ mol^-1 reaction",
                "target_source": "JANAF FeO formation-G values" if row_number == 1 else method,
                "comparison_method": method,
                "temperature_range": _temperature(row),
                "held_out_or_in_sample": "in_sample",
            }
            row["band_basis"] = band_basis("fit_only", note or method, row.get("provenance_class"))
        row["independent_check"] = independent_check("No row-specific independent equilibrium check was identified.")
    # Distin checks the combined buffer, so keep its numeric miss unassigned.
    distin = {
        "status": "tested", "equilibrium": "Fe(metal) + 1/2 O2(g) = FeO(l)",
        "primary_source": "Distin, Whiteway & Masson (1971), DOI 10.1179/cmq.1971.10.1.13",
        "conditions": "1800–2230 K; stoichiometric FeO(l) standard",
        "measured_value": "Oxygen solubility in liquid iron in contact with liquid iron oxide",
        "predicted_value": "Fe/FeO/O2 buffer equilibrium",
        "miss": None, "miss_to_band_ratio": None,
        "reason": "Collective reaction check; the discrepancy cannot be assigned to FeO, Fe(metal), Fe(l), Fe(cr), or O2 individually.",
    }
    pack["rows"][0]["independent_check"] = distin
    for species in ("FeO(l)", "Fe(metal)", "Fe(l)", "Fe(cr)", "O2(g)"):
        pack["thermo_species"][species]["independent_check"] = {**distin, "status": "lead"}
    chen = {
        "status": "tested", "equilibrium": "Mn-bearing silicate/aluminate melt activity",
        "primary_source": "Chen et al. (2021), https://www.jstage.jst.go.jp/article/isijinternational/61/6/61_ISIJINT-2020-730/_pdf; Chen et al. (2022), https://www.jstage.jst.go.jp/article/isijinternational/62/7/62_ISIJINT-2022-018/_pdf",
        "conditions": "Held-out slag activity observations at 1673 K (18 points) and 1723 K (25 points).",
        "measured_value": "Observed Mn activities in held-out slag fixtures",
        "predicted_value": "Coupled OpenIMCC Mn-bearing melt predictions",
        "miss": None, "miss_to_band_ratio": None,
        "reason": "This is a coupled-model check; observations cannot be apportioned to an individual Mn standard or reaction.",
    }
    for species in ("MnO(l)", "MnSiO3(l)", "Mn2SiO4(l)", "MnAl2O4(l)"):
        pack["thermo_species"][species]["independent_check"] = chen
    for row in pack["rows"]:
        if "Mn" in row.get("reaction", ""):
            row["independent_check"] = chen
    changed = True
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
    inputs_path = root / "data-src/liquid-from-solid-inputs.json"
    inputs_digest = hashlib.sha256(inputs_path.read_bytes()).hexdigest()
    source_text = path.read_text()
    # Repair legacy flow mappings before YAML parsing: unquoted commas in a
    # citation are interpreted as mapping separators and silently discard the
    # remainder into null-valued keys.
    citation_fields = (
        "source_record|source_locator|source_url|doi|url|table_id|"
        "anchor|citation_gaps|upstream_source_sha256"
    )
    source_text, _citation_fixes = re.subn(
        rf"source: \{{citation: (?!['\"])(.*?)(, ({citation_fields}):)",
        lambda match: "source: {citation: '"
        + match.group(1).replace("'", "''")
        + "'"
        + match.group(2),
        source_text,
    )
    doc = yaml.safe_load(source_text)
    for row in doc["rows"]:
        source_path = row.get("source_path")
        if not source_path:
            continue
        source_file = root / source_path
        if not source_file.is_file():
            continue
        if source_path.startswith("data-src/janaf/"):
            file_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
            row["source_sha256"] = file_hash
            source_record = (
                yaml.safe_load(source_file.read_text())
                if source_file.suffix in {".yaml", ".yml"}
                else json.loads(source_file.read_text())
                if source_file.suffix == ".json"
                else {}
            )
            upstream_hash = source_record.get("extraction", {}).get("source_sha256")
            if upstream_hash and upstream_hash != file_hash:
                row["upstream_source_sha256"] = upstream_hash
            else:
                row.pop("upstream_source_sha256", None)
        elif source_path.startswith("data-src/published-thermo/"):
            if row.get("source_sha256"):
                row["source_sha256"] = hashlib.sha256(source_file.read_bytes()).hexdigest()
        source = row.get("source")
        if not isinstance(source, dict):
            continue
        citation = source.get("citation", "")
        malformed_citation = any(value is None for value in source.values()) or ": null" in citation
        if not malformed_citation:
            continue
        if source_path.startswith("data-src/published-thermo/pankratz-"):
            published = yaml.safe_load(
                (root / "data-src/published-thermo/PROVENANCE.yaml").read_text()
            )
            record = next(
                record for record in published["records"]
                if record["path"] == source_path
            )
            citation = record["publication"]
        elif source_path.startswith("data-src/konings/"):
            citation = json.loads(source_file.read_text())["citation"]
        else:
            continue
        source["citation"] = citation
        for key in [key for key, value in source.items() if value is None]:
            del source[key]
    parent_checks = {
        "PbO(l)": ("Holman & Fulrath (1973), LBL-1462", "Modified Knudsen effusion; pure PbO vapor-pressure curve", "approximately 1050–1180 K; compared vaporization enthalpy and entropy with JANAF, not liquid Gibbs energy", "Raw Fig. 22 points and Appendix C need transcription before calculating a pressure or Gibbs residual."),
        "Na2O(l)": ("Kobertz (2019), Calphad 64, 327–333, DOI 10.1016/j.calphad.2019.01.006", "KEMS of pure Na2O(c), Na and O2 pressures", "900–1165 K; reported solid-reference reaction Gibbs at 1000 K; no liquid or OpenIMCC difference", "Near-transition vapor evidence; reconcile phase and raw points before a matched comparison."),
        "CaO(l)": ("Halwax et al. (2024), Metallurgical and Materials Transactions B 55, 821–835, DOI 10.1007/s11663-024-02995-6", "Pure-CaO KEMS in an iridium cell", "1825–2125 K; measured Ca, O and O2 pressures compared with FactSage 7.3 FactPS; below the OpenIMCC liquid interval", "Measured solid-vapor evidence; model comparator is FactPS and no OpenIMCC miss was computed."),
        "MgO(l)": ("Halwax et al. (2024), Metallurgical and Materials Transactions B 55, 821–835, DOI 10.1007/s11663-024-02995-6", "Pure-MgO KEMS in an iridium cell", "1675–2075 K; measured Mg, O and O2 pressures compared with FactSage 7.3 FactPS; below the OpenIMCC liquid interval", "Measured solid-vapor evidence; model comparator is FactPS and no OpenIMCC miss was computed."),
        "FeO(l)": ("Distin, Whiteway & Masson (1971), DOI 10.1179/cmq.1971.10.1.13", "Oxygen solubility in liquid iron in contact with liquid iron oxide", "1800–2230 K; combined Fe(l)/FeO(l)/O2 buffer equilibrium", "Coupled redox check is not attributable to the isolated gas-parent FeO Gibbs energy."),
    }
    not_found = "No same-standard independent equilibrium check was located in the 2026-10-15 research sweep."
    for row in doc["rows"]:
        if row.get("construction_inputs_path") == "data-src/liquid-from-solid-inputs.json":
            row["construction_inputs_sha256"] = inputs_digest
        legacy = any(key in row for key in ("max_residual_J_per_mol", "max_residual_J_per_mol_by_interval", "max_residual_log10_K", "uncertainty_band", "additional_uncertainty_bands", "combined_band"))
        if not legacy and all(key in row for key in ("fit_residual", "accuracy_band", "band_basis", "independent_check")):
            fit = row.get("fit_residual")
            if (
                isinstance(fit, dict)
                and fit.get("temperature_range") in ("T_domain_K", "T_range_K or T_interval")
            ):
                fit["temperature_range"] = _temperature(row)
            continue
        old_residuals = {key: row.pop(key) for key in ("max_residual_J_per_mol", "max_residual_J_per_mol_by_interval", "max_residual_log10_K") if key in row}
        additional_bands = row.pop("additional_uncertainty_bands", None)
        old_band = row.pop("uncertainty_band", None)
        if "combined_band" in row:
            old_band = {"combined_band": row.pop("combined_band"), "source_uncertainty": row.pop("source_uncertainty", None)}
        source = row.get("source_locator") or row.get("source_record") or row.get("source_path")
        temperature = _temperature(row)
        if old_residuals:
            row["fit_residual"] = {
                "value": old_residuals,
                "unit": {
                    key: "J mol^-1" if "_J_per_mol" in key else "log10 K"
                    for key in old_residuals
                },
                "target_source": source,
                "comparison_method": row.get("method", "source-function fit"),
                "temperature_range": deepcopy(row.get("fit_intervals_K", temperature)),
                "held_out_or_in_sample": "in_sample",
            }
        else:
            row["fit_residual"] = None
        if old_band is not None:
            kind = str(old_band.get("kind", ""))
            basis_kind, basis_explanation = _source_band_kind(old_band)
            if (
                basis_kind is None
                and _normalized_controlled_value(kind)
                not in _SOURCE_BAND_CLASS_BY_KIND
            ):
                basis_explanation = (
                    "Unlisted source band kind; original band: "
                    + json.dumps(old_band, sort_keys=True, ensure_ascii=False)
                )
            if "combined_band" in old_band:
                basis_kind = "measured_construction"
                basis_explanation = "Tier-2 fusion-entropy and liquid-Cp census envelopes combined with the selected input spreads."
            if basis_kind is None:
                row["accuracy_band"] = None
                row["band_basis"] = band_basis("none", basis_explanation, source)
                row["fit_residual"] = row.get("fit_residual")
                old_band = None
            if old_band is None:
                pass
            else:
                unit = "kJ mol^-1" if "combined_band" in old_band else old_band.get("units", "J mol^-1")
                band_value = old_band
                if additional_bands:
                    band_value = {**old_band, "additional_uncertainty_bands": additional_bands}
                row["accuracy_band"] = {
                    "value": band_value,
                    "unit": unit,
                    "temperature_points_or_range": "values_J_per_mol keys or T_range_K",
                    "coverage": kind or "published source uncertainty or measured construction envelope",
                }
                row["band_basis"] = band_basis(basis_kind, basis_explanation, source)
        else:
            if additional_bands:
                old_band = {"additional_uncertainty_bands": additional_bands}
                row["accuracy_band"] = {"value": old_band, "unit": "J mol^-1", "temperature_points_or_range": temperature, "coverage": "Additional source comparison envelope."}
                row["band_basis"] = band_basis("measured_construction", "Additional source comparison envelope.", source)
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
            row["independent_check"]["status"] = "not_applicable"
    citation_notes = '''    lamoreaux_hildenbrand_1984:
      citation: >-
        Lamoreaux, R. H.; Hildenbrand, D. L. (1984), High Temperature
        Vaporization Behavior of Oxides. I. Alkali Metal Binary Oxides,
        J. Phys. Chem. Ref. Data 13(1), 151-173.
      doi: 10.1063/1.555706
      coefficient_locator: Table 2, p. 154 (K2O(l), 1190-3000 K)
      anchor_locator: Table 4, p. 157 (K2O(s) formation enthalpy)
      records_are: >-
        Primary printed thermodynamic tables. The legacy compatibility row is
        directly checked against Table 2; the default K2O(l) row is the
        separate K-012 liquid-from-solid estimate.
    konings_benes_2010:
      citation: >-
        Konings, R. J. M.; Beneš, O. (2010), The Thermodynamic Properties of
        the f-Elements and Their Compounds. I. The Lanthanide and Actinide
        Metals, J. Phys. Chem. Ref. Data 39(4), 043102.
      doi: 10.1063/1.3474238
    konings_et_al_2014:
      citation: >-
        Konings, R. J. M.; Beneš, O.; Kovács, A.; Manara, D.; Sedmidubský, D.;
        Gorokhov, L.; Iorish, V. S.; Yungman, V.; Shenyavskaya, E.; Osina, E.
        (2014), The Thermodynamic Properties of the f-Elements and Their
        Compounds. Part 2. The Lanthanide and Actinide Oxides, J. Phys. Chem.
        Ref. Data 43(1), 013101.
      doi: 10.1063/1.4825256
    pedley_marshall_1983:
      citation: >-
        Pedley, J. B.; Marshall, E. M. (1983), Thermochemical Data for Gaseous
        Monoxides, J. Phys. Chem. Ref. Data 12(4), 967-1031.
      doi: 10.1063/1.555698
'''
    marker = "    declined_ions:\n"
    if "lamoreaux_hildenbrand_1984:" not in source_text:
        source_text = source_text.replace(marker, citation_notes + marker, 1)
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
        current_band = row.get("accuracy_band")
        current_value = current_band.get("value") if isinstance(current_band, dict) else None
        existing_basis = row.get("band_basis")
        current_kind = str(current_value.get("kind", "")) if isinstance(current_value, dict) else ""
        if (
            isinstance(existing_basis, dict)
            and existing_basis.get("kind") == "measured_construction"
            and current_kind
        ):
            current_class = _liquid_input_band_class(current_kind)
            if current_class is None:
                row["accuracy_band"] = None
                existing_basis["kind"] = "none"
                existing_basis["explanation"] = (
                    "Unlisted liquid-input band kind; original band: "
                    + json.dumps(current_band, sort_keys=True, ensure_ascii=False)
                )
            else:
                existing_basis["kind"] = current_class
        extension_metadata = row.get("crystal_cp_extrapolation_band")
        if extension_metadata is None and isinstance(current_value, dict) and "source_limit_K" in current_value and "values_J_per_mol" not in current_value:
            extension_metadata = current_value
        if (
            not any(key in row for key in ("measured_band", "crystal_cp_extrapolation_band"))
            and extension_metadata is None
            and all(key in row for key in ("fit_residual", "accuracy_band", "band_basis", "independent_check"))
        ):
            continue
        old_measured = row.pop("measured_band", None)
        old_extension = row.pop("crystal_cp_extrapolation_band", None) or extension_metadata
        if old_measured is not None:
            measured_kind = str(old_measured.get("kind", ""))
            basis_kind = _liquid_input_band_class(measured_kind)
            if basis_kind is None:
                row["accuracy_band"] = None
                row["band_basis"] = band_basis(
                    "none",
                    "Unlisted legacy band kind; no accuracy class was inferred.",
                    f"{formula}; unlisted legacy band: "
                    + json.dumps(old_measured, sort_keys=True, ensure_ascii=False),
                )
            else:
                row["accuracy_band"] = {
                    "value": old_measured,
                    "unit": "J mol^-1",
                    "temperature_points_or_range": list(old_measured.get("values_J_per_mol", {}).keys()),
                    "coverage": old_measured.get("kind", "measured construction envelope"),
                }
                row["band_basis"] = band_basis(basis_kind, old_measured.get("source", old_measured.get("kind", "")), formula)
        elif old_extension is not None:
            row["accuracy_band"] = None
            row["band_basis"] = band_basis(
                "none",
                "Cp extrapolation comparison metadata is retained as source provenance; no numerical row band is published.",
                old_extension,
            )
        else:
            row["accuracy_band"] = None
            row["band_basis"] = band_basis(
                "none",
                "Input provenance only; no composed liquid Gibbs band stored on this input row.",
                formula,
            )
        row["fit_residual"] = None
        row["independent_check"] = independent_check("This row supplies an input to a liquid construction; no row-specific equilibrium check is recorded.")
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


def migrate_melt_row(row: dict[str, Any]) -> None:
    provenance = row.get("provenance")
    provenance_ref = (
        provenance.get("source") or provenance.get("source_url")
        if isinstance(provenance, dict)
        else provenance
    )
    def has_legacy_residual(value: Any, *, top: bool = False) -> bool:
        if isinstance(value, dict):
            for key, member in value.items():
                if top and key == "fit_residual":
                    continue
                if "residual" in key.lower() or has_legacy_residual(member):
                    return True
            return False
        if isinstance(value, list):
            return any(has_legacy_residual(member) for member in value)
        return False

    if (
        all(key in row for key in ("fit_residual", "accuracy_band", "band_basis", "independent_check"))
        and not has_legacy_residual(row, top=True)
        and (
            row["fit_residual"] is None
            or row["fit_residual"].get("temperature_range") == _temperature(row)
        )
    ):
        return
    existing_fit = row.get("fit_residual")
    existing_diagnostics = (
        existing_fit.get("value")
        if isinstance(existing_fit, dict) and isinstance(existing_fit.get("value"), dict)
        else {}
    )
    diagnostics: dict[str, Any] = {}

    def move_residuals(value: Any, prefix: str = "", *, top: bool = False) -> None:
        if isinstance(value, dict):
            for key in list(value):
                if top and key == "fit_residual":
                    continue
                label = f"{prefix}/{key}" if prefix else key
                if "residual" in key.lower():
                    diagnostics[label] = value.pop(key)
                else:
                    move_residuals(value[key], label)
        elif isinstance(value, list):
            for index, member in enumerate(value):
                move_residuals(member, f"{prefix}/{index}")

    move_residuals(row, top=True)
    diagnostics = {**existing_diagnostics, **diagnostics}
    split = _melt_held_out_split(diagnostics)
    row["fit_residual"] = None if not diagnostics else {
        "value": diagnostics,
        "unit": "source-specific",
        "target_source": row.get("source") or provenance_ref,
        "comparison_method": "Existing held-out or candidate-fit diagnostic; see original field names in value.",
                "temperature_range": _temperature(row),
        "held_out_or_in_sample": split,
    }
    row["accuracy_band"] = None
    row["band_basis"] = band_basis("none", "Complex association row has no row-specific uncertainty band.", provenance_ref)
    row["independent_check"] = independent_check("No row-specific pure-complex equilibrium check was identified.")


def _melt_packs(root: Path) -> None:
    for path in sorted((root / "src/openimcc/data/packs").glob("imcc-sf04*.json")):
        doc = json.loads(path.read_text())
        for row in doc["rows"]:
            migrate_melt_row(row)
        for row in doc.get("sp_extension", {}).get("rows", []):
            migrate_melt_row(row)
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


def migrate(root: Path = ROOT) -> None:
    _redox(root)
    _redox_source(root)
    _liquid_inputs(root)
    _gas(root)
    _melt_packs(root)


if __name__ == "__main__":
    migrate()
