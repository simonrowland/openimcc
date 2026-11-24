#!/usr/bin/env python3
"""Build the opt-in full-component P2O5 candidate from reviewed source rows."""

from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openimcc.gas import load_gas_datapack, species_thermo


PACKS = ROOT / "src/openimcc/data/packs"
BASE = PACKS / "imcc-sf04-d066-v4.json"
DESTINATION = PACKS / "imcc-sf04-d066-p2o5-candidate-v1.json"
SOURCE = ROOT / "data-src/p2o5-candidate/thermochemistry.json"
JANAF = ROOT / "data-src/janaf"
JUNG_HUDON = ROOT / "data-src/published-thermo/jung-hudon-p2o5.json"
FIT_LOW_K = 1700.0
FIT_HIGH_K = 2300.0
R_J_MOL_K = 8.31446261815324
LN10 = math.log(10.0)
PARENT_ORDER = ("SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O", "K2O", "P2O5")
CLASS_C_J_MOL_ATOM_K = 8.297530569005671


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _array_item_texts(raw: str, key: str) -> list[str]:
    """Return exact JSON source slices for each item in a named array."""
    marker = f'"{key}"'
    marker_start = raw.index(marker)
    array_start = raw.index("[", marker_start + len(marker))
    decoder = json.JSONDecoder()
    cursor = array_start + 1
    items = []
    while True:
        while cursor < len(raw) and raw[cursor].isspace():
            cursor += 1
        if cursor >= len(raw) or raw[cursor] == "]":
            return items
        _value, end = decoder.raw_decode(raw, cursor)
        items.append(raw[cursor:end])
        cursor = end
        while cursor < len(raw) and raw[cursor].isspace():
            cursor += 1
        if cursor < len(raw) and raw[cursor] == ",":
            cursor += 1
        elif cursor < len(raw) and raw[cursor] == "]":
            return items
        else:
            raise ValueError(f"malformed {key} array while checking source bytes")


def _compatibility_material(pack: dict) -> tuple[str, str]:
    base_rows = pack["rows"][:38]
    row_digest = _digest(base_rows)
    standards = pack.get("parent_standard_candidates", {})
    standard_material = {
        "parents": pack["parents"][:8],
        "parent_standard_candidates": standards,
        "feo_liquid_candidate": pack.get("feo_liquid_candidate"),
        "row_standard_state_refits": [row.get("standard_state_refit") for row in base_rows],
    }
    return row_digest, _digest(standard_material)


def _parent_cp(oxide: str, temperature_K: float, gas_pack) -> float:
    return float(species_thermo(oxide, "l", temperature_K, gas_pack).Cp_J_molK)


def _janaf_g(table: dict, temperature_K: float) -> float:
    points = sorted(
        (float(row[0]), float(row[6]) * 1000.0)
        for row in table["values"]
        if len(row) >= 7 and row[6] is not None
    )
    for temperature, gibbs in points:
        if temperature == temperature_K:
            return gibbs
    for (low_t, low_g), (high_t, high_g) in zip(points, points[1:]):
        if low_t < temperature_K < high_t:
            # Premise: JANAF prints Gf at bracketing nodes. Linear interpolation
            # uses G(T)=Glo+(T-Tlo)/(Thi-Tlo)*(Ghi-Glo); input and output G are
            # J/mol, so the temperature ratio is dimensionless. Sanity: either
            # bracket endpoint is recovered exactly as the interpolation limit.
            fraction = (temperature_K - low_t) / (high_t - low_t)
            return low_g + fraction * (high_g - low_g)
    raise ValueError(f"{table['title']} has no formation-Gibbs rows bracketing {temperature_K:g} K")


@lru_cache(maxsize=None)
def _janaf_table(table_id: str) -> dict:
    return yaml.safe_load(
        (JANAF / f"{table_id}.yaml").read_text(encoding="utf-8")
    )["table"]


def _janaf_table_value(table_id: str, temperature_K: float, quantity: str) -> float:
    rows = _janaf_table(table_id)["values"]
    points = sorted(
        (
            float(row["temperature"]["value"]),
            float(row[quantity]["value"]),
        )
        for row in rows
        if row[quantity]["value"] is not None
    )
    for temperature, value in points:
        if temperature == temperature_K:
            return value
    for (low_t, low_v), (high_t, high_v) in zip(points, points[1:]):
        if low_t < temperature_K < high_t:
            fraction = (temperature_K - low_t) / (high_t - low_t)
            return low_v + fraction * (high_v - low_v)
    raise ValueError(f"JANAF {table_id} has no {quantity} rows bracketing {temperature_K:g} K")


def _janaf_formation_g(table_id: str, temperature_K: float) -> float:
    return 1000.0 * _janaf_table_value(
        table_id, temperature_K, "formation_gibbs_energy"
    )


def _p2o5_formation_g(temperature_K: float, jung_hudon: dict) -> float:
    """Convert the Jung–Hudon gas reaction to JANAF elemental-reference ΔGf."""
    equation = jung_hudon["delta_g_standard_j_mol"]
    delta_g_reaction = float(equation["intercept"]) + float(equation["slope"]) * temperature_K

    # Premise: Jung–Hudon's reaction uses inferred 1 atm, while JANAF and the
    # pack use 1 bar. For P2(g)+5/2 O2(g)=P2O5(l), Δνgas=0-(1+2.5)=-3.5,
    # so ΔG°(1 bar)=ΔG°(1 atm)+ΔνRT ln(1/1.01325). RT is J/mol and the log
    # is dimensionless. Sanity at 2000 K: this adds about +0.762 kJ/mol.
    delta_g_nist_pressure = -3.5 * R_J_MOL_K * temperature_K * math.log(1.0 / 1.01325)
    delta_g_reaction += delta_g_nist_pressure

    # ΔGf(P2O5,l)=ΔG°[P2(g)+5/2O2(g)→P2O5(l)]+ΔGf(P2,g), because
    # ΔGf(O2,g)=0. This places the Jung–Hudon parent on the same JANAF
    # elemental reference as Mg3(PO4)2 and the oxide parents. Above the
    # 1180.008 K P2(g) reference marker, JANAF ΔGf(P2,g)=0 (P2(g) is the
    # standard elemental state); reading the table keeps that conversion
    # explicit instead of assuming it. All values are J/mol.
    return delta_g_reaction + _janaf_formation_g("P-012", temperature_K)


def _parent_formation_g(
    oxide: str, temperature_K: float, jung_hudon: dict
) -> float:
    tables = {"MgO": "Mg-009", "CaO": "Ca-028"}
    if oxide == "P2O5":
        return _p2o5_formation_g(temperature_K, jung_hudon)
    try:
        table_id = tables[oxide]
    except KeyError as exc:
        raise ValueError(f"no JANAF formation-Gibbs table for {oxide}") from exc
    return _janaf_formation_g(table_id, temperature_K)


def _mg_crystal_hess_298_kj_mol(source: dict) -> float:
    mg_rows = source["janaf_pages"]["Mg-038"]["values"]
    mg_hf = next(float(row[5]) for row in mg_rows if float(row[0]) == 298.15)
    mgo_hf = _janaf_table_value("Mg-008", 298.15, "formation_enthalpy")
    p2o5 = source["janaf_reference_checks"]["p2o5_crystal_298"]
    p2o5_hf = float(p2o5["formation_enthalpy_kJ_mol"])
    formula_units = float(p2o5["p2o5_formula_units_per_table_formula"])
    return mg_hf - 3.0 * mgo_hf - p2o5_hf / formula_units


def _ca_crystal_formation_g(source: dict, temperature_K: float) -> float:
    rows = source["whitlockite"]["values"]
    points = [(float(row[0]), float(row[6]) * 1000.0) for row in rows]
    if temperature_K <= 1600.0:
        for (low_t, low_g), (high_t, high_g) in zip(points, points[1:]):
            if low_t <= temperature_K <= high_t:
                fraction = (temperature_K - low_t) / (high_t - low_t)
                return low_g + fraction * (high_g - low_g)
        raise ValueError(f"Whitlockite table does not cover {temperature_K:g} K")

    # Above 1600 K the crystal Cp is held at its printed 330.54 J mol^-1 K^-1.
    # The formation enthalpy changes by 4.1 kJ/mol from 1500 to 1600 K, so
    # ΔCp_form = ΔH_form/ΔT = 41.0 J mol^-1 K^-1. With that held constant,
    # H(T)=H0+ΔCp_form(T-T0), S(T)=S0+ΔCp_form ln(T/T0), and G=H-TS, preserving
    # the R&H elemental reference. Units are J/mol and J/(mol K). Sanity:
    # G(1600 K) rejoins the printed -2816.6 kJ/mol formation value.
    last = rows[-1]
    previous = rows[-2]
    reference_T = float(last[0])
    delta_h_ref = float(last[5]) * 1000.0
    delta_g_ref = float(last[6]) * 1000.0
    delta_s_ref = (delta_h_ref - delta_g_ref) / reference_T
    delta_cp = (float(last[5]) - float(previous[5])) * 1000.0 / (
        reference_T - float(previous[0])
    )
    enthalpy = delta_h_ref + delta_cp * (temperature_K - reference_T)
    entropy = delta_s_ref + delta_cp * math.log(temperature_K / reference_T)
    return enthalpy - temperature_K * entropy


def _fit(points: np.ndarray) -> tuple[float, float, float, float]:
    # Premise: log10(K)=A+B/T is linear in regressors [1,1/T]. A is
    # dimensionless, B has units K, and the least-squares residual is dex.
    # Sanity: A+B/2000 reproduces the fitted 2000 K value for the source grid.
    design = np.column_stack((np.ones(len(points)), 1.0 / points[:, 0]))
    coefficients, *_ = np.linalg.lstsq(design, points[:, 1], rcond=None)
    residual = points[:, 1] - design @ coefficients
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(np.max(np.abs(residual))),
        float(np.sqrt(np.mean(residual**2))),
    )


def _reaction_fit(
    parent_oxides: dict[str, float],
    product_gibbs,
    temperatures: np.ndarray,
    jung_hudon: dict,
) -> tuple[float, float, float, float]:
    # Premise: each input here is a formation Gibbs energy from one common
    # elemental reference: JANAF Mg-039 (or the R&H formation rows), JANAF
    # Mg-009/Ca-028 liquid oxide, and Jung–Hudon P2O5(l) converted by
    # _p2o5_formation_g. In ΔGf(product)-3ΔGf(MO)-ΔGf(P2O5), the three Mg/Ca,
    # two P, and eight O elemental chemical potentials cancel exactly. This
    # avoids mixing JANAF ΔGf with the enthalpy-anchored absolute G returned by
    # species_thermo. log10(K)=-ΔG°/(RT ln10); J/mol divided by J/mol is
    # dimensionless. Sanity: the 298.15 K Mg crystal Hess cycle is -436.407 kJ/mol.
    values = []
    for temperature in temperatures:
        product = product_gibbs(float(temperature))
        parents = math.fsum(
            coefficient
            * _parent_formation_g(oxide, float(temperature), jung_hudon)
            for oxide, coefficient in parent_oxides.items()
        )
        log_k = -(product - parents) / (R_J_MOL_K * float(temperature) * LN10)
        values.append((float(temperature), log_k))
    return _fit(np.asarray(values, dtype=float))


def _row(
    *,
    name: str,
    index: int,
    parent_oxides: dict[str, float],
    coefficients: tuple[float, float, float, float],
    paper_domain: list[float],
    source_refs: list[str],
    derivation: str,
    flags: list[str],
) -> dict:
    intercept, slope, max_residual, rms_residual = coefficients
    return {
        "row": index,
        "complex": name,
        "reaction": " + ".join(
            f"{amount:g} {oxide}(liq)" for oxide, amount in parent_oxides.items()
        ) + f" = {name}(liq)",
        "nu": {parent: parent_oxides.get(parent, 0.0) for parent in PARENT_ORDER},
        "A": intercept,
        "B": slope,
        "source": source_refs[0],
        "T_domain_K": [FIT_LOW_K, FIT_HIGH_K],
        "T_domain_basis": "Opt-in candidate fit domain; each associate retains its source window and continuation flags.",
        "provenance_class": "C-independent-thermochemistry",
        "derivation": derivation,
        "T_domain_paper_demonstrated_K": paper_domain,
        "coverage": "C-independent-thermochemistry",
        "flags": flags,
        "fit_residual": None,
        "accuracy_band": None,
        "band_basis": {
            "kind": "none",
            "explanation": "No row-specific uncertainty interval is claimed; provisional constructions are listed in candidate_provenance.",
            "provenance": None,
        },
        "independent_check": {
            "status": "not_found",
            "equilibrium": None,
            "primary_source": None,
            "conditions": None,
            "measured_value": None,
            "predicted_value": None,
            "miss": None,
            "miss_to_band_ratio": None,
            "reason": "No independent liquid association equilibrium was used as a fitting target.",
        },
        "source_refs": source_refs,
        "standard_state_refit": {
            "parents": parent_oxides,
            "temperature_range_K": [FIT_LOW_K, FIT_HIGH_K],
            "fit_method": "least squares on 1001 evenly spaced pointwise re-referenced log10 K values",
            "max_abs_residual_dex": max_residual,
            "rms_residual_dex": rms_residual,
            "flag_over_0_01_dex": bool(max_residual > 0.01),
        },
    }


def build() -> dict:
    pack = json.loads(BASE.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    jung_hudon = json.loads(JUNG_HUDON.read_text(encoding="utf-8"))
    original_rows = copy.deepcopy(pack["rows"])
    if len(original_rows) != 38:
        raise ValueError("D066 v4 must contain exactly 38 rows")
    mg_hess_298 = _mg_crystal_hess_298_kj_mol(source)
    # JANAF prints these three enthalpies to 0.001 kJ/mol. The Hess cycle is
    # -3745.098 - 3(-601.241) - (-3009.936/2) = -436.407 kJ/mol; its negative
    # sign is the required independent check that phosphate formation from
    # MgO(cr)+P2O5(cr) is strongly exothermic.
    if not math.isclose(mg_hess_298, -436.407, rel_tol=0.0, abs_tol=0.01):
        raise ValueError(f"Mg phosphate crystal Hess check is {mg_hess_298:g} kJ/mol")
    gas_pack = load_gas_datapack()
    temperatures = np.linspace(FIT_LOW_K, FIT_HIGH_K, 1001)
    mg_table = source["janaf_pages"]["Mg-039"]
    mg_parents = {"MgO": 3.0, "P2O5": 1.0}
    mg = _reaction_fit(
        mg_parents,
        lambda temperature: _janaf_g(mg_table, temperature),
        temperatures,
        jung_hudon,
    )

    ca_data = source["whitlockite"]
    melting = source["ca3_po4_2_melting"]
    tm = float(melting["temperature_K"])
    ca_parents = {"CaO": 3.0, "P2O5": 1.0}

    def ca_crystal_reaction_g(temperature: float) -> float:
        product = _ca_crystal_formation_g(source, temperature)
        return product - math.fsum(
            coefficient * _parent_formation_g(oxide, temperature, jung_hudon)
            for oxide, coefficient in ca_parents.items()
        )

    delta_g_fusion = ca_crystal_reaction_g(tm)
    # Premise: the crystal reaction entropy is -d(ΔG)/dT at Tm. A symmetric
    # finite difference cancels first-order truncation error; with h in kelvin,
    # the quotient has J mol^-1 K^-1. Sanity: this is evaluated at ±0.01 K
    # around the independently sourced 2083.15 K melting marker.
    step = 0.01
    solid_reaction_entropy = -(
        ca_crystal_reaction_g(tm + step) - ca_crystal_reaction_g(tm - step)
    ) / (2.0 * step)
    # Premise: class-C uses a molar entropy per formula atom. Ca3(PO4)2 has
    # 3+2+8=13 atoms, so ΔSfus = 13 × 8.297530569 J mol^-1 K^-1; units remain
    # J mol^-1 K^-1. Sanity: the estimate is 107.868 J mol^-1 K^-1.
    fusion_entropy = (
        float(source["class_c_fusion_entropy_J_mol_K_atom"])
        * sum({"Ca": 3, "P": 2, "O": 8}.values())
    )
    liquid_reaction_entropy = solid_reaction_entropy + fusion_entropy
    # Premise: the provisional liquid heat capacity follows the Sr construction
    # and is additive in the stoichiometric parent Cp values. Multiplying
    # J mol^-1 K^-1 by dimensionless stoichiometry preserves those units.
    # Sanity at Tm: 3*Cp(CaO,l)+Cp(P2O5,l)=320.402673 J mol^-1 K^-1.
    cp_liquid = math.fsum(
        coefficient * _parent_cp(oxide, tm, gas_pack)
        for oxide, coefficient in ca_parents.items()
    )
    if not math.isfinite(cp_liquid) or cp_liquid <= 0.0:
        raise ValueError("additive Ca-phosphate liquid Cp must be positive and finite")

    def ca_liquid_formation_g(temperature: float) -> float:
        # Premise: at Tm the crystal and liquid chemical potentials agree.
        # Class-C ΔSfus raises the product entropy, while the additive liquid
        # Cp makes ΔCp for M3(PO4)2(l)-3MO(l)-P2O5(l) exactly zero. Algebra:
        # ΔG_rxn,l(T)=ΔG_rxn,cr(Tm)+(Tm-T)(ΔS_rxn,cr+ΔSfus). Units are J/mol.
        # Sanity: at Tm the liquid and crystal reaction Gibbs energies match.
        return (
            delta_g_fusion
            + (tm - temperature) * liquid_reaction_entropy
            + math.fsum(
                coefficient
                * _parent_formation_g(oxide, temperature, jung_hudon)
                for oxide, coefficient in ca_parents.items()
            )
        )

    # _reaction_fit expects product Gf and subtracts parent Gf. The Ca liquid
    # construction above returns product Gf by adding the parent functions.
    ca = _reaction_fit(
        ca_parents,
        ca_liquid_formation_g,
        temperatures,
        jung_hudon,
    )

    row_mg = _row(
        name="Mg3(PO4)2",
        index=39,
        parent_oxides=mg_parents,
        coefficients=mg,
        paper_domain=[1621.0, 3000.0],
        source_refs=[
            "JANAF-MG-PHOSPHATE",
            "JANAF-MGO-LIQUID",
            "JUNG-HUDON-P2O5-LIQUID",
            "JANAF-P2-GAS",
        ],
        derivation=(
            "Re-referenced from JANAF Mg-039(l) to D066 MgO(l) and the existing "
            "Jung–Hudon P2O5(l) parent standard; the P elemental reference cancels "
            "in the formation-from-oxides reaction."
        ),
        flags=["printed_janaf_liquid"],
    )
    row_ca = _row(
        name="Ca3(PO4)2",
        index=40,
        parent_oxides=ca_parents,
        coefficients=ca,
        paper_domain=[298.15, 1600.0],
        source_refs=[
            "ROBIE-HEMINGWAY-1995",
            "WELCH-GUTT-1961",
            "JANAF-CAO-LIQUID",
            "JUNG-HUDON-P2O5-LIQUID",
            "JANAF-P2-GAS",
        ],
        derivation=(
            "Whitlockite crystal formation Gibbs function to 1600 K; constant-Cp "
            "crystal continuation to 2083.15 K; class-C fusion entropy and additive "
            "CaO(l)+P2O5(l) liquid Cp; then refit to the candidate parent standards."
        ),
        flags=[
            "constant_cp_crystal_continuation_above_1600_K",
            "class_c_fusion_entropy_provisional",
            "additive_parent_liquid_cp_provisional",
            "supercooled_liquid_branch_below_2083_15_K",
        ],
    )

    row_digest, standard_digest = _compatibility_material(pack)
    pack["imcc_sf04_datapack_version"] = "1.0.0-d066-p2o5-candidate-v1"
    pack["model_id"] = "IMCC-SF04-D066-P2O5-CANDIDATE"
    pack["model"] = (
        "Opt-in associated-solution candidate with nine full parent oxides, "
        "the unchanged 38-row D066 core, and independently sourced Mg/Ca phosphate associates."
    )
    pack["parents"] = list(PARENT_ORDER)
    pack["pack_arithmetic"] = "38 unchanged D066 v4 core rows + 2 phosphate candidate rows = 40 complexes."
    pack["parent_standard_candidates"]["P2O5"] = {
        "source": "Jung and Hudon (2012), P2O5(l), existing gas/condensate standard",
        "doi": "10.1111/j.1551-2916.2012.05382.x",
        "standard_state": "pure liquid P2O5 at 1 bar",
        "measured_range_K": [1200.0, 3000.0],
        "numeric_uncertainty": None,
        "typed_gap": "no_numeric_uncertainty_published",
        "transfer_note": "Uses the existing P2O5(l) gas and condensate record without rebuilding it from JANAF O-094.",
    }
    pack["candidate_provenance"] = {
        "status": "opt-in provisional candidate; not the default",
        "fit_interval_K": [FIT_LOW_K, FIT_HIGH_K],
        "p_channel_status": "shipped_with_provisional_associates",
        "p_channel_status_basis": (
            "With all Gibbs terms on the same JANAF elemental reference, the "
            "two-associate KREEP check places a(P2O5) many orders below "
            "x(P2O5); P channels are enabled with the provisional Ca3(PO4)2 "
            "flags retained."
        ),
        "source_file": "data-src/p2o5-candidate/thermochemistry.json",
        "d066_core_compatibility": {
            "declared": True,
            "row_count": 38,
            "rows_1_38_sha256": row_digest,
            "existing_parent_liquid_standards_sha256": standard_digest,
            "basis": "The copied D066 rows, their row-level standard-state refits, and common liquid-standard records match D066 v4 by content digest.",
        },
        "parent_standard": "P2O5(l) is the existing Jung–Hudon 1-bar gas/condensate record; its numeric uncertainty remains a typed gap.",
        "formation_from_oxides_derivation": (
            "For 3 MO(l)+P2O5(l)=M3(PO4)2(l), use ΔGf°(product), JANAF ΔGf°(MO,l), "
            "and ΔGf°(P2O5,l)=ΔrG°[P2(g)+5/2O2(g)→P2O5(l)]+ΔGf°(P2,g). "
            "JANAF ΔGf°(O2,g)=0. The 3 M, 2 P, and 8 O elemental chemical potentials cancel from "
            "ΔGf°(product)-3ΔGf°(MO)-ΔGf°(P2O5); the P2 term is explicit so the elemental-P "
            "reference matches the JANAF phosphate row. Convert Jung–Hudon's inferred 1 atm "
            "equation to 1 bar with Δνgas=-3.5. Gibbs energies are J/mol and log10K=-ΔG°/(RTln10) is dimensionless."
        ),
        "mg_crystal_hess_check_298_kJ_mol": mg_hess_298,
        "mg3_po4_2": {
            "tables": ["Mg-038", "Mg-039", "Mg-040"],
            "fusion_temperature_K": 1621.0,
            "fusion_entropy_J_mol_K": 74.853,
            "fusion_enthalpy_kJ_mol": 121.336,
            "liquid_cp_J_mol_K": 475.302,
            "provisional": [],
            "paper_window_K": [1621.0, 3000.0],
            "fit_max_abs_residual_dex": mg[2],
            "fit_rms_residual_dex": mg[3],
            "crystal_formation_from_oxides_298_kJ_mol": mg_hess_298,
        },
        "ca3_po4_2": {
            "crystal_source_window_K": [298.15, 1600.0],
            "rhombohedral_to_monoclinic_K": 1373.0,
            "constant_cp_crystal_J_mol_K": 330.54,
            "constant_cp_formation_reference_J_mol_K": 41.0,
            "melting_temperature_K": tm,
            "melting_source": melting["citation"],
            "fusion_entropy_J_mol_K": fusion_entropy,
            "liquid_cp_J_mol_K_at_fusion": cp_liquid,
            "provisional": ["crystal continuation above 1600 K", "class-C fusion entropy", "additive parent liquid Cp", "liquid association below the melting point"],
            "fit_max_abs_residual_dex": ca[2],
            "fit_rms_residual_dex": ca[3],
        },
        "typed_gaps": [
            "Jung–Hudon P2O5(l) numeric uncertainty",
            "Ca2P2O7",
            "AlPO4 melting",
            "Na phosphates",
            "K phosphates",
            "Fe3(PO4)2",
            "apatite",
        ],
    }
    pack["sources"].update(
        {
            "JANAF-MG-PHOSPHATE": {
                "citation": "Chase (1998), NIST-JANAF Thermochemical Tables, 4th ed., Mg-038, Mg-039, Mg-040.",
                "doi": "10.18434/T42S31",
                "file": "data-src/p2o5-candidate/thermochemistry.json#janaf_pages",
                "locator": "Mg3P2O8 crystal, liquid, and crystal/liquid transition tables.",
            },
            "ROBIE-HEMINGWAY-1995": {
                "citation": ca_data["citation"],
                "doi": ca_data["doi"],
                "file": "data-src/p2o5-candidate/thermochemistry.json#whitlockite",
                "locator": ca_data["locator"],
            },
            "WELCH-GUTT-1961": {
                "citation": melting["citation"],
                "doi": melting["doi"],
                "url": melting["url"],
                "locator": melting["locator"],
            },
            "JANAF-MGO-LIQUID": {
                "citation": "Chase (1998), NIST-JANAF Thermochemical Tables, 4th ed., Mg-009.",
                "doi": "10.18434/T42S31",
                "file": "data-src/janaf/Mg-009.yaml",
                "locator": "MgO(l) formation-Gibbs column, 1700–2300 K; the JANAF pure-liquid standard also underlies the D066 parent standard.",
            },
            "JANAF-CAO-LIQUID": {
                "citation": "Chase (1998), NIST-JANAF Thermochemical Tables, 4th ed., Ca-028.",
                "doi": "10.18434/T42S31",
                "file": "data-src/janaf/Ca-028.yaml",
                "locator": "CaO(l) formation-Gibbs column, 1700–2300 K; the JANAF pure-liquid standard also underlies the D066 parent standard.",
            },
            "JANAF-P2-GAS": {
                "citation": "Chase (1998), NIST-JANAF Thermochemical Tables, 4th ed., P-012.",
                "doi": "10.18434/T42S31",
                "file": "data-src/janaf/P-012.yaml",
                "locator": "P2(g) formation-Gibbs column and its 1180.008 K 1-bar elemental-reference marker.",
            },
            "JUNG-HUDON-P2O5-LIQUID": {
                "citation": "Jung and Hudon (2012), Thermodynamic Assessment of P2O5, Journal of the American Ceramic Society 95(11), 3665-3672.",
                "doi": "10.1111/j.1551-2916.2012.05382.x",
                "file": "data-src/published-thermo/jung-hudon-p2o5.json",
                "locator": "Existing P2(g)+5/2O2(g)=P2O5(l) equation, converted from its inferred 1-atm gas standard to the JANAF 1-bar standard.",
            },
        }
    )
    pack["rows"].extend((row_mg, row_ca))
    pack["errata"].append(
        {
            "version": pack["imcc_sf04_datapack_version"],
            "change": "Adds an opt-in P2O5 parent and two phosphate associates; D066 v4 and all published packs remain unchanged.",
        }
    )
    if pack["rows"][:38] != original_rows:
        raise AssertionError("rows 1–38 changed from D066 v4")
    return pack


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    base_rows = _array_item_texts(BASE.read_text(encoding="utf-8"), "rows")
    candidate_rows = _array_item_texts(generated, "rows")
    if len(base_rows) != 38 or candidate_rows[:38] != base_rows:
        raise AssertionError("candidate rows 1–38 are not byte-equal to D066 v4")
    if args.write:
        DESTINATION.write_text(generated, encoding="utf-8")
        print(f"wrote {DESTINATION.relative_to(ROOT)}")
        return 0
    if not DESTINATION.exists() or DESTINATION.read_text(encoding="utf-8") != generated:
        print(f"{DESTINATION.relative_to(ROOT)} is stale; run with --write")
        return 1
    print(f"{DESTINATION.relative_to(ROOT)} is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
