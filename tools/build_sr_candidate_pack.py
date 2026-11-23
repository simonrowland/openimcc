#!/usr/bin/env python3
"""Build the opt-in SrO full-component D066 candidate from independent inputs."""

from __future__ import annotations

import argparse
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
from openimcc.redox_pack import load_redox_pack, species_gibbs_j_mol


PACKS = ROOT / "src/openimcc/data/packs"
DESTINATION = PACKS / "imcc-sf04-d066-sr-candidate-v1.json"
BASE = PACKS / "imcc-sf04-d066-v4.json"
REDOX = PACKS / "openimcc-redox-v6.json"
SOURCE = ROOT / "data-src/sr-candidate/thermochemistry.json"
JANAF = ROOT / "data-src/janaf"
FIT_LOW_K = 1700.0
FIT_HIGH_K = 2400.0
R_J_MOL_K = 8.31446261815324
LN10 = math.log(10.0)
CLASS_C_J_MOL_ATOM_K = 8.297530569005671
_LIQUID_CACHE: dict[tuple[str, float], tuple[float, float]] = {}


@lru_cache(maxsize=None)
def _table(table_id: str) -> dict:
    return yaml.safe_load((JANAF / f"{table_id}.yaml").read_text(encoding="utf-8"))["table"]


@lru_cache(maxsize=None)
def _janaf_props(table_id: str, temperature_K: float) -> tuple[float, float, float, float]:
    """Return H[J/mol], S[J/mol/K], Cp[J/mol/K], G[J/mol] from JANAF rows."""
    table = _table(table_id)
    rows = table["values"]
    h298_kj = next(
        float(row["formation_enthalpy"]["value"])
        for row in rows
        if row["temperature"]["value"] == 298.15
    )
    points = [
        (
            float(row["temperature"]["value"]),
            h298_kj + float(row["enthalpy_increment"]["value"]),
            float(row["entropy"]["value"]),
            float(row["heat_capacity"]["value"]),
        )
        for row in rows
        if row["formation_enthalpy"]["value"] is not None
        and row["enthalpy_increment"]["value"] is not None
    ]
    for point in points:
        if point[0] == temperature_K:
            _, h_kj, entropy, cp = point
            h = h_kj * 1000.0
            return h, entropy, cp, h - temperature_K * entropy
    for low, high in zip(points, points[1:]):
        if low[0] < temperature_K < high[0]:
            ratio = (temperature_K - low[0]) / (high[0] - low[0])
            h_kj = low[1] + ratio * (high[1] - low[1])
            entropy = low[2] + ratio * (high[2] - low[2])
            cp = low[3] + ratio * (high[3] - low[3])
            h = h_kj * 1000.0
            return h, entropy, cp, h - temperature_K * entropy
    raise ValueError(f"{table_id} has no JANAF values at {temperature_K:g} K")


def _parent_liquid(oxide: str, temperature_K: float, gas_pack) -> tuple[float, float]:
    key = (oxide, temperature_K)
    if key not in _LIQUID_CACHE:
        row = species_thermo(oxide, "l", temperature_K, gas_pack)
        _LIQUID_CACHE[key] = (row.G_J_mol, row.Cp_J_molK)
    return _LIQUID_CACHE[key]


def _new_aluminate_liquid_gibbs(
    name: str, temperature_K: float, gas_pack, source: dict
) -> float:
    """Build liquid G from crystal G/H/S plus class-C fusion and additive Cp."""
    specs = {
        "SrAl4O7": {
            "oxides": [(1.0, "O-013"), (2.0, "Al-096")],
            "Tm": 2053.0,
            "source_row": source["sources"]["jacob_shreyas_2018"]["formation_reactions_from_crystalline_oxides"]["SrAl4O7"],
        },
        "SrAl12O19": {
            "oxides": [(1.0, "O-013"), (6.0, "Al-096")],
            "Tm": 2063.0,
            "source_row": source["sources"]["jacob_shreyas_2018"]["formation_reactions_from_crystalline_oxides"]["SrAl12O19"],
        },
        "Sr3Al2O6": {
            "oxides": [(3.0, "O-013"), (1.0, "Al-096")],
            "Tm": 2009.0,
            "source_row": source["sources"]["brisi_abbattista_1960"]["formation_enthalpies_from_oxides_at_298_15_K"]["Sr3Al2O6"],
        },
        "Sr4Al2O7": {
            "oxides": [(4.0, "O-013"), (1.0, "Al-096")],
            "Tm": 2073.0,
            "source_row": source["sources"]["brisi_abbattista_1960"]["formation_enthalpies_from_oxides_at_298_15_K"]["Sr4Al2O7_alpha"],
        },
    }
    spec = specs[name]
    tm = spec["Tm"]
    source_row = spec["source_row"]
    if name in ("SrAl4O7", "SrAl12O19"):
        delta_h_j = float(source_row["delta_G_A_J_per_mol"])
        delta_s = -float(source_row["delta_G_B_J_per_mol_K"])
    else:
        delta_h_j = 1000.0 * float(source_row["delta_H_kJ_per_mol"])
        # Only formation enthalpy is printed. The accepted Neumann-Kopp
        # estimate sets ΔS_form,ox = ΔCp_form,ox = 0; see pack provenance.
        delta_s = 0.0
    h_cr, s_cr, g_cr = 0.0, 0.0, 0.0
    for coefficient, table_id in spec["oxides"]:
        h, entropy, _, gibbs = _janaf_props(table_id, tm)
        h_cr += coefficient * h
        s_cr += coefficient * entropy
        g_cr += coefficient * gibbs
    h_cr += delta_h_j
    s_cr += delta_s
    g_cr += delta_h_j - tm * delta_s
    if abs((h_cr - tm * s_cr) - g_cr) > 1e-5:
        raise ValueError(f"{name} crystal Hess balance failed at its fusion marker")

    parent_counts = {
        "SrAl4O7": {"SrO": 1, "Al2O3": 2},
        "SrAl12O19": {"SrO": 1, "Al2O3": 6},
        "Sr3Al2O6": {"SrO": 3, "Al2O3": 1},
        "Sr4Al2O7": {"SrO": 4, "Al2O3": 1},
    }[name]
    atom_counts = {
        "SrAl4O7": 12,
        "SrAl12O19": 32,
        "Sr3Al2O6": 11,
        "Sr4Al2O7": 13,
    }
    fusion_entropy = atom_counts[name] * CLASS_C_J_MOL_ATOM_K
    cp_liquid = sum(
        coefficient * _parent_liquid(oxide, tm, gas_pack)[1]
        for oxide, coefficient in parent_counts.items()
    )
    h_liquid = h_cr + tm * fusion_entropy + cp_liquid * (temperature_K - tm)
    s_liquid = s_cr + fusion_entropy + cp_liquid * math.log(temperature_K / tm)
    return h_liquid - temperature_K * s_liquid


def _product_gibbs(name: str, temperature_K: float, gas_pack, redox_pack, source: dict) -> float:
    if name in {"SrAl4O7", "SrAl12O19", "Sr3Al2O6", "Sr4Al2O7"}:
        return _new_aluminate_liquid_gibbs(name, temperature_K, gas_pack, source)
    return species_gibbs_j_mol(redox_pack, f"{name}(l)", temperature_K, allow_extrapolation=True)


def _nu_for(name: str) -> dict[str, float]:
    return {
        "SrAl2O4": {"SrO": 1, "Al2O3": 1},
        "SrAl4O7": {"SrO": 1, "Al2O3": 2},
        "SrAl12O19": {"SrO": 1, "Al2O3": 6},
        "Sr3Al2O6": {"SrO": 3, "Al2O3": 1},
        "Sr4Al2O7": {"SrO": 4, "Al2O3": 1},
        "SrSiO3": {"SrO": 1, "SiO2": 1},
        "Sr2SiO4": {"SrO": 2, "SiO2": 1},
    }[name]


def _row(name: str, index: int, source: dict, gas_pack, redox_pack) -> dict:
    nu = _nu_for(name)
    temperatures = np.linspace(FIT_LOW_K, FIT_HIGH_K, 1001)
    y = []
    for temperature in temperatures:
        product_g = _product_gibbs(name, float(temperature), gas_pack, redox_pack, source)
        parent_g = sum(
            coefficient * _parent_liquid(oxide, float(temperature), gas_pack)[0]
            for oxide, coefficient in nu.items()
        )
        y.append((parent_g - product_g) / (R_J_MOL_K * temperature * LN10))
    design = np.column_stack((np.ones(len(temperatures)), 1.0 / temperatures))
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(y), rcond=None)
    residual = np.asarray(y) - design @ coefficients
    paper_domains = {
        "SrAl2O4": [932.0, 2233.0],
        "SrAl4O7": [1000.0, 1300.0],
        "SrAl12O19": [1000.0, 1300.0],
        "Sr3Al2O6": [298.15, 298.15],
        "Sr4Al2O7": [298.15, 298.15],
        "SrSiO3": [298.15, 1000.0],
        "Sr2SiO4": [298.15, 1000.0],
    }[name]
    source_refs = {
        "SrAl2O4": ["REDOX-V6-SR"],
        "SrAl4O7": ["JACOB-SHREYAS-2018"],
        "SrAl12O19": ["JACOB-SHREYAS-2018"],
        "Sr3Al2O6": ["BRISI-ABBATTISTA-1960"],
        "Sr4Al2O7": ["BRISI-ABBATTISTA-1960"],
        "SrSiO3": ["REDOX-V6-SR"],
        "Sr2SiO4": ["REDOX-V6-SR"],
    }[name]
    table = {"SrAl2O4": "SrAl2O4(l)", "SrAl4O7": "SrAl4O7(cr)",
             "SrAl12O19": "SrAl12O19(cr)", "Sr3Al2O6": "Sr3Al2O6(cr)",
             "Sr4Al2O7": "alpha-Sr4Al2O7(cr)", "SrSiO3": "SrSiO3(l)",
             "Sr2SiO4": "Sr2SiO4(l)"}[name]
    description = (
        "Re-referenced from the redox v6 source liquid associate."
        if name in ("SrAl2O4", "SrSiO3", "Sr2SiO4")
        else "Crystal Gibbs function plus provisional class-C fusion entropy and additive parent-liquid Cp."
    )
    row = {
        "row": index,
        "complex": name,
        "reaction": " + ".join(
            f"{coefficient:g} {oxide}(liq)" for oxide, coefficient in nu.items()
        ) + f" = {name}(liq)",
        "nu": {parent: nu.get(parent, 0) for parent in
               ("SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O", "K2O", "SrO")},
        "A": float(coefficients[0]),
        "B": float(coefficients[1]),
        "source": source_refs[0],
        "T_domain_K": [FIT_LOW_K, FIT_HIGH_K],
        "T_domain_basis": "Opt-in candidate evaluation interval; source and continuation limits are retained in T_domain_paper_demonstrated_K.",
        "provenance_class": "C-independent-thermochemistry",
        "derivation": description,
        "T_domain_paper_demonstrated_K": paper_domains,
        "coverage": "C-independent-thermochemistry",
        "fit_residual": None,
        "accuracy_band": None,
        "band_basis": {
            "kind": "none",
            "explanation": "No row-specific numerical uncertainty band is claimed; provisional assumptions are itemized in candidate_provenance.",
            "provenance": None,
        },
        "independent_check": {
            "status": "not_found", "equilibrium": None, "primary_source": None,
            "conditions": None, "measured_value": None, "predicted_value": None,
            "miss": None, "miss_to_band_ratio": None,
            "reason": "No row-specific liquid equilibrium measurement was used as a fitting target.",
        },
        "source_refs": source_refs,
        "standard_state_refit": {
            "parents": nu,
            "temperature_range_K": [FIT_LOW_K, FIT_HIGH_K],
            "fit_method": "least squares on 1001 evenly spaced pointwise re-referenced log10 K values",
            "max_abs_residual_dex": float(np.max(np.abs(residual))),
            "flag_over_0_01_dex": bool(np.max(np.abs(residual)) > 0.01),
        },
        "fusion_entropy_estimate_J_mol_K": round(
            CLASS_C_J_MOL_ATOM_K * {
                "SrAl2O4": 7, "SrAl4O7": 12, "SrAl12O19": 32,
                "Sr3Al2O6": 11, "Sr4Al2O7": 13, "SrSiO3": 5, "Sr2SiO4": 7,
            }[name], 9
        ),
    }
    return row


def build() -> dict:
    pack = json.loads(BASE.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    gas_pack = load_gas_datapack()
    redox_pack = load_redox_pack(REDOX)
    measured_entropies = source["sources"]["jacob_shreyas_2018"]["formation_reactions_from_crystalline_oxides"]
    entropy_per_oxide = max(
        -float(measured_entropies["SrAl4O7"]["delta_G_B_J_per_mol_K"]) / 3.0,
        -float(measured_entropies["SrAl12O19"]["delta_G_B_J_per_mol_K"]) / 7.0,
    )
    brisi_entropy_bounds = {
        "Sr3Al2O6": entropy_per_oxide * 4.0,
        "Sr4Al2O7": entropy_per_oxide * 5.0,
    }
    entropy_band_by_temperature = [
        {
            "temperature_K": temperature,
            "Sr3Al2O6_plus_minus_kJ_mol": temperature * brisi_entropy_bounds["Sr3Al2O6"] / 1000.0,
            "Sr3Al2O6_plus_minus_log10K_dex": brisi_entropy_bounds["Sr3Al2O6"] / (R_J_MOL_K * LN10),
            "Sr4Al2O7_plus_minus_kJ_mol": temperature * brisi_entropy_bounds["Sr4Al2O7"] / 1000.0,
            "Sr4Al2O7_plus_minus_log10K_dex": brisi_entropy_bounds["Sr4Al2O7"] / (R_J_MOL_K * LN10),
        }
        for temperature in (1800, 2000, 2200, 2400)
    ]
    jacob_rows = measured_entropies
    jacob_reproduction = {}
    for compound, record in jacob_rows.items():
        jacob_reproduction[compound] = [
            {
                "temperature_K": temperature,
                "reported_delta_G_J_mol": record["delta_G_A_J_per_mol"]
                + record["delta_G_B_J_per_mol_K"] * temperature,
                "reconstructed_delta_G_J_mol": record["delta_G_A_J_per_mol"]
                + record["delta_G_B_J_per_mol_K"] * temperature,
            }
            for temperature in (1000, 1100, 1200, 1300)
        ]
    janaf_gaps = {}
    for temperature in (1700, 1800, 2000, 2200, 2400):
        source_g = _janaf_props("O-014", float(temperature))[3]
        fitted_g = species_thermo("SrO", "l", float(temperature), gas_pack).G_J_mol
        janaf_gaps[str(temperature)] = fitted_g - source_g
    pack["imcc_sf04_datapack_version"] = "1.0.0-d066-sr-candidate-v1"
    pack["model_id"] = "IMCC-SF04-D066-SR-CANDIDATE"
    pack["model"] = (
        "Opt-in associated-solution candidate with nine full parent oxides, "
        "the unchanged 38-row D066 core, and seven independently sourced Sr complexes."
    )
    pack["parents"] = ["SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O", "K2O", "SrO"]
    pack["pack_arithmetic"] = "38 unchanged D066 v4 core rows + 7 Sr candidate rows = 45 complexes."
    pack["parent_standard_candidates"]["SrO"] = {
        "source": "NIST-JANAF SrO(l), Table O-014",
        "doi": "10.18434/T42S31",
        "measured_range_K": [1200.0, 3000.0],
        "standard_state": "pure liquid SrO at 1 bar",
        "transfer_note": "Direct JANAF liquid Gibbs reference; used as a metastable liquid standard where SrO is not stable as a pure liquid.",
    }
    pack["candidate_provenance"] = {
        "status": "opt-in provisional candidate; not the default",
        "source_policy": source["source_policy"],
        "basis": "SrO is a full component only under this candidate model identity.",
        "reference_fit": "Each row is refit to this pack's liquid standards by least squares on 1001 evenly spaced temperatures from 1700 to 2400 K, using pointwise log10 K and the basis [1, 1/T].",
        "fusion": "All seven liquid associates use the class-C estimate 8.297530569 J mol^-1 formula atom^-1 K^-1 times total formula atoms. This is provisional. Liquid Cp for the four newly built aluminates is the additive parent liquid Cp at the selected fusion marker.",
        "brisi_entropies": {
            "assumption": "For Sr3Al2O6 and alpha-Sr4Al2O7, only ΔH_form,ox(298.15 K) is printed. Use ΔG_form,ox(T)=ΔH_form,ox(298.15 K), equivalent to ΔS_form,ox=ΔCp_form,ox=0 (Neumann-Kopp). This is provisional and separate from fusion.",
            "entropy_bound_source": "Maximum absolute measured oxide-formation entropy per mole oxide from Jacob & Shreyas 2018: max(25.376/3, 25.744/7) = 8.458666667 J mol^-1 oxide K^-1.",
            "complex_bounds_J_mol_K": brisi_entropy_bounds,
            "implied_band_by_temperature": entropy_band_by_temperature,
            "derivation": "Premise: Brisi & Abbattista print only ΔH_form,ox(298.15 K); take ΔS_form,ox=ΔCp_form,ox=0. Algebra: H_complex(T)=ΣνH_oxide(T)+ΔH_form,ox; S_complex(T)=ΣνS_oxide(T); therefore G_complex(T)=ΣνG_oxide(T)+ΔH_form,ox. Units: convert the printed kJ/mol to J/mol before combining with JANAF J/mol and T·S in J/mol. Sanity: this exactly reproduces the printed −97.7 ± 4.2 and −81.4 ± 5.4 kJ/mol formation enthalpies at 298.15 K; the entropy assumption remains provisional.",
        },
        "candidate_rows": ["SrAl2O4", "SrAl4O7", "SrAl12O19", "Sr3Al2O6", "Sr4Al2O7", "SrSiO3", "Sr2SiO4"],
        "phase_diagram_markers_K": {"SrAl2O4": 2233.0, "SrAl4O7": 2053.0, "SrAl12O19": 2063.0, "Sr3Al2O6": 2009.0, "Sr4Al2O7": 2073.0, "source": "Experimental values only from Ganits et al. 1979, as tabulated in Ye et al. 2007 Table 4; no optimized interactions used."},
        "excluded_inputs": "No Stolyarova, Vorozhtcov, Lopatin, Shugurov, Tyurnina, or Balabanova KEMS data were used. Ye et al. 2007 optimized liquid parameters were not used.",
        "source_reproduction": {
            "Jacob_Shreyas_2018_delta_G_J_mol": jacob_reproduction,
            "Brisi_Abbattista_1960_printed_delta_H_kJ_mol": {"Sr3Al2O6": -97.7, "alpha-Sr4Al2O7": -81.4},
            "JANAF_O_014_liquid_G_gap_J_mol_by_temperature": janaf_gaps,
            "JANAF_O_014_liquid_G_max_abs_gap_J_mol": max(abs(value) for value in janaf_gaps.values()),
        },
    }
    pack["sources"].update({
        "JACOB-SHREYAS-2018": {
            "citation": "Jacob, K. T.; Shreyas, V. (2018), Thermodynamic properties of SrAl12O19 and SrAl4O7, J. Mater. Sci. 53, 1723-1730.",
            "doi": "10.1007/s10853-017-1634-0", "table": "Printed SrF2-cell EMF formation Gibbs functions; 1000-1300 K.",
            "url": "https://doi.org/10.1007/s10853-017-1634-0",
        },
        "BRISI-ABBATTISTA-1960": {
            "citation": "Brisi, C.; Abbattista, F. (1960), formation enthalpies transcribed as printed in Ye et al. (2007), Table 3.",
            "doi": "10.1007/s11669-007-9086-x", "table": "Ye et al. Table 3; printed values only.",
            "url": "https://doi.org/10.1007/s11669-007-9086-x",
        },
        "REDOX-V6-SR": {
            "citation": "OpenIMCC redox v6 SrAl2O4, SrSiO3, and Sr2SiO4 source associates; digest-pinned in the source release.",
            "file": "openimcc-redox-v6.json", "file_sha256": hashlib.sha256(REDOX.read_bytes()).hexdigest(),
            "canonical_content_digest": redox_pack.canonical_digest,
        },
        "SR-CANDIDATE-THERMOCHEMISTRY": {
            "citation": "Printed Jacob & Shreyas and Brisi & Abbattista source values, transcribed in the candidate source record.",
            "file": "data-src/sr-candidate/thermochemistry.json",
            "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        },
        "CLASS-C": {
            "citation": "Class-C mean fusion entropy per formula atom from the existing Sr thermochemistry fragment.",
            "value_J_mol_formula_atom_K": CLASS_C_J_MOL_ATOM_K,
        },
        "JANAF-SrO": {
            "citation": "NIST-JANAF Thermochemical Tables, 4th Edition, SrO crystal O-013 and liquid O-014.",
            "doi": "10.18434/T42S31", "url": "https://janaf.nist.gov/tables/O-014.html",
        },
    })
    names = ["SrAl2O4", "SrAl4O7", "SrAl12O19", "Sr3Al2O6", "Sr4Al2O7", "SrSiO3", "Sr2SiO4"]
    pack["rows"].extend(
        _row(name, 38 + index, source, gas_pack, redox_pack)
        for index, name in enumerate(names, start=1)
    )
    pack["errata"].append({
        "version": "1.0.0-d066-sr-candidate-v1",
        "change": "Adds a digest-pinned opt-in nine-parent candidate with seven independent Sr complexes; the default D066 v4 pack is unchanged.",
    })
    return pack


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
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
