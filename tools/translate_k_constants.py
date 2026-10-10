"""Build the d-066 SF04 K-complex pack against its selected K2O(l) reference."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from openimcc.gas import (
    R_J_MOL_K,
    ImccGasDatapack,
    load_gas_datapack,
    species_thermo,
)
from openimcc.model import (
    _D066_CARRIED_PUBLISHED_ROWS,
    _D066_E25_ROWS,
    _D066_REFERENCE_UNKNOWN_ROWS,
    _D066_ROW_COVERAGE,
    _D066_TRANSLATED_ROWS,
)


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "src" / "openimcc" / "data" / "packs"
PUBLISHED_SOURCE_PACK = PACK_DIR / "imcc-sf04-v1.0.2.json"
EXTENSION_SOURCE_PACK = PACK_DIR / "imcc-sf04-ext-v4.json"
LEGACY_CONDENSATE = PACK_DIR / "sf04-published" / "condensate.csv"
D066_MODEL_ID = "IMCC-SF04-D066"
PRIMARY_VERSION = "1.0.3-d066-v1"
SENSITIVITY_VERSION = "1.0.3-d066-v1-kcaalsi2o7"
EXTENSION_VERSION = "1.0.3-d066-ext-v1"
FIT_POINTS = 10_001
K2O_BAND_DEX_PER_K = 0.4


def fc87_k2o_gibbs_j_mol(T_K: float, gas_pack: ImccGasDatapack) -> float:
    """Return the FC87 Table 1 K2O(l) reference in J/mol."""
    g_k = species_thermo("K", "g", T_K, gas_pack).G_J_mol
    g_o = species_thermo("O", "g", T_K, gas_pack).G_J_mol
    return 2.0 * g_k + g_o + R_J_MOL_K * T_K * math.log(10.0) * (
        15.33 - 36735.0 / T_K
    )


def _fit_reference_shift(
    row: Mapping[str, Any],
    new_gas_pack: ImccGasDatapack,
    reference_pack: ImccGasDatapack,
    reference: str,
) -> dict[str, float]:
    low, high = (float(value) for value in row["T_domain_K"])
    temperatures = np.linspace(low, high, FIT_POINTS, dtype=float)
    inverse_temperature = 1.0 / temperatures
    if reference == "FC87":
        reference_g = np.array(
            [fc87_k2o_gibbs_j_mol(float(T), reference_pack) for T in temperatures]
        )
    elif reference == "LAM":
        reference_g = np.array(
            [
                species_thermo("K2O", "l", float(T), reference_pack).G_J_mol
                for T in temperatures
            ]
        )
    else:
        raise ValueError(f"unsupported K2O(l) reference {reference!r}")

    new_g = np.array(
        [
            species_thermo("K2O", "l", float(T), new_gas_pack).G_J_mol
            for T in temperatures
        ]
    )
    nu_k2o = float(row["nu"]["K2O"])

    # Premise: the absolute complex G stays fixed while its K2O parent standard
    # state changes by DeltaG_ref. Then DeltaG_rxn,new = DeltaG_rxn,old -
    # nu_K2O*DeltaG_ref. Since DeltaG_rxn = -R*T*ln(K), this gives
    # log10(K_new) = log10(K_old) + nu_K2O*DeltaG_ref/(R*T*ln(10)).
    # Unit check: J/mol divided by R*T (J/mol) is dimensionless. Sanity: the
    # packaged new reference is below FC87 over this domain, so DeltaG_ref<0
    # and the translated log10(K) must decrease.
    shift_dex = nu_k2o * (new_g - reference_g) / (
        R_J_MOL_K * temperatures * math.log(10.0)
    )
    design = np.column_stack((np.ones_like(inverse_temperature), inverse_temperature))
    delta_a, delta_b = np.linalg.lstsq(design, shift_dex, rcond=None)[0]
    residual = shift_dex - (delta_a + delta_b * inverse_temperature)
    return {
        "delta_A": float(delta_a),
        "delta_B_K": float(delta_b),
        "max_abs_residual_dex": float(np.max(np.abs(residual))),
        "residual_at_T_min_dex": float(residual[0]),
        "residual_at_T_max_dex": float(residual[-1]),
        "max_abs_complex_G_error_kJ_mol": float(
            np.max(np.abs(residual) * R_J_MOL_K * temperatures * math.log(10.0))
            / 1000.0
        ),
        "fit_error_fraction_of_shift_at_T_min": float(
            abs(residual[0] / shift_dex[0])
        ),
    }


def translate_k_rows(
    source: Mapping[str, Any],
    gas_pack: ImccGasDatapack,
    lam_gas_pack: ImccGasDatapack,
    *,
    arm: str = "primary",
) -> dict[str, Any]:
    """Build one D066 arm from a published or extension source pack."""
    if arm not in {"primary", "sensitivity"}:
        raise ValueError(f"unsupported K-complex arm {arm!r}")
    pack = deepcopy(source)
    has_sp_extension = "sp_extension" in pack
    if has_sp_extension and arm != "primary":
        raise ValueError("the D066 extension pack has only the primary E25 arm")
    citation = pack["sources"]["SF04"]["citation"]
    citation_title, doi_separator, _old_doi = citation.rpartition(" DOI ")
    if not doi_separator:
        raise ValueError("SF04 citation must include a DOI")
    pack["sources"]["SF04"]["citation"] = (
        f"{citation_title} DOI 10.1016/j.icarus.2003.08.023"
    )
    pack["model_id"] = D066_MODEL_ID
    pack["imcc_sf04_datapack_version"] = (
        EXTENSION_VERSION if has_sp_extension else
        SENSITIVITY_VERSION if arm == "sensitivity" else
        PRIMARY_VERSION
    )
    pack["created"] = "2026-10-09"
    pack["errata"] = list(pack.get("errata", [])) + [
        {
            "version": pack["imcc_sf04_datapack_version"],
            "date": "2026-10-09",
            "change": (
                "d-066: four FC87 K rows translated to packaged JANAF K-012 "
                "+ fusion K2O(l) with absolute complex G held fixed; three "
                "unknown-reference rows remain active and flagged; 30 rows "
                "carry the published coefficients; KCaAlSi2O7 follows E25. "
                + (
                    "The ext-v4 S/P extension is carried unchanged."
                    if has_sp_extension else
                    ""
                )
            ),
        }
    ]

    row_gas_pack = gas_pack
    fit_cache: dict[tuple[object, ...], dict[str, float]] = {}

    def reference_fit(row: Mapping[str, Any], reference: str) -> dict[str, float]:
        reference_pack = row_gas_pack if reference == "FC87" else lam_gas_pack
        key = (
            tuple(row["T_domain_K"]),
            float(row["nu"]["K2O"]),
            reference,
            gas_pack.condensate_table_digest,
            reference_pack.condensate_table_digest,
        )
        if key not in fit_cache:
            fit_cache[key] = _fit_reference_shift(
                row, row_gas_pack, reference_pack, reference
            )
        return fit_cache[key]

    for row in pack["rows"]:
        name = row["complex"]
        row["coverage"] = _D066_ROW_COVERAGE[name]
        row["d066_source_row"] = {
            "pack": "imcc-sf04-v1.0.2.json",
            "row": row["row"],
        }
        if name in _D066_TRANSLATED_ROWS:
            fit = reference_fit(row, "FC87")
            old_a = float(row["A"])
            old_b = float(row["B"])
            row["A"] = old_a + fit["delta_A"]
            row["B"] = old_b + fit["delta_B_K"]
            row["k_translation"] = {
                "basis": "FC87 Table 1 p.208 (Glushko/TSIV K2O(l))",
                "new_reference": "packaged JANAF K-012 K2O(l) species_thermo row",
                "operation": "least-squares fit in 1/T over T_domain_K on a 10001-point grid",
                "preserved_quantity": "absolute complex G; no calorimetry re-solve",
                "delta_A": fit["delta_A"],
                "delta_B_K": fit["delta_B_K"],
                "max_abs_residual_dex": fit["max_abs_residual_dex"],
                "residual_at_T_min_dex": fit["residual_at_T_min_dex"],
                "residual_at_T_max_dex": fit["residual_at_T_max_dex"],
                "max_abs_complex_G_error_kJ_mol": fit[
                    "max_abs_complex_G_error_kJ_mol"
                ],
                "fit_error_fraction_of_shift_at_T_min": fit[
                    "fit_error_fraction_of_shift_at_T_min"
                ],
                "tolerance_dex": 2.0 * float(row["nu"]["K2O"]) * K2O_BAND_DEX_PER_K,
                "tolerance_source": (
                    "E24 K2O(l) source-uncertainty budget, +/-0.4 dex per K atom; "
                    "not a fit-precision claim"
                ),
            }
            if name in {"KAlSiO4", "KAlSi2O6"}:
                row["enthalpy_flag"] = "not_closable"
            if name == "KAlSi3O8":
                row["calorimetry_re_solve_candidate"] = (
                    "Not selected: d-066 translates the FC87 row and holds absolute complex G fixed."
                )
        elif name in _D066_REFERENCE_UNKNOWN_ROWS:
            flags = list(row.get("flags", []))
            if "reference_unknown" not in flags:
                flags.append("reference_unknown")
            row["flags"] = flags
            candidates: dict[str, Any] = {"provenance_only": True}
            for basis in ("FC87", "LAM"):
                fit = reference_fit(row, basis)
                candidates[basis] = {
                    "A": float(row["A"]) + fit["delta_A"],
                    "B": float(row["B"]) + fit["delta_B_K"],
                    "max_abs_residual_dex": fit["max_abs_residual_dex"],
                }
            row["translation_candidates"] = candidates
        elif name in _D066_E25_ROWS:
            flags = list(row.get("flags", []))
            if arm == "primary":
                row["active"] = False
                row["inactive_reason"] = "E25: out of liquid domain; owner-confirmed"
                flags = [
                    flag for flag in flags
                    if flag != "kcaalsi2o7_out_of_liquid_domain"
                ]
            else:
                row["active"] = True
                row.pop("inactive_reason", None)
                if "kcaalsi2o7_out_of_liquid_domain" not in flags:
                    flags.append("kcaalsi2o7_out_of_liquid_domain")
            if flags:
                row["flags"] = flags
            else:
                row.pop("flags", None)
        elif name not in _D066_CARRIED_PUBLISHED_ROWS:
            raise ValueError(f"unclassified D066 row {name!r}")

    return pack


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "primary"
    if variant not in {"primary", "sensitivity", "ext"}:
        raise SystemExit("usage: translate_k_constants.py [primary|sensitivity|ext]")
    source_path = (
        EXTENSION_SOURCE_PACK if variant == "ext" else PUBLISHED_SOURCE_PACK
    )
    arm = "primary" if variant == "ext" else variant
    source = json.loads(source_path.read_text(encoding="utf-8"))
    new_gas_pack = load_gas_datapack()
    lam_gas_pack = load_gas_datapack(oxide_path=LEGACY_CONDENSATE)
    pack = translate_k_rows(source, new_gas_pack, lam_gas_pack, arm=arm)
    print(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", end="")


if __name__ == "__main__":
    main()
