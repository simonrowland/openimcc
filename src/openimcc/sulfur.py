"""Opt-in sulfur solubility and sulfide-liquid saturation estimates.

This layer does not change the default melt or gas calculations. Capacities
are reported on the wt% S scale used by Boulliung and Wood; fugacities are
base-10 relative to the package's 1-bar gas standard.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from openimcc.model import (
    ImccLoadedDatapack,
    evaluate,
    load_datapack,
)

_R = 8.31446261815324
_LN10 = math.log(10.0)
_MW_S = 32.065

# Forward notices by their source family and score metadata, never by prose.
_MELT_NOTICE_FORWARD_CODES = {
    ("excess-gibbs-candidate", "scored"): "excess-gibbs-candidate",
    ("excess-gibbs-calibration-range", "scored"): "excess-gibbs-calibration-range",
    ("alkali-bias", "scored"): "alkali-bias",
    ("alkali-bias", "unscored"): "alkali-bias-unscored",
    ("na2o-activity-bias", "scored"): "na2o-activity-bias",
    ("na2o-activity-bias", "unscored"): "na2o-activity-bias-unscored",
    ("cao-sio2-activity-bias", "scored"): "cao-sio2-activity-bias",
    ("cao-sio2-activity-bias", "unscored"): "cao-sio2-activity-bias-unscored",
    ("multicomponent-activity-pattern", "scored"): "multicomponent-activity-pattern",
    ("multicomponent-activity-pattern", "unscored"): (
        "multicomponent-activity-pattern-unscored"
    ),
}


def _melt_notice_forward_code(notice: str) -> str | None:
    return _MELT_NOTICE_FORWARD_CODES.get(
        (getattr(notice, "family", None), getattr(notice, "score", None))
    )

# formula-unit molecular mass, cations per formula, oxygen atoms per formula,
# and the cation symbol used in single-cation mole fractions.
_OXIDES: dict[str, tuple[float, int, int, str]] = {
    "SiO2": (60.0843, 1, 2, "Si"),
    "TiO2": (79.866, 1, 2, "Ti"),
    "Al2O3": (101.961, 2, 3, "Al"),
    "FeO": (71.844, 1, 1, "Fe"),
    "FeOeq": (71.844, 1, 1, "Fe"),
    "MnO": (70.9374, 1, 1, "Mn"),
    "MgO": (40.3044, 1, 1, "Mg"),
    "CaO": (56.0774, 1, 1, "Ca"),
    "Na2O": (61.979, 2, 1, "Na"),
    "K2O": (94.196, 2, 1, "K"),
    "P2O5": (141.9445, 2, 5, "P"),
    "Cr2O3": (151.9904, 2, 3, "Cr"),
    "NiO": (74.6928, 1, 1, "Ni"),
    "CoO": (74.9326, 1, 1, "Co"),
}


@dataclass(frozen=True)
class SulfurNotice:
    """A typed model-domain, assumption, or limitation notice."""

    code: str
    message: str
    model: str | None = None
    axis: str | None = None
    value: float | None = None
    calibrated_interval: tuple[float, float] | None = None
    distance_beyond_interval: float | None = None
    distance_unit: str | None = None


@dataclass(frozen=True)
class SulfurComparisonBand:
    """Separate O'Neill and Mavrogenes comparison predictions."""

    sulfide_capacity_log10_ppm: float | None
    sulfide_s2_ppm: float | None
    sulfate_s6_ppm: float | None


@dataclass(frozen=True)
class SulfurResult:
    """Sulfur result; fields are ``None`` when an input or model is absent."""

    temperature_K: float
    total_pressure_bar: float
    log10_fO2: float | None
    log10_fS2: float | None
    oxygen_source: str
    log10_sulfide_capacity_wt_percent: float | None
    log10_sulfate_capacity_wt_percent: float | None
    capacity_predicted_total_s_ppm: float | None
    dissolved_s2_minus_ppm: float | None
    dissolved_s6_plus_ppm: float | None
    dissolved_total_s_ppm: float | None
    dissolved_total_s_mol_per_mol_melt: float | None
    scss_smythe_2017_ppm: float | None
    scss_oneill_2021_ppm: float | None
    sulfide_liquid_x_fes: float | None
    saturation_state: str | None
    excess_s_ppm: float | None
    excess_s_mol_per_mol_melt: float | None
    comparison_band: SulfurComparisonBand
    notices: tuple[SulfurNotice, ...]


def _notice(
    code: str,
    message: str,
    *,
    model: str | None = None,
    axis: str | None = None,
    value: float | None = None,
    interval: tuple[float, float] | None = None,
    distance: float | None = None,
    distance_unit: str | None = None,
) -> SulfurNotice:
    return SulfurNotice(
        code, message, model, axis, value, interval, distance, distance_unit
    )

def _normalise_composition(
    composition_wt_percent: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float], float]:
    amounts: dict[str, float] = {}
    masses: dict[str, float] = {}
    for raw_name, raw_value in composition_wt_percent.items():
        if raw_name not in _OXIDES:
            raise ValueError(f"unsupported oxide component {raw_name!r}")
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid wt% for {raw_name}") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"wt% for {raw_name} must be finite and nonnegative")
        name = "FeO" if raw_name == "FeOeq" else raw_name
        mass, _cations, _oxygen, _symbol = _OXIDES[raw_name]
        masses[name] = masses.get(name, 0.0) + value
        amounts[name] = amounts.get(name, 0.0) + value / mass
    total_mass = math.fsum(masses.values())
    total_moles = math.fsum(amounts.values())
    if total_mass <= 0.0 or total_moles <= 0.0:
        raise ValueError("composition must contain positive total oxide wt%")
    formula_moles = {name: value / total_moles for name, value in amounts.items()}
    oxide_wt = {name: value / total_mass * 100.0 for name, value in masses.items()}
    mean_molar_mass = total_mass / total_moles
    return formula_moles, oxide_wt, mean_molar_mass


def _fractions(
    formula_moles: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    oxygen_units = math.fsum(
        n * _OXIDES[name][2] for name, n in formula_moles.items()
    )
    cation_total = math.fsum(
        n * _OXIDES[name][1] for name, n in formula_moles.items()
    )
    one_oxygen: dict[str, float] = {}
    cations: dict[str, float] = {}
    for name, n in formula_moles.items():
        _mw, count, oxygen, symbol = _OXIDES[name]
        one_oxygen[name] = one_oxygen.get(name, 0.0) + n * oxygen / oxygen_units
        cations[symbol] = cations.get(symbol, 0.0) + n * count / cation_total
    return one_oxygen, cations


def _log10_sum(log_values: tuple[float, ...]) -> float:
    largest = max(log_values)
    if largest == -math.inf:
        return -math.inf
    return largest + math.log10(math.fsum(10.0 ** (value - largest) for value in log_values))


def _capacity_logs(
    one_oxygen: Mapping[str, float],
    cations: Mapping[str, float],
    T_K: float,
) -> tuple[float, float]:
    # Boulliung & Wood (2023), Eq. 7. X values are single-oxygen-basis oxide
    # fractions; the published concentration unit is wt% S.
    log_cs2 = 0.225 + (
        25237.0 * one_oxygen.get("FeO", 0.0)
        + 5214.0 * one_oxygen.get("CaO", 0.0)
        + 12705.0 * one_oxygen.get("MnO", 0.0)
        + 19829.0 * one_oxygen.get("K2O", 0.0)
        - 1109.0 * one_oxygen.get("SiO2", 0.0)
        - 8879.0
    ) / T_K
    # Boulliung & Wood (2022), Eq. 5. X values are single-cation fractions.
    log_cs6 = -12.659 + (
        3692.0 * cations.get("Ca", 0.0)
        - 7592.0 * cations.get("Si", 0.0)
        - 13736.0 * cations.get("Ti", 0.0)
        + 3762.0 * cations.get("Al", 0.0)
        + 34483.0
    ) / T_K
    return log_cs2, log_cs6


def _sulfur_split(
    log_cs2: float,
    log_cs6: float,
    log_fO2: float,
    log_fS2: float,
) -> tuple[float, float, float]:
    # Fincham–Richardson capacity definitions give log(S6/S2) =
    # log(C6/C2) + 2 log(fO2). Split in log space to avoid overflow.
    log_s2_wt = log_cs2 + 0.5 * (log_fS2 - log_fO2)
    log_s6_wt = log_cs6 + 0.5 * log_fS2 + 1.5 * log_fO2
    total_log_ppm = _log10_sum((log_s2_wt + 4.0, log_s6_wt + 4.0))
    total_ppm = 10.0**total_log_ppm
    log_ratio = log_cs6 - log_cs2 + 2.0 * log_fO2
    if log_ratio > 300.0:
        fraction_s6 = 1.0
    elif log_ratio < -300.0:
        fraction_s6 = 0.0
    else:
        fraction_s6 = 1.0 / (1.0 + 10.0 ** (-log_ratio))
    return total_ppm * (1.0 - fraction_s6), total_ppm * fraction_s6, total_ppm


def _log_fso3(
    log_fO2: float,
    log_fS2: float,
    T_K: float,
    gas_datapack: object,
) -> float:
    from openimcc.gas import species_thermo

    g_s2 = species_thermo("S2", "g", T_K, gas_datapack).G_J_mol
    g_o2 = species_thermo("O2", "g", T_K, gas_datapack).G_J_mol
    g_so3 = species_thermo("SO3", "g", T_K, gas_datapack).G_J_mol
    delta_g = 2.0 * g_so3 - g_s2 - 3.0 * g_o2
    log_k = -delta_g / (_R * T_K * _LN10)
    # S2 + 3 O2 = 2 SO3, so each SO3 fugacity has one half the reaction
    # quotient exponent. All gas fugacities use the same 1-bar standard.
    return 0.5 * (log_k + log_fS2 + 3.0 * log_fO2)


def _scss_smythe(
    cations: Mapping[str, float],
    T_K: float,
    pressure_bar: float,
    x_fes: float,
    ideal_a_feo: float,
) -> float:
    # Smythe et al. (2017), Eq. 11 and Table 2, ideal ΔG° column. The
    # composition fractions are single-cation fractions; P is in GPa and
    # SCSS is in ppm S. The Fe-Si cross term is the printed XSi*XFe term.
    p_gpa = pressure_bar * 1.0e-4
    ln_scss = (
        -14683.0 / T_K
        + 8.03
        - 265.80 * p_gpa / T_K
        + (
            16430.0 * cations.get("Ti", 0.0)
            + 9295.0 * cations.get("Al", 0.0)
            + 13767.0 * cations.get("Mg", 0.0)
            + 19893.0 * cations.get("Ca", 0.0)
            - 7080.0 * cations.get("Fe", 0.0)
            + 14197.0 * cations.get("Na", 0.0)
            + 10189.0 * cations.get("H", 0.0)
            + 117827.0 * cations.get("Si", 0.0) * cations.get("Fe", 0.0)
        ) / T_K
        + math.log(x_fes)
        - math.log(ideal_a_feo)
    )
    return math.exp(ln_scss)


def _melt_activities(
    formula_moles: Mapping[str, float],
    T_K: float,
    pack: ImccLoadedDatapack | object | None,
) -> tuple[object | None, float | None, str | None]:
    active_pack = load_datapack() if pack is None else pack
    parents = set(active_pack.parent_oxides)
    composition = {
        name: n for name, n in formula_moles.items()
        if name in parents and n > 0.0
    }
    try:
        result = evaluate(
            composition,
            T_K,
            active_pack,
            basis_type="mol",
            allow_extrapolation=True,
            allow_out_of_envelope=True,
        )
        return result, result.activity("FeO"), None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _gas_activity_map(melt_result: object | None, log_fS2: float | None) -> dict[str, float]:
    activities: dict[str, float] = {}
    if melt_result is not None:
        for name in melt_result.parent_oxides:
            value = melt_result.activity(name)
            if value > 0.0:
                activities[name] = value
    if log_fS2 is not None:
        activities["S2"] = 10.0**log_fS2
    return activities


def _out_of_range(
    notices: list[SulfurNotice],
    *,
    code: str,
    model: str,
    axis: str,
    value: float,
    lower: float,
    upper: float,
    unit: str,
) -> None:
    if value < lower:
        distance = lower - value
    elif value > upper:
        distance = value - upper
    else:
        return
    notices.append(_notice(
        code,
        f"{model} is outside its {axis} calibration interval",
        model=model,
        axis=axis,
        value=value,
        interval=(lower, upper),
        distance=distance,
        distance_unit=unit,
    ))


def evaluate_sulfur(
    composition_wt_percent: Mapping[str, float],
    T_K: float,
    *,
    log10_fO2: float | None = None,
    log10_fS2: float | None = None,
    total_s_ppm: float | None = None,
    total_s_mol_per_mol_melt: float | None = None,
    total_pressure_bar: float = 1.0,
    melt_pack: ImccLoadedDatapack | object | None = None,
    gas_datapack: object | None = None,
    fe3_fraction: float | None = None,
    sulfide_liquid_composition: Mapping[str, float] | None = None,
) -> SulfurResult:
    """Estimate dissolved sulfur and FeS-matte saturation for one melt.

    Exactly one of ``log10_fS2``, ``total_s_ppm`` and
    ``total_s_mol_per_mol_melt`` is required for sulfur numbers. If it is not
    supplied (or more than one is supplied), the result carries a typed notice
    and leaves sulfur outputs empty. ``log10_fO2=None`` invokes openimcc's
    existing intrinsic gas oxygen-balance root. ``log10_fS2=None`` in a
    total-sulfur call analytically inverts the two capacity equations. The
    SCSS cap is then applied to dissolved S2- only.

    ``composition_wt_percent`` gives oxide weight percentages. Sulfur ppm is
    relative to melt mass; mol/mol is sulfur atoms per mole of oxide formula
    units. ``sulfide_liquid_composition`` gives Fe, Ni, Cu, S, and optional P
    and C mole fractions; it must sum to one. Smythe's FeS activity is
    approximated as Fe/(Fe+Ni+Cu), and metal-rich liquids are flagged.
    """
    T = float(T_K)
    pressure = float(total_pressure_bar)
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("total pressure must be finite and positive")
    for name, value in (("log10_fO2", log10_fO2), ("log10_fS2", log10_fS2)):
        if value is not None and (not math.isfinite(float(value)) or abs(float(value)) > 300):
            raise ValueError(f"{name} must be finite and within [-300, 300]")
    if fe3_fraction is not None and (
        not math.isfinite(float(fe3_fraction)) or not 0.0 <= float(fe3_fraction) <= 1.0
    ):
        raise ValueError("fe3_fraction must be between 0 and 1")

    formula_moles, oxide_wt, mean_molar_mass = _normalise_composition(
        composition_wt_percent
    )
    one_oxygen, cations = _fractions(formula_moles)
    log_cs2, log_cs6 = _capacity_logs(one_oxygen, cations, T)
    notices: list[SulfurNotice] = []
    _out_of_range(
        notices, code="capacity-temperature-outside-calibration",
        model="Boulliung-Wood-2023/2022", axis="temperature_K", value=T,
        lower=1473.0, upper=1773.0, unit="K",
    )
    for model, bounds in (
        ("Boulliung-Wood-2023-Eq-7", {
            "SiO2": (39.9, 61.6), "FeO": (0.0, 16.9), "CaO": (4.1, 22.3),
            "TiO2": (0.0, 8.34), "Na2O": (0.13, 7.77),
        }),
        ("Boulliung-Wood-2022-Eq-5", {
            "SiO2": (39.9, 73.2), "FeO": (0.0, 16.9), "CaO": (1.0, 22.3),
            "TiO2": (0.0, 8.34), "Na2O": (0.0, 7.77),
        }),
    ):
        for oxide, (lo, hi) in bounds.items():
            _out_of_range(
                notices, code="capacity-composition-outside-calibration",
                model=model, axis=f"{oxide}_wt_percent", value=oxide_wt.get(oxide, 0.0),
                lower=lo, upper=hi, unit="wt%",
            )
    for oxide in ("P2O5", "Cr2O3", "NiO", "CoO"):
        value = oxide_wt.get(oxide, 0.0)
        if value > 0.0:
            notices.append(_notice(
                "capacity-composition-component-outside-calibration",
                f"{oxide} is not represented in the Boulliung-Wood capacity calibrations",
                model="Boulliung-Wood-2023/2022", axis=f"{oxide}_wt_percent",
                value=value, interval=(0.0, 0.0), distance=value,
                distance_unit="wt%",
            ))
    _out_of_range(
        notices, code="scss-temperature-outside-calibration",
        model="Smythe-2017-Eq-11", axis="temperature_K", value=T,
        lower=1423.0, upper=2433.0, unit="K",
    )
    _out_of_range(
        notices, code="scss-composition-outside-calibration",
        model="Smythe-2017-Eq-11", axis="FeO_wt_percent",
        value=oxide_wt.get("FeO", 0.0), lower=0.3, upper=40.1, unit="wt%",
    )
    if pressure < 1.0:
        decades = math.log10(1.0 / pressure)
        for model in ("Boulliung-Wood-capacities", "Smythe-2017-SCSS"):
            notices.append(_notice(
                "pressure-below-calibration", "model calibrated at atmospheric pressure",
                model=model, axis="total_pressure_bar", value=pressure,
                interval=(1.0, 1.0), distance=decades, distance_unit="decades below 1 bar",
            ))
    elif pressure > 1.0:
        notices.append(_notice(
            "capacity-pressure-outside-calibration",
            "Boulliung-Wood capacities were calibrated at atmospheric pressure",
            model="Boulliung-Wood-capacities", axis="total_pressure_bar", value=pressure,
            interval=(1.0, 1.0), distance=math.log10(pressure),
            distance_unit="decades above 1 bar",
        ))
    if pressure > 240000.0:
        notices.append(_notice(
            "scss-pressure-outside-calibration", "Smythe SCSS data extend to 24 GPa",
            model="Smythe-2017-Eq-11", axis="total_pressure_bar", value=pressure,
            interval=(1.0, 240000.0), distance=pressure - 240000.0,
            distance_unit="bar above 24 GPa",
        ))

    active_sulfide_liquid = (
        {"Fe": 0.5, "S": 0.5}
        if sulfide_liquid_composition is None
        else {str(k): float(v) for k, v in sulfide_liquid_composition.items()}
    )
    if sulfide_liquid_composition is None:
        notices.append(_notice(
            "default-pure-FeS-matte", "sulfide-liquid composition defaults to stoichiometric FeS",
            model="Smythe-2017-Eq-11",
        ))
    allowed_sulfide_liquid = {"Fe", "Ni", "Cu", "S", "P", "C"}
    if not active_sulfide_liquid or set(active_sulfide_liquid) - allowed_sulfide_liquid:
        raise ValueError("sulfide-liquid composition must use Fe, Ni, Cu, S, and optionally P and C")
    if any(not math.isfinite(v) or v < 0.0 for v in active_sulfide_liquid.values()):
        raise ValueError("sulfide-liquid mole fractions must be finite and nonnegative")
    liquid_total = math.fsum(active_sulfide_liquid.values())
    if not math.isclose(liquid_total, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("sulfide-liquid mole fractions must sum to one")
    metal_total = math.fsum(active_sulfide_liquid.get(name, 0.0) for name in ("Fe", "Ni", "Cu"))
    x_fes = active_sulfide_liquid.get("Fe", 0.0) / metal_total if metal_total > 0.0 else 0.0
    stoichiometric_s = active_sulfide_liquid.get("Fe", 0.0) + active_sulfide_liquid.get("Ni", 0.0) + 0.5 * active_sulfide_liquid.get("Cu", 0.0)
    if stoichiometric_s > 0.0:
        s_to_stoichiometric_matte = active_sulfide_liquid.get("S", 0.0) / stoichiometric_s
        if s_to_stoichiometric_matte < 1.0:
            notices.append(_notice(
                "metal-rich-sulfide-liquid-outside-smythe-calibration",
                "sulfur-poor metal-rich liquid is outside the FeS-NiS-CuS0.5 matte calibration; aFeS still uses Fe/(Fe+Ni+Cu)",
                model="Smythe-2017-Eq-11", axis="matte_S_to_stoichiometric_sulfide_ratio",
                value=s_to_stoichiometric_matte, interval=(1.0, 1.0),
                distance=1.0 - s_to_stoichiometric_matte,
                distance_unit="ratio below stoichiometric matte",
            ))
    for element in ("P", "C"):
        value = active_sulfide_liquid.get(element, 0.0)
        if value > 0.0:
            notices.append(_notice(
                "sulfide-liquid-component-outside-smythe-calibration",
                f"{element} is outside the FeS-NiS-CuS0.5 matte calibration and is omitted from aFeS",
                model="Smythe-2017-Eq-11", axis=f"matte_{element}_mole_fraction",
                value=value, interval=(0.0, 0.0), distance=value,
                distance_unit="mole fraction",
            ))
    if x_fes <= 0.0:
        notices.append(_notice(
            "sulfide-liquid-has-no-Fe", "Smythe SCSS requires a positive FeS component",
            model="Smythe-2017-Eq-11",
        ))
    elif any(active_sulfide_liquid.get(name, 0.0) > 0.0 for name in ("Ni", "Cu")):
        notices.append(_notice(
            "ideal-matte-activity", "non-ideal matte interactions are not included; aFeS is XFeS",
            model="Smythe-2017-Eq-11",
        ))

    melt_result, a_feo, melt_error = _melt_activities(formula_moles, T, melt_pack)
    if melt_error is not None:
        notices.append(_notice(
            "melt-activity-unavailable",
            "openimcc could not evaluate FeO activity: " + melt_error,
            model="openimcc-melt-pack", axis="FeO_activity",
        ))
    elif melt_result is not None:
        for melt_notice in melt_result.labels.notices:
            code = _melt_notice_forward_code(melt_notice)
            if code is not None:
                notices.append(_notice(
                    code, melt_notice, model="openimcc-melt-pack",
                ))
    supported_moles = set(melt_result.parent_oxides) if melt_result is not None else set()
    omitted = sorted(name for name, n in formula_moles.items() if n > 0.0 and name not in supported_moles)
    if omitted:
        notices.append(_notice(
            "oxide-omitted-from-melt-pack",
            "the supplied melt pack does not represent: " + ", ".join(omitted),
            model="openimcc-melt-pack", axis="oxide_composition",
        ))

    scss = None
    oneill2021_scss = None
    ideal_a_feo = cations.get("Fe", 0.0)
    if x_fes > 0.0 and ideal_a_feo > 0.0:
        # Smythe et al. (2017), Eq. 11/Table 2, uses its ideal single-cation
        # FeO activity assumption in the regression. The openimcc activity is
        # reserved for the independent O'Neill thermodynamic cross-check.
        scss = _scss_smythe(cations, T, pressure, x_fes, ideal_a_feo)
        if melt_pack is not None and a_feo is not None and a_feo > 0.0:
            # O'Neill (2021), thermodynamic SCSS relation. The capacity is the
            # central B&W23 model; the tabulated 1-bar FeO-FeS term follows
            # Smythe et al. (2017), Eq. 12. The caller's openimcc pack supplies
            # aFeO, and the ideal matte supplies aFeS = XFeS.
            c2_ppm = 10.0 ** (log_cs2 + 4.0)
            ln_delta_g_over_rt = 14695.0 / T - 9.656 + 1.02 * math.log(T)
            oneill2021_scss = math.exp(
                ln_delta_g_over_rt + math.log(c2_ppm)
                + math.log(x_fes) - math.log(a_feo)
                - 269.40 * (pressure * 1.0e-4) / T
            )
    else:
        notices.append(_notice(
            "scss-unavailable", "SCSS requires positive FeO mole fraction and FeS in the matte",
            model="Smythe-2017-Eq-11", axis="aFeO_or_XFeS",
        ))

    supplied_s = sum(v is not None for v in (
        log10_fS2, total_s_ppm, total_s_mol_per_mol_melt
    ))
    log_fs2 = None if log10_fS2 is None else float(log10_fS2)
    if total_s_ppm is not None and total_s_mol_per_mol_melt is not None:
        notices.append(_notice(
            "sulfur-input-conflict", "supply total sulfur in only one unit",
            axis="sulfur_input",
        ))
        supplied_s = 2
    elif supplied_s == 0:
        notices.append(_notice(
            "sulfur-input-missing", "supply log10_fS2 or one total sulfur value; no S numbers returned",
            axis="sulfur_input",
        ))
    elif supplied_s > 1:
        notices.append(_notice(
            "sulfur-input-conflict", "supply exactly one of log10_fS2 or total sulfur",
            axis="sulfur_input",
        ))

    input_total_ppm: float | None = None
    if total_s_ppm is not None:
        input_total_ppm = float(total_s_ppm)
    elif total_s_mol_per_mol_melt is not None:
        input_mol = float(total_s_mol_per_mol_melt)
        if not math.isfinite(input_mol) or input_mol < 0.0:
            raise ValueError("total_s_mol_per_mol_melt must be finite and nonnegative")
        input_total_ppm = input_mol * _MW_S / mean_molar_mass * 1.0e6
    if input_total_ppm is not None and (
        not math.isfinite(input_total_ppm) or input_total_ppm < 0.0
    ):
        raise ValueError("total sulfur must be finite and nonnegative")

    oxygen_source = "caller-supplied"
    if log10_fO2 is None:
        oxygen_source = "openimcc-intrinsic-oxygen-balance"
    log_fo2 = None if log10_fO2 is None else float(log10_fO2)
    sulfur_fields_available = supplied_s == 1
    if log_fo2 is None:
        try:
            from openimcc.gas import (
                evaluate_gas_oxygen_balance,
                load_gas_datapack,
            )

            gas_pack = load_gas_datapack() if gas_datapack is None else gas_datapack
            if log_fo2 is None and log_fs2 is not None:
                activities = _gas_activity_map(melt_result, log_fs2)
                p_o2, _gas_result, _diagnostics = evaluate_gas_oxygen_balance(
                    activities, T, gas_pack, allow_extrapolation=True,
                )
                log_fo2 = math.log10(p_o2)
            elif log_fo2 is None and sulfur_fields_available:
                # For a total-S input, invert fS2 at each openimcc oxygen root.
                # The outer scalar root enforces the supplied melt inventory;
                # each inner evaluation is the existing gas oxygen-balance path.
                target_ppm = input_total_ppm or 0.0
                target_wt = target_ppm / 1.0e4
                if target_wt == 0.0:
                    log_fs2 = -300.0
                    activities = _gas_activity_map(melt_result, None)
                    p_o2, _gas_result, _diagnostics = evaluate_gas_oxygen_balance(
                        activities, T, gas_pack, allow_extrapolation=True,
                    )
                    log_fo2 = math.log10(p_o2)
                else:
                    from scipy.optimize import brentq

                    def at_log_fs2(candidate: float) -> tuple[float, float]:
                        activities = _gas_activity_map(melt_result, candidate)
                        p_o2, _gas_result, _diagnostics = evaluate_gas_oxygen_balance(
                            activities, T, gas_pack, allow_extrapolation=True,
                        )
                        candidate_fo2 = math.log10(p_o2)
                        l2 = log_cs2 + 0.5 * (candidate - candidate_fo2)
                        l6 = log_cs6 + 0.5 * candidate + 1.5 * candidate_fo2
                        candidate_ppm = 10.0 ** (_log10_sum((l2 + 4.0, l6 + 4.0)))
                        return candidate_ppm - target_ppm, candidate_fo2

                    nodes = (-50.0, -35.0, -25.0, -15.0, -10.0, -5.0, 0.0, 5.0)
                    prior_x = nodes[0]
                    prior_y, _ = at_log_fs2(prior_x)
                    bracket = None
                    for candidate in nodes[1:]:
                        value, _ = at_log_fs2(candidate)
                        if prior_y == 0.0 or prior_y * value < 0.0:
                            bracket = (prior_x, candidate)
                            break
                        prior_x, prior_y = candidate, value
                    if bracket is None:
                        raise ValueError("total sulfur is not bracketed by the oxygen-balance model")
                    log_fs2 = brentq(lambda x: at_log_fs2(x)[0], *bracket, xtol=1.0e-8)
                    _residual, log_fo2 = at_log_fs2(log_fs2)
                    if scss is not None:
                        # The all-dissolved root is valid only below S_max.
                        # Once it is above that limit, saturation fixes fS2 as
                        # a function of fO2. Re-solve the gas oxygen balance
                        # on that relation so the returned fugacity pair
                        # satisfies both equations. The capacity ratio and
                        # therefore S_max still depend on the resulting fO2.
                        log_ratio = log_cs6 - log_cs2 + 2.0 * log_fo2
                        log_scss = math.log10(scss)
                        max_total_ppm = 10.0 ** _log10_sum((
                            log_scss, log_scss + log_ratio,
                        ))
                        if target_ppm > max_total_ppm:
                            def oxygen_residual_at_saturation(
                                candidate_fo2: float,
                            ) -> float:
                                candidate_fs2 = candidate_fo2 + 2.0 * (
                                    log_scss - 4.0 - log_cs2
                                )
                                activities = _gas_activity_map(
                                    melt_result, candidate_fs2,
                                )
                                p_o2, _gas_result, _diagnostics = (
                                    evaluate_gas_oxygen_balance(
                                        activities, T, gas_pack,
                                        allow_extrapolation=True,
                                    )
                                )
                                return candidate_fo2 - math.log10(p_o2)

                            prior_x = nodes[0]
                            prior_y = oxygen_residual_at_saturation(prior_x)
                            bracket = None
                            for candidate in nodes[1:]:
                                value = oxygen_residual_at_saturation(candidate)
                                if prior_y == 0.0 or prior_y * value < 0.0:
                                    bracket = (prior_x, candidate)
                                    break
                                prior_x, prior_y = candidate, value
                            if bracket is None:
                                raise ValueError(
                                    "saturated sulfur fugacity has no bracketed "
                                    "oxygen-balance root"
                                )
                            log_fo2 = brentq(
                                oxygen_residual_at_saturation,
                                *bracket,
                                xtol=1.0e-10,
                            )
                            log_fs2 = log_fo2 + 2.0 * (
                                log_scss - 4.0 - log_cs2
                            )
            elif log_fo2 is None:
                # Even when sulfur input is absent or conflicting, honor the
                # optional-fO2 contract and expose the intrinsic oxygen root.
                activities = _gas_activity_map(melt_result, None)
                p_o2, _gas_result, _diagnostics = evaluate_gas_oxygen_balance(
                    activities, T, gas_pack, allow_extrapolation=True,
                )
                log_fo2 = math.log10(p_o2)
        except Exception as exc:
            notices.append(_notice(
                "oxygen-balance-unavailable",
                "openimcc could not determine the requested gas state: "
                + f"{type(exc).__name__}: {exc}",
                model="openimcc-gas-oxygen-balance", axis="log10_fO2",
            ))
            if log10_fO2 is None:
                log_fo2 = None

    if log_fo2 is None and log10_fO2 is not None:
        log_fo2 = float(log10_fO2)

    if log_fo2 is not None and log_fs2 is None and sulfur_fields_available:
        # The two capacity concentrations are both proportional to sqrt(fS2).
        # Sum their coefficients, then solve the total-S equation in closed form.
        target_ppm = input_total_ppm or 0.0
        if target_ppm == 0.0:
            log_fs2 = -300.0
        else:
            log_a2 = log_cs2 - 0.5 * log_fo2
            log_a6 = log_cs6 + 1.5 * log_fo2
            log_a = _log10_sum((log_a2, log_a6))
            log_fs2 = 2.0 * (math.log10(target_ppm / 1.0e4) - log_a)

    dissolved_s2 = dissolved_s6 = predicted_total = dissolved_total = None
    capacity_predicted_s2 = capacity_predicted_s6 = None
    dissolved_mol = excess_ppm = excess_s_mol = None
    saturation_state = None
    if sulfur_fields_available and log_fo2 is not None and log_fs2 is not None:
        capacity_predicted_s2, capacity_predicted_s6, predicted_total = _sulfur_split(
            log_cs2, log_cs6, log_fo2, log_fs2,
        )
        dissolved_s2, dissolved_s6 = capacity_predicted_s2, capacity_predicted_s6
        if input_total_ppm is not None:
            if scss is not None:
                # The capacity ratio r = S6+/S2- = (C6/C2) fO2^2 is
                # independent of fS2. At any fugacity total dissolved S is
                # S2-(1+r), so SCSS sets the maximum dissolved inventory to
                # S_max = SCSS(1+r). Above S_max, set fS2 to the value that
                # gives S2-=SCSS; this also gives S6+=r*SCSS and makes the
                # returned fugacity regenerate both reported species.
                log_ratio = log_cs6 - log_cs2 + 2.0 * log_fo2
                log_scss = math.log10(scss)
                max_total_ppm = 10.0 ** _log10_sum((
                    log_scss, log_scss + log_ratio,
                ))
                if input_total_ppm > max_total_ppm:
                    log_fs2 = log_fo2 + 2.0 * (log_scss - 4.0 - log_cs2)
                    capacity_predicted_s2, capacity_predicted_s6, predicted_total = _sulfur_split(
                        log_cs2, log_cs6, log_fo2, log_fs2,
                    )
                    dissolved_s2, dissolved_s6 = capacity_predicted_s2, capacity_predicted_s6
                    excess_ppm = input_total_ppm - max_total_ppm
                    saturation_state = "saturated"
                else:
                    excess_ppm = 0.0
                    saturation_state = "undersaturated"
            # Without an SCSS estimate, the capacity inversion is the only
            # available inventory model and no saturation/excess is inferred.
        elif scss is not None:
            # A caller-supplied fS2 is a capacity prediction rather than a
            # closed inventory. SCSS caps only its predicted S2-; any S2-
            # capacity above that limit is a sulfide-liquid overrun.
            if dissolved_s2 >= scss:
                excess_ppm = dissolved_s2 - scss
                dissolved_s2 = scss
                saturation_state = "saturated"
            else:
                excess_ppm = 0.0
                saturation_state = "undersaturated"
        dissolved_total = dissolved_s2 + dissolved_s6
        dissolved_mol = dissolved_total * 1.0e-6 * mean_molar_mass / _MW_S
        excess_s_mol = (excess_ppm or 0.0) * 1.0e-6 * mean_molar_mass / _MW_S
        if excess_ppm and excess_ppm > 0.0:
            notices.append(_notice(
                "sulfur-excess-reported-to-sulfide-liquid",
                "sulfur above the S2- SCSS-limited dissolved inventory is assigned to the caller's sulfide-liquid composition",
                model="Smythe-2017-SCSS", axis="excess_s_ppm", value=excess_ppm,
            ))
        sulfate_dominant = dissolved_s6 > dissolved_s2
        if scss is not None and sulfate_dominant:
            notices.append(_notice(
                "sulfate-saturation-and-so2-degassing-not-modelled",
                "dissolved S6+ exceeds S2-; sulfate saturation (including anhydrite) and SO2 degassing are not modelled",
                model="Boulliung-Wood-2022-Eq-5", axis="dissolved_S6_to_S2_ratio",
                value=dissolved_s6 / dissolved_s2 if dissolved_s2 > 0.0 else math.inf,
            ))
        if (
            log10_fS2 is not None and input_total_ppm is None
            and excess_ppm and excess_ppm > 0.0
        ):
            notices.append(_notice(
                "matte-excess-is-capacity-overrun",
                "with caller-supplied fS2, sulfide excess is the S2- capacity prediction above SCSS, not a bulk-inventory balance",
                model="Boulliung-Wood-2023-Eq-7-plus-Smythe-2017-SCSS",
            ))
        for model, value, maximum in (
            ("Boulliung-Wood-2023-Eq-7", capacity_predicted_s2, 3500.0),
            ("Boulliung-Wood-2022-Eq-5", capacity_predicted_s6, 4090.0),
        ):
            if value is not None and value > maximum:
                notices.append(_notice(
                    "capacity-prediction-above-calibrated-maximum",
                    f"unconstrained capacity-predicted dissolved sulfur exceeds the largest reported {model} calibration value; SCSS is applied separately",
                    model=model, axis="capacity_predicted_S_ppm", value=value,
                    interval=(0.0, maximum), distance=value - maximum, distance_unit="ppm",
                ))

    comparison_sulfide_capacity = None
    comparison_s2 = None
    comparison_s6 = None
    if log_fo2 is not None and log_fs2 is not None:
        # O'Neill & Mavrogenes (2002) sulfide-capacity form at 1400 C. The
        # published basis is cation fractions and ppm S; Smythe et al. (2017),
        # Eq. 12/Table 2 supplies the printed A_M values. Do not extrapolate
        # this one-temperature comparison model to other process temperatures.
        if abs(T - 1673.0) <= 0.5:
            coeff = {
                "Si": -27561.0, "Ti": -11220.0, "Al": -18450.0,
                "Mg": -13970.0, "Ca": -7831.0, "Fe": -34274.0,
                "Na": -13247.0, "K": -29015.0, "H": -17495.0,
            }
            ln_c = 9.087 + math.fsum(coeff[k] * cations.get(k, 0.0) for k in coeff) / T
            ln_c += 116568.0 * cations.get("Si", 0.0) * cations.get("Fe", 0.0) / T
            comparison_sulfide_capacity = ln_c / _LN10
            comparison_s2 = math.exp(ln_c + 0.5 * _LN10 * (log_fs2 - log_fo2))
        else:
            notices.append(_notice(
                "om2002-temperature-outside-calibration",
                "the O'Neill and Mavrogenes sulfide comparison is limited to 1400 C",
                model="ONeill-Mavrogenes-2002", axis="temperature_K", value=T,
                interval=(1673.0, 1673.0), distance=abs(T - 1673.0), distance_unit="K",
            ))
        _out_of_range(
            notices, code="om2022-temperature-outside-calibration",
            model="ONeill-Mavrogenes-2022", axis="temperature_K", value=T,
            lower=1473.0, upper=1773.0, unit="K",
        )
        try:
            from openimcc.gas import load_gas_datapack

            gas_pack = load_gas_datapack() if gas_datapack is None else gas_datapack
            log_so3 = _log_fso3(log_fo2, log_fs2, T, gas_pack)
            cations_om = dict(cations)
            fe2_fraction = 1.0 if fe3_fraction is None else 1.0 - float(fe3_fraction)
            cations_om["Fe"] = cations_om.get("Fe", 0.0) * fe2_fraction
            if fe3_fraction is None:
                notices.append(_notice(
                    "om2022-fe3-assumed-zero",
                    "O'Neill and Mavrogenes (2022) sulfate capacity assumes all iron is Fe2+ unless fe3_fraction is supplied",
                    model="ONeill-Mavrogenes-2022", axis="Fe3_fraction", value=0.0,
                    interval=(0.0, 1.0),
                ))
            ln_c6 = -8.02 + (
                21100.0 + 44000.0 * cations_om.get("Na", 0.0)
                + 18700.0 * cations_om.get("Mg", 0.0)
                + 4300.0 * cations_om.get("Al", 0.0)
                + 44200.0 * cations_om.get("K", 0.0)
                + 35600.0 * cations_om.get("Ca", 0.0)
                + 12600.0 * cations_om.get("Mn", 0.0)
                + 16500.0 * cations_om.get("Fe", 0.0)
            ) / T
            comparison_s6 = math.exp(ln_c6 + _LN10 * log_so3)
        except Exception as exc:
            notices.append(_notice(
                "om2022-comparison-unavailable",
                "could not calculate the O'Neill and Mavrogenes sulfate comparison: "
                + f"{type(exc).__name__}: {exc}",
                model="ONeill-Mavrogenes-2022", axis="fSO3",
            ))

    if fe3_fraction is not None:
        notices.append(_notice(
            "fe3-fraction-only-affects-comparison",
            "fe3_fraction is used only by the O'Neill and Mavrogenes (2022) sulfate comparison",
            model="ONeill-Mavrogenes-2022", axis="Fe3_fraction", value=float(fe3_fraction),
        ))
    notices.extend((
        _notice(
            "metal-silicate-partition-not-modelled",
            "metal-silicate sulfur partitioning is not modelled; sulfide-saturated "
            "excess is assigned to the caller's Fe-Ni-S-P-C liquid without calculating its composition",
        ),
        _notice("cos-h2s-not-in-gas-tables", "COS and H2S gas species are not in the openimcc gas tables"),
    ))
    if log10_fS2 is not None:
        notices.append(_notice(
            "reduced-gas-fugacity-is-caller-input",
            "reduced-gas coupling is represented only through caller-supplied fS2",
        ))

    return SulfurResult(
        T, pressure, log_fo2, log_fs2, oxygen_source,
        log_cs2 if sulfur_fields_available else None,
        log_cs6 if sulfur_fields_available else None,
        predicted_total,
        dissolved_s2, dissolved_s6, dissolved_total, dissolved_mol,
        scss if sulfur_fields_available else None,
        oneill2021_scss if sulfur_fields_available else None,
        x_fes if sulfur_fields_available else None,
        saturation_state,
        excess_ppm, excess_s_mol,
        SulfurComparisonBand(
            comparison_sulfide_capacity if sulfur_fields_available else None,
            comparison_s2 if sulfur_fields_available else None,
            comparison_s6 if sulfur_fields_available else None,
        ),
        tuple(notices),
    )
