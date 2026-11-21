"""Full apparent thermochemistry for the shipped crystal source datasets."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from importlib import resources
import json
import math
import re
from typing import Any, Mapping, Sequence

from openimcc._constants import R_PUBLISHED_GAS_J_MOL_K
from openimcc.diagnostics import _crystal_source_cap_excludes_candidate
from openimcc.janaf_crystals import (
    _janaf_nodes_for_temperature,
    _load_crystal_species as _load_janaf_crystal_species,
    _load_phases as _load_janaf_phases,
    evaluate_janaf_crystal,
)


_FORMULA_TOKEN = re.compile(r"[A-Z][a-z]?|\(|\)|\d+(?:\.\d+)?")
_REFERENCE_POLYMORPHS = frozenset({"ref", "reference", "reference state"})
ELEMENTAL_REFERENCE_CARD_SPECIES = frozenset(
    {"Au", "Pd", "Pt", "Rh", "Ir", "Os", "Ru"}
)
_POLYMORPH_ALIASES = {
    "MgO": {"periclase": "Mg-008"},
    "TiO2": {"rutile": "O-043"},
    "Ti3O5": {"beta": "O-081"},
    "FeS": {"troilite": "Fe-023"},
}

# Berman's thermodynamic functions use a 298.15 K reference, while
# valid_temperature_K on these diagnostic rows remains the fusion-node domain
# for declared-assemblage calculations. The high-temperature polymorph floors
# use Berman (1988), Table 3b and its tridymite transition note; the
# pseudowollastonite floor is the printed 1398 K transition in Robie et al.
# (1978), Bulletin 1452, p. 373.
_BERMAN_PUBLIC_SERVICE_LOWER_K = {
    "forsterite": 298.15,
    "enstatite": 298.15,
    "diopside": 298.15,
    "anorthite": 298.15,
    "spinel": 298.15,
    "akermanite": 298.15,
    "gehlenite": 298.15,
    "merwinite": 298.15,
    "monticellite": 298.15,
    "wollastonite": 298.15,
    "quartz_beta": 848.0,
    "tridymite_high": 432.15,
    "cristobalite_beta": 535.0,
    "pseudowollastonite": 1398.0,
}


def is_reference_polymorph(polymorph: str | None) -> bool:
    return (
        isinstance(polymorph, str)
        and polymorph.strip().casefold() in _REFERENCE_POLYMORPHS
    )


def is_element_formula(formula: str) -> bool:
    return re.fullmatch(r"[A-Z][a-z]?", formula) is not None


@dataclass(frozen=True)
class CrystalSpeciesThermo:
    Cp_J_molK: float
    S_J_molK: float
    H_app_kJ_mol: float
    G_J_mol: float
    source_row_id: str
    source_table_id: str
    T_min: float
    T_max: float
    authority: str
    phase_identity: Mapping[str, Any]
    formation_enthalpy_298_kJ_mol: float | None
    formation_enthalpy_source_id: str | None
    extrapolated: bool = False
    continuation: str | None = None


class CrystalSpeciesTemperatureError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        declared_interval_K: tuple[float, float],
        source_row_id: str,
        available_polymorphs: Sequence[str],
    ) -> None:
        self.declared_interval_K = declared_interval_K
        self.source_row_id = source_row_id
        self.available_polymorphs = tuple(available_polymorphs)
        super().__init__(message)


class CrystalSpeciesPolymorphError(ValueError):
    def __init__(self, message: str, available_polymorphs: Sequence[str]) -> None:
        self.available_polymorphs = tuple(available_polymorphs)
        super().__init__(message)


@dataclass(frozen=True)
class _Candidate:
    dataset: str
    label: str
    phase: Mapping[str, Any]
    polymorph: Mapping[str, Any] | None
    interval_K: tuple[float, float]
    source_row_id: str
    source_table_id: str
    authority: str


def _read_json(resource_name: str) -> dict[str, Any]:
    raw = resources.files("openimcc").joinpath(resource_name).read_bytes()
    return json.loads(raw.decode("utf-8"))


@lru_cache(maxsize=1)
def _berman_phases() -> tuple[dict[str, Any], ...]:
    return tuple(_read_json("data/diagnostics/crystal-phases.json")["phases"])


@lru_cache(maxsize=1)
def _rh_crystal_table_phases() -> tuple[dict[str, Any], ...]:
    return tuple(
        _read_json("data/diagnostics/rh-crystal-species.json")["phases"]
    )


@lru_cache(maxsize=1)
def _barin_phases() -> tuple[dict[str, Any], ...]:
    return tuple(
        _read_json("data/diagnostics/barin-aluminate-crystals.json")["phases"]
    )


@lru_cache(maxsize=1)
def _elemental_reference_card_phases() -> tuple[dict[str, Any], ...]:
    return tuple(
        _read_json(
            "data/diagnostics/elemental-reference-crystal-cards.json"
        )["records"]
    )


def _formula_unit_element_counts(formula: str) -> dict[str, Fraction] | None:
    """Parse one formula unit without correcting OCR forms."""
    tokens = _FORMULA_TOKEN.findall(formula)
    if not tokens or "".join(tokens) != formula:
        return None

    groups: list[dict[str, Fraction]] = [{}]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "(":
            groups.append({})
            index += 1
            continue
        if token == ")":
            if len(groups) == 1 or not groups[-1]:
                return None
            nested = groups.pop()
            index += 1
            multiplier = Fraction(1)
            if index < len(tokens) and tokens[index][0].isdigit():
                count = tokens[index]
                if count.startswith("0") and "." not in count:
                    return None
                multiplier = Fraction(count)
                if multiplier <= 0:
                    return None
                index += 1
            for element, count in nested.items():
                groups[-1][element] = (
                    groups[-1].get(element, Fraction(0)) + count * multiplier
                )
            continue
        if not token[0].isupper():
            return None

        index += 1
        count = Fraction(1)
        if index < len(tokens) and tokens[index][0].isdigit():
            count_token = tokens[index]
            if count_token.startswith("0") and "." not in count_token:
                return None
            count = Fraction(count_token)
            if count <= 0:
                return None
            index += 1
        groups[-1][token] = groups[-1].get(token, Fraction(0)) + count

    if len(groups) != 1:
        return None
    return groups[0] or None


def _formula_element_counts(formula: str) -> dict[str, Fraction] | None:
    """Parse formula units, including common dot-separated hydrate notation."""
    def component_counts(component: str) -> dict[str, Fraction] | None:
        multiplier = Fraction(1)
        prefix = re.match(r"^(\d+)(?=[A-Z(])", component)
        if prefix:
            multiplier = Fraction(prefix.group(1))
            component = component[prefix.end():]
        counts = _formula_unit_element_counts(component)
        if counts is None or multiplier <= 0:
            return None
        return {element: count * multiplier for element, count in counts.items()}

    sections = []
    for explicit_section in re.split(r"[·*]", formula):
        dotted = re.split(r"\.(?=[1-9]\d*[A-Z(])", explicit_section)
        if len(dotted) > 1:
            distinct = []
            for index, section in enumerate(dotted):
                unit = re.sub(r"^\d+(?=[A-Z(])", "", section) if index else section
                distinct.append(set(re.findall(r"[A-Z][a-z]?", unit)))
            if all(len(elements) >= 2 for elements in distinct):
                sections.extend(dotted)
                continue
        sections.append(explicit_section)

    total: dict[str, Fraction] = {}
    for index, section in enumerate(sections):
        counts = (
            component_counts(section)
            if index > 0
            else _formula_unit_element_counts(section)
        )
        if counts is None:
            return None
        for element, count in counts.items():
            total[element] = total.get(element, Fraction(0)) + count
    return total or None


def _linear_value_and_slope(
    nodes: Sequence[Mapping[str, Any]], temperature_K: float
) -> tuple[float, float]:
    """Linearly interpolate a source series and return its selected secant slope."""
    by_temperature = {
        float(node["T_K"]): float(node["value"])
        for node in nodes
    }
    points = sorted(by_temperature.items())
    temperatures = [point[0] for point in points]
    index = bisect_left(temperatures, temperature_K)
    if index < len(points) and points[index][0] == temperature_K:
        value = points[index][1]
        if index > 0:
            left, right = points[index - 1], points[index]
        elif len(points) > 1:
            left, right = points[index], points[index + 1]
        else:
            raise ValueError("a node series needs at least two points for its slope")
    else:
        if index == 0 or index == len(points):
            raise ValueError(f"no source node interval at {temperature_K:g} K")
        left, right = points[index - 1], points[index]
        fraction = (temperature_K - left[0]) / (right[0] - left[0])
        value = left[1] + (right[1] - left[1]) * fraction
    slope = (right[1] - left[1]) / (right[0] - left[0])
    return value, slope


def _node_covered_interval(
    declared_interval_K: tuple[float, float],
    *node_temperature_sets: Sequence[float],
) -> tuple[float, float]:
    """Restrict a declared interval to the common coverage of required nodes."""
    lower_K, upper_K = declared_interval_K
    for temperatures in node_temperature_sets:
        if not temperatures:
            raise ValueError("a crystal node series cannot be empty")
        lower_K = max(lower_K, min(temperatures))
        upper_K = min(upper_K, max(temperatures))
    return lower_K, upper_K


def _cp_primitive(
    coefficients: Sequence[float], temperature_K: float, reference_K: float
) -> tuple[float, float]:
    """Integrate Berman's seven-term Cp polynomial to H and S increments.

    For Cp=k0+k1*T^-1/2+k2*T^-2+k3*T^-3+k4*T^-1+k5*T+k6*T^2,
    ΔH=∫Cp dT and ΔS=∫Cp/T dT, each evaluated from reference_K to T.
    The returned units are J/mol and J/(mol K), respectively.
    """
    k0, k1, k2, k3, k4, k5, k6 = coefficients
    t, r = temperature_K, reference_K
    enthalpy_J_mol = (
        k0 * (t - r)
        + 2.0 * k1 * (math.sqrt(t) - math.sqrt(r))
        - k2 * (1.0 / t - 1.0 / r)
        - 0.5 * k3 * (1.0 / t**2 - 1.0 / r**2)
        + k4 * math.log(t / r)
        + 0.5 * k5 * (t**2 - r**2)
        + (k6 / 3.0) * (t**3 - r**3)
    )
    entropy_J_molK = (
        k0 * math.log(t / r)
        + 2.0 * k1 * (1.0 / math.sqrt(r) - 1.0 / math.sqrt(t))
        - 0.5 * k2 * (1.0 / t**2 - 1.0 / r**2)
        - (k3 / 3.0) * (1.0 / t**3 - 1.0 / r**3)
        - k4 * (1.0 / t - 1.0 / r)
        + k5 * (t - r)
        + 0.5 * k6 * (t**2 - r**2)
    )
    return enthalpy_J_mol, entropy_J_molK


def _berman_properties(
    phase: Mapping[str, Any], temperature_K: float
) -> tuple[float, float, float, float]:
    """Return Berman Cp, S, apparent H, and apparent G."""
    parameters = phase["berman_parameters"]
    reference_K = 298.15
    supported_parameters = {
        "HfPrTr_J_mol", "SPrTr_J_mol_K", "Tlambda_K", "Tref_K",
        "DtH_J_mol", "l1", "l2", "Tmin_K", "Tmax_K",
        *(f"k{index}" for index in range(7)),
    }
    unsupported = {
        key: value
        for key, value in parameters.items()
        if key not in supported_parameters and float(value) != 0.0
    }
    if unsupported:
        raise ValueError(
            f"{phase['id']} has unsupported nonzero Berman terms: {unsupported}"
        )
    coefficients = tuple(
        float(parameters.get(f"k{index}", 0.0)) for index in range(7)
    )

    # Premise: Berman's source form is Cp=k0+k1*T^-1/2+k2*T^-2+k3*T^-3+
    # k4*T^-1+k5*T+k6*T^2. Integrating Cp gives the first primitive below;
    # integrating Cp/T gives the entropy primitive in _cp_primitive. The
    # primitives are ΔH=k0(T-Tr)+2k1(√T-√Tr)-k2(1/T-1/Tr)
    # -k3(1/T²-1/Tr²)/2+k4 ln(T/Tr)+k5(T²-Tr²)/2+k6(T³-Tr³)/3 and
    # ΔS=k0 ln(T/Tr)+2k1(1/√Tr-1/√T)-k2(1/T²-1/Tr²)/2
    # -k3(1/T³-1/Tr³)/3-k4(1/T-1/Tr)+k5(T-Tr)+k6(T²-Tr²)/2.
    # H_app=HfPrTr+ΔH and G_app=H_app-T*(SPrTr+ΔS), the same apparent
    # formation convention as condensates. Units are Cp J mol^-1 K^-1, H
    # J mol^-1, S J mol^-1 K^-1, and G J mol^-1.
    # Berman's lambda Cp=T*(l1+l2*T)^2 is integrated only from Tref to Tlambda;
    # above Tlambda its completed ΔH and ΔS remain as constants. The separate
    # first-order transition contributes ΔH and ΔS=ΔH/Tlambda at/above Tlambda.
    # Table 5 disorder Cp=d0+d1*T^-1/2+d2*T^-2+d3*T+d4*T^2 is integrated from
    # Tmin to the clipped temperature, retaining the completed increment above
    # Tmax. At 298.15 K the forsterite primitives vanish, reproducing its printed
    # HfPrTr=-2174420 J/mol and SPrTr=94.01 J/(mol K) exactly.
    enthalpy_increment_J_mol, entropy_increment_J_molK = _cp_primitive(
        coefficients, temperature_K, reference_K
    )

    lambda_l1 = float(parameters.get("l1", 0.0))
    lambda_l2 = float(parameters.get("l2", 0.0))
    if lambda_l1 != 0.0 or lambda_l2 != 0.0:
        lambda_reference_K = float(parameters.get("Tref_K", 0.0))
        lambda_temperature = float(parameters.get("Tlambda_K", 0.0))
        if not 0.0 < lambda_reference_K < lambda_temperature:
            raise ValueError(
                f"{phase['id']} has lambda Cp without a valid Tref_K/Tlambda_K"
            )
        clipped_K = min(max(temperature_K, lambda_reference_K), lambda_temperature)
        if clipped_K > lambda_reference_K:
            t, r = clipped_K, lambda_reference_K
            # For Cpλ=T*(l1+l2*T)^2, ΔHλ=[l1²T²/2+2l1l2T³/3+
            # l2²T⁴/4]r^t and ΔSλ=[l1²T+l1l2T²+l2²T³/3]r^t.
            enthalpy_increment_J_mol += (
                0.5 * lambda_l1**2 * (t**2 - r**2)
                + (2.0 / 3.0) * lambda_l1 * lambda_l2 * (t**3 - r**3)
                + 0.25 * lambda_l2**2 * (t**4 - r**4)
            )
            entropy_increment_J_molK += (
                lambda_l1**2 * (t - r)
                + lambda_l1 * lambda_l2 * (t**2 - r**2)
                + (lambda_l2**2 / 3.0) * (t**3 - r**3)
            )

    lambda_temperature = float(parameters.get("Tlambda_K", 0.0))
    transition_enthalpy = float(parameters.get("DtH_J_mol", 0.0))
    if transition_enthalpy != 0.0:
        if lambda_temperature <= 0.0:
            raise ValueError(
                f"{phase['id']} has transition enthalpy without Tlambda_K"
            )
        if temperature_K >= lambda_temperature:
            enthalpy_increment_J_mol += transition_enthalpy
            entropy_increment_J_molK += transition_enthalpy / lambda_temperature

    disorder = phase.get("disorder_parameters")
    disorder_heat_capacity = 0.0
    if disorder is not None:
        supported_disorder = {f"d{index}" for index in range(5)}
        unsupported_disorder = {
            key: value
            for key, value in disorder.items()
            if key not in supported_disorder and float(value) != 0.0
        }
        if unsupported_disorder:
            raise ValueError(
                f"{phase['id']} has unsupported nonzero Berman disorder terms: "
                f"{unsupported_disorder}"
            )
        lower_K = float(parameters["Tmin_K"])
        upper_K = float(parameters["Tmax_K"])
        if not lower_K < upper_K:
            raise ValueError(f"{phase['id']} has an invalid disorder interval")
        disorder_coefficients = (
            float(disorder.get("d0", 0.0)),
            float(disorder.get("d1", 0.0)),
            float(disorder.get("d2", 0.0)),
            0.0,
            0.0,
            float(disorder.get("d3", 0.0)),
            float(disorder.get("d4", 0.0)),
        )
        clipped_K = min(max(temperature_K, lower_K), upper_K)
        if clipped_K > lower_K:
            # The disorder coefficients use the same analytic ∫Cp dT and
            # ∫Cp/T dT primitives, anchored at Tmin and clipped at Tmax.
            disorder_h, disorder_s = _cp_primitive(
                disorder_coefficients, clipped_K, lower_K
            )
            enthalpy_increment_J_mol += disorder_h
            entropy_increment_J_molK += disorder_s
        if lower_K <= temperature_K <= upper_K:
            disorder_heat_capacity = (
                disorder_coefficients[0]
                + disorder_coefficients[1] / math.sqrt(temperature_K)
                + disorder_coefficients[2] / temperature_K**2
                + disorder_coefficients[5] * temperature_K
                + disorder_coefficients[6] * temperature_K**2
            )

    lambda_heat_capacity = 0.0
    if (
        (lambda_l1 != 0.0 or lambda_l2 != 0.0)
        and float(parameters.get("Tref_K", 0.0))
        <= temperature_K
        <= lambda_temperature
    ):
        lambda_heat_capacity = temperature_K * (
            lambda_l1 + lambda_l2 * temperature_K
        ) ** 2

    cp = math.fsum((
        coefficients[0],
        coefficients[1] / math.sqrt(temperature_K),
        coefficients[2] / temperature_K**2,
        coefficients[3] / temperature_K**3,
        coefficients[4] / temperature_K,
        coefficients[5] * temperature_K,
        coefficients[6] * temperature_K**2,
        lambda_heat_capacity,
        disorder_heat_capacity,
    ))
    enthalpy_app_kJ_mol = (
        float(parameters["HfPrTr_J_mol"]) + enthalpy_increment_J_mol
    ) / 1000.0
    entropy = (
        float(parameters["SPrTr_J_mol_K"]) + entropy_increment_J_molK
    )
    # H_app is formation-anchored like condensate H_app. Thus G_app=1000H_app-TS.
    gibbs_J_mol = 1000.0 * enthalpy_app_kJ_mol - temperature_K * entropy
    return cp, entropy, enthalpy_app_kJ_mol, gibbs_J_mol


def _node_crystal_properties(
    phase: Mapping[str, Any], temperature_K: float
) -> tuple[float, float, float, float]:
    """Interpolate R&H H/S nodes and return apparent-basis thermochemistry.

    Hinc and S are piecewise linear source node values; Cp is the Hinc secant
    in J/(mol K). H_app=Hf298+Hinc and G_app=1000*H_app-T*S. At each shared
    source node, subtracting the formula's ΣνG_app(element) recovers the
    element-referenced R&H ΔfG; perovskite at 1800 K is the sanity anchor.
    """
    enthalpy_nodes = [
        {"T_K": float(t), "value": float(value)}
        for t, value in phase["source_crystal_enthalpy_increment_nodes_kJ_mol"].items()
    ]
    entropy_nodes = [
        {"T_K": float(t), "value": float(value)}
        for t, value in phase["source_crystal_entropy_nodes_J_mol_K"].items()
    ]
    enthalpy_increment, slope_kJ_molK = _linear_value_and_slope(
        enthalpy_nodes, temperature_K
    )
    entropy, _ = _linear_value_and_slope(entropy_nodes, temperature_K)
    cp = 1000.0 * slope_kJ_molK
    enthalpy_app_kJ_mol = (
        float(phase["formation_enthalpy_298_kJ_mol"]) + enthalpy_increment
    )
    # R&H Hf(298)+Hinc and absolute S give the same condensate apparent form
    # G_app=1000*H_app-T*S. Removing ΣνG_app(element), from the packaged JANAF
    # element_reference_g_app_nodes_kJ_mol, converts this to R&H's printed ΔfG.
    # H nodes are kJ/mol (so multiply by 1000); TS is J/mol. The reconstructed
    # ΔfG matches the perovskite source node within 2.5 kJ/mol at 1800 K.
    gibbs_J_mol = 1000.0 * enthalpy_app_kJ_mol - temperature_K * entropy
    return cp, entropy, enthalpy_app_kJ_mol, gibbs_J_mol


def _barin_properties(
    phase: Mapping[str, Any],
    polymorph: Mapping[str, Any],
    temperature_K: float,
    *,
    continue_from_K: float | None = None,
) -> tuple[float, float, float, float]:
    # Barin supplies ΔfG, H-H298, and absolute S at common T nodes. Between
    # nodes X(T)=X0+(X1-X0)*(T-T0)/(T1-T0) for Hinc and S; Cp is 1000*(H1-H0)/
    # (T1-T0) J/(mol K), except that a printed sparse Cp value wins at its node.
    # H_app=Hf298+Hinc and G_app=1000*H_app-T*S use the condensate apparent
    # convention. Removing ΣνG_app(element) recovers printed ΔfG. The unit
    # check is kJ/mol*1000 versus K*J/(mol K); CaAl2O4 at its 1800 K node is the
    # known-value check.
    series = polymorph["source_series"]
    enthalpy_nodes = series["H_minus_H298"]["nodes"]
    entropy_nodes = series["S"]["nodes"]
    enthalpy_increment, slope_kJ_molK = _linear_value_and_slope(
        enthalpy_nodes, continue_from_K or temperature_K
    )
    entropy, _ = _linear_value_and_slope(
        entropy_nodes, continue_from_K or temperature_K
    )
    heat_capacity_nodes = series.get("Cp", {}).get("nodes", ())
    cp_at_node = next(
        (
            float(node["value"])
            for node in heat_capacity_nodes
            if float(node["T_K"]) == (continue_from_K or temperature_K)
        ),
        None,
    )
    cp = cp_at_node if cp_at_node is not None else 1000.0 * slope_kJ_molK
    enthalpy_app_kJ_mol = (
        float(phase["formation_enthalpy_298_kJ_mol"]) + enthalpy_increment
    )

    if continue_from_K is not None:
        # Premise: the existing Barin diagnostics continuation holds the last
        # printed Cp constant to its source solid limit. Integrating dH/dT=Cp
        # and dS/dT=Cp/T gives ΔH=CpΔT/1000 and ΔS=Cp ln(T/T0), preserving the
        # printed H/S node at T0. Unit check: both H terms are kJ/mol; T*S is J/mol.
        # Sanity: at T=T0 both increments vanish, matching the source node exactly.
        if cp_at_node is None:
            raise ValueError(
                f"{phase['id']}/{polymorph['label']} has no printed Cp at "
                f"continuation edge {continue_from_K:g} K"
            )
        enthalpy_app_kJ_mol, entropy, gibbs_J_mol = continue_constant_cp(
            enthalpy_app_kJ_mol,
            entropy,
            cp,
            temperature_K,
            continue_from_K,
        )
        return cp, entropy, enthalpy_app_kJ_mol, gibbs_J_mol

    # Barin's printed ΔfG omits elemental reference G_app terms. The source Hf,
    # H-H298, and absolute S reconstruct the runtime apparent basis directly;
    # equivalently G_app=ΔfG+ΣνG_app(element). At CaAl2O4/1800 K that conversion
    # recovers the printed -1593.299 kJ/mol ΔfG node from G_app and the element
    # reference nodes. Units: H is kJ/mol; multiply by 1000 before subtracting TS.
    gibbs_J_mol = 1000.0 * enthalpy_app_kJ_mol - temperature_K * entropy
    return cp, entropy, enthalpy_app_kJ_mol, gibbs_J_mol


def _nasa7_card_values(
    coefficients: Sequence[float], temperature_K: float
) -> tuple[float, float, float]:
    """Evaluate a NASA-7 Cp, H and S block in its source units."""
    a1, a2, a3, a4, a5, a6, a7 = map(float, coefficients)
    t = temperature_K
    gas_constant = R_PUBLISHED_GAS_J_MOL_K
    # Premise: Burcat's seven constants give Cp/R=a1+a2*T+...+a5*T^4.
    # Integrating Cp gives H/R=a1*T+a2*T^2/2+...+a5*T^5/5+a6; integrating
    # Cp/T gives S/R=a1*ln(T)+a2*T+...+a5*T^4/4+a7. The elemental cards'
    # absolute NASA H is the apparent H used by the condensate fit; their
    # a6 coefficients put H(298.15 K) at ΔfH298=0 when that temperature is in
    # the fitted block. Do not extrapolate H(298.15 K) from a block whose floor
    # is higher (as Au's is). Then G_app=1000*H_app-T*S. R is J/(mol K); H is
    # converted to kJ/mol only at the returned apparent-enthalpy field.
    cp = gas_constant * (
        a1 + a2 * t + a3 * t**2 + a4 * t**3 + a5 * t**4
    )
    enthalpy_J_mol = gas_constant * (
        a1 * t
        + a2 * t**2 / 2.0
        + a3 * t**3 / 3.0
        + a4 * t**4 / 4.0
        + a5 * t**5 / 5.0
        + a6
    )
    entropy_J_molK = gas_constant * (
        a1 * math.log(t)
        + a2 * t
        + a3 * t**2 / 2.0
        + a4 * t**3 / 3.0
        + a5 * t**4 / 4.0
        + a7
    )
    return cp, enthalpy_J_mol, entropy_J_molK


def _elemental_reference_card_properties(
    phase: Mapping[str, Any], temperature_K: float
) -> tuple[float, float, float, float]:
    """Return apparent thermochemistry for one Burcat or Pankratz crystal card."""
    gas_constant = R_PUBLISHED_GAS_J_MOL_K
    formation_enthalpy = float(phase["formation_enthalpy_298_kJ_mol"])
    if phase["form"] == "nasa7":
        segments = phase["segments"]
        selected_segment = segments[0]
        for segment in segments[1:]:
            if temperature_K >= float(segment["temperature_range_K"][0]):
                selected_segment = segment
        cp, enthalpy_J_mol, entropy = _nasa7_card_values(
            selected_segment["coefficients"], temperature_K
        )
        enthalpy_app_kJ_mol = enthalpy_J_mol / 1000.0
    elif phase["form"] == "tabulated":
        # Pankratz prints Cp in cal/(mol K), S in cal/(mol K), and H-H298 in
        # kcal/mol. Interpolate H-H298 and S linearly between printed nodes;
        # Cp is the H secant between nodes and the printed Cp at a source node.
        # This propagates the table into one consistent H/S/G function while
        # retaining source Cp at the printed temperatures. The factor 4.184
        # converts cal to J and kcal to kJ; G_app=1000*H_app-T*S.
        rows = phase["nodes"]
        enthalpy_nodes = [
            {"T_K": float(row[0]), "value": float(row[3]) * 4.184}
            for row in rows
        ]
        entropy_nodes = [
            {"T_K": float(row[0]), "value": float(row[2]) * 4.184}
            for row in rows
        ]
        enthalpy_increment, slope_kJ_molK = _linear_value_and_slope(
            enthalpy_nodes, temperature_K
        )
        entropy, _ = _linear_value_and_slope(entropy_nodes, temperature_K)
        printed_cp = next(
            (
                float(row[1]) * 4.184
                for row in rows
                if float(row[0]) == temperature_K
            ),
            None,
        )
        cp = printed_cp if printed_cp is not None else 1000.0 * slope_kJ_molK
        enthalpy_app_kJ_mol = formation_enthalpy + enthalpy_increment
    else:
        raise ValueError(f"unsupported elemental crystal card form {phase['form']!r}")

    gibbs_J_mol = 1000.0 * enthalpy_app_kJ_mol - temperature_K * entropy
    return cp, entropy, enthalpy_app_kJ_mol, gibbs_J_mol


def _candidates_for_formula(formula: str) -> list[tuple[str, list[_Candidate]]]:
    formula_counts = _formula_element_counts(formula)
    if formula_counts is None:
        return []

    janaf_sources = (
        ("JANAF pure-oxide crystals", _load_janaf_phases()[0]),
        ("JANAF crystal tables", _load_janaf_crystal_species()[0]),
    )
    for dataset, janaf_rows in janaf_sources:
        janaf = []
        for row in janaf_rows.values():
            if _formula_element_counts(row["formula"]) != formula_counts:
                continue
            source_lo, source_hi = map(float, row["source_temperature_K"])
            interval = (
                max(298.15, source_lo),
                min(source_hi, float(row["crystal_transition_temperature_K"])),
            )
            nodes = [*row["nodes"], *row.get("transition_rows", [])]
            cap_row = row.get("transition_row_at_cap")
            if cap_row is not None:
                nodes.append(cap_row)
            interval = _node_covered_interval(
                interval, [float(node["T_K"]) for node in nodes]
            )
            table_id = str(row["crystal_table_id"])
            janaf.append(_Candidate(
                dataset,
                str(row.get("source_title", table_id)),
                row,
                None,
                interval,
                table_id,
                table_id,
                str(row["source_authority"]),
            ))
        if janaf:
            return [(dataset, janaf)]

    barin = []
    for phase in _barin_phases():
        if _formula_element_counts(phase["formula"]) != formula_counts:
            continue
        for polymorph in phase["polymorphs"]:
            interval = tuple(map(float, polymorph["valid_temperature_K"]))
            source_series = polymorph["source_series"]
            interval = _node_covered_interval(
                interval,
                [float(node["T_K"]) for node in source_series["H_minus_H298"]["nodes"]],
                [float(node["T_K"]) for node in source_series["S"]["nodes"]],
            )
            label = str(polymorph["label"])
            barin.append(_Candidate(
                "Barin aluminates",
                label,
                phase,
                polymorph,
                interval,
                f"{phase['id']}/{label}",
                "Barin-1995",
                str(phase.get("source_authority", "compilation_assessed")),
            ))
    if barin:
        return [("Barin aluminates", barin)]

    minerals = []
    for phase in _berman_phases():
        if _formula_element_counts(phase["formula"]) != formula_counts:
            continue
        interval = tuple(map(float, phase["valid_temperature_K"]))
        public_lower_K = _BERMAN_PUBLIC_SERVICE_LOWER_K.get(str(phase["id"]))
        if phase.get("berman_parameters") is not None and public_lower_K is not None:
            interval = (public_lower_K, interval[1])
        if phase.get("berman_parameters") is None:
            interval = _node_covered_interval(
                interval,
                [
                    float(temperature)
                    for temperature in phase[
                        "source_crystal_enthalpy_increment_nodes_kJ_mol"
                    ]
                ],
                [
                    float(temperature)
                    for temperature in phase["source_crystal_entropy_nodes_J_mol_K"]
                ],
            )
        author = str(phase["source"].get("authors", ""))
        source_table_id = (
            "Berman-1988" if author == "Berman" else "Robie-Hemingway-1995"
        )
        minerals.append(_Candidate(
            "Berman/Robie-Hemingway minerals",
            str(phase["id"]),
            phase,
            None,
            interval,
            str(phase["id"]),
            source_table_id,
            "published_fitted",
        ))
    if minerals:
        return [("Berman/Robie-Hemingway minerals", minerals)]

    robie_hemingway = []
    for phase in _rh_crystal_table_phases():
        if _formula_element_counts(phase["formula"]) != formula_counts:
            continue
        interval = tuple(map(float, phase["valid_temperature_K"]))
        interval = _node_covered_interval(
            interval,
            [
                float(temperature)
                for temperature in phase[
                    "source_crystal_enthalpy_increment_nodes_kJ_mol"
                ]
            ],
            [
                float(temperature)
                for temperature in phase[
                    "source_crystal_entropy_nodes_J_mol_K"
                ]
            ],
        )
        table_id = str(phase["source_table_id"])
        robie_hemingway.append(_Candidate(
            "Robie-Hemingway 1995 tables",
            str(phase["source_title"]),
            phase,
            None,
            interval,
            f"{table_id}:{phase['id']}",
            table_id,
            str(phase["source_authority"]),
        ))
    if robie_hemingway:
        return [("Robie-Hemingway 1995 tables", robie_hemingway)]

    elemental_reference_cards = []
    for phase in _elemental_reference_card_phases():
        if phase["formula"] != formula:
            continue
        interval = tuple(map(float, phase["temperature_range_K"]))
        source_card_id = str(phase["source_card_id"])
        elemental_reference_cards.append(_Candidate(
            "Elemental reference crystal cards",
            str(phase["polymorph"]),
            phase,
            None,
            interval,
            source_card_id,
            str(phase["source_table_id"]),
            "published_fitted",
        ))
    if elemental_reference_cards:
        return [("Elemental reference crystal cards", elemental_reference_cards)]
    return []


def _source_solid_limit(candidate: _Candidate) -> float | None:
    phase = candidate.phase
    limit = phase.get("source_solid_limit")
    if isinstance(limit, Mapping):
        return float(limit["temperature_K"])
    melting = phase.get("melting_point")
    if isinstance(melting, Mapping):
        return float(melting["temperature_K"])
    if candidate.dataset in {
        "JANAF pure-oxide crystals", "JANAF crystal tables",
    }:
        return float(phase["crystal_transition_temperature_K"])
    if candidate.dataset == "Elemental reference crystal cards":
        return float(phase["source_limit_K"])
    return None


def _active_janaf_phase_segment(
    phase: Mapping[str, Any], temperature_K: float,
) -> Mapping[str, Any] | None:
    segments = phase.get("phase_segments", ())
    return next(
        (
            segment
            for index, segment in enumerate(segments)
            if float(segment["temperature_K"][0]) <= temperature_K
            and (
                temperature_K < float(segment["temperature_K"][1])
                or (
                    index == len(segments) - 1
                    and temperature_K == float(segment["temperature_K"][1])
                )
            )
        ),
        None,
    )


def _available_at_temperature(
    candidate: _Candidate,
    temperature_K: float,
    *,
    allow_extrapolation: bool,
) -> tuple[bool, float | None, bool, float | None]:
    lower_K, upper_K = candidate.interval_K
    limit_K = _source_solid_limit(candidate)
    cap_limit_K = _crystal_source_cap_excludes_candidate(
        candidate.phase.get("oxide_stoichiometry", {}), limit_K, temperature_K,
    )
    if cap_limit_K is not None:
        return False, None, False, cap_limit_K
    if lower_K <= temperature_K <= upper_K:
        return True, None, False, None
    if (
        allow_extrapolation
        and candidate.dataset == "Barin aluminates"
        and temperature_K > upper_K
        and limit_K is not None
        and temperature_K <= limit_K
        and candidate.polymorph is not None
    ):
        edge_K = upper_K
        cp_nodes = candidate.polymorph["source_series"].get("Cp", {}).get("nodes", ())
        if any(float(node["T_K"]) == edge_K for node in cp_nodes):
            return True, edge_K, True, None
    return False, None, False, None


def _evaluate_candidate(
    candidate: _Candidate,
    temperature_K: float,
    *,
    continue_from_K: float | None,
) -> tuple[float, float, float, float, float | None]:
    if candidate.dataset in {
        "JANAF pure-oxide crystals", "JANAF crystal tables",
    }:
        row = candidate.phase
        nodes = _janaf_nodes_for_temperature(row, temperature_K)
        enthalpy_nodes = [
            {"T_K": node["T_K"], "value": node["enthalpy_increment_kJ_mol"]}
            for node in nodes
        ]
        entropy_nodes = [
            {"T_K": node["T_K"], "value": node["entropy_J_mol_K"]}
            for node in nodes
        ]
        enthalpy_increment, slope_kJ_molK = _linear_value_and_slope(
            enthalpy_nodes, temperature_K
        )
        entropy, _ = _linear_value_and_slope(entropy_nodes, temperature_K)
        # JANAF supplies H increments in kJ/mol and S in J/(mol K), linearly
        # interpolated between nodes. Cp is the H secant, 1000*(H1-H0)/(T1-T0)
        # J/(mol K), because these crystal rows have no Cp column. H_app=Hf298+
        # Hinc and G_app=1000*H_app-T*S, where 1000 converts kJ/mol to J/mol.
        # This is the condensate convention; at every node it reproduces the
        # existing evaluate_janaf_crystal output exactly (known-value check).
        cp = 1000.0 * slope_kJ_molK
        enthalpy_app = float(row["formation_enthalpy_298_kJ_mol"]) + enthalpy_increment
        # Reuse the established JANAF evaluator so the apparent-G interpolation,
        # cap row, and transition-node tie behavior remain exactly the same.
        gibbs = evaluate_janaf_crystal(str(row["id"]), temperature_K).G_J_mol
        return (
            cp,
            entropy,
            enthalpy_app,
            gibbs,
            float(row["formation_enthalpy_298_kJ_mol"]),
        )

    if candidate.dataset == "Barin aluminates":
        assert candidate.polymorph is not None
        cp, entropy, enthalpy_app, gibbs = _barin_properties(
            candidate.phase,
            candidate.polymorph,
            temperature_K,
            continue_from_K=continue_from_K,
        )
        formation_enthalpy = float(candidate.phase["formation_enthalpy_298_kJ_mol"])
        return cp, entropy, enthalpy_app, gibbs, formation_enthalpy

    if candidate.dataset == "Elemental reference crystal cards":
        cp, entropy, enthalpy_app, gibbs = _elemental_reference_card_properties(
            candidate.phase, temperature_K
        )
        formation_enthalpy = float(
            candidate.phase["formation_enthalpy_298_kJ_mol"]
        )
        return cp, entropy, enthalpy_app, gibbs, formation_enthalpy

    phase = candidate.phase
    if phase.get("berman_parameters") is not None:
        cp, entropy, enthalpy_app, gibbs = _berman_properties(phase, temperature_K)
        formation_enthalpy = float(phase["berman_parameters"]["HfPrTr_J_mol"]) / 1000.0
    else:
        cp, entropy, enthalpy_app, gibbs = _node_crystal_properties(
            phase, temperature_K
        )
        formation_enthalpy = float(phase["formation_enthalpy_298_kJ_mol"])
    return cp, entropy, enthalpy_app, gibbs, formation_enthalpy


def evaluate_crystal_species(
    formula: str,
    temperature_K: float,
    *,
    polymorph: str | None = None,
    allow_extrapolation: bool = False,
) -> CrystalSpeciesThermo | None:
    """Evaluate one shipped crystal source, applying a fixed source precedence.

    Condensate-table precedence is handled by ``species_thermo`` before this
    function. Among shipped sources: JANAF is preferred for pure oxides, Barin
    for its aluminate formulas (including MgAl2O4), then Berman/R&H for mineral
    formulas. This keeps pure-oxide controls and direct aluminate tables on their
    source rails; G is never compared across datasets.
    """
    groups = _candidates_for_formula(formula)
    if not groups:
        return None
    dataset, candidates = groups[0]
    labels = tuple(f"{candidate.label} [{candidate.interval_K[0]:g}-{candidate.interval_K[1]:g} K]" for candidate in candidates)

    named = None
    named_segment = None
    named_segment_interval = None
    is_reference = is_reference_polymorph(polymorph)
    alias_used = False
    if is_reference:
        if not is_element_formula(formula):
            raise CrystalSpeciesPolymorphError(
                f"reference selection is ambiguous for {formula}(cr); "
                f"select one of the available polymorphs: {', '.join(labels)}",
                labels,
            )
        is_elemental_reference_card = (
            len(candidates) == 1
            and candidates[0].dataset == "Elemental reference crystal cards"
            and candidates[0].phase.get("elemental_reference") is True
        )
        is_janaf_elemental_reference = (
            len(candidates) == 1
            and candidates[0].dataset == "JANAF crystal tables"
            and (
                float(
                    candidates[0].phase["formation_enthalpy_298_kJ_mol"]
                ) == 0.0
                or re.search(
                    r"\bref\b",
                    str(candidates[0].phase.get("source_title", "")),
                    re.I,
                )
                is not None
            )
        )
        if (
            len(candidates) != 1
            or not (
                is_elemental_reference_card
                or is_janaf_elemental_reference
            )
        ):
            raise CrystalSpeciesPolymorphError(
                f"reference selection is ambiguous for {formula}(cr) from "
                f"{dataset}; available polymorphs: {', '.join(labels)}",
                labels,
            )
        named = candidates[0]
    elif polymorph is not None:
        folded = polymorph.casefold()
        alias_target = _POLYMORPH_ALIASES.get(formula, {}).get(folded)
        for candidate in candidates:
            source_title = str(candidate.phase.get("source_title", ""))
            if (
                alias_target is not None
                and candidate.source_table_id.casefold() == alias_target.casefold()
            ):
                named = candidate
                alias_used = True
                break
            if (
                candidate.label.casefold() == folded
                or source_title.casefold() == folded
            ):
                named = candidate
                break
            if candidate.dataset == "Robie-Hemingway 1995 tables":
                parenthesized_aliases = {
                    alias.strip().casefold()
                    for alias in re.findall(r"\(([^()]*)\)", source_title)
                }
                if folded.strip("()").strip() in parenthesized_aliases:
                    named = candidate
                    break
            matched_segments = [
                segment
                for segment in candidate.phase.get("phase_segments", ())
                if str(segment.get("phase_label", "")).casefold() == folded
            ]
            if matched_segments:
                named = candidate
                named_segment = str(matched_segments[0]["phase_label"])
                named_segment_interval = tuple(
                    map(float, matched_segments[0]["temperature_K"])
                )
                break
        if named is None:
            raise CrystalSpeciesPolymorphError(
                f"polymorph {polymorph!r} is not available for {formula}(cr) "
                f"from {dataset}; available polymorphs: {', '.join(labels)}",
                labels,
            )

    evaluated = []
    cap_refusals = []
    for candidate in candidates:
        if named is not None and candidate != named:
            continue
        if named_segment is not None:
            segment = _active_janaf_phase_segment(
                candidate.phase, temperature_K,
            )
            if (
                segment is None
                or str(segment["phase_label"]).casefold()
                != named_segment.casefold()
            ):
                continue
        (
            available,
            continue_from_K,
            extrapolated,
            cap_limit_K,
        ) = _available_at_temperature(
            candidate,
            temperature_K,
            allow_extrapolation=allow_extrapolation,
        )
        if not available:
            if cap_limit_K is not None:
                cap_refusals.append((candidate, cap_limit_K))
            continue
        properties = _evaluate_candidate(
            candidate,
            temperature_K,
            continue_from_K=continue_from_K,
        )
        evaluated.append((candidate, properties, continue_from_K, extrapolated))

    if not evaluated:
        available_labels = labels
        candidate = named or candidates[0]
        if named is None and named_segment_interval is None:
            low_K = min(item.interval_K[0] for item in candidates)
            high_K = max(item.interval_K[1] for item in candidates)
        else:
            low_K, high_K = candidate.interval_K
        if named_segment_interval is not None:
            low_K, high_K = named_segment_interval
        if cap_refusals:
            candidate, limit = cap_refusals[0]
            message = (
                f"cannot serve {formula}(cr) at T={temperature_K:g} K: "
                f"above its source solid limit of {limit:g} K; declared "
                f"interval is ({low_K:g}, {high_K:g}) K; available "
                f"polymorphs: {', '.join(available_labels)}"
            )
        elif named_segment_interval is not None:
            message = (
                f"T={temperature_K:g} K is outside the declared G(T) interval "
                f"({low_K:g}, {high_K:g}) K for phase segment "
                f"{named_segment!r} of {formula}(cr) from {dataset}; "
                f"available polymorphs: {', '.join(available_labels)}"
            )
        else:
            message = (
                f"T={temperature_K:g} K is outside the declared G(T) interval "
                f"({low_K:g}, {high_K:g}) K for {formula}(cr) from "
                f"{dataset}; available polymorphs: "
                f"{', '.join(available_labels)}"
            )
        raise CrystalSpeciesTemperatureError(
            message,
            declared_interval_K=(low_K, high_K),
            source_row_id=candidate.source_row_id,
            available_polymorphs=available_labels,
        )

    tied_candidates = []
    if named is not None:
        selected = evaluated[0]
        if is_reference:
            if named.dataset == "Elemental reference crystal cards":
                rule = "selected by the vendored source card's elemental reference crystal"
            else:
                rule = "selected by the source-defined elemental reference-state table"
        elif alias_used:
            rule = (
                f"selected through the {polymorph!r} alias for "
                f"{named.label} in {dataset}"
            )
        else:
            rule = f"selected by polymorph={polymorph!r} within {dataset}"
    else:
        selected = min(evaluated, key=lambda item: item[1][3])
        rule = f"lowest apparent G at T among {dataset} polymorphs covering T"
        selected_gibbs = selected[1][3]
        tied_candidates = [
            item
            for item in evaluated
            if math.isclose(item[1][3], selected_gibbs, rel_tol=1.0e-6)
        ]

    candidate, properties, continue_from_K, extrapolated = selected
    cp, entropy, enthalpy_app, gibbs, formation_enthalpy = properties
    phase_identity = {
        "kind": "single_polymorph",
        "polymorph": candidate.label,
        "dataset": dataset,
        "selection_rule": rule,
        "declared_interval_K": candidate.interval_K,
    }
    if candidate.dataset == "JANAF crystal tables":
        segments = candidate.phase["phase_segments"]
        segment = _active_janaf_phase_segment(
            candidate.phase, temperature_K,
        )
        if segment is None:
            segment = segments[-1]
        phase_identity["polymorph"] = segment["phase_label"]
        phase_identity["source_title"] = candidate.phase["source_title"]
        phase_identity["temperature_segment_K"] = tuple(
            map(float, segment["temperature_K"])
        )
        dedicated_table_phase = (
            len(segments) == 1
            and str(segment["phase_label"]).casefold()
            != candidate.label.casefold()
        )
        if is_reference:
            phase_identity["selection_rule"] = (
                "JANAF reference-state table and transition rows identify the "
                "active segment at T"
            )
        elif dedicated_table_phase:
            if alias_used:
                phase_identity["selection_rule"] = (
                    "selected as an alias of the source table title; the table "
                    "retains its named phase identity"
                )
            else:
                phase_identity["selection_rule"] = (
                    "JANAF source table title identifies its phase"
                )
        elif alias_used:
            phase_identity["selection_rule"] = (
                "selected as an alias of the source table title; transition rows "
                "identify the active segment at T"
            )
        else:
            phase_identity["selection_rule"] = (
                "JANAF source title and transition rows identify the active segment"
            )
    elif candidate.dataset == "Elemental reference crystal cards":
        source_phase = candidate.phase
        phase_identity["source_card_id"] = source_phase["source_card_id"]
        phase_identity["source_path"] = source_phase["source_path"]
        phase_identity["source_title"] = source_phase["source_title"]
        phase_identity["functional_form"] = source_phase["functional_form"]
        phase_identity["source_card_range_K"] = candidate.interval_K
        phase_identity["temperature_segment_K"] = candidate.interval_K
        if source_phase["form"] == "nasa7":
            for segment in source_phase["segments"]:
                lower_K, upper_K = map(float, segment["temperature_range_K"])
                if lower_K <= temperature_K < upper_K or (
                    temperature_K == upper_K
                    and upper_K == candidate.interval_K[1]
                ):
                    phase_identity["temperature_segment_K"] = (
                        lower_K, upper_K
                    )
                    break
    if len(tied_candidates) > 1:
        tied_names = [item[0].label for item in tied_candidates]
        phase_identity["tie_note"] = (
            "Apparent G values are within 1e-6 relative tolerance for "
            f"{', '.join(tied_names)}; selected {candidate.label} by minimum G "
            "(source order breaks exact ties)."
        )
    source_description = candidate.phase.get("source", {}).get(
        "printed_phase_description"
    )
    if isinstance(source_description, str) and source_description:
        has_multiple_forms = "; " in source_description or ". " in source_description
        phase_identity["note"] = (
            f"One row spans a transition: {source_description}"
            if has_multiple_forms
            else source_description
        )
    limit = _source_solid_limit(candidate)
    source_solid_limit = candidate.phase.get("source_solid_limit")
    if isinstance(source_solid_limit, Mapping):
        phase_identity["source_solid_limit_K"] = limit
        phase_identity["source_solid_limit_kind"] = str(
            source_solid_limit.get("kind", "unspecified")
        )
    elif isinstance(candidate.phase.get("melting_point"), Mapping):
        phase_identity["melting_temperature_K"] = limit
    elif candidate.dataset in {
        "JANAF pure-oxide crystals", "JANAF crystal tables",
    }:
        phase_identity["crystal_transition_temperature_K"] = limit
    continuation = None
    if extrapolated:
        continuation = (
            f"constant-Cp crystal continuation within {candidate.label} stability range"
        )
    return CrystalSpeciesThermo(
        Cp_J_molK=float(cp),
        S_J_molK=float(entropy),
        H_app_kJ_mol=float(enthalpy_app),
        G_J_mol=float(gibbs),
        source_row_id=candidate.source_row_id,
        source_table_id=candidate.source_table_id,
        T_min=candidate.interval_K[0],
        T_max=candidate.interval_K[1],
        authority=candidate.authority,
        phase_identity=phase_identity,
        formation_enthalpy_298_kJ_mol=formation_enthalpy,
        formation_enthalpy_source_id=candidate.source_table_id,
        extrapolated=extrapolated,
        continuation=continuation,
    )


def continue_constant_cp(
    enthalpy_app_kJ_mol: float,
    entropy_J_molK: float,
    heat_capacity_J_molK: float,
    temperature_K: float,
    edge_temperature_K: float,
) -> tuple[float, float, float]:
    """Continue H/S/G with the existing bounded crystal constant-Cp rule.

    Integrating dH/dT=Cp and dS/dT=Cp/T from the printed edge T0 gives
    H=H0+Cp*(T-T0)/1000 and S=S0+Cp*ln(T/T0); then G=1000*H-T*S.
    Cp is J/(mol K), H is kJ/mol, and G is J/mol. At T=T0 all increments are
    zero, so the source edge value is recovered exactly.
    """
    continued_h = enthalpy_app_kJ_mol + heat_capacity_J_molK * (
        temperature_K - edge_temperature_K
    ) / 1000.0
    continued_s = entropy_J_molK + heat_capacity_J_molK * math.log(
        temperature_K / edge_temperature_K
    )
    return continued_h, continued_s, 1000.0 * continued_h - temperature_K * continued_s


__all__ = [
    "CrystalSpeciesPolymorphError",
    "CrystalSpeciesTemperatureError",
    "CrystalSpeciesThermo",
    "continue_constant_cp",
    "evaluate_crystal_species",
    "is_element_formula",
    "is_reference_polymorph",
]
