"""Full apparent thermochemistry for the shipped crystal source datasets."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import json
import math
from typing import Any, Mapping, Sequence

from openimcc.janaf_crystals import (
    _load_phases as _load_janaf_phases,
    evaluate_janaf_crystal,
)


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
def _barin_phases() -> tuple[dict[str, Any], ...]:
    return tuple(
        _read_json("data/diagnostics/barin-aluminate-crystals.json")["phases"]
    )


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


def _unique_janaf_nodes(
    row: Mapping[str, Any], *, include_cap: bool = True
) -> list[Mapping[str, Any]]:
    nodes = [*row["nodes"], *row.get("transition_rows", [])]
    cap_row = row.get("transition_row_at_cap")
    if include_cap and cap_row is not None:
        nodes.append(cap_row)
    by_temperature = {float(node["T_K"]): node for node in nodes}
    return [by_temperature[t] for t in sorted(by_temperature)]


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


def _candidates_for_formula(formula: str) -> list[tuple[str, list[_Candidate]]]:
    janaf_rows, _ = _load_janaf_phases()
    janaf = []
    for row in janaf_rows.values():
        if row["formula"] != formula:
            continue
        source_lo, source_hi = map(float, row["source_temperature_K"])
        interval = (
            max(298.15, source_lo),
            min(source_hi, float(row["crystal_transition_temperature_K"])),
        )
        nodes = _unique_janaf_nodes(row)
        interval = _node_covered_interval(
            interval, [float(node["T_K"]) for node in nodes]
        )
        table_id = str(row["crystal_table_id"])
        janaf.append(_Candidate(
            "JANAF pure-oxide crystals",
            table_id,
            row,
            None,
            interval,
            table_id,
            table_id,
            str(row["source_authority"]),
        ))
    if janaf:
        return [("JANAF pure-oxide crystals", janaf)]

    barin = []
    for phase in _barin_phases():
        if phase["formula"] != formula:
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
        if phase["formula"] != formula:
            continue
        interval = tuple(map(float, phase["valid_temperature_K"]))
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
    return []


def _source_solid_limit(candidate: _Candidate) -> float | None:
    phase = candidate.phase
    limit = phase.get("source_solid_limit")
    if isinstance(limit, Mapping):
        return float(limit["temperature_K"])
    melting = phase.get("melting_point")
    if isinstance(melting, Mapping):
        return float(melting["temperature_K"])
    if candidate.dataset == "JANAF pure-oxide crystals":
        return float(phase["crystal_transition_temperature_K"])
    return None


def _available_at_temperature(
    candidate: _Candidate,
    temperature_K: float,
    *,
    allow_extrapolation: bool,
) -> tuple[bool, float | None, bool]:
    lower_K, upper_K = candidate.interval_K
    limit_K = _source_solid_limit(candidate)
    if (
        len(candidate.phase["oxide_stoichiometry"]) > 1
        and limit_K is not None
        and temperature_K > limit_K
    ):
        return False, None, False
    if lower_K <= temperature_K <= upper_K:
        return True, None, False
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
            return True, edge_K, True
    return False, None, False


def _evaluate_candidate(
    candidate: _Candidate,
    temperature_K: float,
    *,
    continue_from_K: float | None,
) -> tuple[float, float, float, float, float | None]:
    if candidate.dataset == "JANAF pure-oxide crystals":
        row = candidate.phase
        nodes = _unique_janaf_nodes(row)
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
    if polymorph is not None:
        folded = polymorph.casefold()
        named = next(
            (candidate for candidate in candidates if candidate.label.casefold() == folded),
            None,
        )
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
        available, continue_from_K, extrapolated = _available_at_temperature(
            candidate,
            temperature_K,
            allow_extrapolation=allow_extrapolation,
        )
        if not available:
            limit = _source_solid_limit(candidate)
            if (
                len(candidate.phase["oxide_stoichiometry"]) > 1
                and limit is not None
                and temperature_K > limit
            ):
                cap_refusals.append((candidate, limit))
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
        low_K, high_K = candidate.interval_K
        if cap_refusals:
            candidate, limit = cap_refusals[0]
            message = (
                f"cannot serve {formula}(cr) at T={temperature_K:g} K: "
                f"above its source solid limit of {limit:g} K; available "
                f"polymorphs: {', '.join(available_labels)}"
            )
        else:
            message = (
                f"T={temperature_K:g} K is outside the declared G(T) interval "
                f"for {formula}(cr) from {dataset}; available polymorphs: "
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
        phase_identity["note"] = (
            f"One row spans a transition: {source_description}"
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
    elif candidate.dataset == "JANAF pure-oxide crystals":
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
]
