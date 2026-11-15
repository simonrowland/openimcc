"""Independent pure-oxide JANAF and packaged-standard checks.

The three values in :class:`PureOxideStandardCheck` are deliberately separate:
the JANAF row-pair closure, the selected packaged liquid-minus-crystal Gibbs
residual, and the root predicted by those packaged functions.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from openimcc.kernel import ImccRefusal
from openimcc.janaf_crystals import (
    JanafCrystalTemperatureError,
    JanafRowTemperatureError,
    evaluate_janaf_apparent_gibbs,
    evaluate_janaf_crystal,
)


JANAF_PAIR_TOLERANCE_KJ_MOL = 0.01
PACKAGED_SIO2_ROUNDING_TOLERANCE_KJ_MOL = 0.002
SWITCH_BRACKET_TOLERANCE_K = 0.1
SWITCH_SOLVER_XTOL_K = 1e-8
_ROOT_SCAN_STEP_K = 1.0
_SLOPE_STEP_K = 0.01
_RESOLUTION_ULPS = 8.0
_FLOAT_EPSILON = sys.float_info.epsilon
_J_TO_KJ = 1000.0


@dataclass(frozen=True)
class _OxideDefinition:
    oxide: str
    crystal_table: str
    liquid_table: str | None
    transition_K: float
    construction: str | None = None


_OXIDES = (
    _OxideDefinition("CaO", "Ca-027", "Ca-028", 3200.0),
    _OxideDefinition("Al2O3", "Al-096", "Al-100", 2327.0),
    _OxideDefinition("MgO", "Mg-008", "Mg-009", 3105.0),
    _OxideDefinition("FeO", "Fe-018", "Fe-019", 1650.0),
    _OxideDefinition("Na2O", "Na-012", "Na-013", 1405.2),
    _OxideDefinition("SiO2", "O-035", "O-038", 1996.0),
    _OxideDefinition("TiO2", "O-043", "O-044", 2130.0),
    _OxideDefinition(
        "K2O", "K-012", None, 1013.0,
        "constructed liquid standard; selected Tfus=1013 K, source band 919–1190 K",
    ),
)
_OXIDE_BY_NAME = MappingProxyType({row.oxide: row for row in _OXIDES})


class StandardCheckRefusal(ImccRefusal):
    """Typed refusal when a required standard function cannot be evaluated."""

    code = "pure_oxide_standard_check_refused"

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        axis: str = "temperature_K",
        valid_range_K: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.axis = axis
        self.valid_range_K = valid_range_K


@dataclass(frozen=True)
class SwitchPrediction:
    status: Literal["root", "no_switch_in_range", "resolution_indeterminate", "out_of_range"]
    temperature_K: float | None
    bracket_K: tuple[float, float] | None
    bracket_residuals_J_mol: tuple[float, float] | None
    residual_J_mol: float | None
    uncertainty_K: float | None
    search_range_K: tuple[float, float] | None
    temperature_axis_label: str
    diagnostic: str | None = None
    bracket_resolution_bounds_J_mol: tuple[float, float] | None = None


@dataclass(frozen=True)
class PackagedClosure:
    status: Literal["available", "out_of_range"]
    delta_kJ_mol: float | None
    crystal_source: str
    crystal_range_K: tuple[float, float] | None
    liquid_source: str
    liquid_range_K: tuple[float, float] | None
    refusal: str | None = None


@dataclass(frozen=True)
class PureOxideStandardCheck:
    oxide: str
    temperature_K: float
    janaf_transition_K: float
    janaf_pair_status: Literal["available", "construction_case"]
    delta_janaf_kJ_mol: float | None
    packaged_closure: PackagedClosure
    switch_prediction: SwitchPrediction
    janaf_rows: tuple[str, ...]
    pack_id: str
    pack_digest: str
    source_crystal_id: str
    source_crystal_digest: str | None
    least_authoritative_input: str
    authority: str
    construction_flag: str | None = None


def _janaf_crystal(oxide: str, temperature_K: float):
    try:
        return evaluate_janaf_crystal(oxide, temperature_K, apply_phase_cap=False)
    except JanafCrystalTemperatureError as exc:
        definition = _OXIDE_BY_NAME.get(oxide)
        table_id = definition.crystal_table if definition is not None else exc.phase_id
        raise StandardCheckRefusal(
            str(exc), phase=table_id, axis="temperature_K",
            valid_range_K=exc.domain_K,
        ) from exc


def _janaf_liquid(table_id: str, temperature_K: float):
    try:
        return evaluate_janaf_apparent_gibbs(table_id, temperature_K)
    except JanafRowTemperatureError as exc:
        raise StandardCheckRefusal(
            str(exc), phase=exc.table_id, axis="temperature_K",
            valid_range_K=exc.domain_K,
        ) from exc


def _packaged_crystal_source(
    oxide: str, gas_pack: Any
) -> tuple[str, tuple[float, float]] | None:
    name = f"{oxide}(cr)"
    if name not in gas_pack.oxide_df.index:
        return None
    rows = gas_pack.oxide_df.loc[name]
    if not hasattr(rows, "iterrows"):
        rows = rows.to_frame().T
    low = min(float(row["T_min"]) for _, row in rows.iterrows())
    high = max(float(row["T_max"]) for _, row in rows.iterrows())
    return name, (low, high)


def _interval_rows(gas_pack: Any, oxide: str, phase: Literal["l", "cr"]) -> tuple[Any, ...]:
    name = f"{oxide}({phase})"
    if name not in gas_pack.oxide_df.index:
        return ()
    rows = gas_pack.oxide_df.loc[name]
    if not hasattr(rows, "iterrows"):
        return (rows,)
    return tuple(row for _, row in rows.iterrows())


def _selected_crystal_source(
    oxide: str,
    definition: _OxideDefinition,
    gas_pack: Any,
    crystal_source: Literal["auto", "janaf", "packaged"],
) -> tuple[str, tuple[float, float], str | None, bool]:
    packaged = _packaged_crystal_source(oxide, gas_pack)
    if crystal_source == "packaged" and packaged is None:
        raise StandardCheckRefusal(
            f"no packaged {oxide}(cr) row is available on the source-crystal axis",
            phase=f"{oxide}(cr)",
            axis="source_crystal",
        )
    if packaged is not None and crystal_source in {"auto", "packaged"}:
        row_name, bounds = packaged
        thermo = _thermo(oxide, "cr", max(bounds[0], min(definition.transition_K, bounds[1])), gas_pack)
        source_id = f"packaged:{thermo.source_row_id}"
        return row_name, bounds, source_id, True
    crystal = _janaf_crystal(oxide, definition.transition_K)
    return definition.oxide, crystal.domain_K, definition.crystal_table, False


def _thermo(oxide: str, phase: Literal["l", "cr"], temperature_K: float, gas_pack: Any):
    from openimcc.gas import species_thermo

    return species_thermo(oxide, phase, temperature_K, gas_pack)


def _gibbs_pair(
    oxide: str,
    temperature_K: float,
    gas_pack: Any,
    crystal_source_name: str,
    crystal_is_packaged: bool,
) -> tuple[float, float]:
    liquid = _thermo(oxide, "l", temperature_K, gas_pack)
    if crystal_is_packaged:
        crystal = _thermo(oxide, "cr", temperature_K, gas_pack)
        crystal_gibbs = crystal.G_J_mol
    else:
        crystal_gibbs = _janaf_crystal(oxide, temperature_K).G_J_mol
    # Premise: both standards are apparent Gibbs energies on the same 298.15 K
    # element reference. Algebra: δpack = Gpack,l − Gsource,cr. Both operands
    # are J/mol oxide, so the result remains J/mol oxide; the switch is δpack=0.
    # Sanity: changing the selected pack or crystal row changes the measured
    # residual, while identical functions give a zero residual without a shift.
    return liquid.G_J_mol, crystal_gibbs


def _axis_label(temperature_K: float | None, transition_K: float | None = None) -> str:
    if temperature_K is None:
        return "temperature_K; switch not found in the supported source/pack overlap"
    if temperature_K < 1700.0:
        label = "temperature_K; liquid side extrapolated below 1700 K (OUT_OF_DOMAIN)"
    elif temperature_K > 3000.0:
        label = "temperature_K; above 3000 K liquid-model domain (OUT_OF_DOMAIN)"
    else:
        label = "temperature_K; in the 1700–3000 K liquid-model domain"
    if transition_K is not None and temperature_K > transition_K:
        label += "; metastable/superheated crystal leg above JANAF transition"
    return label


def _switch_prediction(
    oxide: str,
    gas_pack: Any,
    crystal_source_name: str,
    crystal_is_packaged: bool,
    crystal_range: tuple[float, float],
) -> SwitchPrediction:
    from openimcc.gas import ImccGasTemperatureOutsideDomainError

    transition_K = _OXIDE_BY_NAME[oxide].transition_K
    liquid_rows = _interval_rows(gas_pack, oxide, "l")
    liquid_ranges = tuple((float(row["T_min"]), float(row["T_max"])) for row in liquid_rows)
    if not liquid_ranges:
        return SwitchPrediction(
            "out_of_range", None, None, None, None, None, None, _axis_label(None),
            f"{oxide}(l) has no packaged liquid function on temperature_K",
        )
    low = max(crystal_range[0], min(bounds[0] for bounds in liquid_ranges))
    high = min(crystal_range[1], max(bounds[1] for bounds in liquid_ranges))
    if high <= low:
        return SwitchPrediction(
            "out_of_range", None, None, None, None, None, (low, high), _axis_label(None),
            f"{oxide}(l) and {oxide}(cr) have no overlapping temperature_K range",
        )

    def delta(temp: float) -> tuple[float, float, float, float]:
        try:
            liquid_g, crystal_g = _gibbs_pair(
                oxide, temp, gas_pack, crystal_source_name, crystal_is_packaged
            )
        except ImccGasTemperatureOutsideDomainError as exc:
            raise StandardCheckRefusal(
                f"{oxide}(l) packaged row is outside temperature_K at {temp:g} K: {exc}",
                phase=f"{oxide}(l)", axis="temperature_K",
                valid_range_K=(low, high),
            ) from exc
        residual = liquid_g - crystal_g
        bound = _RESOLUTION_ULPS * _FLOAT_EPSILON * (abs(liquid_g) + abs(crystal_g))
        return residual, liquid_g, crystal_g, bound

    points = {low, high}
    first_integer = math.ceil(low)
    last_integer = math.floor(high)
    points.update(float(temp) for temp in range(first_integer, last_integer + 1, int(_ROOT_SCAN_STEP_K)))
    for row in liquid_rows:
        for key in ("T_min", "T_max"):
            boundary = float(row[key])
            if low <= boundary <= high:
                points.add(boundary)
    ordered = sorted(points)
    sampled = [(temp, *delta(temp)) for temp in ordered]
    brackets: list[tuple[float, float]] = []
    def sign(item: tuple[float, float, float, float, float]) -> int:
        if item[1] > item[4]:
            return 1
        if item[1] < -item[4]:
            return -1
        return 0

    for left, right in zip(sampled, sampled[1:]):
        if sign(left) * sign(right) == -1:
            brackets.append((left[0], right[0]))

    if not brackets:
        near = min(sampled, key=lambda item: abs(item[1]))
        if sign(near) == 0:
            near_index = ordered.index(near[0])
            neighbors = sampled[max(0, near_index - 1): near_index] + sampled[
                near_index + 1: near_index + 2
            ]
            resolved_neighbors = [item for item in neighbors if sign(item) != 0]
            resolved = (
                min(resolved_neighbors, key=lambda item: abs(item[0] - near[0]))
                if resolved_neighbors else near
            )
            unresolved = near
            while abs(unresolved[0] - resolved[0]) > SWITCH_BRACKET_TOLERANCE_K:
                mid = (unresolved[0] + resolved[0]) / 2.0
                mid_eval = delta(mid)
                midpoint = (mid, *mid_eval)
                midpoint_sign = sign(midpoint)
                if midpoint_sign == 0:
                    unresolved = midpoint
                elif midpoint_sign == sign(resolved):
                    resolved = midpoint
                else:
                    brackets.append(tuple(sorted((resolved[0], mid))))
                    break
            if not brackets:
                endpoints = sorted((resolved, unresolved), key=lambda item: item[0])
                return SwitchPrediction(
                    "resolution_indeterminate", None,
                    (endpoints[0][0], endpoints[1][0]),
                    (endpoints[0][1], endpoints[1][1]), unresolved[1], None, (low, high),
                    _axis_label(None),
                    f"endpoint residual {unresolved[1]:.9g} J/mol at {unresolved[0]:g} K is within "
                    f"its floating-point resolution bound {unresolved[4]:.9g} J/mol",
                    (endpoints[0][4], endpoints[1][4]),
                )
        else:
            return SwitchPrediction(
                "no_switch_in_range", None, None, None, None, None, (low, high), _axis_label(None),
                f"no sign-changing zero in the source/pack overlap [{low:g}, {high:g}] K",
            )

    lo, hi = brackets[0]
    left_eval, right_eval = delta(lo), delta(hi)
    flo, fhi = left_eval[0], right_eval[0]
    if sign((lo, *left_eval)) == 0 or sign((hi, *right_eval)) == 0:
        unresolved = (lo, *left_eval) if sign((lo, *left_eval)) == 0 else (hi, *right_eval)
        return SwitchPrediction(
            "resolution_indeterminate", None, (lo, hi), (flo, fhi), None, None,
            (low, high), _axis_label(None),
            f"bracket endpoint at {unresolved[0]:g} K has unresolved residual "
            f"{unresolved[1]:.9g} J/mol within bound {unresolved[4]:.9g} J/mol",
            (left_eval[3], right_eval[3]),
        )
    # Refine the sign-changing interval using resolved endpoint signs. The
    # bracket width is a temperature tolerance, not a Gibbs-energy tolerance.
    while hi - lo > SWITCH_BRACKET_TOLERANCE_K:
        mid = lo + (hi - lo) / 2.0
        mid_eval = delta(mid)
        fmid = mid_eval[0]
        mid_item = (mid, *mid_eval)
        if sign(mid_item) == 0:
            quarter = (lo + mid) / 2.0
            three_quarter = (mid + hi) / 2.0
            q_eval, tq_eval = delta(quarter), delta(three_quarter)
            q_item, tq_item = (quarter, *q_eval), (three_quarter, *tq_eval)
            if sign(q_item) * sign((lo, *left_eval)) == -1:
                hi, right_eval, fhi = quarter, q_eval, q_eval[0]
                continue
            if sign((hi, *right_eval)) * sign(tq_item) == -1:
                lo, left_eval, flo = three_quarter, tq_eval, tq_eval[0]
                continue
            if sign(q_item) * sign(tq_item) == -1:
                lo, left_eval, flo = quarter, q_eval, q_eval[0]
                hi, right_eval, fhi = three_quarter, tq_eval, tq_eval[0]
                continue
            return SwitchPrediction(
                "resolution_indeterminate", None, (lo, hi), (flo, fhi), fmid, None,
                (low, high), _axis_label(None),
                f"midpoint residual {fmid:.9g} J/mol is within its resolution bound "
                f"{mid_eval[3]:.9g} J/mol; resolved probes do not narrow the bracket",
                (left_eval[3], right_eval[3]),
            )
        if sign((lo, *left_eval)) * sign(mid_item) == -1:
            hi, right_eval, fhi = mid, mid_eval, fmid
        else:
            lo, left_eval, flo = mid, mid_eval, fmid
    root = lo + (hi - lo) / 2.0
    residual, liquid_g, crystal_g, root_resolution = delta(root)
    left_room, right_room = root - low, high - root
    central_step = min(_SLOPE_STEP_K, left_room, right_room)
    if left_room >= _SLOPE_STEP_K and right_room >= _SLOPE_STEP_K:
        slope = abs(
            (delta(root + central_step)[0] - delta(root - central_step)[0])
            / (2.0 * central_step)
        )
    elif left_room >= right_room and left_room > 0.0:
        # Near a tabulated endpoint, estimate the local slope from the in-range
        # side only; this does not evaluate or imply an extrapolated crystal.
        slope_step = min(_SLOPE_STEP_K, left_room)
        slope = abs((residual - delta(root - slope_step)[0]) / slope_step)
    elif right_room > 0.0:
        slope_step = min(_SLOPE_STEP_K, right_room)
        slope = abs((delta(root + slope_step)[0] - residual) / slope_step)
    else:
        slope = 0.0
    resolution = root_resolution
    residual_uncertainty = (abs(residual) + resolution) / slope if slope > 0.0 else math.inf
    bracket_uncertainty = hi - lo
    uncertainty = max(bracket_uncertainty, residual_uncertainty)
    if not math.isfinite(uncertainty) or uncertainty > SWITCH_BRACKET_TOLERANCE_K:
        return SwitchPrediction(
            "resolution_indeterminate", None, (lo, hi), (flo, fhi), residual, uncertainty,
            (low, high), _axis_label(None),
            f"root uncertainty {uncertainty:g} K exceeds the {SWITCH_BRACKET_TOLERANCE_K:g} K temperature tolerance",
            (left_eval[3], right_eval[3]),
        )
    return SwitchPrediction(
        "root", root, (lo, hi), (flo, fhi), residual, uncertainty, (low, high),
        _axis_label(root, transition_K), None, (left_eval[3], right_eval[3]),
    )


def pure_oxide_standard_check(
    oxide: str,
    temperature_K: float | None = None,
    *,
    condensate_path: str | Path | None = None,
    crystal_source: Literal["auto", "janaf", "packaged"] = "auto",
) -> PureOxideStandardCheck:
    """Return separate JANAF-pair, packaged-closure, and switch results.

    ``temperature_K`` defaults to the oxide's JANAF transition (or the selected
    K2O construction temperature). The condensate file and crystal-source mode
    are independent inputs. ``auto`` uses a crystal row carried by that exact
    condensate table when present, otherwise the named JANAF crystal row.
    """
    from openimcc.gas import (
        ImccGasTemperatureOutsideDomainError,
        default_condensate_database_path,
        default_gas_database_path,
        load_gas_datapack,
    )
    from openimcc.liquid_reference import liquid_reference_authority

    definition = _OXIDE_BY_NAME.get(str(oxide))
    if definition is None:
        raise ValueError(f"unsupported pure parent oxide {oxide!r}")
    temp = definition.transition_K if temperature_K is None else float(temperature_K)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    if crystal_source not in {"auto", "janaf", "packaged"}:
        raise ValueError("crystal_source must be 'auto', 'janaf', or 'packaged'")
    path = Path(condensate_path) if condensate_path is not None else default_condensate_database_path()
    gas_pack = load_gas_datapack(default_gas_database_path(), path)
    crystal_name, crystal_range, crystal_id, crystal_is_packaged = _selected_crystal_source(
        definition.oxide, definition, gas_pack, crystal_source
    )
    source_crystal_id = crystal_id or crystal_name

    if definition.liquid_table is None:
        # K2O has no JANAF liquid partner, but its crystal table still defines
        # the source range for this construction case.
        _janaf_crystal(definition.oxide, temp)
        pair_status: Literal["available", "construction_case"] = "construction_case"
        delta_janaf = None
        janaf_rows = (definition.crystal_table,)
    else:
        pair_status = "available"
        liquid_janaf = _janaf_liquid(definition.liquid_table, temp)
        crystal_janaf = _janaf_crystal(definition.oxide, temp)
        delta_janaf = (liquid_janaf.G_J_mol - crystal_janaf.G_J_mol) / _J_TO_KJ
        janaf_rows = (definition.crystal_table, definition.liquid_table)

    liquid_source = ""
    liquid_range: tuple[float, float] | None = None
    packaged_closure: PackagedClosure
    crystal_source_display = source_crystal_id
    try:
        liquid = _thermo(definition.oxide, "l", temp, gas_pack)
        liquid_source = f"{liquid.source_table_id or liquid.source_row_id} ({liquid.source_row_id})"
        liquid_range = (liquid.T_min, liquid.T_max)
        if crystal_is_packaged:
            crystal = _thermo(definition.oxide, "cr", temp, gas_pack)
            crystal_g = crystal.G_J_mol
            source_crystal_id = f"packaged:{crystal.source_row_id}"
            crystal_range = (crystal.T_min, crystal.T_max)
            crystal_source_display = source_crystal_id
        else:
            crystal_g = _janaf_crystal(definition.oxide, temp).G_J_mol
            crystal_source_display = source_crystal_id
        packaged_closure = PackagedClosure(
            "available", (liquid.G_J_mol - crystal_g) / _J_TO_KJ,
            crystal_source_display, crystal_range,
            f"{liquid.source_table_id or liquid.source_row_id} ({liquid.source_row_id})",
            liquid_range,
        )
    except (ImccGasTemperatureOutsideDomainError, StandardCheckRefusal) as exc:
        message = str(exc)
        packaged_closure = PackagedClosure(
            "out_of_range", None, source_crystal_id, crystal_range,
            liquid_source, liquid_range, message,
        )

    switch = _switch_prediction(
        definition.oxide, gas_pack, crystal_name, crystal_is_packaged, crystal_range,
    )

    authority_record = None
    if liquid_range is not None and liquid_source:
        selected_source_id = liquid_source.split(" (", 1)[0]
        try:
            authority_record = liquid_reference_authority(
                f"{definition.oxide}(l)", temp,
                selected_source_id=selected_source_id,
            )
        except KeyError:
            authority_record = None
    if authority_record is not None:
        authority = authority_record.authority.value
        least_authoritative = f"packaged liquid standard {liquid_source}"
    elif liquid_source:
        authority = "source-declared; no separate authority record for the selected row"
        least_authoritative = f"packaged liquid standard {liquid_source}"
    elif crystal_is_packaged:
        authority = "source-declared; packaged liquid row is outside the requested temperature"
        least_authoritative = f"packaged source crystal {source_crystal_id}"
    else:
        authority = "machine-transcribed JANAF source data"
        least_authoritative = f"JANAF source crystal {source_crystal_id}"
    if definition.construction:
        authority = f"constructed liquid standard; {authority}"
        least_authoritative = f"constructed liquid standard {liquid_source or 'K2O(l)'}"

    crystal_digest = (
        str(_janaf_crystal(definition.oxide, min(temp, crystal_range[1])).provenance["source_sha256"])
        if not crystal_is_packaged
        else gas_pack.condensate_table_digest
    )
    return PureOxideStandardCheck(
        oxide=definition.oxide,
        temperature_K=temp,
        janaf_transition_K=definition.transition_K,
        janaf_pair_status=pair_status,
        delta_janaf_kJ_mol=delta_janaf,
        packaged_closure=packaged_closure,
        switch_prediction=switch,
        janaf_rows=janaf_rows,
        pack_id=path.name,
        pack_digest=gas_pack.condensate_table_digest,
        source_crystal_id=source_crystal_id,
        source_crystal_digest=crystal_digest,
        least_authoritative_input=least_authoritative,
        authority=authority,
        construction_flag=definition.construction,
    )


def pure_oxide_standard_checks(
    condensate_path: str | Path | None = None,
) -> tuple[PureOxideStandardCheck, ...]:
    """Evaluate all eight supported pure-oxide standard checks at their anchors."""
    return tuple(
        pure_oxide_standard_check(
            row.oxide, row.transition_K, condensate_path=condensate_path
        )
        for row in _OXIDES
    )


def _cmd_diagnose_standard_checks(args: Any) -> int:
    """CLI handler registered by ``openimcc.cli`` for the new diagnostic."""
    import csv

    rows = pure_oxide_standard_checks(args.condensate)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pure-oxide-standard-checks.csv"
    fields = (
        "oxide", "temperature_K", "janaf_transition_K", "janaf_pair_status",
        "delta_janaf_kJ_mol", "packaged_closure_delta_kJ_mol",
        "packaged_closure_status", "switch_status", "switch_temperature_K",
        "switch_bracket_K", "switch_residual_J_mol", "switch_uncertainty_K",
        "switch_bracket_residuals_J_mol", "switch_bracket_resolution_bounds_J_mol",
        "switch_search_range_K", "temperature_axis_label", "janaf_rows",
        "pack_id", "pack_digest", "source_crystal_id", "source_crystal_digest",
        "least_authoritative_input", "authority", "construction_flag",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            closure = data.pop("packaged_closure")
            switch = data.pop("switch_prediction")
            writer.writerow({
                **data,
                "packaged_closure_delta_kJ_mol": closure["delta_kJ_mol"],
                "packaged_closure_status": closure["status"],
                "switch_status": switch["status"],
                "switch_temperature_K": switch["temperature_K"],
                "switch_bracket_K": json.dumps(switch["bracket_K"]),
                "switch_bracket_residuals_J_mol": json.dumps(switch["bracket_residuals_J_mol"]),
                "switch_bracket_resolution_bounds_J_mol": json.dumps(switch["bracket_resolution_bounds_J_mol"]),
                "switch_residual_J_mol": switch["residual_J_mol"],
                "switch_uncertainty_K": switch["uncertainty_K"],
                "switch_search_range_K": json.dumps(switch["search_range_K"]),
                "temperature_axis_label": switch["temperature_axis_label"],
                "janaf_rows": json.dumps(data["janaf_rows"]),
            })
    print(f"wrote {csv_path}")
    return 0


__all__ = [
    "JANAF_PAIR_TOLERANCE_KJ_MOL",
    "PACKAGED_SIO2_ROUNDING_TOLERANCE_KJ_MOL",
    "SWITCH_BRACKET_TOLERANCE_K",
    "SWITCH_SOLVER_XTOL_K",
    "PackagedClosure",
    "PureOxideStandardCheck",
    "StandardCheckRefusal",
    "SwitchPrediction",
    "pure_oxide_standard_check",
    "pure_oxide_standard_checks",
]
