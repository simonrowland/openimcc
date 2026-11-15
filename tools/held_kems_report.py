"""Compare fixed-activity vapour predictions with crystalline-source KEMS rows.

The input is a public literature extract YAML file. PyYAML is the optional
parser dependency used by the benchmark tools.
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on the optional tools extra
    raise SystemExit("held_kems_report.py requires the optional PyYAML dependency") from exc

from openimcc import gas as gas_module
from openimcc.equilibrium import equilibrate
from openimcc.gas import load_gas_datapack
from openimcc.model import evaluate, load_datapack


_STANDARD_PRESSURE_PA = 100_000.0
_LIQUID_DOMAIN_K = (1700.0, 3000.0)
_WOLLASTONITE_SOURCE_ID = "shornikov-2026-wollastonite-kems"
_WOLLASTONITE_SOLID_SOURCE_LIMIT_K = 1817.0


@lru_cache(maxsize=1)
def _melt_pack() -> Any:
    return load_datapack()


def _unwrap(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if value.get("tag") == "value":
        return _unwrap(value.get("value"))
    if value.get("kind") == "point":
        return _unwrap(value.get("point"))
    if "state" in value:
        return _unwrap(value["state"])
    if "value" in value:
        return _unwrap(value["value"])
    return value


def _composition(row: dict[str, Any]) -> dict[str, float] | None:
    raw = _unwrap(row.get("identity", {}).get("composition"))
    if not isinstance(raw, dict):
        raw = _unwrap(row.get("point_conditions", {}).get("composition"))
    components = raw.get("components") if isinstance(raw, dict) else None
    if not isinstance(components, list) or not components:
        return None
    try:
        values = {str(name): float(amount) for name, amount in components}
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        return None
    if not set(values).issubset(_melt_pack().parent_oxides):
        return None
    # Equilibrium is invariant to a common bulk scale; normalize mixed source
    # mole fractions and formula inventories onto one mole of parent oxides.
    total = math.fsum(values.values())
    return {name: value / total for name, value in values.items()}


def _temperature(row: dict[str, Any]) -> float | None:
    value = _unwrap(row.get("identity", {}).get("temperature_K"))
    if value is None:
        value = _unwrap(row.get("point_conditions", {}).get("temperature_K"))
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        return None
    return temperature if math.isfinite(temperature) else None


def _numeric_point(row: dict[str, Any]) -> float | None:
    try:
        value = _unwrap(row.get("value"))
        point = float(value)
    except (TypeError, ValueError):
        return None
    return point if math.isfinite(point) else None


def _oxygen_pressure_pa(row: dict[str, Any]) -> float | None:
    value = _unwrap(row.get("identity", {}).get("fO2_Pa"))
    if value is None:
        value = _unwrap(row.get("point_conditions", {}).get("fO2_Pa"))
    try:
        pressure = float(value)
    except (TypeError, ValueError):
        return None
    return pressure if math.isfinite(pressure) and pressure > 0.0 else None


def _composition_key(composition: dict[str, float]) -> tuple[tuple[str, float], ...]:
    # Rounding only coalesces repeated YAML encodings of the same composition;
    # the unrounded values remain the inputs to each thermodynamic solve.
    return tuple(sorted((name, round(value, 12)) for name, value in composition.items()))


def _solid_source(row: dict[str, Any], source_id: str, temperature_K: float) -> bool:
    provenance = row.get("provenance") or {}
    if provenance.get("measured_condensed_form") == "solid":
        return True
    sample_phase = str(provenance.get("sample_phase", "")).lower()
    if "solid" in sample_phase:
        return True
    # The extract identifies the initial charge as stoichiometric wollastonite,
    # but leaves its pointwise measurement-state composition unstated. Keep
    # only temperatures through the packaged crystal-source limit (1817 K).
    return (
        source_id == _WOLLASTONITE_SOURCE_ID
        and temperature_K <= _WOLLASTONITE_SOLID_SOURCE_LIMIT_K
    )


def _row_label(row: dict[str, Any]) -> str:
    observation_id = str(row.get("observation_id", "unknown"))
    source_id = str(row.get("source_id", "unknown"))
    suffix = observation_id.split("::")[-1].split(":T=", 1)[0]
    return f"{source_id}::{suffix}"


def _observed_bar(row: dict[str, Any]) -> str:
    point = _numeric_point(row)
    if point is None:
        value = row.get("value") or {}
        reason = str(value.get("unavailable_reason", "unavailable"))
        return "not detected" if "not detected" in reason.lower() else "unavailable"
    unit = str((row.get("derivation") or {}).get("output_unit", "Pa")).lower()
    if unit in {"pa", "pascal", "pascals"}:
        # evaluate_gas reports p/p°; with p°=1 bar, 1 bar=100000 Pa.
        point /= _STANDARD_PRESSURE_PA
    elif unit != "bar":
        return f"{point:g} {unit}"
    return f"{point:.6g}"


def _pressure(result: Any, species: str) -> str:
    if species in result.gas_depends:
        return "DEPENDS"
    if result.gas is None or species not in result.gas:
        return "—"
    return f"{result.gas[species]:.6g}"


def _all_liquid_pressures(
    composition: dict[str, float],
    temperature_K: float,
    fO2: float,
    gas_pack: Any,
) -> dict[str, float] | None:
    try:
        melt = evaluate(composition, temperature_K, _melt_pack(), basis_type="mol")
        activities = {
            oxide: melt.activity(oxide)
            for oxide in composition
        }
        return dict(gas_module.evaluate_gas(
            activities,
            temperature_K,
            fO2,
            gas_pack,
            parent_oxides=tuple(activities),
        ))
    except gas_module.ImccGasTemperatureOutsideDomainError:
        return None


def _phase_state(result: Any) -> str:
    fraction = result.liquid_mass_fraction
    present = sorted(
        phase for phase, status in result.phase_presence.items()
        if status == "present"
    )
    if fraction is None:
        state = "unresolved"
    elif fraction == 1.0:
        state = "all-liquid"
    elif fraction == 0.0:
        state = "fully-solid" + (f" ({','.join(present)})" if present else "")
    else:
        state = f"partial-liquid={fraction:.6g}"
        if present:
            state += f" ({','.join(present)})"
    if result.labels.validity != "IN":
        state += f" / {result.labels.validity}"
    return state


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("observations"), list):
        raise ValueError(f"{path}: expected a battery_observations YAML extract")
    return document["observations"]


def report(paths: list[Path]) -> None:
    gas_pack = load_gas_datapack()
    condition_count = 0
    skipped = 0
    for path in paths:
        rows = _read_rows(path)
        oxygen_by_condition: dict[tuple[float, tuple[tuple[str, float], ...]], float] = {}
        for row in rows:
            quantity = _unwrap(row.get("identity", {}).get("quantity"))
            species = (row.get("identity", {}).get("species") or {}).get("formula")
            if quantity != "p_partial" or species != "O2":
                continue
            temperature_K = _temperature(row)
            composition = _composition(row)
            pressure_pa = _numeric_point(row)
            if temperature_K is not None and composition and pressure_pa and pressure_pa > 0.0:
                oxygen_by_condition[(temperature_K, _composition_key(composition))] = pressure_pa

        grouped: dict[
            tuple[float, tuple[tuple[str, float], ...], float],
            tuple[dict[str, float], list[dict[str, Any]]],
        ] = {}
        for row in rows:
            quantity = _unwrap(row.get("identity", {}).get("quantity"))
            species = (row.get("identity", {}).get("species") or {}).get("formula")
            if quantity != "p_partial" or not species or species == "O2":
                continue
            temperature_K = _temperature(row)
            composition = _composition(row)
            if (
                temperature_K is None
                or composition is None
                or not _LIQUID_DOMAIN_K[0] <= temperature_K <= _LIQUID_DOMAIN_K[1]
                or not _solid_source(row, str(row.get("source_id", "")), temperature_K)
            ):
                skipped += 1
                continue
            oxygen_pa = _oxygen_pressure_pa(row)
            if oxygen_pa is None:
                oxygen_pa = oxygen_by_condition.get(
                    (temperature_K, _composition_key(composition))
                )
            observed = row.get("value") or {}
            if oxygen_pa is None or (
                observed.get("kind") not in {"point", "unavailable"}
            ):
                skipped += 1
                continue
            # fO2 is the dimensionless pO2/p° input expected by evaluate_gas;
            # the extracted oxygen pressures are in Pa and p° is 1 bar.
            fO2 = oxygen_pa / _STANDARD_PRESSURE_PA
            key = (temperature_K, _composition_key(composition), fO2)
            if key not in grouped:
                grouped[key] = (composition, [])
            grouped[key][1].append(row)

        for (temperature_K, composition_key, fO2), (composition, observations) in grouped.items():
            state = equilibrate(composition, temperature_K, fO2=fO2)
            all_liquid = _all_liquid_pressures(
                composition, temperature_K, fO2, gas_pack,
            )
            condition_count += 1
            for row in observations:
                species = str(row["identity"]["species"]["formula"])
                eq_pressure = _pressure(state, species)
                liquid_pressure = (
                    "OUT_OF_DOMAIN" if all_liquid is None else
                    f"{all_liquid[species]:.6g}" if species in all_liquid else "—"
                )
                print(
                    f"{_row_label(row)} | {temperature_K:g} | {fO2:.6g} | "
                    f"{_phase_state(state)} | {eq_pressure} | {liquid_pressure} | "
                    f"{_observed_bar(row)}"
                )
    if not condition_count:
        print("No rows had a supported composition, crystalline source, and stated oxygen condition.")
    print(f"# eligible conditions: {condition_count}; skipped observations: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extract", nargs="+", type=Path, help="public literature extract YAML path(s)")
    args = parser.parse_args()
    print("row | T_K | fO2 | phase_state | vapour_equilibrium_bar | vapour_all_liquid_bar | observed_bar")
    print("--- | ---: | ---: | --- | ---: | ---: | ---:")
    report(args.extract)


if __name__ == "__main__":
    main()
