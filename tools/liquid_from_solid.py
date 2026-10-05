"""Construct apparent liquid Gibbs functions from crystal JANAF tables."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _number(cell: Any) -> float | None:
    if isinstance(cell, Mapping):
        cell = cell.get("value")
    if cell is None:
        return None
    return float(cell)


def _crystal_rows(crystal_table: Mapping[str, Any]) -> list[dict[str, float]]:
    table = crystal_table.get("table", crystal_table)
    rows = []
    for source in table["values"]:
        temperature = _number(source.get("temperature"))
        cp = _number(source.get("heat_capacity"))
        entropy = _number(source.get("entropy"))
        enthalpy_increment = _number(source.get("enthalpy_increment"))
        formation_enthalpy = _number(source.get("formation_enthalpy"))
        if None in (temperature, cp, entropy, enthalpy_increment):
            continue
        row = {
            "temperature": temperature,
            "heat_capacity": cp,
            "entropy": entropy,
            "enthalpy_increment": enthalpy_increment,
        }
        if formation_enthalpy is not None:
            row["formation_enthalpy"] = formation_enthalpy
        rows.append(row)
    return sorted(rows, key=lambda row: row["temperature"])


def _crystal_state(
    rows: Sequence[dict[str, float]], temperature: float
) -> tuple[float, float, float]:
    """Return crystal H-H(298), S, and Cp at a temperature.

    Premise: JANAF gives H-H(298), S, and Cp at tabulated temperatures.
    Algebra: between adjacent nodes Cp is linear; integrating Cp gives the
    enthalpy increment, and integrating Cp/T gives entropy. Unit check: the
    first integral is J/mol and is divided by 1000 before adding to the JANAF
    kJ/mol increment. Sanity: at a table node this returns the printed H, S,
    and Cp exactly.
    """
    exact = next(
        (row for row in rows if row["temperature"] == temperature), None
    )
    if exact is not None:
        return (
            exact["enthalpy_increment"],
            exact["entropy"],
            exact["heat_capacity"],
        )

    lower = next(
        (row for row in reversed(rows) if row["temperature"] < temperature), None
    )
    upper = next(
        (row for row in rows if row["temperature"] > temperature), None
    )
    if lower is None or upper is None:
        raise ValueError(f"crystal table does not bracket {temperature:g} K")

    t0 = lower["temperature"]
    t1 = upper["temperature"]
    cp0 = lower["heat_capacity"]
    cp1 = upper["heat_capacity"]
    slope = (cp1 - cp0) / (t1 - t0)
    delta_t = temperature - t0
    log_ratio = math.log(temperature / t0)
    enthalpy_increment = lower["enthalpy_increment"] + (
        cp0 * delta_t + 0.5 * slope * delta_t**2
    ) / 1000.0
    entropy = lower["entropy"] + cp0 * log_ratio + slope * (
        delta_t - t0 * log_ratio
    )
    heat_capacity = cp0 + slope * delta_t
    return enthalpy_increment, entropy, heat_capacity


@dataclass(frozen=True)
class LiquidPoint:
    temperature_k: float
    gibbs_kj_mol: float
    branch: str


@dataclass(frozen=True)
class LiquidConstruction:
    species: str
    inputs: Mapping[str, Any]
    dfh298_kj_mol: float
    crystal_enthalpy_increment_at_fusion_kj_mol: float
    crystal_entropy_at_fusion_j_mol_k: float
    crystal_cp_at_fusion_j_mol_k: float
    _crystal_rows: tuple[dict[str, float], ...] = field(repr=False, compare=False)

    @property
    def fusion_temperature_k(self) -> float:
        return float(self.inputs["fusion_temperature"]["value"])

    @property
    def fusion_entropy_j_mol_k(self) -> float:
        return float(self.inputs["fusion_entropy"]["value"])

    @property
    def liquid_cp_j_mol_k(self) -> float:
        return float(self.inputs["liquid_heat_capacity"]["value"])

    @property
    def diagnostics(self) -> dict[str, Any]:
        delta_cp = self.liquid_cp_j_mol_k - self.crystal_cp_at_fusion_j_mol_k
        return {
            "fusion_delta_cp_j_mol_k": delta_cp,
            "negative_fusion_delta_cp": delta_cp < 0.0,
            "negative_fusion_delta_cp_note": (
                "Liquid Cp is below crystal Cp at fusion; check whether the "
                "crystal high-temperature Cp extrapolation runs high or the "
                "liquid Cp estimate runs low."
                if delta_cp < 0.0
                else None
            ),
        }

    def at(self, temperature_k: float) -> LiquidPoint:
        """Evaluate G_app, labelling a below-fusion liquid as supercooled."""
        temperature = float(temperature_k)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        tfus = self.fusion_temperature_k
        cp = self.liquid_cp_j_mol_k
        delta_s = self.fusion_entropy_j_mol_k
        h_increment = (
            self.crystal_enthalpy_increment_at_fusion_kj_mol
            + tfus * delta_s / 1000.0
            + cp * (temperature - tfus) / 1000.0
        )
        entropy = (
            self.crystal_entropy_at_fusion_j_mol_k
            + delta_s
            + cp * math.log(temperature / tfus)
        )
        # Premise: apparent JANAF Gibbs energy uses the crystal's ΔfH°298
        # anchor. Algebra: G_app = ΔfH°298 + [H-H(298)] - T*S. Unit check:
        # H is kJ/mol, S is J/(mol K), so TS is divided by 1000. Sanity: at
        # T_fus the construction adds T_fus*ΔS_fus to H and ΔS_fus to S.
        gibbs = self.dfh298_kj_mol + h_increment - temperature * entropy / 1000.0
        branch = "liquid" if temperature >= tfus else "supercooled_extrapolation"
        return LiquidPoint(temperature, gibbs, branch)

    def input_spread_kj_mol(self, temperature_k: float) -> float:
        """Propagate input spreads using a conservative first-order bound.

        Premise: the selected fusion entropy and liquid heat capacity have
        independent bounded spreads. Algebra: the sensitivities of G to those
        inputs are (T_fus-T)/1000 and ((T-T_fus)-T*ln(T/T_fus))/1000.
        Unit check: multiplying either sensitivity by J/(mol K) gives kJ/mol.
        Sanity: both contributions vanish at T_fus, where the fusion jump
        cancels from G by construction.
        """
        temperature = float(temperature_k)
        tfus = self.fusion_temperature_k
        ds = self.fusion_entropy_j_mol_k
        cp = self.liquid_cp_j_mol_k
        fusion_spread = float(self.inputs["fusion_temperature"].get("spread", 0.0))
        entropy_spread = float(self.inputs["fusion_entropy"].get("spread", 0.0))
        cp_spread = float(self.inputs["liquid_heat_capacity"].get("spread", 0.0))

        spread = abs((tfus - temperature) * entropy_spread / 1000.0)
        spread += abs(
            cp_spread
            * ((temperature - tfus) - temperature * math.log(temperature / tfus))
            / 1000.0
        )
        if fusion_spread:
            central = self.at(temperature).gibbs_kj_mol
            for shifted in (tfus - fusion_spread, tfus + fusion_spread):
                if shifted <= 0.0:
                    continue
                h_cr, s_cr, _ = _crystal_state(
                    self._crystal_rows, shifted
                )
                h = (
                    h_cr
                    + shifted * ds / 1000.0
                    + cp * (temperature - shifted) / 1000.0
                )
                s = s_cr + ds + cp * math.log(temperature / shifted)
                shifted_g = self.dfh298_kj_mol + h - temperature * s / 1000.0
                spread += abs(shifted_g - central)
        return spread

    def band_kj_mol(
        self, temperature_k: float, measured_tier_band_kj_mol: float
    ) -> float:
        """Add the measured tier band to this row's propagated input spread."""
        return float(measured_tier_band_kj_mol) + self.input_spread_kj_mol(
            temperature_k
        )

    def source_rows(
        self, temperatures_k: Sequence[float]
    ) -> dict[float, dict[str, float]]:
        """Return H-H(298) and S nodes for the existing condensate fitter.

        The source liquid enthalpy node is the crystal increment at fusion,
        plus T_fus*ΔS_fus, plus the integrated constant liquid Cp. Its entropy
        node adds ΔS_fus and Cp*ln(T/T_fus). The returned units match the
        fitter's JANAF inputs: kJ/mol and J/(mol K).
        """
        rows = {}
        for temperature in temperatures_k:
            h_increment = (
                self.crystal_enthalpy_increment_at_fusion_kj_mol
                + self.fusion_temperature_k * self.fusion_entropy_j_mol_k / 1000.0
                + self.liquid_cp_j_mol_k
                * (temperature - self.fusion_temperature_k)
                / 1000.0
            )
            entropy = (
                self.crystal_entropy_at_fusion_j_mol_k
                + self.fusion_entropy_j_mol_k
                + self.liquid_cp_j_mol_k
                * math.log(temperature / self.fusion_temperature_k)
            )
            rows[float(temperature)] = {
                "enthalpy_increment": h_increment,
                "entropy": entropy,
            }
        return rows

    def fit_condensate_row(self) -> dict[str, str]:
        """Fit generated liquid nodes with the existing condensate fitter."""
        fit = self.inputs["condensate_fit"]
        from tools.build_gas_tables import _fit_condensate_values

        t_min = float(fit["fit_t_min"])
        t_max = float(fit["fit_t_max"])
        if t_min % 100.0 or t_max % 100.0:
            raise ValueError("condensate fit endpoints must lie on the 100 K grid")
        nodes = [float(t) for t in range(int(t_min), int(t_max) + 1, 100)]
        return _fit_condensate_values(
            f"{self.species}(l)",
            str(fit["table_id"]),
            str(fit["cation"]),
            int(fit["cat_num"]),
            int(fit["oxy_num"]),
            self.source_rows(nodes),
            self.dfh298_kj_mol,
            fit_t_min=t_min,
            fit_t_max=t_max,
            runtime_t_min=float(fit["runtime_t_min"]),
            runtime_t_max=float(fit["runtime_t_max"]),
            ref=str(fit["ref"]),
        )


def liquid_from_solid(
    species: str, crystal_table: Mapping[str, Any], inputs: Mapping[str, Any]
) -> LiquidConstruction:
    """Build a liquid Gibbs function from one crystal table and tiered inputs."""
    rows = _crystal_rows(crystal_table)
    if not rows:
        raise ValueError(f"{species}: crystal table has no usable thermal rows")
    tfus = float(inputs["fusion_temperature"]["value"])
    h_cr, s_cr, cp_cr = _crystal_state(rows, tfus)
    reference = next(
        row.get("formation_enthalpy")
        for row in rows
        if row["temperature"] == 298.15 and "formation_enthalpy" in row
    )
    result = LiquidConstruction(
        species=species,
        inputs=inputs,
        dfh298_kj_mol=float(reference),
        crystal_enthalpy_increment_at_fusion_kj_mol=h_cr,
        crystal_entropy_at_fusion_j_mol_k=s_cr,
        crystal_cp_at_fusion_j_mol_k=cp_cr,
        _crystal_rows=tuple(rows),
    )
    return result
