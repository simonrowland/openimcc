"""IMCC-only empirical residual runner.

Run a tracked ``melt-activity-bench.v1`` set against one IMCC datapack and
print per-point residuals. This is not a replacement for
``benchmarks/melt_activity_benchmark.py`` (multi-engine comparison, coverage
maps, latch detection). It is the small tool: one pack, the tracked points,
the residuals.

Basis (must match the harness; a mismatched basis is silently wrong):

- ``activity`` and ``activity_coefficient`` compare IMCC parent-formula
  values directly. Gamma is ``a/x`` on the parent-oxide formula-unit basis.
- ``partial_pressure`` passes those parent-formula activities to
  ``openimcc.gas.evaluate_gas`` and converts its bar result to the Pa basis of
  the tracked measurements. Activity and pressure are never coerced into each
  other.
- Residual is ``log10(predicted/measured)`` (the bench-set fair-comparison
  convention). ``ratio`` is the linear ``predicted/measured``.

Engine status mapping reuses ``ImccEngine.evaluate`` semantics in
``benchmarks/melt_activity_benchmark.py``. Refusals are counted data; they
never enter RMSE.

Subcommand hook (no package CLI module; call this)::

    openimcc-bench BENCH_SET PACK
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml

from openimcc import (
    ImccCompositionOutsideValidatedEnvelopeError,
    ImccComponentOutsideDomainError,
    ImccCompositionIncompleteError,
    ImccDatapack,
    ImccFerricInputUnsupportedError,
    ImccLoadedDatapack,
    ImccMalformedDatapackError,
    ImccNonconvergenceError,
    ImccRefusal,
    ImccTOutsideDatapackDomainError,
    evaluate,
    label_research_datapack,
    load_datapack,
)
from openimcc.model import DEFAULT_MELT_PACK_NAME
from openimcc.json_utils import reject_duplicate_json_keys

_YAML_BASE_LOADER = yaml.SafeLoader


class _BenchYAML12SafeLoader(_YAML_BASE_LOADER):
    """Safe loader with YAML 1.2 boolean and integer scalars.

    YAML 1.1 resolves bare ``NO``/``yes``/``on``/``off`` to booleans, which
    would turn chemistry identifiers such as ``species: NO`` (nitric oxide)
    or a composition id ``no`` into ``False``. YAML 1.1 also treats leading
    zero, hexadecimal, and sexagesimal forms as integers. Copy the resolver
    table, then add only YAML 1.2 bool spellings and decimal integers.
    """

    yaml_implicit_resolvers = {
        key: [
            (tag, regexp)
            for tag, regexp in resolvers
            if tag != "tag:yaml.org,2002:bool"
        ]
        for key, resolvers in _YAML_BASE_LOADER.yaml_implicit_resolvers.items()
    }


_BenchYAML12SafeLoader.yaml_implicit_resolvers = {
    key: [
        (tag, regexp)
        for tag, regexp in resolvers
        if tag != "tag:yaml.org,2002:int"
    ]
    for key, resolvers in _BenchYAML12SafeLoader.yaml_implicit_resolvers.items()
}

_BenchYAML12SafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)
_BenchYAML12SafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^[-+]?(?:0|[1-9][0-9_]*)$"),
    list("-+0123456789"),
)

_BENCH_POINT_KEYS = frozenset(
    {
        "activity_basis_conversion",
        "as_published_tridymite_activity",
        "composition_id",
        "composition_wt_pct",
        "convention",
        "dropped_reason",
        "evidence_class",
        "extrapolation_flag",
        "extrapolation_note",
        "fO2_bar",
        "fO2_condition",
        "fO2_note",
        "id",
        "material_class",
        "measured",
        "measurement_technique",
        "observable",
        "parent_oxide",
        "population",
        "provenance",
        "published_a_AlO1.5",
        "published_coefficients",
        "published_mass_pct",
        "published_mole_fraction",
        "reduction_class",
        "score",
        "scoring_status",
        "source_evidence_class",
        "species",
        "standard_state_conversion",
        "temperature_K",
        "uncertainty",
        "units",
    }
)

# Bare pack names that resolve to the packaged resource when no file of that
# name exists relative to the working directory.
_PACKAGED_PACK_NAMES = frozenset(
    {
        "imcc-sf04-v1.0.2.json",
        "imcc-sf04-ext-v4.json",
        "imcc-sf04-d066-v1.json",
        "imcc-sf04-d066-v2.json",
        "imcc-sf04-d066-v3.json",
        DEFAULT_MELT_PACK_NAME,
        "imcc-sf04-d066-v5.json",
        "imcc-sf04-d066-v1-kcaalsi2o7.json",
        "imcc-sf04-d066-ext-v1.json",
    }
)

POINT_STATUSES = (
    "ok",
    "out_of_domain",
    "refused",
    "not_converged",
    "unsupported_observable",
)
_SUPPORTED_OBSERVABLES = frozenset(
    {"activity", "activity_coefficient", "partial_pressure"}
)
_ID_ENCODED_XTOKEN = re.compile(r"_x\d{3,}(?:_|$)")
BAR_TO_PA = 1.0e5
"""Pressure conversion used at the bench boundary: 1 bar = 100,000 Pa."""
_BINARY_POPULATIONS = frozenset(
    {"tsaplin2000_kems_na2o_sio2", "yamaguchi1983_emf_na2o_sio2"}
)
_KUME_LIQUID_ACTIVITY_WARNING = (
    "unconverted solid standard state; not an accuracy figure for liquid activities"
)
_CONVERTED_LIQUID_STANDARD_STATE = (
    "pure-liquid (converted from pure-solid; Gibbs conversion v2)"
)
_CONTROLLED_TEXT_SEPARATORS = str.maketrans({"-": " ", "_": " "})


def _controlled_text(value: Any) -> str:
    return " ".join(str(value or "").lower().translate(_CONTROLLED_TEXT_SEPARATORS).split())


_STANDARD_STATE_BY_CONVENTION = {
    # Each complete normalized convention and observable pair occurs in the
    # tracked basalt benchmark set; free-text variants remain unspecified.
    (_controlled_text("Printed activity against the authors' pure-solid oxide standard state (column headers a SiO2(s), a CaO(s), a AlO1.5(s), a MgO(s)). Entered as printed. Alumina is stored on the Al2O3 formula-unit basis as a(AlO1.5)^2; the printed a(AlO1.5) is retained in published_a_AlO1.5. No solid-to-liquid conversion is applied."), "activity"): "pure-solid",
    (_controlled_text("a(Na2O) on the parent-oxide formula-unit basis against a pure-liquid-Na2O standard state, evaluated from the published authors-own least-squares coefficients and retained without proxy conversion. No single-cation conversion or square root applied."), "activity"): "pure-liquid",
    (_controlled_text("Parent-oxide formula-unit activity. The as-published Tridymite-reference value is preserved above; measured is the controller-directed pure-liquid reference conversion. No single-cation conversion or square root applied."), "activity"): "pure-liquid",
    (_controlled_text("a(SiO2) on the parent-oxide formula-unit basis against a pure-liquid-SiO2 standard state, entered exactly as published with no single-cation conversion."), "activity"): "pure-liquid",
    (_controlled_text("a(Na2O) on the parent-oxide formula-unit basis against a pure-liquid-Na2O standard state, entered exactly as published and retained without proxy conversion."), "activity"): "pure-liquid",
    (_controlled_text("measured K(g); common analytical gas layer; IW+2.4 fO2 pin"), "partial_pressure"): "pure-liquid parent",
    (_controlled_text("measured SiO(g); common analytical gas layer; IW+2.4 fO2 pin"), "partial_pressure"): "pure-liquid parent",
    (_controlled_text("gamma_Mg on Berman CMAS pure-liquid parent reference"), "activity_coefficient"): "pure-liquid",
    (_controlled_text("gamma_Si inferred with SiO(g) carrier on the published CMAS basis"), "activity_coefficient"): "CMAS basis",
    (_controlled_text(""), "evaporation_flux"): "unspecified",
}

# The runner's own controlled conventions, for callers declaring a point's
# standard state directly. These are API vocabulary, not free text.
_CALLER_STANDARD_STATE_CONVENTIONS = {
    (_controlled_text("pure-solid"), "activity"): "pure-solid",
    (_controlled_text("pure-liquid"), "activity"): "pure-liquid",
    (_controlled_text(_CONVERTED_LIQUID_STANDARD_STATE), "activity"): "pure-liquid",
}

_PACK_OVERLAY_BY_VERSION = {
    "1.0.3-d066-ext-v1": True,
    "1.0.9-d066-feo-hess-candidate": False,
    "1.0.3-d066-feo-liquid-candidate": False,
    "1.0.6-d066-sio2-na2o-candidate": False,
    "1.0.3-d066-v1-kcaalsi2o7": False,
    "1.0.3-d066-v1": False,
    "1.0.4-d066-v2": False,
    "1.0.5-d066-v3": False,
    "1.0.6-d066-v4": False,
    "1.0.7-d066-v5": False,
    "1.0.2-ext-sp-2": True,
    "1.0.2": False,
    "1.5.0-cr-candidate": False,
    "1.6.0-cr-candidate": False,
    "1.7.0-cr-candidate": False,
    "1.4.0-cr-candidate": False,
    "1.7.0-cr-ti-candidate": False,
    "1.8.0-cr-ti-candidate": False,
    "1.9.0-cr-ti-candidate": False,
    "1.6.0-cr-ti-candidate": False,
    "1.6.0-feo-liquid-candidate": False,
    "1.5.0-feo-liquid-candidate": False,
    "1.7.0-sio2-na2o-candidate": False,
    "1.6.0-ti-candidate": False,
    "1.7.0-ti-candidate": False,
    "1.8.0-ti-candidate": False,
    "1.5.0-ti-candidate": False,
    "1.4.0": False,
    "1.5.0": False,
    "1.6.0": False,
    "1.7.0": False,
    "1.8.0-cr-candidate": False,
}
_PACK_OVERLAY_MODEL_BY_VERSION = {
    "1.0.3-d066-ext-v1": "IMCC-SF04-D066",
    "1.0.2-ext-sp-2": "IMCC-SF04-EXT",
}
_SOLID_STANDARD_JANAF_CRYSTAL_TABLES = {
    "CaO": "Ca-027",
    "MgO": "Mg-008",
    "Al2O3": "Al-096",
}
_SOLID_STANDARD_CONDENSATE_CSV = (
    "data/gas/condensate-sio2-janaf-cr-candidate.csv"
)


@dataclass(frozen=True)
class PointResult:
    """One bench point versus one IMCC pack."""

    point_id: str
    population: str
    species: str
    parent_oxide: str
    observable: str
    temperature_K: float
    measured: float
    predicted: float | None
    residual: float | None
    ratio: float | None
    status: str
    reason: str
    convention: str
    units: str
    standard_state: str
    score: bool
    domain_flag: str | None = None
    extrapolation_flag: str | None = None
    binary_subslice: str | None = None
    provenance_class: str | None = None
    notices: tuple[str, ...] = ()
    source_table: str | None = None


@dataclass(frozen=True)
class Aggregate:
    """Counts and residuals for one species or one population."""

    key: str
    n: int
    n_ok: int
    n_flagged: int
    n_predicted: int
    rmse: float | None
    flagged_rmse: float | None
    median_residual: float | None
    mean_residual: float | None
    median_abs_residual: float | None
    status_counts: Mapping[str, int]


@dataclass(frozen=True)
class BenchReport:
    """Full IMCC residual run."""

    bench_set_path: str
    pack_path: str
    pack_model_id: str
    pack_version: str
    n: int
    n_ok: int
    n_flagged: int
    rmse: float | None
    flagged_rmse: float | None
    median_abs_residual: float | None
    status_counts: Mapping[str, int]
    per_slice: tuple[Aggregate, ...]
    binary_slices: tuple[Aggregate, ...]
    solid_standard_slices: tuple[Aggregate, ...]
    converted_slices: tuple[Aggregate, ...]
    kume_table_slices: tuple[Aggregate, ...]
    out_of_domain_counts: Mapping[str, int]
    points: tuple[PointResult, ...]
    converted_points: tuple[PointResult, ...]
    notices: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # These totals combine incompatible standard states. Keep them for
        # the explicitly labelled text arithmetic total, but never publish
        # them as an unlabeled JSON accuracy headline.
        for key in ("rmse", "flagged_rmse", "median_abs_residual"):
            payload.pop(key, None)
        return payload


@dataclass(frozen=True)
class _PackedEngine:
    pack: ImccLoadedDatapack | ImccDatapack
    enable_sp_extension: bool
    model_id: str
    version: str


def _resolve_path(value: str | Path) -> Path:
    """Resolve a user-supplied bench-set or pack path.

    Relative paths resolve against the CWD, which is what a shell user means.
    The previous form fell back to a repository root derived from __file__;
    installed as a package there is no such root, and the fallback silently
    produced a path inside site-packages' parent instead of reporting that the
    file was not found.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _reason_line(value: Any) -> str:
    return " ".join(str(value or "").split())[:800]


def _standard_state_for_point(point: Mapping[str, Any]) -> str:
    """Classify only complete conventions in the controlled vocabulary."""
    key = (
        _controlled_text(point.get("convention")),
        str(point.get("observable") or "").strip().casefold(),
    )
    return _STANDARD_STATE_BY_CONVENTION.get(
        key, _CALLER_STANDARD_STATE_CONVENTIONS.get(key, "unspecified")
    )


def _is_binary_point(point: Mapping[str, Any]) -> bool:
    return str(point.get("population")) in _BINARY_POPULATIONS


def _binary_subslice(point: Mapping[str, Any]) -> str | None:
    if not _is_binary_point(point):
        return None
    mole_fraction = point.get("published_mole_fraction")
    if not isinstance(mole_fraction, Mapping) or "Na2O" not in mole_fraction:
        return "X unknown"
    # Use the paper's published mole fraction, not the converted wt% vector;
    # this keeps nominal X=0.5 rows out of the X>0.5 envelope sub-slice while
    # the model's 0.500003947 fencepost is being fixed in model.py.
    x_na2o = float(mole_fraction["Na2O"])
    if not math.isfinite(x_na2o):
        return "X unknown"
    return "X>0.5" if x_na2o > 0.5 else "X<=0.5"


def _slice_key(row: PointResult) -> str:
    return f"{row.population} × {row.observable} × {row.standard_state}"


def _oxide_slice_key(row: PointResult) -> str:
    return (
        f"{row.population} × {row.parent_oxide} × {row.observable} × "
        f"{row.standard_state}"
    )


def _kume_table_slice_key(row: PointResult) -> str:
    return (
        f"{row.population} × {row.source_table or 'Table unknown'} × "
        f"{row.parent_oxide} × {row.observable} × {row.standard_state}"
    )


@lru_cache(maxsize=1)
def _solid_standard_condensate_rows() -> dict[str, tuple[dict[str, float], ...]]:
    """Read the closed SiO2 pair and default liquid rows from the shipped table."""
    resource = resources.files("openimcc").joinpath(
        _SOLID_STANDARD_CONDENSATE_CSV
    )
    with resource.open("r", encoding="utf-8", newline="") as handle:
        rows: dict[str, list[dict[str, float]]] = defaultdict(list)
        for raw in csv.DictReader(handle):
            species = str(raw["species_name"])
            parsed = {
                name: float(raw[name])
                for name in (
                    "T_min",
                    "T_max",
                    "dH298_R",
                    "dG_A",
                    "dG_B",
                    "dG_C",
                    "dG_D",
                    "dG_E",
                )
            }
            rows[species].append(parsed)
    return {species: tuple(values) for species, values in rows.items()}


def _condensate_gibbs_j_mol(species: str, temperature_K: float) -> float:
    """Evaluate a row with the same LAMOR G(T) implementation used by gas.py."""
    from openimcc.gas import _lamor_gibbs

    candidates = [
        row
        for row in _solid_standard_condensate_rows().get(species, ())
        if row["T_min"] <= temperature_K <= row["T_max"]
    ]
    if not candidates:
        raise ValueError(
            f"no packaged condensate reference row for {species} at "
            f"{temperature_K:g} K"
        )
    # Match the gas runtime's interval rule at shared boundaries.
    row = max(candidates, key=lambda item: item["T_min"])
    return float(_lamor_gibbs(temperature_K, row))


@lru_cache(maxsize=3)
def _janaf_apparent_gibbs_table(
    table_id: str,
) -> tuple[float, tuple[tuple[float, float, float], ...]]:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "data-src"
        / "janaf"
        / f"{table_id}.yaml"
    )
    table = json.loads(source_path.read_text(encoding="utf-8"))["table"]
    rows = table["values"]
    reference_row = next(
        row for row in rows if float(row["temperature"]["value"]) == 298.15
    )
    formation_enthalpy_298_kj_mol = reference_row["formation_enthalpy"]["value"]
    if formation_enthalpy_298_kj_mol is None:
        raise ValueError(f"{table_id} has no 298.15 K formation enthalpy")
    nodes = tuple(
        sorted(
            (
                float(row["temperature"]["value"]),
                float(row["enthalpy_increment"]["value"]),
                float(row["entropy"]["value"]),
            )
            for row in rows
            if row["enthalpy_increment"]["value"] is not None
            and row["entropy"]["value"] is not None
        )
    )
    return float(formation_enthalpy_298_kj_mol), nodes


def _janaf_apparent_gibbs_j_mol(table_id: str, temperature_K: float) -> float:
    """Evaluate JANAF apparent G° from its 298 K anchor, H increment, and S."""
    formation_enthalpy_298_kj_mol, nodes = _janaf_apparent_gibbs_table(table_id)
    if not nodes or temperature_K < nodes[0][0] or temperature_K > nodes[-1][0]:
        raise ValueError(
            f"{table_id} has no Gibbs value at {temperature_K:g} K"
        )
    # JANAF apparent G° uses the same 298.15 K elemental references as the
    # packaged condensate rows: Hf(298) + [H(T)-H(298)] - T*S(T). The enthalpy
    # cells are kJ/mol, entropy is J/(mol K), and the returned G is J/mol.
    temperatures = [node[0] for node in nodes]
    enthalpy_increment_kj_mol = float(
        np.interp(temperature_K, temperatures, [node[1] for node in nodes])
    )
    entropy_j_mol_k = float(
        np.interp(temperature_K, temperatures, [node[2] for node in nodes])
    )
    return (
        (formation_enthalpy_298_kj_mol + enthalpy_increment_kj_mol) * 1000.0
        - temperature_K * entropy_j_mol_k
    )


def _solid_to_liquid_log10_shift(
    parent_oxide: str, temperature_K: float
) -> float:
    """Return the additive log10 activity shift from solid to liquid reference.

    Premise: at equilibrium the same oxide chemical potential is
    G_cr(T) + RT ln(a_solid) = G_liq(T) + RT ln(a_liq). Rearranging gives
    log10(a_liq) = log10(a_solid) - [G_liq(T)-G_cr(T)]/[RT ln(10)]. The
    Gibbs energies and RT are J/mol, T is K, R is J/(mol K), and activity is
    dimensionless, so the shift is dimensionless dex. At the melting point
    G_liq = G_cr and the shift is zero; the closed SiO2 pair reaches that
    sanity check at 1996 K using JANAF O-035 high-cristobalite with default
    SiO2(l). The other liquid G values come from the packaged CaO(l), MgO(l),
    and Al2O3(l) rows; their JANAF crystal rows are Ca-027, Mg-008, and Al-096.
    Ca-028 and Mg-009 are not used as liquid functions below their glass joins.
    """
    from openimcc.gas import R_J_MOL_K

    if parent_oxide == "SiO2":
        liquid_gibbs = _condensate_gibbs_j_mol("SiO2(l)", temperature_K)
        # The default LH87 crystal row is +2.9 kJ/mol from the default liquid
        # at 1996 K. This candidate replaces only that row with closed JANAF O-035.
        crystal_gibbs = _condensate_gibbs_j_mol("SiO2(cr)", temperature_K)
    else:
        crystal_table = _SOLID_STANDARD_JANAF_CRYSTAL_TABLES.get(parent_oxide)
        if crystal_table is None:
            raise ValueError(
                f"no JANAF crystal reference pair for {parent_oxide!r}"
            )
        liquid_gibbs = _condensate_gibbs_j_mol(
            f"{parent_oxide}(l)", temperature_K
        )
        crystal_gibbs = _janaf_apparent_gibbs_j_mol(
            crystal_table, temperature_K
        )
    return (liquid_gibbs - crystal_gibbs) / (
        R_J_MOL_K * temperature_K * math.log(10.0)
    )


def _convert_solid_activity_to_liquid(
    activity_solid: float, parent_oxide: str, temperature_K: float
) -> float:
    if activity_solid <= 0.0 or not math.isfinite(activity_solid):
        raise ValueError("solid-standard activity must be finite and positive")
    return 10.0 ** (
        math.log10(activity_solid)
        - _solid_to_liquid_log10_shift(parent_oxide, temperature_K)
    )


def _domain_partition(domain_flag: str | None) -> str | None:
    if not domain_flag:
        return None
    causes = frozenset(
        re.findall(r"(?:^|;\s*)(temperature|envelope):", domain_flag)
    )
    return {
        frozenset({"temperature"}): "temperature_only",
        frozenset({"envelope"}): "envelope_only",
        frozenset({"temperature", "envelope"}): "both",
    }.get(causes)


def _positive_finite(values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    nonfinite: list[str] = []
    for key, raw in values.items():
        value = float(raw)
        if not math.isfinite(value):
            nonfinite.append(f"{key}={value!r}")
            continue
        if value > 0.0:
            result[str(key)] = value
    if nonfinite:
        raise ValueError(
            "non-finite melt-activity value refused (would silently shrink "
            f"the sample): {', '.join(nonfinite)}"
        )
    return result


def _normalize_wt(values: Mapping[str, Any]) -> dict[str, float]:
    positive = _positive_finite(values)
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {key: 100.0 * value / total for key, value in positive.items()}


def id_encodes_composition_token(point_id: str) -> bool:
    return bool(_ID_ENCODED_XTOKEN.search(str(point_id)))


def missing_explicit_composition_fields(
    point: Mapping[str, Any],
) -> tuple[str, ...]:
    missing: list[str] = []
    wt = point.get("composition_wt_pct")
    if not isinstance(wt, Mapping) or not wt:
        missing.append("composition_wt_pct")
    mole_fraction = point.get("published_mole_fraction")
    if not isinstance(mole_fraction, Mapping) or not mole_fraction:
        missing.append("published_mole_fraction")
    return tuple(missing)


def assert_xtoken_points_carry_explicit_composition(
    points: Sequence[Mapping[str, Any]],
) -> None:
    offenders: list[str] = []
    for point in points:
        point_id = str(point.get("id", ""))
        if not id_encodes_composition_token(point_id):
            continue
        missing = missing_explicit_composition_fields(point)
        if missing:
            offenders.append(f"{point_id} missing {', '.join(missing)}")
    if offenders:
        raise ValueError(
            "melt-activity bench point encodes composition only in its id; "
            "carry composition_wt_pct and published_mole_fraction on the "
            f"point: {'; '.join(offenders)}"
        )


def composition_wt_pct_for_point(
    point: Mapping[str, Any],
    compositions: Mapping[str, Any],
) -> dict[str, float]:
    """Prefer the point's explicit wt% vector; never parse the point id."""
    inline = point.get("composition_wt_pct")
    if isinstance(inline, Mapping) and inline:
        return _normalize_wt(inline)
    composition_id = str(point["composition_id"])
    return _normalize_wt(compositions[composition_id]["composition_wt_pct"])


def load_bench_set(path: Path) -> dict[str, Any]:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_BenchYAML12SafeLoader)
    if not isinstance(data, dict) or data.get("schema_version") != "melt-activity-bench.v1":
        raise ValueError(f"unsupported melt activity bench set: {path}")
    if not data.get("compositions") or not data.get("points"):
        raise ValueError(f"bench set lacks compositions or points: {path}")
    for point in data["points"]:
        if not isinstance(point, Mapping):
            raise ValueError("bench points must be mappings")
        unknown = set(point) - _BENCH_POINT_KEYS
        if unknown:
            point_id = point.get("id", "<unknown>")
            raise ValueError(
                f"bench point {point_id!r} has unknown point keys: "
                f"{', '.join(sorted(map(str, unknown)))}"
            )
        _point_is_scored(point)
    assert_xtoken_points_carry_explicit_composition(data["points"])
    return data


def _point_is_scored(point: Mapping[str, Any]) -> bool:
    """Resolve scoring from the serialized Boolean and exclusion reason."""
    score = point.get("score", True)
    if not isinstance(score, bool):
        raise ValueError(
            f"bench point {point.get('id', '<unknown>')!r} score must be a boolean"
        )
    return score and not bool(point.get("dropped_reason"))


def _is_research_overlay(raw: Mapping[str, Any]) -> bool:
    model = str(raw.get("model_id") or raw.get("model") or "").strip().casefold()
    version = str(raw.get("imcc_sf04_datapack_version") or "").strip().lower()
    if not _PACK_OVERLAY_BY_VERSION.get(version, False):
        return False
    return model == _PACK_OVERLAY_MODEL_BY_VERSION[version].casefold()


def _load_research_overlay(raw: Mapping[str, Any]) -> ImccDatapack:
    parents = tuple(str(value) for value in raw["parents"])
    rows = list(raw["rows"])
    datapack = ImccDatapack(
        reactions=[str(row["complex"]) for row in rows],
        nu=np.asarray(
            [
                [float(row["nu"].get(parent, 0.0)) for row in rows]
                for parent in parents
            ],
            dtype=float,
        ),
        A=np.asarray([float(row["A"]) for row in rows], dtype=float),
        B=np.asarray([float(row["B"]) for row in rows], dtype=float),
        domains=[tuple(float(v) for v in row["T_domain_K"]) for row in rows],
        version=str(raw["imcc_sf04_datapack_version"]),
        parent_oxides=parents,
    )
    return label_research_datapack(
        datapack,
        model_id="IMCC-SF04-EXT",
        coverage="reviewed-central-table-research-extension",
    )


def load_pack(pack_path: Path) -> _PackedEngine:
    """Load a published pack or an IMCC-SF04-EXT overlay.

    Published and ``sp_extension`` packs go through ``load_datapack``.
    Research overlays that fail that gate (ext-v1/v2/v3) use the same
    ``label_research_datapack`` path as harness ``ImccEngine(published=False)``.

    A bare packaged pack name (e.g. ``imcc-sf04-v1.0.2.json``) with no file of
    that name in the working directory resolves to the packaged resource.
    """
    pack_path = Path(pack_path)
    try:
        pack_path.lstat()
    except FileNotFoundError:
        if str(pack_path) != pack_path.name or pack_path.name not in _PACKAGED_PACK_NAMES:
            raise
        resource = resources.files("openimcc").joinpath("data", "packs", pack_path.name)
        with resources.as_file(resource) as packaged_path:
            return load_pack(packaged_path)
    raw = json.loads(
        pack_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_json_keys,
    )
    if not isinstance(raw, dict):
        raise ImccMalformedDatapackError("datapack JSON root must be an object")
    try:
        pack = load_datapack(pack_path)
        enable_sp_extension = bool(pack.extension_parents)
        return _PackedEngine(
            pack=pack,
            enable_sp_extension=enable_sp_extension,
            model_id=str(pack.model_id),
            version=str(pack.version),
        )
    except ImccMalformedDatapackError:
        if not _is_research_overlay(raw):
            raise
        pack = _load_research_overlay(raw)
        return _PackedEngine(
            pack=pack,
            enable_sp_extension=True,
            model_id=str(pack.model_id),
            version=str(pack.version),
        )


def _evaluate_imcc(
    engine: _PackedEngine,
    composition_wt_pct: Mapping[str, float],
    temperature_K: float,
) -> tuple[str, dict[str, float], dict[str, float], str, str | None, tuple[str, ...]]:
    """Return status, values, reason, and a visible physics-domain flag."""

    def maps(result: Any) -> tuple[dict[str, float], dict[str, float]]:
        activities = {
            oxide: float(value)
            for oxide, value in zip(result.parent_oxides, result.parent_activity)
        }
        gammas = {
            oxide: float(value)
            for oxide, value in zip(result.parent_oxides, result.parent_gamma)
        }
        return activities, gammas

    def evaluate_with_overrides(
        *, allow_extrapolation: bool, allow_out_of_envelope: bool
    ) -> Any:
        return evaluate(
            composition_wt_pct,
            float(temperature_K),
            engine.pack,
            basis_type="wt",
            enable_sp_extension=engine.enable_sp_extension,
            allow_extrapolation=allow_extrapolation,
            allow_out_of_envelope=allow_out_of_envelope,
        )

    try:
        result = evaluate_with_overrides(
            allow_extrapolation=False, allow_out_of_envelope=False
        )
    except (
        ImccCompositionOutsideValidatedEnvelopeError,
        ImccTOutsideDatapackDomainError,
    ) as exc:
        temperature_refusal: ImccTOutsideDatapackDomainError | None = None
        envelope_refusal: ImccCompositionOutsideValidatedEnvelopeError | None = None

        # Probe the two opt-ins independently. The envelope-only call tells us
        # whether temperature extrapolation is required; the temperature-only
        # call tells us whether the validated-envelope override is required.
        try:
            evaluate_with_overrides(
                allow_extrapolation=False, allow_out_of_envelope=True
            )
        except ImccTOutsideDatapackDomainError as probe_exc:
            temperature_refusal = probe_exc
        except ImccRefusal:
            pass
        try:
            evaluate_with_overrides(
                allow_extrapolation=True, allow_out_of_envelope=False
            )
        except ImccCompositionOutsideValidatedEnvelopeError as probe_exc:
            envelope_refusal = probe_exc
        except ImccRefusal:
            pass

        flags: list[str] = []
        if temperature_refusal is not None:
            flags.append(f"temperature: {_reason_line(temperature_refusal)}")
        if envelope_refusal is not None:
            flags.append(f"envelope: {_reason_line(envelope_refusal)}")
        if not flags:
            cause = (
                "envelope"
                if isinstance(exc, ImccCompositionOutsideValidatedEnvelopeError)
                else "temperature"
            )
            flags.append(f"{cause}: {_reason_line(exc)}")
        domain_flag = "; ".join(flags)

        try:
            result = evaluate_with_overrides(
                allow_extrapolation=True, allow_out_of_envelope=True
            )
        except ImccNonconvergenceError as retry_exc:
            return "not_converged", {}, {}, _reason_line(retry_exc), domain_flag, ()
        except ImccRefusal as retry_exc:
            return "out_of_domain", {}, {}, _reason_line(retry_exc), domain_flag, ()
        activities, gammas = maps(result)
        return "ok", activities, gammas, "", domain_flag, tuple(result.labels.notices)
    except (
        ImccComponentOutsideDomainError,
        ImccCompositionIncompleteError,
        ImccFerricInputUnsupportedError,
    ) as exc:
        return "out_of_domain", {}, {}, _reason_line(exc), None, ()
    except ImccNonconvergenceError as exc:
        return "not_converged", {}, {}, _reason_line(exc), None, ()
    except ImccRefusal as exc:
        return "refused", {}, {}, _reason_line(exc), None, ()
    activities, gammas = maps(result)
    return "ok", activities, gammas, "", None, tuple(result.labels.notices)


def _prediction_for_point(
    point: Mapping[str, Any],
    status: str,
    activities: Mapping[str, float],
    gammas: Mapping[str, float],
    reason: str,
    *,
    gas_datapack: Any = None,
    gas_evaluate: Any = None,
    gas_species_provenance: Any = None,
    gas_reason: str = "",
    gas_extrapolation_labels: Mapping[str, str] | None = None,
    initial_domain_flag: str | None = None,
) -> tuple[float | None, str, str, str | None, str | None]:
    """Return (predicted, reason, status, provenance class, domain flag).

    ``unsupported_observable`` is a runner verdict, not an engine crash.
    Activity stays on the parent-formula basis; pressure goes through the
    shared analytical gas layer, which also takes parent-formula activities.
    """
    observable = str(point["observable"])
    if observable not in _SUPPORTED_OBSERVABLES:
        return (
            None,
            f"unsupported observable {observable!r}",
            "unsupported_observable",
            None,
            None,
        )
    if status != "ok":
        return None, reason, status, None, initial_domain_flag
    parent = str(point["parent_oxide"])
    provenance_class: str | None = None
    domain_flag = initial_domain_flag
    if observable == "activity":
        value: float | None = activities.get(parent)
    elif observable == "activity_coefficient":
        value = gammas.get(parent)
    else:
        species = str(point.get("species") or "")
        if gas_species_provenance is not None and species:
            try:
                provenance_class = str(gas_species_provenance(species)["authority"])
            except ImccRefusal:
                pass
        if point.get("fO2_bar") is None:
            return (
                None,
                "gas comparison refused: observation has no independent fO2 pin",
                "refused",
                provenance_class,
                None,
            )
        units = str(point.get("units") or "")
        if units != "Pa":
            return (
                None,
                f"partial_pressure refused: expected units 'Pa', got {units!r}",
                "refused",
                provenance_class,
                None,
            )
        if gas_reason:
            return None, gas_reason, "refused", provenance_class, None
        if (
            gas_datapack is None
            or gas_evaluate is None
            or gas_species_provenance is None
        ):
            return (
                None,
                "partial_pressure refused: gas layer is unavailable",
                "refused",
                provenance_class,
                None,
            )

        try:
            gas_result = gas_evaluate(
                activities,
                float(point["temperature_K"]),
                float(point["fO2_bar"]),
                gas_datapack,
                gas_species=(species,),
                composition_wt_pct=point.get("composition_wt_pct"),
            )
            value = gas_result.get(species)
            gas_flag = gas_result.domain_flags.get(species)
            if gas_flag is not None:
                label = (gas_extrapolation_labels or {}).get(species)
                if label:
                    gas_flag = f"{gas_flag}; {label}"
                domain_flag = (
                    f"gas: {gas_flag}"
                    if domain_flag is None
                    else f"{domain_flag}; gas: {gas_flag}"
                )
        except ImccRefusal as exc:
            return None, _reason_line(exc), "refused", provenance_class, None
        except (TypeError, ValueError) as exc:
            return (
                None,
                f"gas comparison refused: {_reason_line(exc)}",
                "refused",
                provenance_class,
                None,
            )

        # Premise: evaluate_gas returns p_i / p° numerically in bar, while
        # every partial_pressure measurement is in Pa. Algebra: p_Pa =
        # p_bar * (1 bar / 1) * 1e5 Pa/bar. Unit check: BAR_TO_PA has units
        # Pa/bar, so the product is Pa. Sanity: 1 bar becomes 100000 Pa;
        # omitting this factor shifts every log10 residual by exactly -5 dex.
        value = None if value is None else float(value) * BAR_TO_PA
    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        return (
            None,
            f"engine returned no positive {observable} for {parent}",
            "refused",
            provenance_class,
            None,
        )
    return float(value), "", "ok", provenance_class, domain_flag


def _residual_and_ratio(
    predicted: float | None,
    measured: float,
    *,
    score: bool,
) -> tuple[float | None, float | None]:
    if predicted is None or measured <= 0.0 or not score:
        return None, None
    ratio = predicted / measured
    if ratio <= 0.0 or not math.isfinite(ratio):
        return None, None
    return math.log10(ratio), ratio


def _rmse(values: Iterable[float]) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return math.sqrt(sum(value * value for value in materialized) / len(materialized))


def _median_abs(values: Iterable[float]) -> float | None:
    materialized = [abs(value) for value in values]
    if not materialized:
        return None
    return float(statistics.median(materialized))


def _median(values: Iterable[float]) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return float(statistics.median(materialized))


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return float(statistics.fmean(materialized))


def _status_counts(rows: Sequence[PointResult]) -> dict[str, int]:
    counts = Counter(row.status for row in rows)
    return {status: int(counts[status]) for status in POINT_STATUSES}


def _aggregate(key: str, rows: Sequence[PointResult]) -> Aggregate:
    residuals = [row.residual for row in rows if row.residual is not None]
    flagged_residuals = [
        row.residual
        for row in rows
        if row.domain_flag is not None and row.residual is not None
    ]
    unflagged_residuals = [
        row.residual
        for row in rows
        if row.domain_flag is None and row.residual is not None
    ]
    return Aggregate(
        key=key,
        n=len(rows),
        n_ok=sum(row.status == "ok" for row in rows),
        n_flagged=sum(row.domain_flag is not None for row in rows),
        n_predicted=sum(row.predicted is not None for row in rows),
        rmse=_rmse(value for value in residuals if value is not None),
        flagged_rmse=_rmse(
            value for value in flagged_residuals if value is not None
        ),
        median_residual=_median(value for value in residuals if value is not None),
        mean_residual=_mean(value for value in residuals if value is not None),
        median_abs_residual=_median_abs(
            value for value in unflagged_residuals if value is not None
        ),
        status_counts=_status_counts(rows),
    )


def _select_points(
    points: Sequence[Mapping[str, Any]],
    *,
    populations: Sequence[str] | None,
    species: Sequence[str] | None,
    limit: int | None,
) -> list[Mapping[str, Any]]:
    selected = list(points)
    if populations is not None:
        wanted = set(populations)
        selected = [point for point in selected if str(point["population"]) in wanted]
    if species is not None:
        wanted_species = set(species)
        selected = [point for point in selected if str(point["species"]) in wanted_species]
    if limit is not None:
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        selected = selected[:limit]
    return selected


def run_bench(
    bench_set_path: str | Path,
    pack_path: str | Path,
    *,
    populations: Sequence[str] | None = None,
    species: Sequence[str] | None = None,
    limit: int | None = None,
) -> BenchReport:
    """Evaluate tracked empirical points against one IMCC datapack."""
    bench_path = _resolve_path(bench_set_path)
    resolved_pack = _resolve_path(pack_path)
    fixture = load_bench_set(bench_path)
    engine = load_pack(resolved_pack)
    compositions = dict(fixture["compositions"])
    selected = _select_points(
        fixture["points"],
        populations=populations,
        species=species,
        limit=limit,
    )
    gas_datapack: Any = None
    gas_evaluate: Any = None
    gas_species_provenance: Any = None
    gas_reason = ""
    gas_extrapolation_labels: Mapping[str, str] | None = None
    if any(str(point.get("observable")) == "partial_pressure" for point in selected):
        try:
            from openimcc.gas import (
                IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS as _gas_extrapolation_labels,
                evaluate_gas as _evaluate_gas,
                gas_species_provenance as _gas_species_provenance,
                load_gas_datapack,
            )

            gas_evaluate = _evaluate_gas
            gas_species_provenance = _gas_species_provenance
            gas_extrapolation_labels = _gas_extrapolation_labels
            gas_datapack = load_gas_datapack()
        except ImportError as exc:
            gas_reason = (
                "partial_pressure refused: gas layer unavailable; install "
                f'"openimcc[gas]" (pandas import failed: {_reason_line(exc)})'
            )
        except ImccRefusal as exc:
            gas_reason = f"partial_pressure refused: {_reason_line(exc)}"
    cache: dict[
        tuple[tuple[tuple[str, float], ...], float],
        tuple[str, dict[str, float], dict[str, float], str, str | None, tuple[str, ...]],
    ] = {}
    rows: list[PointResult] = []
    converted_rows: list[PointResult] = []
    for point in selected:
        composition_id = str(point["composition_id"])
        composition = composition_wt_pct_for_point(point, compositions)
        temperature_K = float(point["temperature_K"])
        cache_key = (tuple(sorted(composition.items())), temperature_K)
        if cache_key not in cache:
            cache[cache_key] = _evaluate_imcc(engine, composition, temperature_K)
        (
            status,
            activities,
            gammas,
            engine_reason,
            engine_domain_flag,
            engine_notices,
        ) = cache[cache_key]
        score = _point_is_scored(point)
        if not score and point.get("dropped_reason") and status == "ok":
            status = "refused"
            engine_reason = str(point["dropped_reason"])
        engine_status = status
        enriched = {**point, "composition_wt_pct": composition}
        (
            predicted,
            prediction_reason,
            status,
            provenance_class,
            domain_flag,
        ) = _prediction_for_point(
            enriched,
            status,
            activities,
            gammas,
            engine_reason,
            gas_datapack=gas_datapack,
            gas_evaluate=gas_evaluate,
            gas_species_provenance=gas_species_provenance,
            gas_reason=gas_reason,
            gas_extrapolation_labels=gas_extrapolation_labels,
            initial_domain_flag=engine_domain_flag,
        )
        point_notices = engine_notices
        if (
            str(point.get("observable")) not in _SUPPORTED_OBSERVABLES
            and engine_status != "ok"
        ):
            point_notices = (
                *point_notices,
                f"melt evaluation {engine_status}: {engine_reason}",
            )
        measured = float(point["measured"])
        residual, ratio = _residual_and_ratio(predicted, measured, score=score)
        provenance = point.get("provenance")
        source_table = (
            str(provenance["table"])
            if isinstance(provenance, Mapping) and provenance.get("table")
            else None
        )
        row = PointResult(
            point_id=str(point["id"]),
            population=str(point["population"]),
            species=str(point["species"]),
            parent_oxide=str(point["parent_oxide"]),
            observable=str(point["observable"]),
            temperature_K=temperature_K,
            measured=measured,
            predicted=predicted,
            residual=residual,
            ratio=ratio,
            status=status,
            reason=prediction_reason or engine_reason,
            convention=str(point.get("convention") or ""),
            units=str(point.get("units") or ""),
            standard_state=_standard_state_for_point(point),
            score=score,
            domain_flag=domain_flag,
            extrapolation_flag=(
                str(point["extrapolation_flag"])
                if point.get("extrapolation_flag")
                else None
            ),
            binary_subslice=_binary_subslice(point),
            provenance_class=provenance_class,
            notices=point_notices,
            source_table=source_table,
        )
        rows.append(row)
        if row.standard_state == "pure-solid" and row.observable == "activity":
            measured_liquid = _convert_solid_activity_to_liquid(
                row.measured, row.parent_oxide, row.temperature_K
            )
            converted_residual, converted_ratio = _residual_and_ratio(
                row.predicted, measured_liquid, score=row.score
            )
            if row.parent_oxide == "SiO2":
                conversion_sources = (
                    "liquid and crystal Gibbs from the closed O-035 SiO2 "
                    "condensate pair"
                )
            else:
                crystal_table = _SOLID_STANDARD_JANAF_CRYSTAL_TABLES[
                    row.parent_oxide
                ]
                conversion_sources = (
                    "liquid Gibbs from packaged condensate rows and "
                    f"JANAF {crystal_table} apparent crystal Gibbs"
                )
            converted_rows.append(
                replace(
                    row,
                    measured=measured_liquid,
                    residual=converted_residual,
                    ratio=converted_ratio,
                    standard_state=_CONVERTED_LIQUID_STANDARD_STATE,
                    convention=(
                        f"{row.convention}; measured activity converted from "
                        "pure-solid to pure-liquid (Gibbs conversion v2), "
                        f"using {conversion_sources}"
                    ),
                )
            )
    overall = _aggregate("", rows)
    headline_rows = [
        row
        for row in rows
        if row.binary_subslice is None
        and row.domain_flag is None
        and row.residual is not None
    ]
    headline_residuals = [row.residual for row in headline_rows]
    flagged_residuals = [
        row.residual
        for row in rows
        if row.domain_flag is not None and row.residual is not None
    ]
    by_slice: dict[str, list[PointResult]] = defaultdict(list)
    by_binary_slice: dict[str, list[PointResult]] = defaultdict(list)
    by_solid_oxide: dict[str, list[PointResult]] = defaultdict(list)
    by_converted_oxide: dict[str, list[PointResult]] = defaultdict(list)
    by_kume_table: dict[str, list[PointResult]] = defaultdict(list)
    for row in rows:
        key = _slice_key(row)
        if row.binary_subslice is None:
            by_slice[key].append(row)
        else:
            by_binary_slice[
                f"{key} × {row.parent_oxide} × {row.binary_subslice}"
            ].append(row)
        if row.standard_state == "pure-solid" and row.observable == "activity":
            by_solid_oxide[_oxide_slice_key(row)].append(row)
    for row in converted_rows:
        by_converted_oxide[_oxide_slice_key(row)].append(row)
    for row in (*rows, *converted_rows):
        if (
            row.population == "kume2000_slag_si_alloy"
            and row.observable == "activity"
            and row.source_table is not None
        ):
            by_kume_table[_kume_table_slice_key(row)].append(row)
    out_of_domain_counts = Counter(
        cause
        for row in rows
        for cause in [_domain_partition(row.domain_flag)]
        if cause is not None
    )
    return BenchReport(
        bench_set_path=str(bench_path),
        pack_path=str(resolved_pack),
        pack_model_id=engine.model_id,
        pack_version=engine.version,
        n=overall.n,
        n_ok=overall.n_ok,
        n_flagged=overall.n_flagged,
        rmse=_rmse(value for value in headline_residuals if value is not None),
        flagged_rmse=_rmse(value for value in flagged_residuals if value is not None),
        median_abs_residual=_median_abs(
            value for value in headline_residuals if value is not None
        ),
        status_counts=overall.status_counts,
        per_slice=tuple(
            _aggregate(key, by_slice[key]) for key in sorted(by_slice)
        ),
        binary_slices=tuple(
            _aggregate(key, by_binary_slice[key])
            for key in sorted(by_binary_slice)
        ),
        solid_standard_slices=tuple(
            _aggregate(key, by_solid_oxide[key]) for key in sorted(by_solid_oxide)
        ),
        converted_slices=tuple(
            _aggregate(key, by_converted_oxide[key])
            for key in sorted(by_converted_oxide)
        ),
        kume_table_slices=tuple(
            _aggregate(key, by_kume_table[key]) for key in sorted(by_kume_table)
        ),
        out_of_domain_counts={
            key: int(out_of_domain_counts[key])
            for key in ("temperature_only", "envelope_only", "both")
            if out_of_domain_counts.get(key, 0)
        },
        points=tuple(rows),
        converted_points=tuple(converted_rows),
        notices=tuple(dict.fromkeys(notice for row in rows for notice in row.notices)),
    )


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}g}"


def _fmt_status_counts(counts: Mapping[str, int]) -> str:
    return "  ".join(f"{status}={counts.get(status, 0)}" for status in POINT_STATUSES)


def _render_aggregate_table(title: str, aggregates: Sequence[Aggregate]) -> list[str]:
    lines = [title, ""]
    header = (
        f"{'slice':<78} {'n':>5} {'pred':>6} {'flagged':>8} "
        f"{'median(r)':>10} {'mean(r)':>10} {'RMSE':>10} {'flagRMSE':>10} "
        f"{'refused':>8} {'ood':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in aggregates:
        lines.append(
            f"{row.key:<78} {row.n:>5} {row.n_predicted:>6} "
            f"{row.n_flagged:>8} {_fmt_float(row.median_residual):>10} "
            f"{_fmt_float(row.mean_residual):>10} {_fmt_float(row.rmse):>10} "
            f"{_fmt_float(row.flagged_rmse):>10} "
            f"{row.status_counts.get('refused', 0):>8} "
            f"{row.status_counts.get('out_of_domain', 0):>8}"
        )
    lines.append("")
    return lines


def render_report(report: BenchReport) -> str:
    """Readable text table for stdout."""
    headline_residuals = [
        row.residual
        for row in report.points
        if row.binary_subslice is None
        and row.domain_flag is None
        and row.residual is not None
    ]
    predicted = sum(row.predicted is not None for row in report.points)
    domain_total = sum(report.out_of_domain_counts.values())
    lines = [
        "IMCC empirical residual runner",
        f"  bench: {report.bench_set_path}",
        f"  pack:  {report.pack_path}",
        f"  model: {report.pack_model_id}  version={report.pack_version}",
        "  Results split by dataset × observable × standard state; no single accuracy headline.",
        (
            f"  N={report.n}  N_ok={report.n_ok}  predictions={predicted}  "
            f"flagged={report.n_flagged}  "
            f"statuses: {_fmt_status_counts(report.status_counts)}"
        ),
        (
            "  arithmetic total across incompatible slices (not an accuracy figure): "
            f"n={len(headline_residuals)} "
            f"median={_fmt_float(_median(value for value in headline_residuals if value is not None))} "
            f"mean={_fmt_float(_mean(value for value in headline_residuals if value is not None))} "
            f"RMSE={_fmt_float(report.rmse)} "
            f"flagged_RMSE={_fmt_float(report.flagged_rmse)}"
        ),
        (
            f"  out_of_domain flags: temperature-only={report.out_of_domain_counts.get('temperature_only', 0)}  "
            f"envelope-only={report.out_of_domain_counts.get('envelope_only', 0)}  "
            f"both={report.out_of_domain_counts.get('both', 0)}  total={domain_total}"
        ),
        "",
    ]
    if report.pack_model_id == "IMCC-SF04-D066-EXCESS-CANDIDATE":
        lines.extend(f"  notice: {notice}" for notice in report.notices)
        if report.notices:
            lines.append("")
    lines.extend(
        _render_aggregate_table(
            "Per dataset × observable × standard state", report.per_slice
        )
    )
    if report.binary_slices:
        lines.append(
            "Sodium binary predict-and-flag (separate; never averaged into another slice)"
        )
        lines.append(
            "  Na2O rows are verified pure-liquid parent-oxide activities; "
            "temperature, envelope, and source extrapolation flags remain visible."
        )
        lines.append("")
        lines.extend(_render_aggregate_table("Binary sub-slices", report.binary_slices))
    if report.solid_standard_slices:
        lines.extend(
            _render_aggregate_table(
                "Solid-standard activity slices by oxide (as published)",
                report.solid_standard_slices,
            )
        )
    if report.converted_slices:
        lines.extend(
            _render_aggregate_table(
                "Converted activity slices by oxide (pure-liquid standard state)",
                report.converted_slices,
            )
        )
    if report.kume_table_slices:
        lines.append("Kume 2000 activity slices by source table and oxide")
        lines.append(
            "  Table 4 CaO and Al2O3 activities are derived via Rein–Chipman "
            "silica activities; Table 2 and Table 4 remain separate."
        )
        lines.append("")
        lines.extend(
            _render_aggregate_table(
                "Kume table slices (raw pure-solid and converted pure-liquid)",
                report.kume_table_slices,
            )
        )
    if any(
        row.population == "kume2000_slag_si_alloy"
        and row.standard_state == "pure-solid"
        for row in report.points
    ):
        lines.append(
            f"Kume notice: {_KUME_LIQUID_ACTIVITY_WARNING}."
        )
        lines.append("")
    lines.append("Points")
    lines.append("")
    point_header = (
        f"{'id':<28} {'pop':<24} {'sp':<6} {'obs':<18} "
        f"{'meas':>10} {'pred':>10} {'resid':>10} {'ratio':>10} "
        f"{'status':<22} {'domain_flag':<88} {'source_xtrap':<34} "
        f"{'standard_state':<18} {'provenance_class':<44}"
    )
    lines.append(point_header)
    lines.append("-" * len(point_header))
    for row in report.points:
        lines.append(
            f"{row.point_id:<28} {row.population:<24} {row.species:<6} "
            f"{row.observable:<18} {_fmt_float(row.measured):>10} "
            f"{_fmt_float(row.predicted):>10} {_fmt_float(row.residual):>10} "
            f"{_fmt_float(row.ratio):>10} {row.status:<22} "
            f"{row.domain_flag or '—':<88} {row.extrapolation_flag or '—':<34} "
            f"{row.standard_state:<18} {row.provenance_class or '—':<44}"
        )
    return "\n".join(lines) + "\n"


def _csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openimcc-bench",
        description=(
            "Run tracked empirical melt-activity points against one IMCC "
            "datapack and print residuals."
        ),
    )
    parser.add_argument("bench_set", help="Path to a melt-activity-bench.v1 YAML file")
    parser.add_argument("pack", help="Path to an IMCC datapack JSON file")
    parser.add_argument(
        "--populations",
        default=None,
        help="Comma-separated population ids to keep (default: all)",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Comma-separated species labels to keep (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most N points after filtering",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the BenchReport as JSON instead of the text table",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_bench(
        args.bench_set,
        args.pack,
        populations=_csv_list(args.populations),
        species=_csv_list(args.species),
        limit=args.limit,
    )
    stream = sys.stdout if out is None else out
    if args.json:
        json.dump(report.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
    else:
        stream.write(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
