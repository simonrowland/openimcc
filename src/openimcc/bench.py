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
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
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
    score: bool
    domain_flag: str | None = None
    provenance_class: str | None = None


@dataclass(frozen=True)
class Aggregate:
    """Counts and residuals for one species or one population."""

    key: str
    n: int
    n_ok: int
    n_flagged: int
    rmse: float | None
    flagged_rmse: float | None
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
    per_species: tuple[Aggregate, ...]
    per_population: tuple[Aggregate, ...]
    points: tuple[PointResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
            f"point (t-691): {'; '.join(offenders)}"
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
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != "melt-activity-bench.v1":
        raise ValueError(f"unsupported melt activity bench set: {path}")
    if not data.get("compositions") or not data.get("points"):
        raise ValueError(f"bench set lacks compositions or points: {path}")
    assert_xtoken_points_carry_explicit_composition(data["points"])
    return data


def _is_research_overlay(raw: Mapping[str, Any]) -> bool:
    model = str(raw.get("model_id") or raw.get("model") or "")
    version = str(raw.get("imcc_sf04_datapack_version") or "")
    return "EXT" in model.upper() or "EXT" in version.upper()


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
    """
    raw = json.loads(pack_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ImccMalformedDatapackError("datapack JSON root must be an object")
    try:
        pack = load_datapack(pack_path)
        enable_sp_extension = pack.model_id != "IMCC-SF04"
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
) -> tuple[str, dict[str, float], dict[str, float], str]:
    """Return (status, activities, gammas, reason) with ImccEngine mapping."""
    try:
        result = evaluate(
            composition_wt_pct,
            float(temperature_K),
            engine.pack,
            basis_type="wt",
            enable_sp_extension=engine.enable_sp_extension,
            allow_extrapolation=False,
            allow_out_of_envelope=False,
        )
    except (
        ImccCompositionOutsideValidatedEnvelopeError,
        ImccComponentOutsideDomainError,
        ImccCompositionIncompleteError,
        ImccFerricInputUnsupportedError,
        ImccTOutsideDatapackDomainError,
    ) as exc:
        return "out_of_domain", {}, {}, _reason_line(exc)
    except ImccNonconvergenceError as exc:
        return "not_converged", {}, {}, _reason_line(exc)
    except ImccRefusal as exc:
        return "refused", {}, {}, _reason_line(exc)
    activities = {
        oxide: float(value)
        for oxide, value in zip(result.parent_oxides, result.parent_activity)
    }
    gammas = {
        oxide: float(value)
        for oxide, value in zip(result.parent_oxides, result.parent_gamma)
    }
    return "ok", activities, gammas, ""


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
    gas_domain_error: Any = None,
    gas_extrapolation_labels: Mapping[str, str] | None = None,
) -> tuple[float | None, str, str, str | None, str | None]:
    """Return (predicted, reason, status, provenance class, domain flag).

    ``unsupported_observable`` is a runner verdict, not an engine crash.
    Activity stays on the parent-formula basis; pressure goes through the
    shared analytical gas layer, which also takes parent-formula activities.
    """
    if status != "ok":
        return None, reason, status, None, None
    observable = str(point["observable"])
    if observable not in _SUPPORTED_OBSERVABLES:
        return (
            None,
            f"unsupported observable {observable!r}",
            "unsupported_observable",
            None,
            None,
        )
    parent = str(point["parent_oxide"])
    provenance_class: str | None = None
    domain_flag: str | None = None
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
            gas_values = gas_evaluate(
                activities,
                float(point["temperature_K"]),
                float(point["fO2_bar"]),
                gas_datapack,
                gas_species=(species,),
            )
            value = gas_values.get(species)
        except ImccRefusal as exc:
            if gas_domain_error is not None and isinstance(exc, gas_domain_error):
                try:
                    gas_values = gas_evaluate(
                        activities,
                        float(point["temperature_K"]),
                        float(point["fO2_bar"]),
                        gas_datapack,
                        gas_species=(species,),
                        allow_extrapolation=True,
                    )
                    value = gas_values.get(species)
                except (ImccRefusal, TypeError, ValueError) as retry_exc:
                    return (
                        None,
                        _reason_line(retry_exc),
                        "refused",
                        provenance_class,
                        None,
                    )
                domain_flag = _reason_line(exc)
                label = (gas_extrapolation_labels or {}).get(species)
                if label:
                    domain_flag = f"{domain_flag}; {label}"
            else:
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


def _status_counts(rows: Sequence[PointResult]) -> dict[str, int]:
    counts = Counter(row.status for row in rows)
    return {status: int(counts[status]) for status in POINT_STATUSES}


def _aggregate(key: str, rows: Sequence[PointResult]) -> Aggregate:
    flagged_residuals = [
        row.residual
        for row in rows
        if row.domain_flag is not None and row.residual is not None
    ]
    residuals = [
        row.residual
        for row in rows
        if row.domain_flag is None and row.residual is not None
    ]
    return Aggregate(
        key=key,
        n=len(rows),
        n_ok=sum(row.status == "ok" for row in rows),
        n_flagged=sum(row.domain_flag is not None for row in rows),
        rmse=_rmse(value for value in residuals if value is not None),
        flagged_rmse=_rmse(
            value for value in flagged_residuals if value is not None
        ),
        median_abs_residual=_median_abs(
            value for value in residuals if value is not None
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
    gas_domain_error: Any = None
    gas_extrapolation_labels: Mapping[str, str] | None = None
    if any(str(point.get("observable")) == "partial_pressure" for point in selected):
        try:
            from openimcc.gas import (
                IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS as _gas_extrapolation_labels,
                ImccGasTemperatureOutsideDomainError as _gas_domain_error,
                evaluate_gas as _evaluate_gas,
                gas_species_provenance as _gas_species_provenance,
                load_gas_datapack,
            )

            gas_evaluate = _evaluate_gas
            gas_species_provenance = _gas_species_provenance
            gas_domain_error = _gas_domain_error
            gas_extrapolation_labels = _gas_extrapolation_labels
            gas_datapack = load_gas_datapack()
        except ImportError as exc:
            gas_reason = (
                "partial_pressure refused: gas layer unavailable; install "
                f'"openimcc[gas]" (pandas import failed: {_reason_line(exc)})'
            )
        except ImccRefusal as exc:
            gas_reason = f"partial_pressure refused: {_reason_line(exc)}"
    cache: dict[tuple[str, float], tuple[str, dict[str, float], dict[str, float], str]] = {}
    rows: list[PointResult] = []
    for point in selected:
        composition_id = str(point["composition_id"])
        composition = composition_wt_pct_for_point(point, compositions)
        temperature_K = float(point["temperature_K"])
        cache_key = (composition_id, temperature_K)
        if cache_key not in cache:
            cache[cache_key] = _evaluate_imcc(engine, composition, temperature_K)
        status, activities, gammas, engine_reason = cache[cache_key]
        if (
            not bool(point.get("score", True))
            and point.get("dropped_reason")
            and status == "ok"
        ):
            status = "refused"
            engine_reason = str(point["dropped_reason"])
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
            gas_domain_error=gas_domain_error,
            gas_extrapolation_labels=gas_extrapolation_labels,
        )
        measured = float(point["measured"])
        score = bool(point.get("score", True))
        residual, ratio = _residual_and_ratio(predicted, measured, score=score)
        rows.append(
            PointResult(
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
                score=score,
                domain_flag=domain_flag,
                provenance_class=provenance_class,
            )
        )
    by_species: dict[str, list[PointResult]] = defaultdict(list)
    by_population: dict[str, list[PointResult]] = defaultdict(list)
    for row in rows:
        by_species[row.species].append(row)
        by_population[row.population].append(row)
    overall = _aggregate("", rows)
    return BenchReport(
        bench_set_path=str(bench_path),
        pack_path=str(resolved_pack),
        pack_model_id=engine.model_id,
        pack_version=engine.version,
        n=overall.n,
        n_ok=overall.n_ok,
        n_flagged=overall.n_flagged,
        rmse=overall.rmse,
        flagged_rmse=overall.flagged_rmse,
        median_abs_residual=overall.median_abs_residual,
        status_counts=overall.status_counts,
        per_species=tuple(
            _aggregate(key, by_species[key]) for key in sorted(by_species)
        ),
        per_population=tuple(
            _aggregate(key, by_population[key]) for key in sorted(by_population)
        ),
        points=tuple(rows),
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
        f"{'key':<32} {'N':>5} {'N_ok':>5} {'RMSE':>10} {'med|r|':>10} "
        f"{'flagged':>8} {'flagRMSE':>10} {'refused':>8} {'ood':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in aggregates:
        lines.append(
            f"{row.key:<32} {row.n:>5} {row.n_ok:>5} "
            f"{_fmt_float(row.rmse):>10} {_fmt_float(row.median_abs_residual):>10} "
            f"{row.n_flagged:>8} {_fmt_float(row.flagged_rmse):>10} "
            f"{row.status_counts.get('refused', 0):>8} "
            f"{row.status_counts.get('out_of_domain', 0):>8}"
        )
    lines.append("")
    return lines


def render_report(report: BenchReport) -> str:
    """Readable text table for stdout."""
    lines = [
        "IMCC empirical residual runner",
        f"  bench: {report.bench_set_path}",
        f"  pack:  {report.pack_path}",
        f"  model: {report.pack_model_id}  version={report.pack_version}",
        (
            f"  N={report.n}  N_ok={report.n_ok}  "
            f"scored={report.n_ok - report.n_flagged}  "
            f"flagged={report.n_flagged}  "
            f"RMSE={_fmt_float(report.rmse)}  "
            f"flagged_RMSE={_fmt_float(report.flagged_rmse)}  "
            f"median|residual|={_fmt_float(report.median_abs_residual)}"
        ),
        f"  statuses: {_fmt_status_counts(report.status_counts)}",
        "",
    ]
    lines.extend(_render_aggregate_table("Per species", report.per_species))
    lines.extend(_render_aggregate_table("Per population", report.per_population))
    lines.append("Points")
    lines.append("")
    point_header = (
        f"{'id':<28} {'pop':<24} {'sp':<6} {'obs':<18} "
        f"{'meas':>10} {'pred':>10} {'resid':>10} {'ratio':>10} "
        f"{'status':<22} {'domain_flag':<88} {'provenance_class':<44}"
    )
    lines.append(point_header)
    lines.append("-" * len(point_header))
    for row in report.points:
        lines.append(
            f"{row.point_id:<28} {row.population:<24} {row.species:<6} "
            f"{row.observable:<18} {_fmt_float(row.measured):>10} "
            f"{_fmt_float(row.predicted):>10} {_fmt_float(row.residual):>10} "
            f"{_fmt_float(row.ratio):>10} {row.status:<22} "
            f"{row.domain_flag or '—':<88} {row.provenance_class or '—':<44}"
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
