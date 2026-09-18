"""IMCC-only empirical residual runner.

Run a tracked ``melt-activity-bench.v1`` set against one IMCC datapack and
print per-point residuals. This is not a replacement for
``benchmarks/melt_activity_benchmark.py`` (multi-engine comparison, coverage
maps, latch detection). It is the small tool: one pack, the tracked points,
the residuals.

Basis (must match the harness; a mismatched basis is silently wrong):

- ``activity`` and ``activity_coefficient`` compare IMCC parent-formula
  values directly. Gamma is ``a/x`` on the parent-oxide formula-unit basis.
- ``partial_pressure`` converts those parent-formula activities onto each
  rail's single-cation basis, then holds the tracked analytical gas layer
  constant. Activity and pressure are never coerced into each other.
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


@dataclass(frozen=True)
class Aggregate:
    """Counts and residuals for one species or one population."""

    key: str
    n: int
    n_ok: int
    rmse: float | None
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
    rmse: float | None
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


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def _parse_oxide_formula(formula: str) -> dict[str, int]:
    """Element counts for a simple oxide formula.

    Deliberately narrow: parent oxides are flat formulae like SiO2, Al2O3,
    Na2O, P2O5 -- no brackets, no hydrates, no charges. A general chemical
    parser would be more code and more ways to be subtly wrong on input this
    module never sees. Anything the narrow grammar does not consume raises
    rather than being silently skipped, because a dropped element would change
    the cation count and therefore the basis.
    """
    counts: dict[str, int] = {}
    consumed = 0
    for match in _FORMULA_TOKEN.finditer(formula):
        if match.start() != consumed:
            break
        element, digits = match.groups()
        counts[element] = counts.get(element, 0) + (int(digits) if digits else 1)
        consumed = match.end()
    if consumed != len(formula) or not counts:
        raise ValueError(f"cannot parse oxide formula {formula!r}")
    return counts


def _oxide_cations_per_formula(parent_oxide: str) -> float:
    """Cations per formula unit, e.g. Na2O -> 2, SiO2 -> 1, Al2O3 -> 2.

    This is what converts a parent-oxide activity to the single-cation basis,
    so getting it wrong rescales every gas comparison silently.
    """
    elements = _parse_oxide_formula(str(parent_oxide))
    cations = [float(n) for el, n in elements.items() if el != "O"]
    if "O" not in elements or len(cations) != 1:
        raise ValueError(
            f"expected a one-cation-family oxide formula, got {parent_oxide!r}"
        )
    return cations[0]


def _single_cation_gas_activities(
    parent_oxide_activities: Mapping[str, float],
) -> dict[str, float]:
    converted: dict[str, float] = {}
    for parent_oxide, activity in parent_oxide_activities.items():
        cations = _oxide_cations_per_formula(parent_oxide)
        # Premise: M_nO_m contains n units of the single-cation component
        # MO_(m/n). Algebra: a_parent = a_single**n, so
        # a_single = a_parent**(1/n). Sanity: K2O/Na2O/Li2O use sqrt(a_M2O).
        converted[str(parent_oxide)] = float(activity) ** (1.0 / cations)
    return _positive_finite(converted)


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
) -> tuple[float | None, str, str]:
    """Return (predicted, reason, status).

    ``unsupported_observable`` is a runner verdict, not an engine crash.
    Activity stays on the parent-formula basis; pressure goes through the
    single-cation conversion plus the shared analytical gas layer.
    """
    if status != "ok":
        return None, reason, status
    observable = str(point["observable"])
    if observable not in _SUPPORTED_OBSERVABLES:
        return None, f"unsupported observable {observable!r}", "unsupported_observable"
    parent = str(point["parent_oxide"])
    if observable == "activity":
        value: float | None = activities.get(parent)
    elif observable == "activity_coefficient":
        value = gammas.get(parent)
    else:
        if "fO2_bar" not in point:
            return (
                None,
                "gas comparison refused: observation has no independent fO2 pin",
                "refused",
            )
        # The gas observable is not wired in this package yet.
        #
        # Upstream this branch called the simulator's analytical vapour stack,
        # which reads a large vapour-pressure catalogue through two further
        # simulator subsystems. That stack is not worth porting: openimcc.gas
        # already covers every species this bench uses (Al, Ca, K, Mg, Na, SiO
        # are all in IMCC_GAS_CHANNEL_SPECIES), and its evaluate_gas takes
        # parent-oxide activities, T and fO2 -- exactly what is in scope here.
        # So the move is a rewire, not a transplant.
        #
        # ★ WHOEVER DOES THAT REWIRE: assert the units first. evaluate_gas
        # returns BAR; the upstream layer returned Pa, and every `measured`
        # value in a melt-activity-bench.v1 partial_pressure point is in Pa.
        # That is exactly 5 dex. Miss it and every gas residual shifts by five
        # orders of magnitude while the log-residual table still looks
        # entirely plausible.
        return (
            None,
            "partial_pressure is not wired in openimcc yet: the gas layer "
            "rewire onto openimcc.gas.evaluate_gas is pending (see "
            "docs/EXTRACTION.md)",
            "refused",
        )
    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        return (
            None,
            f"engine returned no positive {observable} for {parent}",
            "refused",
        )
    return float(value), "", "ok"


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
    residuals = [row.residual for row in rows if row.residual is not None]
    return Aggregate(
        key=key,
        n=len(rows),
        n_ok=sum(row.status == "ok" for row in rows),
        rmse=_rmse(value for value in residuals if value is not None),
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
        predicted, prediction_reason, status = _prediction_for_point(
            enriched, status, activities, gammas, engine_reason
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
        rmse=overall.rmse,
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
        f"{'refused':>8} {'ood':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in aggregates:
        lines.append(
            f"{row.key:<32} {row.n:>5} {row.n_ok:>5} "
            f"{_fmt_float(row.rmse):>10} {_fmt_float(row.median_abs_residual):>10} "
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
            f"RMSE={_fmt_float(report.rmse)}  "
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
        f"{'meas':>10} {'pred':>10} {'resid':>10} {'ratio':>10} {'status':<22}"
    )
    lines.append(point_header)
    lines.append("-" * len(point_header))
    for row in report.points:
        lines.append(
            f"{row.point_id:<28} {row.population:<24} {row.species:<6} "
            f"{row.observable:<18} {_fmt_float(row.measured):>10} "
            f"{_fmt_float(row.predicted):>10} {_fmt_float(row.residual):>10} "
            f"{_fmt_float(row.ratio):>10} {row.status:<22}"
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
