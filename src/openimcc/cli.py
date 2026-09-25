"""Standalone command-line interface to the IMCC-SF04 model.

Input-output only: a datapack plus a composition and a temperature in, melt
activities out. Deliberately depends on the MODEL half (``adapter`` +
``kernel``) and never on ``backend.py``, so it travels with the package if the
model is ever extracted from this repository.

Exit codes are part of the contract:

    0   solved
    2   typed refusal (out of domain, out of envelope, malformed pack, ...)
    1   usage / IO error

A refusal is a RESULT, not a crash -- it exits 2 with its typed ``code`` and
reason on stdout, in the same shape as a solve, so a caller can script against
it. Callers must not treat exit 2 as a failed invocation.

Examples
--------
    openimcc describe

    openimcc solve \\
        --temperature 1800 \\
        --oxide SiO2=45.4 --oxide MgO=8.1 --oxide FeO=10.9 --oxide CaO=11.4 \\
        --oxide Al2O3=14.2 --oxide TiO2=3.2 --oxide Na2O=0.4 --oxide K2O=0.1 \\
        --basis-type wt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from openimcc.model import (
    ImccLoadedDatapack,
    evaluate,
    load_datapack,
)
from openimcc.kernel import ImccRefusal

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_REFUSED = 2


def _parse_oxides(pairs: Sequence[str]) -> dict[str, float]:
    """Parse repeated ``--oxide NAME=VALUE`` arguments."""
    out: dict[str, float] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--oxide expects NAME=VALUE, got {item!r}")
        name, _, raw = item.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"--oxide has an empty name: {item!r}")
        if name in out:
            raise ValueError(f"--oxide {name} given more than once")
        try:
            out[name] = float(raw)
        except ValueError as exc:
            raise ValueError(f"--oxide {name} value {raw!r} is not numeric") from exc
    return out


def _load_composition(
    composition_path: str | None, oxides: Sequence[str]
) -> dict[str, float]:
    """Resolve the composition from a JSON file or repeated --oxide flags."""
    if composition_path and oxides:
        raise ValueError("pass --composition or --oxide, not both")
    if composition_path:
        data = json.loads(Path(composition_path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(
                f"{composition_path}: expected a JSON object of oxide -> value"
            )
        bad = sorted(k for k, v in data.items() if not isinstance(v, (int, float)))
        if bad:
            raise ValueError(f"{composition_path}: non-numeric values for {bad}")
        return {str(k): float(v) for k, v in data.items()}
    if not oxides:
        raise ValueError("no composition given; pass --composition or --oxide")
    return _parse_oxides(oxides)


def _pack_metadata(pack: ImccLoadedDatapack) -> dict[str, Any]:
    return {
        "model_id": pack.model_id,
        "version": pack.version,
        "parent_oxides": list(pack.parent_oxides),
        "domain_basis": list(pack.domain_basis),
        "extension_parents": list(pack.extension_parents),
        "extension_species": list(pack.extension_species),
        "reactions": len(pack.kernel_datapack.reactions),
    }


def _result_payload(result: Any) -> dict[str, Any]:
    """Project an ImccResult into a JSON-safe dict."""
    parents = [str(x) for x in result.parent_oxides]

    def _vec(attr: str) -> list[float]:
        values = getattr(result, attr, None)
        return [] if values is None else [float(v) for v in values]

    labels = getattr(result, "labels", None)
    label_payload: Any
    if labels is None:
        label_payload = None
    elif hasattr(labels, "__dict__"):
        label_payload = {k: _jsonable(v) for k, v in vars(labels).items()}
    else:
        label_payload = _jsonable(labels)

    return {
        "temperature_K": float(result.temperature_K),
        "basis": float(result.basis),
        "D": float(result.D),
        "extrapolated": bool(getattr(result, "extrapolated", False)),
        "parent_oxides": parents,
        "parent_mol": _vec("parent_mol"),
        "parent_x": _vec("parent_x"),
        "parent_x_star": _vec("parent_x_star"),
        "parent_activity": _vec("parent_activity"),
        "parent_gamma": _vec("parent_gamma"),
        "species_names": [str(s) for s in getattr(result, "species_names", ())],
        "species_x": _vec("species_x"),
        "labels": label_payload,
        "convergence": _jsonable(getattr(result, "convergence", None)),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "_asdict"):
        return {k: _jsonable(v) for k, v in value._asdict().items()}
    if hasattr(value, "__dict__") and vars(value):
        return {k: _jsonable(v) for k, v in vars(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _refusal_payload(exc: ImccRefusal) -> dict[str, Any]:
    return {
        "status": "refused",
        "code": getattr(exc, "code", type(exc).__name__),
        "type": type(exc).__name__,
        "reason": str(exc),
    }


def _render_solve_text(payload: Mapping[str, Any], *, basis_type: str) -> str:
    lines: list[str] = []
    labels = payload.get("labels") or {}
    model = labels.get("model_id") if isinstance(labels, Mapping) else None
    basis_label = "mole total (from wt%)" if basis_type == "wt" else "mole total"
    lines.append(
        f"IMCC solve  T = {payload['temperature_K']:.2f} K"
        f"   basis type = {basis_type}"
        f"   {basis_label} = {payload['basis']:.6g}"
        + (f"   model = {model}" if model else "")
    )
    if payload.get("extrapolated"):
        lines.append("  ** EXTRAPOLATED: outside a declared T domain **")
    # Symmetry with the line above. --allow-out-of-envelope turns a refusal
    # into an answer, so the answer has to SAY it is outside the validated
    # composition envelope; JSON already carries labels.envelope_status, and
    # text mode was printing a clean-looking result with no flag at all.
    if isinstance(labels, Mapping) and labels.get("envelope_status") == "outside_validated":
        lines.append("  ** OUTSIDE VALIDATED COMPOSITION ENVELOPE **")
    if isinstance(labels, Mapping):
        for flag in labels.get("flags") or ():
            lines.append(f"  ** FLAG: {flag} **")
        for notice in labels.get("notices") or ():
            lines.append(f"  notice: {notice}")
    lines.append("")
    header = f"  {'parent':<8} {'x':>12} {'x*':>12} {'activity':>14} {'gamma':>12}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    parents = payload["parent_oxides"]
    for i, name in enumerate(parents):
        def _at(key: str) -> float:
            seq = payload.get(key) or []
            return float(seq[i]) if i < len(seq) else float("nan")

        lines.append(
            f"  {name:<8} {_at('parent_x'):>12.6g} {_at('parent_x_star'):>12.6g} "
            f"{_at('parent_activity'):>14.6g} {_at('parent_gamma'):>12.6g}"
        )
    lines.append("")
    lines.append(f"  degree of association D = {payload['D']:.6g}")
    conv = payload.get("convergence")
    if conv is not None:
        lines.append(f"  convergence: {conv}")
    return "\n".join(lines)


def _cmd_describe(args: argparse.Namespace) -> int:
    pack = load_datapack(args.pack)
    meta = _pack_metadata(pack)
    if args.json:
        print(json.dumps(meta, indent=2, sort_keys=True))
        return EXIT_OK
    print(f"model_id       : {meta['model_id']}")
    print(f"version        : {meta['version']}")
    print(f"reactions      : {meta['reactions']}")
    print(f"parent_oxides  : {', '.join(meta['parent_oxides'])}")
    # domain_basis is PER-ROW and highly repetitive (one long string per
    # reaction). Joining it verbatim prints a screenful of duplicates, so
    # collapse to distinct values with their row counts.
    bases = meta["domain_basis"]
    if not bases:
        print("domain_basis   : (none)")
    else:
        counts: dict[str, int] = {}
        for basis in bases:
            counts[str(basis)] = counts.get(str(basis), 0) + 1
        print(f"domain_basis   : {len(counts)} distinct over {len(bases)} rows")
        for basis, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            shown = basis if len(basis) <= 96 else basis[:93] + "..."
            print(f"    [{n:>3}] {shown}")
    if meta["extension_parents"]:
        print(f"extension      : {', '.join(meta['extension_parents'])}")
    return EXIT_OK


def _cmd_validate_pack(args: argparse.Namespace) -> int:
    pack = load_datapack(args.pack)
    pack_name = str(args.pack) if args.pack else f"packaged imcc-sf04-v{pack.version}"
    payload = {"status": "ok", "pack": pack_name}
    print(json.dumps(payload) if args.json else f"ok: {pack_name} loads and validates")
    return EXIT_OK


def _cmd_solve(args: argparse.Namespace) -> int:
    composition = _load_composition(args.composition, args.oxide)
    pack = load_datapack(args.pack)
    result = evaluate(
        composition,
        args.temperature,
        pack,
        basis=args.basis,
        basis_type=args.basis_type,
        enable_sp_extension=args.enable_sp_extension,
        allow_extrapolation=args.allow_extrapolation,
        allow_out_of_envelope=args.allow_out_of_envelope,
        tol=args.tol,
        max_iter=args.max_iter,
    )
    payload = _result_payload(result)
    payload["status"] = "ok"
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_solve_text(payload, basis_type=args.basis_type))
    return EXIT_OK


def _resolve_version() -> str:
    """Code version for --version.

    Reads installed distribution metadata when this runs as the standalone
    package. In-tree -- executed out of a simulator checkout -- there is no
    distribution to query, so say so rather than invent a number: a wrong
    version string attached to a scientific result is worse than an honest
    "unknown", because it looks reproducible and is not.
    """
    try:
        from importlib.metadata import version

        return version("openimcc")
    except Exception:
        return "in-tree (no installed distribution)"


__version__ = _resolve_version()


# The exit-code contract is the reason anyone would script this, so it belongs
# in --help and not only in the module docstring. A caller that cannot tell a
# refusal from a usage error has to parse stderr, which is exactly what the
# typed codes exist to avoid.
_EPILOG = """\
exit codes:
  0  solved
  1  usage error -- the invocation was wrong
  2  typed refusal -- the model declined, with a machine-readable code

  A refusal is a RESULT, not a crash: out-of-domain temperature, a composition
  outside the validated envelope and a malformed datapack each carry their own
  code. Use --json to read it as {"status": "refused", "code": ...}.
  --allow-extrapolation and --allow-out-of-envelope turn a refusal into an
  answer; the answer is always flagged, never silently extrapolated.

examples:
  openimcc describe

  openimcc solve --temperature 1800 \\
      --basis-type wt \\
      --oxide SiO2=45.4 --oxide MgO=8.1 --oxide FeO=10.9 --oxide CaO=11.4 \\
      --oxide Al2O3=14.2 --oxide TiO2=3.2 --oxide Na2O=0.4 --oxide K2O=0.1

  openimcc --json solve --temperature 2200 \\
      --composition melt.json --allow-extrapolation
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openimcc",
        description=(
            "IMCC-SF04 melt-activity model: datapack in, activities out. "
            "Solves ideal mixing of complex components for a silicate liquid, "
            "returning per-parent-oxide activity and activity coefficient plus "
            "the degree of association D."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit JSON on stdout")
    parser.add_argument(
        "--version",
        action="version",
        version=f"openimcc {__version__}",
        help="print the engine version and exit",
    )

    # --json is accepted BOTH before and after the subcommand. Users reach for
    # the trailing form ("solve ... --json") and argparse would otherwise reject
    # it. SUPPRESS keeps the subparser from clobbering a leading --json with its
    # own default when the flag was given up front.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit JSON on stdout",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    describe = sub.add_parser(
        "describe", parents=[common], help="print datapack metadata"
    )
    describe.add_argument(
        "--pack", default=None, help="path to a datapack JSON (default: packaged SF04)"
    )
    describe.set_defaults(func=_cmd_describe)

    validate = sub.add_parser(
        "validate-pack", parents=[common], help="load and validate a datapack, then exit"
    )
    validate.add_argument(
        "--pack", default=None, help="path to a datapack JSON (default: packaged SF04)"
    )
    validate.set_defaults(func=_cmd_validate_pack)

    solve = sub.add_parser(
        "solve", parents=[common], help="solve melt activities at one T"
    )
    solve.add_argument(
        "--pack", default=None, help="path to a datapack JSON (default: packaged SF04)"
    )
    solve.add_argument(
        "--temperature", type=float, required=True, metavar="K", help="temperature in K"
    )
    solve.add_argument(
        "--composition", help="JSON file: {oxide: value} on the declared basis"
    )
    solve.add_argument(
        "--oxide",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="inline composition entry; repeatable",
    )
    solve.add_argument(
        "--basis-type",
        choices=("mol", "wt"),
        default="mol",
        help="units of the composition values (default: mol)",
    )
    solve.add_argument(
        "--basis",
        type=float,
        default=None,
        help="declared normalization basis; default is the composition sum",
    )
    solve.add_argument(
        "--enable-sp-extension",
        action="store_true",
        help="enable S and P2O5 parents (IMCC-SF04-EXT packs only)",
    )
    solve.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="evaluate outside declared T domains and mark the result",
    )
    solve.add_argument(
        "--allow-out-of-envelope",
        action="store_true",
        help="evaluate outside the validated composition envelope and mark it",
    )
    solve.add_argument(
        "--tol", type=float, default=1.0e-12,
        help="residual tolerance for the speciation solve (default: 1e-12)",
    )
    solve.add_argument(
        "--max-iter", type=int, default=100,
        help="iteration cap before the solve is declared non-convergent (default: 100)",
    )
    solve.set_defaults(func=_cmd_solve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse's own usage errors exit 2, which is OUR typed-refusal code.
        # Left alone, `imcc solve` with a missing --pack would be indis-
        # tinguishable from the model declining to answer, and main() would
        # raise instead of returning at all. Remap: --help/-h keeps its 0,
        # everything else becomes EXIT_USAGE so exit 2 means exactly one thing.
        code = exc.code if isinstance(exc.code, int) else EXIT_USAGE
        return EXIT_OK if code == 0 else EXIT_USAGE
    try:
        return int(args.func(args))
    except ImccRefusal as exc:
        # A typed refusal is a result. Emit it in the same shape as a solve and
        # exit 2 so a caller can distinguish "the model declined" from "the
        # invocation was wrong" (exit 1).
        payload = _refusal_payload(exc)
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"refused [{payload['code']}]: {payload['reason']}")
        return EXIT_REFUSED
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
