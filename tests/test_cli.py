"""IMCC-SF04 CLI and package-split tests.

Two things are pinned here:

1. The CLI contract -- exit 0 solved, exit 2 typed refusal, exit 1 usage error.
   The refusal code matters: a caller scripting against this needs to tell
   "the model declined" apart from "the invocation was wrong".
2. The MODEL/GLUE split itself. ``adapter.py`` + ``kernel.py`` + ``gas.py`` +
   ``cli.py`` must not import simulator policy. That is the property that makes
   the package extractable, and it is invisible to every other test -- nothing
   fails if someone re-adds a ``simulator.backend_names`` import to the model
   half, it just quietly re-welds the seam.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from openimcc import cli

PACK = Path("src/openimcc/data/packs/imcc-sf04-v1.0.2.json")

BASALT = [
    "--oxide", "SiO2=45.4",
    "--oxide", "MgO=8.1",
    "--oxide", "FeO=10.9",
    "--oxide", "CaO=11.4",
    "--oxide", "Al2O3=14.2",
    "--oxide", "TiO2=3.2",
    "--oxide", "Na2O=0.4",
    "--oxide", "K2O=0.1",
]

# The published pack declares [1700, 3000] K. 900 K is far outside it and no
# --allow-extrapolation is passed, so the model must refuse rather than answer.
T_IN_DOMAIN = "1800"
T_OUT_OF_DOMAIN = "900"


def _solve_argv(temperature: str, *extra: str) -> list[str]:
    return [
        "solve",
        "--pack", str(PACK),
        "--temperature", temperature,
        "--basis-type", "wt",
        *BASALT,
        *extra,
    ]


def test_describe_reports_the_pack_identity(capsys):
    assert cli.main(["describe", "--pack", str(PACK)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "IMCC-SF04" in out
    assert "1.0.2" in out


def test_describe_collapses_the_repetitive_per_row_domain_basis(capsys):
    """domain_basis is one long string PER REACTION; printing it verbatim is a
    screenful of duplicates. The published pack has 38 rows and 2 distinct
    values, so the summary must be far shorter than the raw join."""
    assert cli.main(["describe", "--pack", str(PACK)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "distinct over" in out
    # The raw join would repeat the long ADOPTED string ~34 times.
    assert out.count("sf04-exercised-ADOPTED") <= 2


def test_solve_in_domain_exits_ok_and_emits_activities(capsys):
    assert cli.main(_solve_argv(T_IN_DOMAIN, "--json")) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["temperature_K"] == pytest.approx(1800.0)
    assert payload["parent_oxides"][0] == "SiO2"
    # One activity/gamma per parent oxide, all finite and non-negative.
    n = len(payload["parent_oxides"])
    assert len(payload["parent_activity"]) == n
    assert len(payload["parent_gamma"]) == n
    assert all(a >= 0.0 for a in payload["parent_activity"])
    assert payload["D"] > 1.0  # an associated solution, not ideal mixing


def test_solve_and_describe_use_the_packaged_default(capsys):
    argv = [
        "solve",
        "--temperature", T_IN_DOMAIN,
        "--basis-type", "wt",
        *BASALT,
        "--json",
    ]
    assert cli.main(argv) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["labels"]["identity"]["datapack_version"] == "1.0.2"
    assert payload["labels"]["notices"]

    assert cli.main(["describe"]) == cli.EXIT_OK
    assert "1.0.2" in capsys.readouterr().out


def test_out_of_domain_is_a_typed_refusal_not_a_crash(capsys):
    """Exit 2 with a machine-readable code. A refusal is a RESULT."""
    assert cli.main(_solve_argv(T_OUT_OF_DOMAIN, "--json")) == cli.EXIT_REFUSED
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "refused"
    assert payload["code"] == "imcc_T_outside_datapack_domain"
    assert "900" in payload["reason"]


def test_extrapolation_flag_turns_the_refusal_into_a_flagged_answer(capsys):
    """The same point answers when extrapolation is explicitly allowed, and the
    answer carries the flag. Silent extrapolation would be the real defect."""
    assert cli.main(_solve_argv(T_OUT_OF_DOMAIN, "--allow-extrapolation", "--json")) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["extrapolated"] is True


def test_usage_errors_exit_one_not_two(capsys):
    """Exit 1 is 'you called it wrong'; exit 2 is reserved for the model
    declining. Collapsing them would make the CLI unscriptable."""
    assert cli.main(["solve", "--pack", str(PACK), "--temperature", "1800"]) == cli.EXIT_USAGE
    assert "no composition given" in capsys.readouterr().err


@pytest.mark.parametrize(
    "bad, fragment",
    [
        ("SiO2", "NAME=VALUE"),
        ("=45", "empty name"),
        ("SiO2=lots", "not numeric"),
    ],
)
def test_oxide_parsing_refuses_malformed_pairs(bad, fragment):
    with pytest.raises(ValueError) as excinfo:
        cli._parse_oxides([bad])
    assert fragment in str(excinfo.value)


def test_oxide_parsing_refuses_a_repeated_oxide():
    """A repeated --oxide is ambiguous; last-wins would silently discard input."""
    with pytest.raises(ValueError, match="more than once"):
        cli._parse_oxides(["SiO2=40", "SiO2=50"])


# The package these modules live in, used to resolve relative imports. A
# relative import records only the tail -- `from ...accounting.formulas import X`
# has node.module == "accounting.formulas" -- so a scanner that tests
# `startswith("simulator")` on the raw value sees nothing at all. Resolving
# against the package first is what makes the relative and absolute spellings
# compare equal, which is the whole point of these guards.
_PKG_PARTS = ("simulator", "melt_backend", "imcc_sf04")


def _resolve(module: str | None, level: int) -> str:
    """Absolute dotted name for one import, relative or not.

    level 0 is already absolute. level 1 is this package, level 2 its parent,
    and so on -- the same arithmetic importlib does:
        from .backend        (level 1) -> simulator.melt_backend.imcc_sf04.backend
        from ...accounting.x (level 3) -> simulator.accounting.x
    """
    if level == 0:
        return module or ""
    base = ".".join(_PKG_PARTS[: len(_PKG_PARTS) - (level - 1)])
    return f"{base}.{module}" if module else base


def _import_statements(tree: ast.AST, *, import_time_only: bool):
    """Yield import nodes, optionally only those that RUN on import.

    When import_time_only, descends through every statement container except
    function bodies. A module-level `try: import x except ImportError:` or a
    `class C: import x` executes the moment the module loads, so excluding them
    -- as a plain `tree.body` scan does -- leaves the guard blind in exactly the
    place someone writes a graceful-degradation import.
    """
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif import_time_only and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue  # a function body runs when called, not on import
        else:
            yield from _import_statements(node, import_time_only=import_time_only)


def _imported_modules(module_path: Path, *, import_time_only: bool = False) -> set[str]:
    """Resolved absolute module names imported by this file."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in _import_statements(tree, import_time_only=import_time_only):
        if isinstance(node, ast.ImportFrom):
            resolved = _resolve(node.module, node.level)
            if resolved.startswith("simulator"):
                found.add(resolved)
        else:
            for alias in node.names:
                if alias.name.startswith("simulator"):
                    found.add(alias.name)
    return found


def _simulator_imports(module_path: Path) -> set[str]:
    """Every simulator module this file imports, however deeply nested."""
    return _imported_modules(module_path)


_PKG = Path("src/openimcc")
_MODULES = ("kernel.py", "model.py", "gas.py", "cli.py", "bench.py", "scalar_boundary.py",
            "__init__.py")

# Upstream, these guards policed a MODEL/GLUE split: the model half had to stay
# free of the simulator's policy modules so the package could be lifted out.
# The lift has happened, so that framing no longer applies -- there is no glue
# here. What survives is the property the split existed to protect, now stated
# directly: nothing in this package may reach back into the simulator, and the
# core solve must not pull the optional dependencies.


def _third_party_imports(module_path: Path) -> set[str]:
    """Top-level names imported by a module, at any nesting depth."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
    return found


@pytest.mark.parametrize("module", _MODULES)
def test_nothing_reaches_back_into_the_simulator(module):
    """Backslide guard.

    This package was extracted from a larger simulator. Copying a helper back
    in from upstream is the easy mistake, and it would not fail anything else
    here -- it would fail for USERS, at import time, with ModuleNotFoundError.
    """
    offending = sorted(
        name for name in _third_party_imports(_PKG / module) if name == "simulator"
    )
    assert not offending, (
        f"{module} imports the simulator this package was extracted from. "
        "Port the helper or refuse cleanly; do not reach upstream."
    )


@pytest.mark.parametrize("module", ["kernel.py", "model.py", "cli.py", "scalar_boundary.py"])
def test_core_modules_do_not_import_the_optional_extras(module):
    """The dependency tiering is a promise in pyproject; this keeps it true.

    `pip install openimcc` brings numpy and scipy only. pandas belongs to the
    [gas] extra and pyyaml to [bench]. One convenience import in a core module
    silently breaks every core install, and the failure appears on the user's
    machine rather than here.
    """
    extras = {"pandas", "yaml"} & _third_party_imports(_PKG / module)
    assert not extras, (
        f"{module} imports {sorted(extras)}, which are optional extras. "
        "Core install would break; move the use or defer the import."
    )


# --- exit-code contract: argparse must not squat on the refusal code ---------

# An alkali fraction above the pack's 0.5 bound; refused unless explicitly
# allowed. See tests/test_imcc_adapter.py::test_allow_out_of_envelope_labels_result.
_OUT_OF_ENVELOPE = ["--oxide", "Na2O=70.0", "--oxide", "SiO2=30.0"]


@pytest.mark.parametrize(
    "argv, why",
    [
        (["solve"], "required --pack/--temperature missing"),
        (["solve", "--nonesuch"], "unknown flag"),
        ([], "no subcommand"),
        (["solve", "--pack", str(PACK), "--temperature", "notanumber"], "bad type"),
    ],
)
def test_argparse_usage_errors_do_not_return_the_refusal_code(argv, why):
    """argparse exits 2 by default -- the same code this CLI reserves for a
    typed refusal. Unmapped, a caller scripting `rc == 2` would read "you
    called it wrong" as "the model declined", and main() would raise
    SystemExit instead of returning at all."""
    assert cli.main(argv) == cli.EXIT_USAGE, why


def test_help_still_exits_zero():
    """The remap above must not swallow --help's successful exit."""
    assert cli.main(["--help"]) == cli.EXIT_OK


def test_json_is_accepted_before_or_after_the_subcommand(capsys):
    """The parent-parser + SUPPRESS pattern is load-bearing: dropping SUPPRESS
    would let the subparser default clobber a leading --json, and every other
    test here passes --json trailing so nothing would notice."""
    argv = ["--pack", str(PACK), "--temperature", T_IN_DOMAIN, "--basis-type", "wt", *BASALT]
    assert cli.main(["--json", "solve", *argv]) == cli.EXIT_OK
    leading = json.loads(capsys.readouterr().out)
    assert cli.main(["solve", *argv, "--json"]) == cli.EXIT_OK
    trailing = json.loads(capsys.readouterr().out)
    assert leading == trailing


def test_out_of_envelope_refuses_then_answers_with_the_flag_visible(capsys):
    """Composition envelope, the sibling of the temperature-domain case above.

    The flag half is the point: the mandate's posture is predict AND flag, so
    an allowed out-of-envelope answer must carry its notice. Text mode was
    printing a clean-looking result with nothing to distinguish it.
    """
    argv = ["solve", "--pack", str(PACK), "--temperature", T_IN_DOMAIN,
            "--basis-type", "wt", *_OUT_OF_ENVELOPE]

    assert cli.main([*argv, "--json"]) == cli.EXIT_REFUSED
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["code"] == "imcc_composition_outside_validated_envelope"

    assert cli.main([*argv, "--allow-out-of-envelope", "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["labels"]["envelope_status"] == "outside_validated"

    assert cli.main([*argv, "--allow-out-of-envelope"]) == cli.EXIT_OK
    assert "OUTSIDE VALIDATED COMPOSITION ENVELOPE" in capsys.readouterr().out


def test_the_backslide_scanner_is_not_bypassable_by_nesting(tmp_path):
    """Pin the detector, not just its current verdict.

    A scan of module-level statements only would miss every one of these, and
    they are not exotic: a `try: import x except ImportError:` is the standard
    way someone reintroduces an upstream helper "safely", and a function-body
    import is the standard way someone hides one. Both still fail for a user
    who does not have the simulator installed -- the try/except at import time,
    the function-body one the first time that code path runs.
    """
    hidden = {
        "bare": "import simulator\n",
        "from-import": "from simulator.accounting import formulas\n",
        "try/except": "try:\n    import simulator\nexcept ImportError:\n    simulator = None\n",
        "if": "if True:\n    import simulator\n",
        "class body": "class C:\n    import simulator\n",
        "function body": "def go():\n    import simulator\n    return simulator\n",
        "method body": "class C:\n    def go(self):\n        from simulator import x\n",
        "nested": "try:\n    if True:\n        import simulator\nexcept ImportError:\n    pass\n",
    }
    for label, source in hidden.items():
        probe = tmp_path / "probe.py"
        probe.write_text(source, encoding="utf-8")
        assert "simulator" in _third_party_imports(probe), (
            f"{label!r} hides a simulator import from the backslide guard"
        )

    for label, source in {
        "own package": "from openimcc.kernel import solve\n",
        "relative": "from .kernel import solve\n",
        "stdlib": "import json\n",
        "lookalike": "import simulator_utils\n",
    }.items():
        probe = tmp_path / "benign.py"
        probe.write_text(source, encoding="utf-8")
        assert "simulator" not in _third_party_imports(probe), (
            f"false positive on {label!r}"
        )
