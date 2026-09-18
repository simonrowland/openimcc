#!/usr/bin/env bash
#
# Run the CI gates on this machine.
#
# GitHub Actions is the nominal home for these, but it is not always available
# (a billing lock stops jobs before they start, and a workflow that never runs
# is decorative). This script is the same set of gates, runnable locally, so
# "CI passed" means something either way.
#
# It is deliberately NOT just `pytest`. Two of the three gates catch failures
# the test suite cannot see on a developer machine:
#
#   drift gate  - a datapack edited under an unchanged version string silently
#                 moves every activity the engine reports.
#   core-only   - the package promises `pip install openimcc` needs numpy and
#                 scipy alone. One convenience import in a core module breaks
#                 that for users while every local test still passes, because
#                 the dev environment has the extras installed.
#
# Usage:  scripts/ci-local.sh [--quick]
#           --quick   current interpreter only; skip the version matrix
set -uo pipefail

cd "$(dirname "$0")/.."
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
FAILED=(); SKIPPED=()

pass() { printf '%s  PASS%s  %s\n' "$GRN" "$OFF" "$1"; }
fail() { printf '%s  FAIL%s  %s\n' "$RED" "$OFF" "$1"; FAILED+=("$1"); }
skip() { printf '%s  SKIP%s  %s %s(%s)%s\n' "$YEL" "$OFF" "$1" "$DIM" "$2" "$OFF"; SKIPPED+=("$1"); }
head() { printf '\n%s== %s ==%s\n' "$DIM" "$1" "$OFF"; }

# Prefer uv: it creates throwaway venvs in ~a second and can fetch interpreters
# we do not have. Without it we fall back to whatever is already on PATH.
if command -v uv >/dev/null 2>&1; then HAVE_UV=1; else HAVE_UV=0; fi

TMPROOT="$(mktemp -d)"
trap 'rm -rf "$TMPROOT"' EXIT

# --------------------------------------------------------------------------
head "datapack drift"
# First on purpose: a drifted pack makes every residual below it meaningless,
# so failing here is clearer than a confusing test failure downstream.
if python3 tools/packmanifest.py --check >/dev/null 2>&1; then
    pass "packs match MANIFEST.json"
else
    fail "datapack drift (run: python3 tools/packmanifest.py --write)"
    python3 tools/packmanifest.py --check 2>&1 | sed 's/^/        /'
fi

# --------------------------------------------------------------------------
head "test matrix"
if [ "$QUICK" = "1" ]; then
    VERSIONS=("$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')")
else
    VERSIONS=(3.11 3.12 3.13 3.14)
fi

for V in "${VERSIONS[@]}"; do
    VENV="$TMPROOT/py$V"
    if [ "$HAVE_UV" = "1" ]; then
        uv venv --python "$V" "$VENV" >/dev/null 2>&1 || { skip "tests on $V" "interpreter unavailable"; continue; }
        VPY="$VENV/bin/python"
        uv pip install --python "$VPY" -q -e ".[dev]" >/dev/null 2>&1 || { fail "install on $V"; continue; }
    else
        command -v "python$V" >/dev/null 2>&1 || { skip "tests on $V" "no python$V and no uv"; continue; }
        "python$V" -m venv "$VENV" >/dev/null 2>&1 || { skip "tests on $V" "venv creation failed"; continue; }
        VPY="$VENV/bin/python"
        "$VPY" -m pip install -q -e ".[dev]" >/dev/null 2>&1 || { fail "install on $V"; continue; }
    fi
    OUT="$("$VPY" -m pytest tests/ -q 2>&1)"
    if printf '%s' "$OUT" | tail -1 | grep -qE '^[0-9]+ passed'; then
        pass "tests on $V  $DIM$(printf '%s' "$OUT" | tail -1)$OFF"
    else
        fail "tests on $V"
        printf '%s' "$OUT" | tail -12 | sed 's/^/        /'
    fi
done

# --------------------------------------------------------------------------
head "core-only install"
# The gate the dev environment can never catch: install WITHOUT extras and
# prove the core solve still works. pandas and yaml must be absent, or the
# check proves nothing.
VENV="$TMPROOT/core"
if [ "$HAVE_UV" = "1" ]; then
    uv venv "$VENV" >/dev/null 2>&1 && VPY="$VENV/bin/python" && uv pip install --python "$VPY" -q -e . >/dev/null 2>&1
else
    python3 -m venv "$VENV" >/dev/null 2>&1 && VPY="$VENV/bin/python" && "$VPY" -m pip install -q -e . >/dev/null 2>&1
fi
if [ -x "${VPY:-}" ]; then
    OUT="$("$VPY" - <<'PY' 2>&1
import importlib.util, sys
present = [m for m in ("pandas", "yaml") if importlib.util.find_spec(m)]
if present:
    sys.exit(f"vacuous: {present} installed; this gate must run without them")
from openimcc import load_datapack, evaluate
pack = load_datapack("src/openimcc/data/packs/imcc-sf04-v1.0.2.json")
r = evaluate({"SiO2": 0.45, "MgO": 0.10, "CaO": 0.15,
              "Al2O3": 0.15, "FeO": 0.15}, 1800.0, pack)
assert r.D > 1.0, r.D
print(f"core-only solve OK, D = {r.D:.6f}")
PY
)"
    if printf '%s' "$OUT" | grep -q 'core-only solve OK'; then
        pass "core install needs numpy+scipy only  $DIM$(printf '%s' "$OUT" | tail -1)$OFF"
    else
        fail "core-only install"
        printf '%s' "$OUT" | tail -8 | sed 's/^/        /'
    fi
else
    fail "core-only install (could not build the venv)"
fi

# --------------------------------------------------------------------------
printf '\n'
if [ ${#FAILED[@]} -eq 0 ]; then
    printf '%sall gates passed%s' "$GRN" "$OFF"
    [ ${#SKIPPED[@]} -gt 0 ] && printf ' %s(%d skipped)%s' "$DIM" "${#SKIPPED[@]}" "$OFF"
    printf '\n'; exit 0
fi
printf '%s%d gate(s) failed:%s\n' "$RED" "${#FAILED[@]}" "$OFF"
printf '  - %s\n' "${FAILED[@]}"
exit 1
