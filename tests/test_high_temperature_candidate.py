from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

from openimcc import evaluate, load_datapack
from openimcc.gas import (
    evaluate_gas,
    load_gas_datapack,
)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import build_gas_tables
from tools.build_gas_tables import _condensate_row_g_s


GAS_CANDIDATE = ROOT / "src/openimcc/data/gas/gas-shomate-hight-candidate.csv"
OXIDE_CANDIDATE = ROOT / "src/openimcc/data/gas/condensate-hight-candidate.csv"
COMPOSITION = {
    "SiO2": 51.85068,
    "MgO": 4.78527,
    "FeO": 13.77307,
    "CaO": 9.02862,
    "Al2O3": 14.80572,
    "TiO2": 1.73824,
    "Na2O": 3.23108,
    "K2O": 0.78732,
}
ACTIVITIES = {
    "SiO2": 0.55,
    "MgO": 0.05,
    "FeO": 0.10,
    "CaO": 0.10,
    "Al2O3": 0.16,
    "TiO2": 0.01,
    "Na2O": 0.03,
    "K2O": 0.01,
}


def _assert_close(left: float, right: float) -> None:
    assert abs(left - right) <= 1e-6 * max(abs(left), abs(right), 1.0)


def test_default_tables_remain_byte_identical_and_candidate_prefixes_match() -> None:
    default_gas = ROOT / "src/openimcc/data/gas/gas-shomate.csv"
    default_oxide = ROOT / "src/openimcc/data/gas/condensate.csv"
    assert hashlib.sha256(default_gas.read_bytes()).hexdigest() == (
        "e01c363ac6bb526b5e8ee6df300963652e0771585e9348e6456afef5cacb8058"
    )
    assert hashlib.sha256(default_oxide.read_bytes()).hexdigest() == (
        "1b9149eb9d53ffd7d96872b25b27144bca902a71b963fcec9b6bf9cec37127a0"
    )
    for default, candidate in ((default_gas, GAS_CANDIDATE), (default_oxide, OXIDE_CANDIDATE)):
        assert candidate.read_bytes().splitlines()[: len(default.read_bytes().splitlines())] == (
            default.read_bytes().splitlines()
        )


def test_candidate_tables_load_and_preserve_gas_and_liquid_seams() -> None:
    default = load_gas_datapack()
    candidate = load_gas_datapack(str(GAS_CANDIDATE), str(OXIDE_CANDIDATE))

    for species, *_ in build_gas_tables.HIGH_TEMP_GAS_CARDS:
        base = default.gas_df.loc[species].sort_values("T_min").iloc[-1]
        high = candidate.gas_df.loc[species].sort_values("T_min").iloc[-1]
        for left, right in zip(
            build_gas_tables._gas_row_h_s(base, 3000.0),
            build_gas_tables._gas_row_h_s(high, 3000.0),
        ):
            _assert_close(float(left), float(right))
        hb, sb = build_gas_tables._gas_row_h_s(base, 3000.0)
        hh, sh = build_gas_tables._gas_row_h_s(high, 3000.0)
        _assert_close(float(hb - 3000.0 * sb), float(hh - 3000.0 * sh))

    for species, _, _, _, _, bridge in build_gas_tables.HIGH_TEMP_LIQUID_CARDS:
        rows = candidate.oxide_df.loc[species].sort_values("T_min")
        base = default.oxide_df.loc[species].sort_values("T_min").iloc[-1]
        at_3000 = rows.loc[rows["T_min"] == 3000.0].iloc[-1]
        gb, sb = _condensate_row_g_s(base, 3000.0)
        gh, sh = _condensate_row_g_s(at_3000, 3000.0)
        _assert_close(gb, gh)
        _assert_close(sb, sh)
        _assert_close(gb + 3000.0 * sb, gh + 3000.0 * sh)
        if bridge > 3000.0:
            left = rows.loc[rows["T_max"] == bridge].iloc[-1]
            right = rows.loc[rows["T_min"] == bridge].iloc[-1]
            gl, sl = _condensate_row_g_s(left, bridge)
            gr, sr = _condensate_row_g_s(right, bridge)
            _assert_close(gl, gr)
            _assert_close(sl, sr)
            _assert_close(gl + bridge * sl, gr + bridge * sr)

    feo_base = default.oxide_df.loc[["FeO(l)"]].sort_values("T_min").iloc[-1]
    feo_high = candidate.oxide_df.loc[["FeO(l)"]].sort_values("T_min").iloc[-1]
    gb, sb = _condensate_row_g_s(feo_base, 3000.0)
    gh, sh = _condensate_row_g_s(feo_high, 3000.0)
    _assert_close(gb, gh)
    _assert_close(sb, sh)
    _assert_close(gb + 3000.0 * sb, gh + 3000.0 * sh)


def test_candidate_results_match_defaults_below_3000_k() -> None:
    default = load_gas_datapack()
    candidate = load_gas_datapack(str(GAS_CANDIDATE), str(OXIDE_CANDIDATE))
    base = evaluate_gas(ACTIVITIES, 2500.0, 1.0e-10, default)
    high = evaluate_gas(ACTIVITIES, 2500.0, 1.0e-10, candidate)
    assert dict(base) == dict(high)
    assert dict(base.domain_flags) == dict(high.domain_flags)


def test_high_temperature_gas_and_basalt_melt_runs_are_flagged() -> None:
    candidate = load_gas_datapack(str(GAS_CANDIDATE), str(OXIDE_CANDIDATE))
    for temperature in (3500.0, 4500.0, 6000.0):
        result = evaluate_gas(
            ACTIVITIES, temperature, 1.0e-10, candidate,
            allow_extrapolation=True,
        )
        assert result
        assert all(math.isfinite(value) and value >= 0.0 for value in result.values())
        assert any(
            "rough high-temperature extension" in (flag or "")
            for flag in result.domain_flags.values()
        )

    melt_pack = load_datapack(ROOT / "src/openimcc/data/packs/imcc-sf04-v1.0.2.json")
    for temperature in (3500.0, 4500.0):
        melt = evaluate(
            COMPOSITION,
            temperature,
            pack=melt_pack,
            basis_type="wt",
            allow_extrapolation=True,
        )
        assert melt.extrapolated
        activities = dict(zip(melt.parent_oxides, melt.parent_activity))
        result = evaluate_gas(
            activities, temperature, 1.0e-10, candidate,
            allow_extrapolation=True,
            composition_wt_pct=COMPOSITION,
        )
        assert all(math.isfinite(value) and value >= 0.0 for value in result.values())
        assert any("rough high-temperature extension" in (flag or "") for flag in result.domain_flags.values())
