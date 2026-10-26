from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

from openimcc import evaluate, load_datapack
import openimcc.gas as gas_module
from openimcc.gas import (
    evaluate_gas,
    load_gas_datapack,
    _source_model_flag,
)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import build_gas_tables
from tools.build_gas_tables import _condensate_row_g_s


GAS_CANDIDATE = ROOT / "src/openimcc/data/gas/gas-shomate-hight-candidate.csv"
OXIDE_CANDIDATE = ROOT / "src/openimcc/data/gas/condensate-hight-candidate.csv"
GAS_CANDIDATE_V2 = ROOT / "src/openimcc/data/gas/gas-shomate-hight-candidate-v2.csv"
OXIDE_CANDIDATE_V2 = ROOT / "src/openimcc/data/gas/condensate-hight-candidate-v2.csv"
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


def test_v2_candidate_loads_preserves_prefixes_and_all_seams() -> None:
    default_gas = ROOT / "src/openimcc/data/gas/gas-shomate.csv"
    default_oxide = ROOT / "src/openimcc/data/gas/condensate.csv"
    v1_gas = ROOT / "src/openimcc/data/gas/gas-shomate-hight-candidate.csv"
    v1_oxide = ROOT / "src/openimcc/data/gas/condensate-hight-candidate.csv"
    assert hashlib.sha256(v1_gas.read_bytes()).hexdigest() == "4dc41e150e22beb90f6b6b8dec67b0f4d993b5984e2e3ce1ec792d52b4c0f036"
    assert hashlib.sha256(v1_oxide.read_bytes()).hexdigest() == "aca172c1d435e520e8a16f5a58ce5ae67c191c1fafc34bf98c080d447efb973d"
    assert hashlib.sha256(GAS_CANDIDATE_V2.read_bytes()).hexdigest() == "55769977fd5ea4c549796be7b23baeb916d76c7904ace9cb05e0cd159918d972"
    for default, v1, v2 in ((default_gas, v1_gas, GAS_CANDIDATE_V2),
                            (default_oxide, v1_oxide, OXIDE_CANDIDATE_V2)):
        lines = default.read_bytes().splitlines()
        assert v1.read_bytes().splitlines()[:len(lines)] == lines
        assert v2.read_bytes().splitlines()[:len(lines)] == lines

    default = load_gas_datapack()
    v2 = load_gas_datapack(str(GAS_CANDIDATE_V2), str(OXIDE_CANDIDATE_V2))
    for species, *_ in build_gas_tables.HIGH_TEMP_GAS_CARDS:
        base = default.gas_df.loc[species].sort_values("T_min").iloc[-1]
        high = v2.gas_df.loc[species].sort_values("T_min").iloc[-1]
        for left, right in zip(build_gas_tables._gas_row_h_s(base, 3000),
                                build_gas_tables._gas_row_h_s(high, 3000)):
            _assert_close(float(left), float(right))

    for species in v2.oxide_df.index.unique():
        rows = v2.oxide_df.loc[[species]].sort_values("T_min")
        candidates = rows.loc[rows["Ref"].astype(str).str.contains("HT-CANDIDATE")]
        if candidates.empty:
            continue
        first = candidates.iloc[0]
        if float(first["T_min"]) == 3000:
            base = default.oxide_df.loc[[species]].sort_values("T_min").iloc[-1]
            base_g, base_s = _condensate_row_g_s(base, 3000)
            first_g, first_s = _condensate_row_g_s(first, 3000)
            for left, right in ((base_g, first_g), (base_s, first_s),
                                (base_g + 3000*base_s, first_g + 3000*first_s)):
                _assert_close(left, right)
        for (_, left), (_, right) in zip(candidates.iloc[:-1].iterrows(), candidates.iloc[1:].iterrows()):
            seam = float(left["T_max"])
            if seam != float(right["T_min"]):
                continue
            gl, sl = _condensate_row_g_s(left, seam)
            gr, sr = _condensate_row_g_s(right, seam)
            for a, b in ((gl, gr), (sl, sr), (gl + seam*sl, gr + seam*sr)):
                _assert_close(a, b)

    low_default = evaluate_gas(ACTIVITIES, 2500.0, 1.0e-10, default)
    low_v2 = evaluate_gas(ACTIVITIES, 2500.0, 1.0e-10, v2)
    assert dict(low_default) == dict(low_v2)
    assert dict(low_default.domain_flags) == dict(low_v2.domain_flags)


def test_v2_candidate_flags_show_per_species_tier_and_band() -> None:
    candidate = load_gas_datapack(str(GAS_CANDIDATE_V2), str(OXIDE_CANDIDATE_V2))
    expected = {
        "SiO(g)": (
            "corroborated by calculation over 3000-6000 K",
            "within at most 0.18033 kJ/mol",
            "at most ±0.00157 dex",
            "over that range",
        ),
        "Mg(g)": (
            "corroborated by calculation over 3000-6000 K",
            "within at most 0.00218 kJ/mol",
            "at most ±2.8e-05 dex",
            "over that range",
        ),
        "Fe(g)": (
            "corroborated by calculation over 3000-6000 K",
            "within at most 0.100721 kJ/mol",
            "at most ±0.000877 dex",
            "over that range",
        ),
        "Na(g)": (
            "corroborated by calculation over 3000-6000 K",
            "within at most 0.02452 kJ/mol",
            "at most ±0.000213 dex",
            "over that range",
        ),
        "Al2O3(l)": (
            "rough above 4000 K",
            "diverge by up to 4.45627 kJ/mol",
            "maximum pressure band ±0.058192 dex",
            "over the reported comparison range",
        ),
    }
    for species, parts in expected.items():
        frame = candidate.gas_df if species.endswith("(g)") else candidate.oxide_df
        phase = "g" if species.endswith("(g)") else "l"
        for temperature in (3500.0, 4500.0, 5500.0):
            row = frame.loc[species].sort_values("T_min")
            row = row.loc[row["T_min"] <= temperature].iloc[-1]
            flag = _source_model_flag(species, temperature, row, phase=phase)
            assert flag is not None
            assert all(part in flag for part in parts), (species, temperature, flag)


def test_v2_candidate_ions_cover_6000_k_and_close_charge_for_k_bearing_melt() -> None:
    candidate = load_gas_datapack(str(GAS_CANDIDATE_V2), str(OXIDE_CANDIDATE_V2))
    for temperature in (4000.0, 5000.0, 6000.0):
        melt = evaluate(
            COMPOSITION, T_K=temperature, basis_type="wt",
            allow_extrapolation=True,
        )
        activities = dict(zip(melt.parent_oxides, melt.parent_activity))
        result = evaluate_gas(
            activities, temperature, 1.0e-10, candidate,
            include_ions=True, allow_extrapolation=True,
        )
        assert math.isfinite(result["K+"]) and result["K+"] > 0.0
        assert math.isfinite(result["e-"]) and result["e-"] > 0.0
        for species in ("K+", "e-"):
            flag = result.domain_flags[species] or ""
            assert "high-temperature extension" in flag
            assert "out-of-domain extrapolation" not in flag
            assert "outside its declared" not in flag

        positive = sum(species_value for species, species_value in result.items() if species.endswith("+"))
        negative = sum(
            result.get(species, 0.0)
            for species in (
                "Na-", "K-", "O-", "Al-", "Fe-", "Si-", "Ti-", "O2-",
                "AlO-", "AlO2-", "KO-", "NaO-", "Cr-", "V-", "Nb-",
            )
        )
        residual = result["e-"] + negative - positive
        scale = max(result["e-"] + negative, positive)
        assert abs(residual) <= 1.0e-12 * scale


def test_v2_major_ion_rows_cover_298_to_6000_k_and_join_at_3000() -> None:
    candidate = load_gas_datapack(str(GAS_CANDIDATE_V2), str(OXIDE_CANDIDATE_V2))
    default = load_gas_datapack()
    new_ions = {"Al+(g)", "Fe+(g)", "Mg+(g)", "Si+(g)", "Ti+(g)", "Cr+(g)"}
    for species, _card, _table, _cation, _cat_num, _oxy_num in build_gas_tables.HIGH_TEMP_V2_ION_CARDS:
        rows = candidate.gas_df.loc[[species]].sort_values("T_min")
        high = rows.loc[rows["T_min"] == 3000.0].iloc[-1]
        assert float(rows["T_min"].min()) == (298.15 if species in new_ions else float(default.gas_df.loc[species]["T_min"].min()))
        assert float(high["T_max"]) == 6000.0
        low = rows.loc[rows["T_max"] == 3000.0].iloc[-1]
        for left, right in zip(
            build_gas_tables._gas_row_h_s(low, 3000.0),
            build_gas_tables._gas_row_h_s(high, 3000.0),
        ):
            _assert_close(float(left), float(right))
        low_h, low_s = build_gas_tables._gas_row_h_s(low, 3000.0)
        high_h, high_s = build_gas_tables._gas_row_h_s(high, 3000.0)
        _assert_close(float(low_h - 3000.0 * low_s), float(high_h - 3000.0 * high_s))


def test_default_ion_outputs_match_the_pre_expansion_pair_set(monkeypatch) -> None:
    default = load_gas_datapack()
    expanded_pairs = gas_module._GAS_IONIZATION_PAIRS
    added_pairs = tuple(
        pair for pair in expanded_pairs
        if pair[1] in {"Al+", "Fe+", "Mg+", "Si+", "Ti+", "Cr+"}
    )
    old_pairs = tuple(pair for pair in expanded_pairs if pair not in added_pairs)
    for include_ions in (False, True):
        monkeypatch.setattr(gas_module, "_GAS_IONIZATION_PAIRS", old_pairs)
        before = evaluate_gas(
            ACTIVITIES, 3000.0, 1.0e-10, default, include_ions=include_ions,
        )
        monkeypatch.setattr(gas_module, "_GAS_IONIZATION_PAIRS", expanded_pairs)
        after = evaluate_gas(
            ACTIVITIES, 3000.0, 1.0e-10, default, include_ions=include_ions,
        )
        assert dict(after) == dict(before)
        assert dict(after.domain_flags) == dict(before.domain_flags)
        assert dict(after.provenance_class) == dict(before.provenance_class)
    assert not {ion for _neutral, ion in added_pairs} & set(after)


def test_v2_major_ions_are_present_in_the_6000_k_basalt_closure() -> None:
    candidate = load_gas_datapack(str(GAS_CANDIDATE_V2), str(OXIDE_CANDIDATE_V2))
    melt = evaluate(
        COMPOSITION, T_K=6000.0, basis_type="wt", allow_extrapolation=True,
    )
    activities = dict(zip(melt.parent_oxides, melt.parent_activity))
    result = evaluate_gas(
        activities, 6000.0, 1.0e-10, candidate,
        include_ions=True, allow_extrapolation=True,
    )
    for element in ("Al", "Fe", "Mg"):
        assert result[element + "+"] > 0.0
        assert result[element] > 0.0
        fraction = result[element + "+"] / (result[element + "+"] + result[element])
        assert fraction > 0.0
