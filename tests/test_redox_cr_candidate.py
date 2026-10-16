from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from openimcc import evaluate_redox
from openimcc.redox import RedoxNumericalError
from openimcc.redox_pack import canonical_redox_digest, load_redox_pack


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "src/openimcc/data/packs/openimcc-redox-cr-candidate.json"
DEFAULT = ROOT / "src/openimcc/data/packs/openimcc-redox-v1.json"


def _candidate_result(T: float, log10_f_o2: float, inventory=None):
    feed = inventory or {
        "Si": 0.3, "Mg": 0.1, "Fe": 0.1, "Ca": 0.15, "Al": 0.1,
        "Ti": 0.01, "Na": 0.01, "K": 0.001, "Cr": 0.002, "O": 1.05,
    }
    return evaluate_redox(
        feed, T, "imposed", log10_f_o2 * math.log(10.0),
        redox_pack_path=CANDIDATE,
    )


def test_candidate_cr_fraction_matches_independent_2021_equation_at_three_points():
    for T, logf, inventory in [
        (1673.0, -10.0, None),
        (1700.0, -8.0, None),
        (1773.0, -6.0, None),
    ]:
        result = _candidate_result(T, logf, inventory)
        total_cr = result.chromous_cr_mol + result.chromic_cr_mol
        actual_fraction = result.chromous_cr_mol / total_cr

        # Independent single-cation composition calculation from input totals
        # and the returned Fe valence split; Cr contributes to the denominator.
        feed = inventory or {
            "Si": 0.3, "Mg": 0.1, "Fe": 0.1, "Ca": 0.15, "Al": 0.1,
            "Ti": 0.01, "Na": 0.01, "K": 0.001, "Cr": 0.002, "O": 1.05,
        }
        total_cations = sum(value for key, value in feed.items() if key != "O")
        x = {
            "NaO0.5": feed.get("Na", 0.0) / total_cations,
            "MgO": feed.get("Mg", 0.0) / total_cations,
            "Fe2+O": result.ferrous_fe_mol / total_cations,
            "AlO1.5": feed.get("Al", 0.0) / total_cations,
            "Fe3+O1.5": result.ferric_fe_mol / total_cations,
            "KO0.5": feed.get("K", 0.0) / total_cations,
            "CaO": feed.get("Ca", 0.0) / total_cations,
            "TiO2": feed.get("Ti", 0.0) / total_cations,
        }
        composition = sum(
            coefficient * x[name]
            for name, coefficient in {
                "NaO0.5": 2.00, "MgO": 1.00, "Fe2+O": 1.00,
                "AlO1.5": 0.37, "Fe3+O1.5": 0.37, "KO0.5": 2.12,
                "CaO": 2.44, "TiO2": 3.69,
            }.items()
        )
        qfm = 8.58 - 25050.0 / T
        log10_cr3_cr2 = 0.25 * (logf - qfm) + 3031.0 / T - 2.26 + composition
        expected = 1.0 / (1.0 + 10.0**log10_cr3_cr2)
        assert actual_fraction == pytest.approx(expected, abs=2e-10)
        assert result.composition_redox_iterations >= 2
        assert result.cr_oxygen_capacity_mol == pytest.approx(0.5 * result.chromous_cr_mol)


def test_algebraic_berry_oneill_2004_fixed_slope_positions_for_two_glasses():
    """Check the source equation against two Table 4 glass positions; no solver call."""
    # Table 1 oxide wt% converted independently to single-cation fractions.
    # The fixed-slope Table 4 log10 K' values are 1.871 (AD) and 1.921 (AD+Fo).
    mw = {"SiO2": 60.0843, "Al2O3": 101.9613, "MgO": 40.3044, "CaO": 56.0774}
    glasses = [
        ({"SiO2": 50.6, "Al2O3": 15.1, "MgO": 10.3, "CaO": 24.0}, 1.871, 0.06),
        ({"SiO2": 49.4, "Al2O3": 13.0, "MgO": 17.1, "CaO": 20.5}, 1.921, 0.03),
    ]
    T = 1673.0
    qfm = 8.58 - 25050.0 / T
    for oxide_wt, observed_log_kprime, tolerance in glasses:
        single_cation_moles = {
            "SiO2": oxide_wt["SiO2"] / mw["SiO2"],
            "AlO1.5": 2.0 * oxide_wt["Al2O3"] / mw["Al2O3"],
            "MgO": oxide_wt["MgO"] / mw["MgO"],
            "CaO": oxide_wt["CaO"] / mw["CaO"],
        }
        total = sum(single_cation_moles.values())
        x = {name: value / total for name, value in single_cation_moles.items()}
        composition = 0.37 * x["AlO1.5"] + x["MgO"] + 2.44 * x["CaO"]
        predicted_log_kprime = 3031.0 / T - 2.26 + composition - 0.25 * qfm
        assert predicted_log_kprime == pytest.approx(
            observed_log_kprime, abs=tolerance
        )


def test_algebraic_source_equation_has_one_quarter_log_fugacity_slope():
    """Check the source equation's algebraic slope; this does not exercise the solver."""
    T = 1673.0
    fixed_composition = 0.71749
    qfm = 8.58 - 25050.0 / T
    logits = [
        0.25 * (logf - qfm) + 3031.0 / T - 2.26 + fixed_composition
        for logf in (-12.0, -11.0)
    ]
    assert logits[1] - logits[0] == pytest.approx(0.25)


def test_apollo_14072_like_composition_reproduces_reported_fraction():
    oxide_wt = {
        "SiO2": 45.15, "TiO2": 2.57, "Al2O3": 11.07, "FeO": 17.82,
        "MnO": 0.27, "MgO": 12.16, "CaO": 9.84, "Na2O": 0.32, "K2O": 0.08,
    }
    mw = {
        "SiO2": 60.0843, "TiO2": 79.866, "Al2O3": 101.9613,
        "FeO": 71.844, "MnO": 70.9374, "MgO": 40.3044, "CaO": 56.0774,
        "Na2O": 61.979, "K2O": 94.196,
    }
    inventory = {
        "Si": oxide_wt["SiO2"] / mw["SiO2"],
        "Ti": oxide_wt["TiO2"] / mw["TiO2"],
        "Al": 2.0 * oxide_wt["Al2O3"] / mw["Al2O3"],
        "Fe": oxide_wt["FeO"] / mw["FeO"],
        "Mn": oxide_wt["MnO"] / mw["MnO"],
        "Mg": oxide_wt["MgO"] / mw["MgO"],
        "Ca": oxide_wt["CaO"] / mw["CaO"],
        "Na": 2.0 * oxide_wt["Na2O"] / mw["Na2O"],
        "K": 2.0 * oxide_wt["K2O"] / mw["K2O"],
        "Cr": 0.00671,
        "O": 0.0,
    }
    # Neal (2006), as tabulated in NASA Lunar Sample Compendium Table 1:
    # SiO2 45.15, TiO2 2.57, Al2O3 11.07, FeO 17.82, MnO 0.27, MgO 12.16,
    # CaO 9.84, Na2O 0.32, K2O 0.08 wt%. The source calculation uses
    # log10(fO2)=-10.729 (1-bar IW-1).
    result = _candidate_result(1673.0, -10.729, inventory)
    fraction = result.chromous_cr_mol / (
        result.chromous_cr_mol + result.chromic_cr_mol
    )
    assert fraction == pytest.approx(0.87515, abs=0.0001)
    assert result.cr_oxygen_capacity_mol == pytest.approx(0.5 * result.chromous_cr_mol)


def test_candidate_oxygen_capacity_matches_total_fixed_point_difference():
    inventory = {
        "Si": 0.3, "Mg": 0.1, "Fe": 0.1, "Ca": 0.15, "Al": 0.1,
        "Ti": 0.01, "Na": 0.01, "K": 0.001, "Cr": 0.005, "O": 1.05,
    }
    temperature_K = 1700.0
    qfm_log10 = 8.58 - 25050.0 / temperature_K
    log10_f_o2 = qfm_log10 - 2.0
    step = 1.0e-4
    center = _candidate_result(temperature_K, log10_f_o2, inventory)
    oxidized = _candidate_result(
        temperature_K, log10_f_o2 + step / math.log(10.0), inventory
    )
    reduced = _candidate_result(
        temperature_K, log10_f_o2 - step / math.log(10.0), inventory
    )
    finite_difference = (oxidized.oxygen_mol - reduced.oxygen_mol) / (2.0 * step)
    relative_error = abs(
        (center.oxygen_capacity_mol_per_ln_f_o2 - finite_difference)
        / finite_difference
    )
    assert relative_error < 1.0e-4


def test_candidate_loader_and_default_outputs_remain_identical_across_sweep():
    candidate = load_redox_pack(CANDIDATE)
    default = load_redox_pack(DEFAULT)
    assert candidate.rows[-1]["composition_coefficients"]["Fe3+O1.5"] == 0.37
    assert not any("Cr" in row["complex"] for row in default.rows)
    for T, inventory in [
        (1800.0, {"Fe": 1.0, "O": 1.25}),
        (2200.0, {"Fe": 1.0, "Mn": 0.1, "Si": 0.2, "Al": 0.1, "O": 1.65}),
        (2500.0, {"Fe": 0.7, "Si": 0.2, "Mg": 0.1, "O": 1.0}),
    ]:
        default_result = evaluate_redox(inventory, T, "closed")
        assert default_result == evaluate_redox(
            inventory, T, "closed", redox_pack_path=DEFAULT
        )
        assert default_result.oxygen_capacity_mol_per_ln_f_o2 == evaluate_redox(
            inventory, T, "closed", redox_pack_path=CANDIDATE
        ).oxygen_capacity_mol_per_ln_f_o2


def test_composition_coefficients_are_rejected_by_default_pack(tmp_path):
    content = json.loads(DEFAULT.read_text())
    content["rows"][1]["composition_coefficients"] = {"MgO": 1.0}
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path = tmp_path / "default-with-candidate-field.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="candidate-pack only"):
        load_redox_pack(path)


def test_candidate_composition_fixed_point_nonconvergence_is_typed(monkeypatch):
    import openimcc.redox as redox_module

    original = redox_module._cr_composition_effective_ln_k
    flip = [False]

    def oscillating(*args, **kwargs):
        values = original(*args, **kwargs)
        flip[0] = not flip[0]
        return {name: value + (1.0 if flip[0] else -1.0) for name, value in values.items()}

    monkeypatch.setattr(redox_module, "_cr_composition_effective_ln_k", oscillating)
    with pytest.raises(RedoxNumericalError, match="fixed point did not converge"):
        _candidate_result(1673.0, -10.0)
