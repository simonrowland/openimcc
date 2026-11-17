from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from openimcc.diagnostics import CrystalStatus, _crystal_thermo_context
from openimcc.equilibrium import (
    _source_cap_excludes_candidate,
    _source_solid_limit_K,
    equilibrate,
)
from openimcc.gas import species_thermo


ROOT = Path(__file__).resolve().parents[1]
CRYSTAL_PATH = ROOT / "src/openimcc/data/diagnostics/crystal-phases.json"
RH_PHASES = {"fayalite", "hercynite", "ilmenite", "perovskite", "ulvospinel"}

V1_FUSION_NODES = {
    "fayalite": {"1300": 41.479529006584485, "1400": 34.462932360988844},
    "hercynite": {"1300": 62.124145741, "1400": 58.708471805},
    "ilmenite": {
        "1300": 51.64567852, "1400": 46.841511921,
        "1500": 42.006564761, "1600": 36.95535081,
    },
    "perovskite": {
        "1500": 148.45171145, "1600": 144.885880243,
        "1700": 140.980904016, "1800": 136.968897478,
    },
    "ulvospinel": {
        "1300": 70.518639447, "1400": 64.161921252,
        "1500": 57.7758078, "1600": 51.384302812,
        "1700": 45.185683843, "1800": 39.073685231,
    },
}
NODE_SHIFTS_V2_MINUS_V1 = {
    ("fayalite", "1300"): -0.6372,
    ("hercynite", "1300"): -0.2356,
    ("ilmenite", "1300"): -0.31895,
    ("ilmenite", "1500"): -0.3755,
    ("perovskite", "1500"): 1.9920009,
    ("perovskite", "1700"): 2.217000336,
    ("perovskite", "1800"): 2.338703025,
    ("ulvospinel", "1300"): -0.6374,
    ("ulvospinel", "1500"): -0.7375,
    ("ulvospinel", "1700"): -0.8426,
    ("ulvospinel", "1800"): -0.869,
}
SWEEP_RESIDUALS_KJ_MOL = {
    "fayalite": ("1400", -0.72),
    "hercynite": ("1400", -0.40),
    "ilmenite": ("1500", -0.41),
    "perovskite": ("1700", 2.31),
    "ulvospinel": ("1800", -0.89),
}


def _dataset() -> dict:
    return json.loads(CRYSTAL_PATH.read_text(encoding="utf-8"))


def _semantic_digest(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def test_v2_identity_and_untouched_crystal_controls() -> None:
    data = _dataset()
    assert data["schema"] == "openimcc-crystal-phases-v2"
    assert data["source"]["source_commit"] == "9aacc2f6308e49a5a1988c25487518ba346d9ad0"

    berman_rows = [phase for phase in data["phases"] if phase["id"] not in RH_PHASES]
    controls = data["pure_oxide_controls"]
    # Canonical snapshots of the v1 Berman rows and JANAF pure-oxide controls.
    assert _semantic_digest(berman_rows) == (
        "74b0050d14d923c1d0c007e7d83503382a03a0064560b9734dd8afad5ee484c3"
    )
    assert _semantic_digest(controls) == (
        "123215d1537f16f56e22f3fe662273a97bbf7e8cebda28e7c9cdbe614a6154de"
    )
    assert data["element_reference_g_app_nodes_kJ_mol"]["Si"]["1700"] == -69.4688
    assert data["element_reference_g_app_nodes_kJ_mol"]["Si"]["1800"] == -78.7284


def test_rh_node_shifts_match_the_sweep_residuals() -> None:
    phases = {phase["id"]: phase for phase in _dataset()["phases"]}
    for phase_id, nodes in V1_FUSION_NODES.items():
        phase = phases[phase_id]
        assert set(phase["fusion_gibbs_nodes_kJ_mol"]) == set(nodes)
        for temperature, old_value in nodes.items():
            if temperature not in {"1300", "1500", "1700", "1800"}:
                continue
            shift = phase["fusion_gibbs_nodes_kJ_mol"][temperature] - old_value
            assert shift == pytest.approx(
                NODE_SHIFTS_V2_MINUS_V1[(phase_id, temperature)], abs=1e-6,
            )

    for phase_id, (temperature, expected_shift) in SWEEP_RESIDUALS_KJ_MOL.items():
        shift = (
            phases[phase_id]["fusion_gibbs_nodes_kJ_mol"][temperature]
            - V1_FUSION_NODES[phase_id][temperature]
        )
        assert shift == pytest.approx(expected_shift, abs=0.10)


def test_rh_printed_ranges_and_formation_enthalpy_anchors() -> None:
    phases = {phase["id"]: phase for phase in _dataset()["phases"]}
    expected_h298 = {
        "fayalite": -1478.2,
        "hercynite": -1950.5,
        "ilmenite": -1232.0,
        "perovskite": -1660.6,
        "ulvospinel": -1493.8,
    }
    expected_descriptions = {
        "fayalite": "Orthorhombic crystals 298.15 to 1490 K. Fayalite melts incongruently at 1490 K.\n            Liquid 1490 to 1800 K.",
        "hercynite": "CUbic crystals 298.15 to 1400 K.",
        "ilmenite": "Rhombohedral crystals 298.15 to 1640 K.",
        "perovskite": "Orthorhombic crystals 1530 K; cubic crystals 1530 to 1800 K.",
        "ulvospinel": "Cubic crystals 298.15 to 1800 K.",
    }
    for phase_id in RH_PHASES:
        phase = phases[phase_id]
        assert phase["formation_enthalpy_298_kJ_mol"] == expected_h298[phase_id]
        assert phase["source"]["printed_phase_description"] == expected_descriptions[phase_id]
        assert phase["valid_temperature_K"] == {
            "fayalite": [1300, 1400],
            "hercynite": [1300, 1400],
            "ilmenite": [1300, 1600],
            "perovskite": [1300, 1800],
            "ulvospinel": [1300, 1800],
        }[phase_id]


def test_fusion_secants_agree_with_entropy_slopes() -> None:
    data = _dataset()
    gas_pack = _crystal_thermo_context(None)[0]
    for phase in data["phases"]:
        if phase["id"] not in RH_PHASES:
            continue
        gibbs_nodes = phase["fusion_gibbs_nodes_kJ_mol"]
        entropy_nodes = phase["source_crystal_entropy_nodes_J_mol_K"]
        temperatures = sorted(float(value) for value in gibbs_nodes)
        for low_t, high_t in zip(temperatures, temperatures[1:]):
            # The perovskite 1500–1600 K interval crosses its printed 1530 K
            # structural transition and is not a single-phase entropy secant.
            if phase["id"] == "perovskite" and (low_t, high_t) == (1500.0, 1600.0):
                continue
            middle_t = (low_t + high_t) / 2
            crystal_s = (
                float(entropy_nodes[str(int(low_t))])
                + float(entropy_nodes[str(int(high_t))])
            ) / 2.0
            parent_s = math.fsum(
                float(amount) * species_thermo(oxide, "l", middle_t, gas_pack).S_J_molK
                for oxide, amount in phase["oxide_stoichiometry"].items()
            )
            secant = (
                float(gibbs_nodes[str(int(high_t))])
                - float(gibbs_nodes[str(int(low_t))])
            ) / (high_t - low_t)
            expected_slope = (crystal_s - parent_s) / 1000.0
            assert secant == pytest.approx(expected_slope, abs=0.0025)


@pytest.mark.parametrize(
    ("phase_id", "temperature", "h298", "h_function", "entropy", "expected_fusion"),
    (
        ("fayalite", 1400.0, -1478.2, 141.30, 415.77, 33.74813236092609),
        ("perovskite", 1700.0, -1660.6, 106.95, 310.58, 143.19790435201708),
    ),
)
def test_absolute_g_hess_cycle_independently_at_two_temperatures(
    phase_id: str,
    temperature: float,
    h298: float,
    h_function: float,
    entropy: float,
    expected_fusion: float,
) -> None:
    phase = next(row for row in _dataset()["phases"] if row["id"] == phase_id)
    gas_pack = _crystal_thermo_context(None)[0]
    h_increment = h_function * temperature / 1000.0
    crystal_absolute_g = h298 + h_increment - temperature * entropy / 1000.0
    parent_absolute_g = math.fsum(
        float(amount) * species_thermo(oxide, "l", temperature, gas_pack).G_J_mol / 1000.0
        for oxide, amount in phase["oxide_stoichiometry"].items()
    )
    assert parent_absolute_g - crystal_absolute_g == pytest.approx(
        expected_fusion, abs=1e-9,
    )


def test_rh_source_solid_limits_apply_to_melted_crystals() -> None:
    phases = {phase["id"]: phase for phase in _dataset()["phases"]}
    expected_limits = {"fayalite": 1490.0, "perovskite": 2188.0}
    for phase_id, temperature in expected_limits.items():
        phase = phases[phase_id]
        assert _source_solid_limit_K(phase) == temperature
        assert phase["source_solid_limit"]["temperature_K"] == temperature
        assert not _source_cap_excludes_candidate(
            phase["oxide_stoichiometry"], phase, temperature,
        )
        assert _source_cap_excludes_candidate(
            phase["oxide_stoichiometry"], phase, temperature + 1.0,
        )
    ilmenite = phases["ilmenite"]
    assert "source_solid_limit" not in ilmenite
    assert _source_solid_limit_K(ilmenite) is None
    assert ilmenite["valid_temperature_K"] == [1300, 1600]
    assert max(map(float, ilmenite["fusion_gibbs_nodes_kJ_mol"])) == 1600.0


def test_equilibrium_sanity_bulk_cases_use_crystal_phases_v2() -> None:
    molecular_weights = {
        "SiO2": 60.0843, "TiO2": 79.866, "Al2O3": 101.961,
        "FeO": 71.844, "MgO": 40.304, "CaO": 56.077,
        "Na2O": 61.979, "K2O": 94.196,
    }
    mare_weight_percent = {
        "SiO2": 45.0, "TiO2": 3.0, "Al2O3": 10.0, "FeO": 19.0,
        "MgO": 10.0, "CaO": 11.0, "Na2O": 0.4, "K2O": 0.1,
    }
    bulks = (
        (
            "mare-basalt-like",
            {
                oxide: amount / molecular_weights[oxide]
                for oxide, amount in mare_weight_percent.items()
            },
        ),
        ("CaO-TiO2-SiO2", {"CaO": 1.0, "TiO2": 1.0, "SiO2": 1.0}),
    )
    for name, bulk in bulks:
        for temperature in (1700.0, 1800.0):
            result = equilibrate(bulk, temperature)
            assert result.labels.identity["crystal_phases"] == "openimcc-crystal-phases-v2"
            assert result.receipt.data_digests["crystal_phases"]
            assert next(
                row for row in result.phase_saturation_indices
                if row.phase == "perovskite"
            )
            if name == "mare-basalt-like":
                assert result.refusal is not None
                assert result.refusal.code == "missing_phase_data"
                assert "crystal.ilmenite" in result.refusal.axes
                assert result.labels.coverage["ilmenite"] == "temperature_out_of_range"
                ilmenite = next(
                    row for row in result.phase_saturation_indices
                    if row.phase == "ilmenite"
                )
                assert ilmenite.status is CrystalStatus.OUT_OF_RANGE
                assert ilmenite.SI is None
