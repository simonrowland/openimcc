from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import yaml

from openimcc.declared_assemblage import evaluate_declared_assemblage
from openimcc import diagnostics, equilibrium as equilibrium_module
from openimcc.diagnostics import _interpolate_crystal_fusion_nodes
from openimcc.equilibrium import equilibrate
from openimcc.gas import species_thermo


ROOT = Path(__file__).resolve().parents[1]
BARIN_PATH = ROOT / "src/openimcc/data/diagnostics/barin-aluminate-crystals.json"
CRYSTAL_PATH = ROOT / "src/openimcc/data/diagnostics/crystal-phases.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atoms(formula: str) -> dict[str, int]:
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*)", formula)
    assert "".join(element + count for element, count in tokens) == formula
    return {element: int(count or 1) for element, count in tokens}


def _source_formation_gibbs(table_id: str) -> list[tuple[float, float]]:
    table = yaml.safe_load(
        (ROOT / f"data-src/janaf/{table_id}.yaml").read_text(encoding="utf-8")
    )["table"]
    return [
        (float(row["temperature"]["value"]), float(row["formation_gibbs_energy"]["value"]))
        for row in table["values"]
        if row["formation_gibbs_energy"]["value"] is not None
    ]


def _interpolate(nodes: list[tuple[float, float]], temperature: float) -> float:
    by_temperature = {t: value for t, value in nodes}
    if temperature in by_temperature:
        return by_temperature[temperature]
    ordered = sorted(by_temperature.items())
    for (low_t, low_value), (high_t, high_value) in zip(ordered, ordered[1:]):
        if low_t < temperature < high_t:
            return low_value + (high_value - low_value) * (
                (temperature - low_t) / (high_t - low_t)
            )
    raise ValueError(f"no source interval at {temperature:g} K")


def _source_series(phase: dict, quantity: str) -> list[tuple[float, float]]:
    return sorted(
        (
            float(node["T_K"]), float(node["value"])
        )
        for polymorph in phase["polymorphs"]
        for node in polymorph["source_series"][quantity]["nodes"]
    )


def test_barin_phase_formulas_and_printed_node_ranges() -> None:
    data = _json(BARIN_PATH)
    assert data["schema"] == "openimcc-barin-aluminate-crystals-v2"
    assert len(data["phases"]) == 6

    for phase in data["phases"]:
        atoms = {}
        for oxide, coefficient in phase["oxide_stoichiometry"].items():
            for element, count in _atoms(oxide).items():
                atoms[element] = atoms.get(element, 0.0) + coefficient * count
        assert atoms == pytest.approx(_atoms(phase["formula"]))

        source_temperatures = [
            temperature for temperature, _value in _source_series(phase, "delta_fG")
        ]
        assert phase["valid_temperature_K"] == [
            min(source_temperatures),
            max(source_temperatures),
        ]
        assert all(polymorph["label"].startswith("SOL") for polymorph in phase["polymorphs"])
        for polymorph in phase["polymorphs"]:
            series = polymorph["source_series"]
            assert {"delta_fG", "H_minus_H298", "S"} <= set(series)
            assert set(series) <= {"delta_fG", "H_minus_H298", "S", "Cp"}
            for quantity in series.values():
                assert quantity["nodes"]
                assert all(
                    {"page", "folio", "pdf_page_index", "row", "column"}
                    <= node["locator"].keys()
                    for node in quantity["nodes"]
                )
            assert [node["T_K"] for node in series["delta_fG"]["nodes"]] == [
                node["T_K"] for node in series["H_minus_H298"]["nodes"]
            ] == [node["T_K"] for node in series["S"]["nodes"]]
            if "Cp" in series:
                assert [node["T_K"] for node in series["Cp"]["nodes"]] == [
                    series["delta_fG"]["nodes"][-1]["T_K"]
                ]

    cp_nodes = {
        phase["id"]: phase["polymorphs"][0]["source_series"]["Cp"]["nodes"][0]["value"]
        for phase in data["phases"] if "Cp" in phase["polymorphs"][0]["source_series"]
    }
    assert cp_nodes == {"CA": 193.101, "CA2": 336.802, "C3A": 310.778}

    sodium = next(phase for phase in data["phases"] if phase["id"] == "NaAlO2")
    assert [(p["label"], p["valid_temperature_K"]) for p in sodium["polymorphs"]] == [
        ("SOL-A", [298.15, 740.0]),
        ("SOL-B", [740.0, 1900.0]),
    ]
    source_solid_limits = {
        phase["id"]: phase["source_solid_limit"]
        for phase in data["phases"] if "source_solid_limit" in phase
    }
    assert {
        phase: limit["temperature_K"]
        for phase, limit in source_solid_limits.items()
    } == {"CA": 1875.0, "CA2": 2035.0, "C3A": 1808.0, "MgAl2O4": 2408.0}
    assert {phase: limit["kind"] for phase, limit in source_solid_limits.items()} == {
        "CA": "decomposition", "CA2": "decomposition", "C3A": "decomposition",
        "MgAl2O4": "melting",
    }
    assert "C12A7" not in source_solid_limits
    assert "NaAlO2" not in source_solid_limits


@pytest.mark.parametrize("phase_id", ("CA", "CA2", "C3A"))
def test_constant_cp_fusion_continuation_is_continuous_and_has_sane_slope(
    phase_id: str,
) -> None:
    data = _json(BARIN_PATH)
    phase = next(row for row in data["phases"] if row["id"] == phase_id)
    node_temperatures = sorted(float(t) for t in phase["fusion_gibbs_nodes_kJ_mol"])
    t0, previous_temperature = node_temperatures[-1], node_temperatures[-2]
    node_gibbs = float(phase["fusion_gibbs_nodes_kJ_mol"][f"{t0:.2f}"])
    gas_pack = diagnostics._crystal_thermo_context(None)[0]

    def parent_gibbs(temperature: float) -> dict[str, float]:
        return {
            oxide: species_thermo(oxide, "l", temperature, gas_pack).G_J_mol
            for oxide in phase["oxide_stoichiometry"]
        }

    parent_at_node = parent_gibbs(t0)
    at_node = diagnostics._crystal_fusion_gibbs_kj_mol(
        phase, t0, parent_at_node, parent_at_node,
    )
    assert at_node == pytest.approx(node_gibbs, abs=1.0e-12)

    epsilon = 1.0e-6
    just_above = diagnostics._crystal_fusion_gibbs_kj_mol(
        phase, t0 + epsilon, parent_gibbs(t0 + epsilon), parent_at_node,
    )
    no_jump = just_above - node_gibbs
    assert abs(no_jump) < 1.0e-5

    source_slope = (
        node_gibbs
        - float(phase["fusion_gibbs_nodes_kJ_mol"][f"{previous_temperature:.2f}"])
    ) / (t0 - previous_temperature)
    slope_step = 1.0e-2
    continued_slope = (
        diagnostics._crystal_fusion_gibbs_kj_mol(
            phase, t0 + slope_step, parent_gibbs(t0 + slope_step), parent_at_node,
        ) - at_node
    ) / slope_step
    assert continued_slope == pytest.approx(source_slope, abs=0.1)


@pytest.mark.parametrize(
    ("phase_id", "expected_fusion_at_limit"),
    (
        ("CA", 107.046660942),
        ("CA2", 152.878711274),
        ("C3A", 202.864404047),
    ),
)
def test_continued_fusion_gibbs_at_source_limit(
    phase_id: str, expected_fusion_at_limit: float,
) -> None:
    data = _json(BARIN_PATH)
    phase = next(row for row in data["phases"] if row["id"] == phase_id)
    t0, limit = diagnostics._crystal_continuation_range(phase)
    gas_pack = diagnostics._crystal_thermo_context(None)[0]

    def parent_gibbs(temperature: float) -> dict[str, float]:
        return {
            oxide: species_thermo(oxide, "l", temperature, gas_pack).G_J_mol
            for oxide in phase["oxide_stoichiometry"]
        }

    fusion_at_limit = diagnostics._crystal_fusion_gibbs_kj_mol(
        phase, limit, parent_gibbs(limit), parent_gibbs(t0),
    )
    assert math.isfinite(fusion_at_limit)
    assert fusion_at_limit == pytest.approx(expected_fusion_at_limit, abs=1.0e-6)


def test_equilibrium_inside_ca_continuation_has_plausible_crystal_standard() -> None:
    result = equilibrate({"CaO": 1.0, "Al2O3": 1.0}, 1850.0)
    notice = "crystal_function_continued:CA:1800-1875K"
    assert result.labels.validity in ("IN", "DEPENDS")
    assert notice in result.labels.notices
    assert notice in result.receipt.solver["crystal_function_continued"]
    if result.labels.validity == "IN":
        assert result.receipt.solver["phase_amount_ranges_certified"] is True
    ca_row = next(row for row in result.phase_saturation_indices if row.phase == "CA")
    assert ca_row.status.value == "available"
    phase = next(
        row for row in _json(BARIN_PATH)["phases"] if row["id"] == "CA"
    )
    gas_pack = diagnostics._crystal_thermo_context(None)[0]
    parent_at_node = {
        oxide: species_thermo(oxide, "l", 1800.0, gas_pack).G_J_mol
        for oxide in phase["oxide_stoichiometry"]
    }
    parent_at_temperature = {
        oxide: species_thermo(oxide, "l", 1850.0, gas_pack).G_J_mol
        for oxide in phase["oxide_stoichiometry"]
    }
    fusion = diagnostics._crystal_fusion_gibbs_kj_mol(
        phase, 1850.0, parent_at_temperature, parent_at_node,
    )
    assert math.isfinite(fusion)
    assert 90.0 < fusion < 120.0
    if ca_row.SI is not None:
        assert math.isfinite(ca_row.SI)
        assert abs(ca_row.SI) < 10.0


@pytest.mark.parametrize(
    ("phase_id", "last_node", "limit", "bulk"),
    (
        ("CA", 1800.0, 1875.0, {"CaO": 1.0, "Al2O3": 1.0}),
        ("CA2", 2000.0, 2035.0, {"CaO": 1.0, "Al2O3": 2.0}),
        ("C3A", 1800.0, 1808.0, {"CaO": 3.0, "Al2O3": 1.0}),
    ),
)
def test_continuation_availability_ends_at_source_limit(
    phase_id: str, last_node: float, limit: float, bulk: dict[str, float],
) -> None:
    below_limit = math.nextafter(limit, -math.inf)
    states = equilibrium_module._default_candidate_states(below_limit)[0]
    state = next(row for row in states if row.candidate.phase_id == phase_id)
    assert state.available
    assert state.domain == pytest.approx((298.15, limit))
    assert state.continued_range_K == pytest.approx((last_node, limit))

    at_limit = equilibrium_module._default_candidate_states(limit)[0]
    at_limit_state = next(row for row in at_limit if row.candidate.phase_id == phase_id)
    assert at_limit_state.available
    assert at_limit_state.continued_range_K == pytest.approx((last_node, limit))

    above_limit = math.nextafter(limit, math.inf)
    capped_states = equilibrium_module._default_candidate_states(above_limit)[0]
    capped = next(row for row in capped_states if row.candidate.phase_id == phase_id)
    assert not capped.available
    assert capped.reason == "above_source_melting_point"

    # A single hotter result exercises all three source caps in its receipt.
    if phase_id == "CA2":
        hot = equilibrate(bulk, 2040.0)
        assert {"CA", "CA2", "C3A"} <= set(
            hot.receipt.solver["crystals_above_source_melting_point"]
        )
        assert not {"CA", "CA2", "C3A"} & set(
            hot.receipt.solver["missing_candidate_phase_ids"]
        )


def test_continued_notice_and_receipt_cover_equilibrium_and_declared_results() -> None:
    notice = "crystal_function_continued:CA:1800-1875K"
    result = equilibrate({"CaO": 1.0, "Al2O3": 1.0}, 1810.0)
    assert notice in result.labels.notices
    assert result.receipt.solver["crystal_function_continued"] == (notice,)

    declared = evaluate_declared_assemblage(("CA",), 1810.0)
    assert notice in declared.labels.notices
    assert declared.receipt.solver["crystal_function_continued"] == (notice,)


def test_phases_without_a_printed_limit_remain_missing_above_their_last_node() -> None:
    for phase_id, last_temperature in (("C12A7", 1700.0), ("NaAlO2", 1900.0)):
        states, _digest, _pure_digest, _gas_pack, missing = (
            equilibrium_module._default_candidate_states(
                math.nextafter(last_temperature, math.inf)
            )
        )
        state = next(row for row in states if row.candidate.phase_id == phase_id)
        assert not state.available
        assert state.reason == "temperature_out_of_range"
        assert phase_id in missing


def test_formation_gibbs_reconstruction_and_independent_oxide_hess_cycles() -> None:
    data = _json(BARIN_PATH)
    references = data["conversion"]["parent_oxide_references"]
    oxide_nodes = {
        oxide: _source_formation_gibbs(reference["table_id"])
        for oxide, reference in references.items()
    }
    reconstruction_errors = {}

    for phase in data["phases"]:
        max_delta_f_g_error = 0.0
        for polymorph in phase["polymorphs"]:
            series = polymorph["source_series"]
            for node in series["delta_fG"]["nodes"]:
                temperature = float(node["T_K"])
                oxide_formation_g = sum(
                    float(coefficient) * _interpolate(oxide_nodes[oxide], temperature)
                    for oxide, coefficient in phase["oxide_stoichiometry"].items()
                )
                fusion = float(phase["fusion_gibbs_nodes_kJ_mol"][f"{temperature:.2f}"])
                reconstructed_delta_f_g = oxide_formation_g - fusion
                max_delta_f_g_error = max(
                    max_delta_f_g_error, abs(reconstructed_delta_f_g - float(node["value"]))
                )
                assert reconstructed_delta_f_g == pytest.approx(node["value"], abs=1e-9)

            # H−H298 and S are carried as the printed nodes, not inferred by
            # differentiating Gibbs nodes. Thus their node reconstruction
            # error is zero by representation; check each aligned value exists.
            temperatures = [node["T_K"] for node in series["delta_fG"]["nodes"]]
            for quantity in ("H_minus_H298", "S"):
                nodes = series[quantity]["nodes"]
                assert [node["T_K"] for node in nodes] == temperatures
                assert all(math.isfinite(float(node["value"])) for node in nodes)

        reconstruction_errors[phase["id"]] = max_delta_f_g_error
        assert reconstruction_errors[phase["id"]] <= 1e-9


def test_spinel_is_compared_with_berman_and_janaf_without_averaging() -> None:
    barin = _json(BARIN_PATH)
    existing = _json(CRYSTAL_PATH)
    phase = next(row for row in barin["phases"] if row["id"] == "MgAl2O4")
    berman = next(row for row in existing["phases"] if row["id"] == "spinel")
    barin_formation = _source_series(phase, "delta_fG")
    janaf_crystal = _source_formation_gibbs("Al-089")
    overlap = (1323.0, 2408.0)
    barin_fusion = [
        (float(t), float(value))
        for t, value in phase["fusion_gibbs_nodes_kJ_mol"].items()
    ]
    berman_fusion = [
        (float(t), float(value))
        for t, value in berman["fusion_gibbs_nodes_kJ_mol"].items()
    ]
    common_temperatures = sorted(
        {overlap[0], overlap[1]}
        | {t for t, _ in barin_fusion if overlap[0] < t < overlap[1]}
        | {t for t, _ in berman_fusion if overlap[0] < t < overlap[1]}
    )

    barin_minus_berman = []
    barin_minus_janaf = []
    for temperature in common_temperatures:
        barin_fusion_value = _interpolate_crystal_fusion_nodes(phase, temperature)
        berman_fusion_value = _interpolate_crystal_fusion_nodes(berman, temperature)
        barin_minus_berman.append(berman_fusion_value - barin_fusion_value)
        barin_minus_janaf.append(
            _interpolate(barin_formation, temperature)
            - _interpolate(janaf_crystal, temperature)
        )
    assert min(barin_minus_berman) == pytest.approx(5.67698852374042)
    assert max(barin_minus_berman) == pytest.approx(9.326397581285839)
    assert min(barin_minus_janaf) == pytest.approx(9.77944)
    assert max(barin_minus_janaf) == pytest.approx(21.9622)


@pytest.mark.parametrize(
    ("bulk", "temperature", "expected_crystals"),
    [
        ({"CaO": 1.0}, 1800.0, {"CaO"}),
        ({"CaO": 3.0, "Al2O3": 1.0}, 1800.0, {"CaO"}),
        ({"CaO": 12.0, "Al2O3": 7.0}, 1700.0, {"CaO"}),
        ({"CaO": 1.0, "Al2O3": 1.0}, 1800.0, set()),
        ({"CaO": 1.0, "Al2O3": 2.0}, 1900.0, set()),
        ({"Al2O3": 1.0}, 1800.0, {"Al2O3"}),
    ],
)
def test_ca_al2o3_join_returns_certified_states(bulk, temperature, expected_crystals) -> None:
    result = equilibrate(bulk, temperature)
    sequence = {"CA", "CA2", "C3A", "C12A7", "CaO", "Al2O3"}
    barin_ids = {"CA", "CA2", "C3A", "C12A7", "NaAlO2", "MgAl2O4"}
    present = {
        phase for phase, status in result.phase_presence.items() if status == "present"
    }
    summary = {
        "sequence_members_present": sorted(present),
        "validity": result.labels.validity,
        "refusal": result.refusal.code if result.refusal else None,
        "refusal_axes": result.refusal.axes if result.refusal else (),
        "kkt_recheck": dict(result.kkt_recheck),
        "phase_amount_ranges_certified": result.receipt.solver.get(
            "phase_amount_ranges_certified"
        ),
    }
    assert result.kkt_recheck.get("independent_kkt_passed") == 1.0, summary
    assert result.receipt.solver["phase_amount_ranges_certified"] is True
    assert present <= sequence, summary
    assert barin_ids <= result.phase_presence.keys()
    assert present == expected_crystals, summary
    assert result.labels.identity["barin_aluminate_crystals"] == (
        "openimcc-barin-aluminate-crystals-v2"
    )
    assert "barin_aluminate_crystals" in result.receipt.data_digests
    if "CaO" in bulk and "Al2O3" in bulk:
        assert "CA6" in result.receipt.solver["missing_candidate_phase_ids"]


def test_active_set_replacement_tries_lime_after_failed_c3a() -> None:
    result = equilibrate({"CaO": 3.0, "Al2O3": 1.0}, 1700.0)
    summary = {
        "validity": result.labels.validity,
        "refusal": result.refusal.code if result.refusal else None,
        "active_set_history": result.active_set_history,
        "solid_mol": result.solid_mol,
        "liquid_parent_mol": result.liquid_parent_mol,
        "kkt_recheck": dict(result.kkt_recheck),
    }
    assert result.refusal is None or result.refusal.code == "missing_phase_data", summary
    assert result.kkt_recheck.get("independent_kkt_passed") == 1.0, summary
    assert result.solid_mol is not None
    assert result.solid_mol["CaO"] == pytest.approx(1.343335, abs=1.0e-5)
    assert result.liquid_parent_mol == pytest.approx(
        {"CaO": 1.656665, "Al2O3": 1.0}, abs=1.0e-5
    )
    assert result.phase_presence["C3A"] == "absent"
    assert ("C3A",) in result.active_set_history
    assert ("CaO",) in result.active_set_history


def test_source_molten_crystals_are_named_and_not_missing() -> None:
    calcium_aluminate = equilibrate({"CaO": 3.0, "Al2O3": 1.0}, 1900.0)
    calcium_molten = ("C3A", "CA")
    assert calcium_aluminate.receipt.solver["crystals_above_source_melting_point"] == (
        calcium_molten
    )
    assert any(
        notice == "crystals_above_source_melting_point:C3A,CA"
        for notice in calcium_aluminate.labels.notices
    )
    missing = calcium_aluminate.receipt.solver["missing_candidate_phase_ids"]
    assert {"CA6", "C12A7"} <= set(missing)
    assert not set(calcium_molten) & set(missing)
    for phase in calcium_molten:
        assert calcium_aluminate.phase_presence[phase] == "absent"
        assert calcium_aluminate.phase_amount_ranges_mol[phase] == (0.0, 0.0)

    magnesium_aluminate = equilibrate({"MgO": 1.0, "Al2O3": 1.0}, 2500.0)
    assert magnesium_aluminate.receipt.solver[
        "crystals_above_source_melting_point"
    ] == ("MgAl2O4", "spinel")
    assert magnesium_aluminate.labels.validity == "IN"
    assert magnesium_aluminate.refusal is None
    assert magnesium_aluminate.receipt.solver["missing_candidate_phase_ids"] == ()


def test_missing_sodium_and_magnesium_exclude_barins_candidates() -> None:
    result = equilibrate({"CaO": 1.0, "Al2O3": 1.0}, 1700.0)
    assert result.phase_presence["NaAlO2"] == "infeasible"
    assert result.phase_presence["MgAl2O4"] == "infeasible"


def test_missing_beta_alumina_keeps_sodium_alumina_bulk_dependent() -> None:
    result = equilibrate({"Na2O": 1.0, "Al2O3": 1.0}, 1800.0)
    assert result.labels.validity == "DEPENDS"
    assert result.labels.coverage["beta-alumina"] == "missing_phase_data"
    assert "beta-alumina" in result.receipt.solver["missing_candidate_phase_ids"]


def test_declared_barin_phase_receipt_names_its_data_identity() -> None:
    result = evaluate_declared_assemblage(("CA",), 1800.0)
    assert result.labels.identity["barin_aluminate_crystals"] == (
        "openimcc-barin-aluminate-crystals-v2"
    )
    assert "barin_aluminate_crystals" in result.receipt.data_digests
