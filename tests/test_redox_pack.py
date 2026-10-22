"""Schema, source-row, and reaction checks for the redox component."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

import openimcc.redox_pack as redox_pack_module
from openimcc.redox_pack import (
    _finite_number,
    RedoxSchemaError,
    canonical_redox_digest,
    load_redox_pack,
    ln_k_prime,
    reaction_gibbs_j_mol,
    species_gibbs_j_mol,
)
from openimcc.gas import R_J_MOL_K
from tools.build_gas_tables import _load_record, build_redox_metal_standard_rows
from tools.liquid_from_solid import liquid_from_solid


ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "src/openimcc/data/packs/openimcc-redox-v4.json"
SOURCE_PATH = ROOT / "data-src/redox-thermochemistry.json"
PROVENANCE_SUCCESSOR_PATHS = (
    PACK_PATH,
    ROOT / "src/openimcc/data/packs/openimcc-redox-feo-liquid-candidate-v2.json",
    ROOT / "src/openimcc/data/packs/openimcc-redox-cr-candidate-v3.json",
    ROOT / "src/openimcc/data/packs/openimcc-redox-ti-candidate-v3.json",
    ROOT / "src/openimcc/data/packs/openimcc-redox-cr-ti-candidate-v3.json",
)
INDEPENDENT_CHECK_NOT_FOUND_REASON = (
    "No independent FeO(l) standard-state check was identified. Darken & Gurry "
    "(1946) is the fit source; the held-out activity comparisons inherit that "
    "D&G FeO reference and do not independently validate the FeO(l) standard "
    "Gibbs energy."
)


def _band_value(record: dict) -> float:
    band = record["accuracy_band"] or record["fit_residual"]
    return float(band["value"])


def test_redox_pack_loads_with_n_parent_registry_and_digest() -> None:
    pack = load_redox_pack(PACK_PATH)
    elements = [parent["element"] for parent in pack.parents]

    assert pack.version == "1.7.0"
    assert pack.content["schema"] == "openimcc-redox-pack.v2"
    assert elements == [
        "Si", "Mg", "Fe", "Ca", "Al", "Ti", "Na", "K", "Cr", "Ni",
        "Co", "P", "V", "Mn", "Eu", "Ce", "S",
    ]
    assert pack.canonical_digest == canonical_redox_digest(pack.content)
    assert pack.content["canonical_content_digest"] == pack.canonical_digest
    assert sum(
        parent["redox_thermochemistry_status"] == "registered_unpopulated"
        for parent in pack.parents
    ) == 15
    with pytest.raises(TypeError):
        pack.content["version"] = "changed"


def test_feo_fit_source_is_not_reported_as_an_independent_check() -> None:
    for path in PROVENANCE_SUCCESSOR_PATHS:
        content = json.loads(path.read_text(encoding="utf-8"))
        records = (content["thermo_species"]["FeO(l)"], content["rows"][0])
        for record in records:
            check = record["independent_check"]
            assert check["status"] == "not_found", path.name
            assert check["primary_source"] is None, path.name
            assert check["reason"] == INDEPENDENT_CHECK_NOT_FOUND_REASON, path.name
            assert all(
                check[field] is None
                for field in (
                    "equilibrium",
                    "conditions",
                    "measured_value",
                    "predicted_value",
                    "miss",
                    "miss_to_band_ratio",
                )
            ), path.name
        assert content["errata"][-1]["version"] == content["version"], path.name


def test_finite_number_refuses_integer_beyond_float_range() -> None:
    with pytest.raises(ValueError, match="must be a finite number"):
        _finite_number(10**400, "test value")


def test_default_redox_pack_is_pinned_to_manifest_bytes(tmp_path, monkeypatch):
    content = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    content["thermo_species"]["FeO(l)"]["row"]["dG_A"] += 1.0
    content["canonical_content_digest"] = canonical_redox_digest(content)
    altered_bytes = json.dumps(content).encode("utf-8")

    package_root = tmp_path / "package"
    pack_dir = package_root / "packs"
    pack_dir.mkdir(parents=True)
    (pack_dir / PACK_PATH.name).write_bytes(altered_bytes)
    (pack_dir / "MANIFEST.json").write_bytes(
        (PACK_PATH.parent / "MANIFEST.json").read_bytes()
    )
    monkeypatch.setattr(redox_pack_module, "files", lambda _package: package_root)

    with pytest.raises(ValueError, match="SHA-256 does not match MANIFEST.json"):
        redox_pack_module.load_redox_pack()

    external_path = tmp_path / "external-redox.json"
    external_path.write_bytes(altered_bytes)
    assert redox_pack_module.load_redox_pack(external_path).canonical_digest == (
        content["canonical_content_digest"]
    )


def test_parent_registry_oxidation_states_and_oxide_charge_are_validated(
    tmp_path: Path,
) -> None:
    content = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    content["parent_registry"][0]["known_oxidation_states"] = [4, 4]
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path = tmp_path / "invalid-parent-states.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="known_oxidation_states"):
        load_redox_pack(path)

    content["parent_registry"][0]["known_oxidation_states"] = [4]
    content["parent_registry"][0]["parent_oxidation_state"] = 5
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="parent_oxidation_state"):
        load_redox_pack(path)

    content["parent_registry"][0]["parent_oxidation_state"] = 4
    sulfur = next(parent for parent in content["parent_registry"] if parent["element"] == "S")
    sulfur["parent_oxidation_state"] = 4
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="does not balance its oxide formula"):
        load_redox_pack(path)


def test_redox_pack_v1_schema_is_refused(tmp_path: Path) -> None:
    content = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    content["schema"] = "openimcc-redox-pack.v1"
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path = tmp_path / "legacy-schema.json"
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported redox pack schema"):
        load_redox_pack(path)


def test_multi_interval_species_are_contiguous_and_allow_legacy_rows(
    tmp_path: Path,
) -> None:
    from openimcc.gas import _janaf_gibbs, _lamor_gibbs

    pack = load_redox_pack(PACK_PATH)
    species = pack.content["thermo_species"]["MnO(l)"]
    assert "row" not in species
    assert [interval["T_domain_K"] for interval in species["intervals"]] == [
        (1200, 1600), (1600, 2200), (2200, 3000),
    ]
    for temperature in (1200.0, 1600.0, 2200.0, 3000.0):
        assert math.isfinite(species_gibbs_j_mol(pack, "MnO(l)", temperature))
    assert math.isfinite(
        species_gibbs_j_mol(pack, "MnO(l)", 1100.0, allow_extrapolation=True)
    )
    assert math.isfinite(
        species_gibbs_j_mol(pack, "MnO(l)", 3100.0, allow_extrapolation=True)
    )
    with pytest.raises(ValueError, match="outside its fitted table domain"):
        species_gibbs_j_mol(pack, "O2(g)", 1199.0, allow_extrapolation=True)
    for name, record in pack.content["thermo_species"].items():
        intervals = record.get("intervals")
        if intervals is None:
            continue
        evaluator = (
            _janaf_gibbs
            if record["evaluator"] == "janaf_shomate"
            else _lamor_gibbs
        )
        for lower, upper in zip(intervals, intervals[1:]):
            join = float(lower["T_domain_K"][1])
            gap = abs(evaluator(join, lower["row"]) - evaluator(join, upper["row"]))
            assert gap <= 10.0, (name, join, gap)

    content = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    content["thermo_species"]["MnO(l)"]["intervals"][1]["T_domain_K"][0] = 1601
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path = tmp_path / "non-contiguous-species-intervals.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="does not continue the preceding interval"):
        load_redox_pack(path)

    content = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    content["thermo_species"]["MnO(l)"]["row"] = dict(
        content["thermo_species"]["MnO(l)"]["intervals"][1]["row"]
    )
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot have both row and intervals"):
        load_redox_pack(path)

    content = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    content["thermo_species"]["MnO(l)"]["intervals"] = None
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path = tmp_path / "null-species-intervals.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="intervals must contain at least two rows"):
        load_redox_pack(path)


def test_fe_valence_metadata_is_validated_for_species_and_combinations(
    tmp_path: Path,
) -> None:
    content = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    content["thermo_species"]["FeO1.5(l)"]["ferric_fe_units"] = 0.5
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path = tmp_path / "invalid-fe-valence.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="ferric_fe_units does not match constituents"):
        load_redox_pack(path)

    content["thermo_species"]["FeO1.5(l)"]["ferric_fe_units"] = 1.0
    content["thermo_species"]["FeO(l)"]["ferrous_fe_units"] = 0.0
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="does not assign every Fe atom a valence"):
        load_redox_pack(path)


def test_reaction_rows_are_balanced_and_have_evidence_and_range_flags() -> None:
    pack = load_redox_pack(PACK_PATH)

    assert [row["complex"] for row in pack.rows] == [
        "FeO(l)", "FeO1.5(l)", "MnO(l)", "MnSiO3(l)", "Mn2SiO4(l)",
        "MnAl2O4(l)"
    ]
    for name, species in pack.content["thermo_species"].items():
        assert _band_value(species) >= 0.0
        assert isinstance(species["range_flags"], tuple)
        assert species["provenance"]["source_url"].startswith("https://")
        assert species["provenance"]["tier"].startswith("tier")
        assert species["provenance"]["selection_rationale"]
        low, high = species["T_domain_K"]
        value = species_gibbs_j_mol(pack, name, (low + high) / 2.0)
        assert math.isfinite(value)
    for row in pack.rows:
        expected_domains = {
            "FeO(l)": (1700, 3000),
            "FeO1.5(l)": (1500, 3000),
            "MnO(l)": (1673, 2000),
            "MnSiO3(l)": (1673, 2000),
            "Mn2SiO4(l)": (1673, 2000),
            "MnAl2O4(l)": (1673, 2000),
        }
        assert row["T_domain_K"] == expected_domains[row["complex"]]
        assert _band_value(row) >= 0.0
        assert row["range_flags"]
        assert row["phase_or_construction"]
        assert row["method"]
        assert row["band_basis"]["explanation"]
        assert row["uncertainty_note"]
        assert row["provenance"]["source_url"].startswith("https://")
        assert row["provenance"]["tier"].startswith("tier")
        assert row["provenance"]["selection_rationale"]
        for temperature in row["T_domain_K"]:
            assert math.isfinite(reaction_gibbs_j_mol(pack, row, temperature))

    ferric = pack.rows[1]
    assert ferric["reaction"] == "FeO(l) + 1/4 O2(g) = FeO1.5(l)"
    assert ferric["nu"] == {"FeO(l)": 1}
    assert ferric["external_oxygen_stoich_product_positive"] == -0.25


def test_ferric_reaction_gibbs_and_oxygen_fugacity_shift() -> None:
    pack = load_redox_pack(PACK_PATH)
    ferric = pack.rows[1]
    temperature = 2000.0
    delta_g = reaction_gibbs_j_mol(pack, ferric, temperature)
    expected = (
        species_gibbs_j_mol(pack, "FeO1.5(l)", temperature)
        - species_gibbs_j_mol(pack, "FeO(l)", temperature)
        - 0.25 * species_gibbs_j_mol(pack, "O2(g)", temperature)
    )
    assert delta_g == pytest.approx(expected, abs=1e-10)

    shift = -3.2
    ln_k = ln_k_prime(pack, ferric, temperature)
    ln_k_shifted = ln_k_prime(pack, ferric, temperature, ln_f_o2=shift)
    assert ln_k == pytest.approx(-delta_g / (R_J_MOL_K * temperature))
    assert ln_k_shifted - ln_k == pytest.approx(shift / 4.0)


def test_pure_iron_reference_switches_continuously_at_janaf_fusion_temperature() -> None:
    pack = load_redox_pack(PACK_PATH)
    species = pack.content["thermo_species"]["Fe(metal)"]
    assert species["reference_activity"] == 1.0
    assert species_gibbs_j_mol(pack, "Fe(metal)", 1800.0) == pytest.approx(
        species_gibbs_j_mol(pack, "Fe(cr)", 1800.0)
    )
    assert species_gibbs_j_mol(pack, "Fe(metal)", 1809.0) == pytest.approx(
        species_gibbs_j_mol(pack, "Fe(l)", 1809.0)
    )
    transition_gap_j = abs(
        species_gibbs_j_mol(pack, "Fe(cr)", 1809.0)
        - species_gibbs_j_mol(pack, "Fe(l)", 1809.0)
    )
    # The direct JANAF fits retain a 2.345 J/mol seam residual at fusion;
    # this is within the 10 J/mol fit gate and is not removed by an anchor.
    assert transition_gap_j < 10.0

    for transition in (900.0, 1184.0, 1665.0):
        lower_phase = species_gibbs_j_mol(
            pack, "Fe(cr)", math.nextafter(transition, 0.0)
        )
        upper_phase = species_gibbs_j_mol(pack, "Fe(cr)", transition)
        assert abs(upper_phase - lower_phase) < 10.0


def test_metal_standard_rows_reproduce_the_existing_fitter() -> None:
    sources = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    generated = build_redox_metal_standard_rows(sources["tables"])
    pack = load_redox_pack(PACK_PATH)
    coefficients = ("dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E")

    for species_name, fitted in generated.items():
        record = pack.content["thermo_species"][species_name]
        if species_name == "Fe(cr)":
            assert [item["T_domain_K"] for item in fitted["intervals"]] == [
                [500.0, 900.0], [900.0, 1184.0],
                [1184.0, 1665.0], [1665.0, 1809.0],
            ]
            assert [item["T_domain_K"] for item in record["intervals"]] == [
                (500, 900), (900, 1184),
                (1184, 1665), (1665, 1809),
            ]
            for source_interval, packed_interval in zip(
                fitted["intervals"], record["intervals"]
            ):
                assert float(source_interval["row"]["_max_residual_J_per_mol"]) < 10.0
                for coefficient in coefficients:
                    assert float(packed_interval["row"][coefficient]) == pytest.approx(
                        float(source_interval["row"][coefficient]),
                        rel=1e-9,
                        abs=1e-12,
                    )
            assert _band_value(record) >= float(
                fitted["_max_residual_J_per_mol"]
            ) / 1000.0
            continue
        if species_name == "Mn(l)":
            assert [item["T_domain_K"] for item in fitted["intervals"]] == [
                [1200.0, 1600.0], [1600.0, 2200.0], [2200.0, 3000.0],
            ]
            for source_interval, packed_interval in zip(
                fitted["intervals"], record["intervals"]
            ):
                assert float(source_interval["row"]["_max_residual_J_per_mol"]) < 10.0
                for coefficient in coefficients:
                    assert float(packed_interval["row"][coefficient]) == pytest.approx(
                        float(source_interval["row"][coefficient]),
                        rel=1e-9,
                        abs=1e-12,
                    )
            continue
        packed = record["row"]
        for coefficient in coefficients:
            expected = float(fitted[coefficient])
            # Least-squares coefficients differ in the last few ulps across
            # NumPy/BLAS builds (4e-13 relative measured on NumPy 1.26 vs 2.x),
            # so compare relatively. Sanity of the bound: G = 1000*R*dH298_R
            # - R*T*P(tau); every individual term of G in this pack is below
            # 1.9e6 J/mol in magnitude over 298-3000 K (measured), so a 1e-9
            # relative change in any one coefficient moves G by < 2e-3 J/mol,
            # at least 500x below the smallest Fe fit band (1 J/mol).
            assert float(packed[coefficient]) == pytest.approx(
                expected, rel=1e-9, abs=1e-12
            )
        residual = float(fitted["_max_residual_J_per_mol"])
        assert _band_value(record) >= residual / 1000.0


def test_ferrous_parent_reports_janaf_offset_without_using_it_as_a_gate() -> None:
    pack = load_redox_pack(PACK_PATH)
    row = next(
        row for row in pack.rows
        if row["role"] == "parent_reference" and row["complex"] == "FeO(l)"
    )
    source = _load_record(ROOT / "data-src/janaf/Fe-019.yaml")["table"]["values"]
    residuals = []

    for point in source:
        temperature = float(point["temperature"]["value"])
        formation_gibbs = point["formation_gibbs_energy"]["value"]
        if 1700.0 <= temperature <= 3000.0 and formation_gibbs is not None:
            reaction = reaction_gibbs_j_mol(pack, row, temperature)
            residuals.append(abs(reaction - float(formation_gibbs) * 1000.0) / 1000.0)

    assert residuals
    # Fe-019's fusion construction is the rejected source for this standard.
    # The offset is reported as source-comparison evidence; it is not a gate
    # on the criterion-2 primary-measurement selection.
    assert max(residuals) == pytest.approx(8.05585033661028, abs=1.0e-9)


def test_oxygen_standard_row_band_covers_janaf_nodes() -> None:
    pack = load_redox_pack(PACK_PATH)
    species = pack.content["thermo_species"]["O2(g)"]
    source = _load_record(ROOT / "data-src/janaf/O-029.yaml")["table"]["values"]
    residuals = []

    for point in source:
        temperature = float(point["temperature"]["value"])
        enthalpy = point["enthalpy_increment"]["value"]
        entropy = point["entropy"]["value"]
        if 1200.0 <= temperature <= 3000.0 and enthalpy is not None and entropy is not None:
            expected = float(enthalpy) * 1000.0 - temperature * float(entropy)
            residuals.append(
                abs(species_gibbs_j_mol(pack, "O2(g)", temperature) - expected) / 1000.0
            )

    assert residuals
    assert max(residuals) <= _band_value(species)


def test_elemental_metal_functions_match_their_source_nodes_within_ten_joules() -> None:
    sources = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))["tables"]
    pack = load_redox_pack(PACK_PATH)
    table_by_species = {
        "Fe(cr)": "Fe-007",
        "Fe(l)": "Fe-006",
        "Si(l)": "Si-003",
        "Cr(l)": "Cr-003",
        "Ni(l)": "Ni-003",
        "Co(l)": "Co-003",
        "Mn(l)": "Mn-003",
    }
    for species_name, table_id in table_by_species.items():
        table = sources[table_id]
        reference = float(table["formation_enthalpy_298_kj_mol"])
        record = pack.content["thermo_species"][species_name]
        nodes = list(table["nodes"])
        for node in table.get("transition_nodes", []):
            nodes.append(
                [
                    node["temperature_K"],
                    node["heat_capacity"],
                    node["entropy"],
                    node["enthalpy_increment"],
                ]
            )
        residuals = []
        for temperature, _cp, entropy, enthalpy_increment in nodes:
            temperature = float(temperature)
            low, high = map(float, record["T_domain_K"])
            if not low <= temperature <= high:
                continue
            expected = (
                (reference + float(enthalpy_increment)) * 1000.0
                - temperature * float(entropy)
            )
            actual = species_gibbs_j_mol(pack, species_name, temperature)
            residuals.append(abs(actual - expected))
        assert residuals, species_name
        assert max(residuals) <= 10.0, (species_name, max(residuals))


def test_hematite_row_reproduces_liquid_from_solid_construction_and_band() -> None:
    sources = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    pack = load_redox_pack(PACK_PATH)
    record = pack.content["thermo_species"]["Fe2O3(l)"]
    spec = record["construction"]

    def make_construction(enthalpy_kj_mol: float):
        fusion_entropy = enthalpy_kj_mol * 1000.0 / spec["fusion_temperature_K"]
        return liquid_from_solid(
            "Fe2O3",
            sources["hematite_crystal_table"],
            {
                "fusion_temperature": {"value": spec["fusion_temperature_K"]},
                "fusion_entropy": {
                    "value": fusion_entropy,
                    "spread": spec["fusion_entropy_spread_j_mol_K"],
                },
                "liquid_heat_capacity": {
                    "value": spec["liquid_heat_capacity_j_mol_K"],
                    "spread": spec["liquid_heat_capacity_spread_j_mol_K"],
                },
                "condensate_fit": {
                    "fit_t_min": spec["fit_T_K"][0],
                    "fit_t_max": spec["fit_T_K"][1],
                    "runtime_t_min": spec["runtime_T_K"][0],
                    "runtime_t_max": spec["runtime_T_K"][1],
                    "table_id": "Fe2O3-redox-1",
                    "cation": "Fe",
                    "cat_num": 2,
                    "oxy_num": 3,
                    "ref": "Fe-030 + hypothetical fusion",
                },
            },
        )

    nominal = make_construction(spec["fusion_enthalpy_kj_mol"])
    fitted = nominal.fit_condensate_row()
    retained_row = next(
        interval["row"]
        for interval in record["intervals"]
        if interval["T_domain_K"] == (1500, 3000)
    )
    for key, value in retained_row.items():
        # Relative bound for the same cross-build roundoff reason as the
        # metal-standard test above (1e-9 relative moves G by < 2e-3 J/mol).
        assert float(fitted[key]) == pytest.approx(float(value), rel=1e-9, abs=1e-12)

    temperatures = (1500.0, 1895.0, 2195.0, 2695.0, 3000.0)
    # The input-spread helper conservatively propagates the primary upper
    # fusion estimate and measured Cp uncertainty through this construction.
    spread = max(nominal.input_spread_kj_mol(temperature) for temperature in temperatures)
    spread += float(fitted["_max_residual_J_per_mol"]) / 1000.0
    assert _band_value(record) == pytest.approx(spread, abs=1e-3)
    assert record["provenance"]["tier"] == "tier3_flagged_estimate"
    assert record["provenance"]["provenance_class"] == "secondary_transcription_unverified_primary"
    assert record["construction"]["liquid_heat_capacity_j_mol_K"] == 240.9
    assert record["construction"]["liquid_heat_capacity_flag"].startswith("amber;")
    assert record["construction"]["fusion_enthalpy_kj_mol"] == 114.5


def test_manganese_liquid_rows_reproduce_crystal_constructions_and_bands():
    inputs = json.loads(
        (ROOT / "data-src/liquid-from-solid-inputs.json").read_text(
            encoding="utf-8"
        )
    )
    crystals = json.loads(
        (ROOT / "data-src/crystal-thermochemistry.json").read_text(
            encoding="utf-8"
        )
    )["tables"]
    pack = load_redox_pack(PACK_PATH)
    loo_entropy = inputs["tier_error_bands"]["fusion_entropy_tier3"]["max"]
    loo_cp = inputs["tier_error_bands"]["liquid_heat_capacity_tier3"]["max"]
    coefficients = ("dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E")

    for name in ("MnO", "MnSiO3", "Mn2SiO4", "MnAl2O4"):
        construction_row = inputs["rows"][name]
        species = pack.content["thermo_species"][f"{name}(l)"]
        construction = liquid_from_solid(
            name,
            crystals[construction_row["crystal_table_id"]],
            construction_row["inputs"],
        )
        fitted = construction.fit_condensate_row()
        fit = construction_row["inputs"]["condensate_fit"]
        retained = next(
            interval
            for interval in species["intervals"]
            if interval["T_domain_K"]
            == (fit["fit_t_min"], fit["fit_t_max"])
        )
        for coefficient in coefficients:
            assert float(retained["row"][coefficient]) == pytest.approx(
                float(fitted[coefficient]), abs=1e-12
            )

        evidence_row = next(
            row for row in pack.rows if row["complex"] == f"{name}(l)"
        )
        low, high = map(float, evidence_row["T_domain_K"])
        temperatures = [low, high, *range(int(low), int(high) + 1, 25)]
        residual = float(fitted["_max_residual_J_per_mol"]) / 1000.0
        construction_residuals = []
        for temperature in range(1200, 3001, 100):
            expected_g = construction.at(float(temperature)).gibbs_kj_mol * 1000.0
            actual_g = species_gibbs_j_mol(
                pack, f"{name}(l)", float(temperature)
            )
            construction_residuals.append(abs(actual_g - expected_g))
        assert max(construction_residuals) <= 10.0

        if name in {"MnO", "MnAl2O4"}:
            expected_at = {
                t: construction.below_fusion_band_kj_mol(
                    t, float(loo_entropy[0])
                )
                + residual
                for t in temperatures
            }
            if name == "MnAl2O4":
                expected_at = {
                    t: band
                    + float(crystals[construction_row["crystal_table_id"]][
                        "formation_enthalpy_spread_kj_mol"
                    ])
                    + t
                    * float(crystals[construction_row["crystal_table_id"]][
                        "entropy_spread_j_mol_k"
                    ])
                    / 1000.0
                    for t, band in expected_at.items()
                }
        else:
            def loo_at_offset(columns, offset):
                if offset <= 300.0:
                    fraction = offset / 300.0
                    return float(columns[0]) + fraction * (
                        float(columns[1]) - float(columns[0])
                    )
                fraction = (offset - 300.0) / 500.0
                return float(columns[1]) + fraction * (
                    float(columns[2]) - float(columns[1])
                )

            input_entropy_spread = float(
                construction_row["inputs"]["fusion_entropy"]["spread"]
            )
            input_cp_spread = float(
                construction_row["inputs"]["liquid_heat_capacity"]["spread"]
            )
            expected_at = {}
            for t in temperatures:
                tfus = construction.fusion_temperature_k
                shape = (t - tfus) - t * math.log(t / tfus)
                # The LOO columns already measure these entropy and Cp
                # residuals. Keep only the independent fusion-temperature
                # interval from input propagation, then add the fitted-row
                # residual and any direct crystal formation-enthalpy error.
                entropy_input = abs(
                    (tfus - t) * input_entropy_spread / 1000.0
                )
                cp_input = abs(input_cp_spread * shape / 1000.0)
                temperature_input = max(
                    0.0,
                    construction.input_spread_kj_mol(t)
                    - entropy_input
                    - cp_input,
                )
                expected_at[t] = (
                    loo_at_offset(loo_entropy, t - tfus)
                    + loo_at_offset(loo_cp, t - tfus)
                    + temperature_input
                    + residual
                    + float(
                        crystals[construction_row["crystal_table_id"]].get(
                            "formation_enthalpy_spread_kj_mol", 0.0
                        )
                    )
                )
        expected_band = max(expected_at.values())
        assert _band_value(species) == pytest.approx(expected_band, abs=1e-6)

    with (ROOT / "src/openimcc/data/gas/condensate.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        gas_rows = list(csv.DictReader(stream))
    for redox_name, gas_name in (
        ("Al2O3(l)", "Al2O3(l)"),
        ("SiO2(l)-supercooled", "SiO2(l)"),
        ("SiO2(l)-direct", "SiO2(l)"),
    ):
        alumina_or_silica = pack.content["thermo_species"][redox_name]
        fitted_intervals = alumina_or_silica.get("intervals") or [
            {
                "T_domain_K": alumina_or_silica["T_domain_K"],
                "row": alumina_or_silica["row"],
            }
        ]
        gas_intervals = sorted(
            (
                row
                for row in gas_rows
                if row["species_name"] == gas_name
                and int(row["T_min"]) >= 1200
                and (
                    gas_name != "SiO2(l)"
                    or (
                        redox_name == "SiO2(l)-supercooled"
                        and int(row["T_max"]) <= 1800
                    )
                    or (
                        redox_name == "SiO2(l)-direct"
                        and int(row["T_min"]) >= 1800
                    )
                )
            ),
            key=lambda row: int(row["T_min"]),
        )
        assert [interval["T_domain_K"] for interval in fitted_intervals] == [
            (int(row["T_min"]), int(row["T_max"])) for row in gas_intervals
        ]
        for fitted, source in zip(fitted_intervals, gas_intervals):
            for coefficient in coefficients:
                assert float(fitted["row"][coefficient]) == pytest.approx(
                    float(source[coefficient]), abs=1e-12
                )

    alumina = pack.content["thermo_species"]["Al2O3(l)"]
    assert _band_value(alumina) == pytest.approx(
        inputs["tier_error_bands"]["fusion_entropy_tier1"]["max"][0]
    )

    generated = build_redox_metal_standard_rows(
        json.loads(SOURCE_PATH.read_text(encoding="utf-8"))["tables"]
    )
    metal = pack.content["thermo_species"]["Mn(l)"]
    for source_interval, packed_interval in zip(
        generated["Mn(l)"]["intervals"], metal["intervals"]
    ):
        for coefficient in coefficients:
            assert float(packed_interval["row"][coefficient]) == pytest.approx(
                float(source_interval["row"][coefficient]), abs=1e-13
            )


def test_manganese_parent_and_association_reactions_have_expected_constants():
    pack = load_redox_pack(PACK_PATH)
    by_reaction = {row["reaction"]: row for row in pack.rows}
    manganese = next(parent for parent in pack.parents if parent["element"] == "Mn")
    assert manganese["redox_thermochemistry_status"] == "populated"
    assert manganese["parent_standard_state"] == "pure liquid MnO at 1 bar"
    assert manganese["elemental_reference_state"] == "Mn(l)"
    parent_row = by_reaction["Mn(l) + 1/2 O2(g) = MnO(l)"]
    assert parent_row["role"] == "parent_reference"
    assert parent_row["external_oxygen_stoich_product_positive"] == -0.5
    expected = {
        "Mn(l) + 1/2 O2(g) = MnO(l)": (18.0069480442, 13.6595399686),
        "MnO(l) + SiO2(l) = MnSiO3(l)": (2.5910868901, 2.5120668805),
        "2 MnO(l) + SiO2(l) = Mn2SiO4(l)": (4.1916008326, 3.6718251563),
        "MnO(l) + Al2O3(l) = MnAl2O4(l)": (4.6353020697, 4.0276646397),
    }
    for reaction, (at_1673, at_2000) in expected.items():
        row = by_reaction[reaction]
        assert ln_k_prime(pack, row, 1673.0) == pytest.approx(at_1673, abs=1e-9)
        assert ln_k_prime(pack, row, 2000.0) == pytest.approx(at_2000, abs=1e-9)


def test_manganese_reaction_bands_sum_each_species_band_once():
    pack = load_redox_pack(PACK_PATH)
    for row in pack.rows:
        if row["row"] < 3:
            continue
        band = sum(
            float(coefficient)
            * _band_value(pack.content["thermo_species"][species])
            for species, coefficient in row["reactants"].items()
        )
        band += sum(
            float(coefficient)
            * _band_value(pack.content["thermo_species"][species])
            for species, coefficient in row["products"].items()
        )
        assert _band_value(row) == pytest.approx(
            round(band, 3), abs=1e-12
        )


def test_loader_rejects_a_rehashed_unbalanced_row(tmp_path: Path) -> None:
    content = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    content["rows"][1]["reactants"]["O2(g)"] = 0.24
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path = tmp_path / "unbalanced.json"
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ValueError, match="unbalanced for O"):
        load_redox_pack(path)


@pytest.mark.parametrize(
    ("coefficient", "message"),
    [
        (1.0, "combination formula_atoms do not match constituents"),
        (float("nan"), "finite number"),
    ],
)
def test_loader_rejects_a_rehashed_derived_formula_contradiction(
    tmp_path: Path, coefficient: float, message: str
) -> None:
    content = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    content["thermo_species"]["FeO1.5(l)"]["terms"][0]["coefficient"] = coefficient
    content["canonical_content_digest"] = canonical_redox_digest(content)
    path = tmp_path / "derived-formula-contradiction.json"
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_redox_pack(path)


def test_loader_rejects_a_digest_mismatch(tmp_path: Path) -> None:
    content = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    content["version"] = "1.0.1"
    path = tmp_path / "digest-mismatch.json"
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        load_redox_pack(path)
