"""Schema, source-row, and reaction checks for the redox component."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from openimcc.redox_pack import (
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
PACK_PATH = ROOT / "src/openimcc/data/packs/openimcc-redox-v1.json"
SOURCE_PATH = ROOT / "data-src/redox-thermochemistry.json"


def test_redox_pack_loads_with_n_parent_registry_and_digest() -> None:
    pack = load_redox_pack(PACK_PATH)
    elements = [parent["element"] for parent in pack.parents]

    assert pack.version == "1.0.0"
    assert elements == [
        "Si", "Mg", "Fe", "Ca", "Al", "Ti", "Na", "K", "Cr", "Ni",
        "Co", "P", "V", "Mn", "Eu", "Ce", "S",
    ]
    assert pack.canonical_digest == canonical_redox_digest(pack.content)
    assert pack.content["canonical_content_digest"] == pack.canonical_digest
    assert sum(
        parent["redox_thermochemistry_status"] == "registered_unpopulated"
        for parent in pack.parents
    ) == 16
    with pytest.raises(TypeError):
        pack.content["version"] = "changed"


def test_reaction_rows_are_balanced_and_have_evidence_and_range_flags() -> None:
    pack = load_redox_pack(PACK_PATH)

    assert [row["complex"] for row in pack.rows] == ["FeO(l)", "FeO1.5(l)"]
    for name, species in pack.content["thermo_species"].items():
        assert species["band_kj_mol"] >= 0.0
        assert isinstance(species["range_flags"], tuple)
        assert species["provenance"]["source_url"].startswith("https://")
        assert species["provenance"]["tier"].startswith("tier")
        assert species["provenance"]["selection_rationale"]
        low, high = species["T_domain_K"]
        value = species_gibbs_j_mol(pack, name, (low + high) / 2.0)
        assert math.isfinite(value)
    for row in pack.rows:
        expected_low = 1500 if row["complex"] == "FeO1.5(l)" else 1700
        assert row["T_domain_K"] == (expected_low, 3000)
        assert row["band_kj_mol"] >= 0.0
        assert row["range_flags"]
        assert row["phase_or_construction"]
        assert row["method"]
        assert row["band_method"]
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
    assert transition_gap_j < 1.0


def test_metal_standard_rows_reproduce_the_existing_fitter() -> None:
    sources = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    generated = build_redox_metal_standard_rows(sources["tables"])
    pack = load_redox_pack(PACK_PATH)
    coefficients = ("dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E")

    for species_name, fitted in generated.items():
        packed = pack.content["thermo_species"][species_name]["row"]
        for coefficient in coefficients:
            expected = float(fitted[coefficient])
            if species_name == "Fe(l)" and coefficient == "dH298_R":
                # G = 1000*R*dH298_R - R*T*P(tau); add
                # ΔG_anchor/(1000*R) to the stored kK-scaled enthalpy anchor.
                adjustment = pack.content["thermo_species"][species_name][
                    "gibbs_anchor_adjustment_j_mol"
                ]
                expected += adjustment / (R_J_MOL_K * 1000.0)
            # Least-squares coefficients differ in the last few ulps across
            # NumPy/BLAS builds (4e-13 relative measured on NumPy 1.26 vs 2.x),
            # so compare relatively. Sanity of the bound: G = 1000*R*dH298_R
            # - R*T*P(tau); every individual term of G in this pack is below
            # 1.9e6 J/mol in magnitude over 298-3000 K (measured), so a 1e-9
            # relative change in any one coefficient moves G by < 2e-3 J/mol,
            # 1000x below the smallest band in the pack (2 J/mol).
            assert float(packed[coefficient]) == pytest.approx(
                expected, rel=1e-9, abs=1e-12
            )
        assert pack.content["thermo_species"][species_name]["band_kj_mol"] >= float(
            fitted["_max_residual_J_per_mol"]
        ) / 1000.0


def test_ferrous_parent_reaction_band_covers_janaf_reference_residual() -> None:
    pack = load_redox_pack(PACK_PATH)
    row = pack.rows[0]
    source = _load_record(ROOT / "data-src/janaf/Fe-019.yaml")["table"]["values"]
    residuals = []

    for point in source:
        temperature = float(point["temperature"]["value"])
        formation_gibbs = point["formation_gibbs_energy"]["value"]
        if 1700.0 <= temperature <= 3000.0 and formation_gibbs is not None:
            reaction = reaction_gibbs_j_mol(pack, row, temperature)
            residuals.append(abs(reaction - float(formation_gibbs) * 1000.0) / 1000.0)

    assert residuals
    assert max(residuals) <= row["band_kj_mol"]


def test_oxygen_standard_row_band_covers_janaf_nodes() -> None:
    pack = load_redox_pack(PACK_PATH)
    species = pack.content["thermo_species"]["O2(g)"]
    source = _load_record(ROOT / "data-src/janaf/O-029.yaml")["table"]["values"]
    residuals = []

    for point in source:
        temperature = float(point["temperature"]["value"])
        enthalpy = point["enthalpy_increment"]["value"]
        entropy = point["entropy"]["value"]
        if 1700.0 <= temperature <= 3000.0 and enthalpy is not None and entropy is not None:
            expected = float(enthalpy) * 1000.0 - temperature * float(entropy)
            residuals.append(
                abs(species_gibbs_j_mol(pack, "O2(g)", temperature) - expected) / 1000.0
            )

    assert residuals
    assert max(residuals) <= species["band_kj_mol"]


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
    for key, value in record["row"].items():
        # Relative bound for the same cross-build roundoff reason as the
        # metal-standard test above (1e-9 relative moves G by < 2e-3 J/mol).
        assert float(fitted[key]) == pytest.approx(float(value), rel=1e-9, abs=1e-12)

    temperatures = (1500.0, 1895.0, 2195.0, 2695.0, 3000.0)
    # The input-spread helper conservatively propagates the primary upper
    # fusion estimate and measured Cp uncertainty through this construction.
    spread = max(nominal.input_spread_kj_mol(temperature) for temperature in temperatures)
    spread += float(fitted["_max_residual_J_per_mol"]) / 1000.0
    assert record["band_kj_mol"] == pytest.approx(spread, abs=1e-3)
    assert record["provenance"]["tier"] == "tier3_flagged_estimate"
    assert record["provenance"]["provenance_class"] == "secondary_transcription_unverified_primary"
    assert record["construction"]["liquid_heat_capacity_j_mol_K"] == 240.9
    assert record["construction"]["liquid_heat_capacity_flag"].startswith("amber;")
    assert record["construction"]["fusion_enthalpy_kj_mol"] == 114.5


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
