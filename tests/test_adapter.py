"""Adapter-level gates for IMCC-SF04 (chunk 3)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import openimcc as imcc_sf04
# The simulator this engine was extracted from owns a "fidelity vocabulary"
# deciding which label strings deny authority. That predicate is deny-by-default
# -- it returns True for almost any string, including "inside" -- so porting it
# would assert close to nothing. Every call site here sat next to an assertion
# on the label's exact VALUE, which is strictly stronger, so the checks below
# assert the value directly. The trust policy itself stays upstream; this
# package has no view on which backends a host considers authoritative.
from openimcc import (
    ImccAdapterLabels,
    ImccComponentOutsideDomainError,
    ImccCompositionOutsideValidatedEnvelopeError,
    ImccCompositionIncompleteError,
    ImccDatapack,
    ImccFerricInputUnsupportedError,
    ImccLoadedDatapack,
    ImccMalformedDatapackError,
    ImccNonconvergenceError,
    ImccTOutsideDatapackDomainError,
    ImccUnprovenDatapackError,
    evaluate,
    label_research_datapack,
    load_datapack,
)
from openimcc.kernel import (
    ImccRefusal,
    _label_loaded_datapack,
    solve_imcc_sf04,
)


DATAPACK_PATH = Path(
    "src/openimcc/data/packs/imcc-sf04-v1.0.2.json"
)


def _make_uniform_composition(pack: ImccLoadedDatapack) -> dict[str, float]:
    return {name: 0.125 for name in pack.parent_oxides}


def _make_alkali_composition(
    pack: ImccLoadedDatapack, x_me2o: float
) -> dict[str, float]:
    composition = {name: 0.0 for name in pack.parent_oxides}
    composition["SiO2"] = 1.0 - x_me2o
    composition["Na2O"] = x_me2o
    return composition


def test_load_datapack_roundtrip() -> None:
    pack = load_datapack(DATAPACK_PATH)
    assert isinstance(pack, ImccLoadedDatapack)
    assert pack.version == "1.0.2"
    assert pack.model_id == "IMCC-SF04"
    assert set(pack.kernel_datapack.coverage.values()) == {"A-published-imcc"}
    assert pack.parent_oxides == (
        "SiO2",
        "MgO",
        "FeO",
        "CaO",
        "Al2O3",
        "TiO2",
        "Na2O",
        "K2O",
    )
    assert len(pack.domain_basis) == 38
    assert set(pack.domain_basis) <= {
        "paper-demonstrated",
        "SF04-as-exercised",
        (
            "sf04-exercised-ADOPTED (v1.0.2: FC87 fits were exercised by "
            "SF04 over 1700-3000 K; the FC87-paper-demonstrated span is "
            "preserved in T_domain_paper_demonstrated_K)"
        ),
    }

    kernel = pack.kernel_datapack
    assert kernel.n_parents == 8
    assert kernel.n_complexes == 38
    assert kernel.n_species == 46
    assert kernel.version == "1.0.2"

    # Exact rationals: 0.5 survives as one-half (fractional Na/K/Al stoichiometry).
    half = Fraction(1, 2)
    by_name = {name: i for i, name in enumerate(kernel.reactions)}
    for complex_name in (
        "NaAlSiO4",
        "NaAlSi3O8",
        "NaAlO2",
        "NaAlSi2O6",
    ):
        idx = by_name[complex_name]
        assert Fraction(kernel.nu[pack.parent_oxides.index("Al2O3"), idx]) == half
        assert Fraction(kernel.nu[pack.parent_oxides.index("Na2O"), idx]) == half
        assert Fraction(kernel.nu[pack.parent_oxides.index("K2O"), idx]) == 0
    for complex_name in (
        "KAlSiO4",
        "KAlSi3O8",
        "KAlO2",
        "KAlSi2O6",
        "KCaAlSi2O7",
    ):
        idx = by_name[complex_name]
        assert Fraction(kernel.nu[pack.parent_oxides.index("Al2O3"), idx]) == half
        assert Fraction(kernel.nu[pack.parent_oxides.index("K2O"), idx]) == half
        assert Fraction(kernel.nu[pack.parent_oxides.index("Na2O"), idx]) == 0

    # The three corrected A signs from datapack v1.0.1 errata.
    assert kernel.A[by_name["Mg2SiO4"]] == pytest.approx(-0.94)
    assert kernel.A[by_name["CaTiO3"]] == pytest.approx(-0.08)
    assert kernel.A[by_name["Na2Si2O5"]] == pytest.approx(-1.39)


def test_wt_to_mol_known_value() -> None:
    # Derivation: molar mass SiO2 = 60.0843 g/mol, FeO = 71.844 g/mol.
    # 60.0843 g SiO2 = 1.0000 mol SiO2; 71.844 g FeO = 1.0000 mol FeO.
    # A 60.0843 g + 71.844 g feed at 100 wt-% basis therefore yields exactly
    # 1 mol SiO2 + 1 mol FeO = 2 mol total, with the other six parents zero.
    pack = load_datapack(DATAPACK_PATH)
    composition = {
        "SiO2": 60.0843,
        "MgO": 0.0,
        "FeO": 71.844,
        "CaO": 0.0,
        "Al2O3": 0.0,
        "TiO2": 0.0,
        "Na2O": 0.0,
        "K2O": 0.0,
    }
    result = evaluate(
        composition,
        2500.0,
        pack,
        basis=60.0843 + 71.844,
        basis_type="wt",
    )
    # parent_oxides order: SiO2=0, MgO=1, FeO=2, ...
    assert np.isclose(result.parent_mol[0], 1.0, rtol=1.0e-12)
    assert np.isclose(result.parent_mol[1], 0.0, atol=1.0e-15)
    assert np.isclose(result.parent_mol[2], 1.0, rtol=1.0e-12)
    assert np.isclose(result.basis, 2.0, rtol=1.0e-12)
    assert result.labels.identity["datapack_version"] == "1.0.2"


def test_mol_basis_with_declared_basis() -> None:
    pack = load_datapack(DATAPACK_PATH)
    composition = {name: 0.125 for name in pack.parent_oxides}
    result = evaluate(composition, 2500.0, pack, basis=1.0, basis_type="mol")
    assert np.isclose(result.basis, 1.0, rtol=1.0e-12)
    assert np.isclose(result.parent_mol.sum(), 1.0, rtol=1.0e-12)


def test_refusal_composition_outside_validated_envelope() -> None:
    pack = load_datapack(DATAPACK_PATH)
    with pytest.raises(ImccCompositionOutsideValidatedEnvelopeError) as exc:
        evaluate(_make_alkali_composition(pack, 0.51), 2500.0, pack)
    assert exc.value.code == "imcc_composition_outside_validated_envelope"
    assert "X_Me2O=0.51" in str(exc.value)
    assert "bound 0.5" in str(exc.value)


def test_composition_envelope_boundary_is_inside() -> None:
    pack = load_datapack(DATAPACK_PATH)
    result = evaluate(_make_alkali_composition(pack, 0.5), 2500.0, pack)
    assert result.labels.envelope_status == "inside"


def test_allow_out_of_envelope_labels_result() -> None:
    pack = load_datapack(DATAPACK_PATH)
    result = evaluate(
        _make_alkali_composition(pack, 0.51),
        2500.0,
        pack,
        allow_out_of_envelope=True,
    )
    assert result.labels.envelope_status == "outside_validated"


def test_in_envelope_composition_labels_result() -> None:
    pack = load_datapack(DATAPACK_PATH)
    result = evaluate(_make_uniform_composition(pack), 2500.0, pack)
    assert result.labels.envelope_status == "inside"


def test_refusal_ferric_input_in_composition() -> None:
    pack = load_datapack(DATAPACK_PATH)
    composition = {"SiO2": 1.0, "Fe2O3": 0.1}
    with pytest.raises(ImccFerricInputUnsupportedError):
        evaluate(composition, 2500.0, pack)


def test_refusal_component_outside_domain_in_composition() -> None:
    pack = load_datapack(DATAPACK_PATH)
    composition = {"SiO2": 1.0, "P2O5": 0.1}
    with pytest.raises(ImccComponentOutsideDomainError):
        evaluate(composition, 2500.0, pack)


def test_refusal_composition_incomplete_basis_mismatch() -> None:
    pack = load_datapack(DATAPACK_PATH)
    composition = {"SiO2": 0.5, "FeO": 0.6}
    with pytest.raises(ImccCompositionIncompleteError):
        evaluate(composition, 2500.0, pack, basis=1.0)


def test_refusal_composition_incomplete_negative_value() -> None:
    pack = load_datapack(DATAPACK_PATH)
    composition = {"SiO2": -0.1, "FeO": 1.1}
    with pytest.raises(ImccCompositionIncompleteError):
        evaluate(composition, 2500.0, pack)


def test_refusal_T_outside_domain() -> None:
    pack = load_datapack(DATAPACK_PATH)
    composition = _make_uniform_composition(pack)
    with pytest.raises(ImccTOutsideDatapackDomainError):
        evaluate(composition, 500.0, pack)


def test_refusal_nonconvergence() -> None:
    pack = load_datapack(DATAPACK_PATH)
    composition = _make_uniform_composition(pack)
    with pytest.raises(ImccNonconvergenceError) as exc:
        evaluate(composition, 2500.0, pack, max_iter=0)
    assert exc.value.diagnostics["iterations"] == 0


def test_refusal_malformed_datapack(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"invalid": True}))
    with pytest.raises(ImccMalformedDatapackError):
        load_datapack(bad_path)


def test_refusal_malformed_datapack_bad_json(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("not json")
    with pytest.raises(ImccMalformedDatapackError):
        load_datapack(bad_path)


def test_refusal_contaminated_published_core_extension_row(tmp_path: Path) -> None:
    data = json.loads(DATAPACK_PATH.read_text())
    data["rows"][0].update(
        {
            "complex": "FeS",
            "nu": {"FeO": 1, "S": 1},
            "provenance_class": "extension-compound-thermo",
        }
    )
    bad_path = tmp_path / "contaminated.json"
    bad_path.write_text(json.dumps(data))

    with pytest.raises(ImccMalformedDatapackError) as exc:
        load_datapack(bad_path)
    assert exc.value.code == "imcc_malformed_datapack"
    assert "canonical hash mismatch" in str(exc.value)


def test_refusal_published_core_unknown_nu_key(tmp_path: Path) -> None:
    data = json.loads(DATAPACK_PATH.read_text())
    data["rows"][1]["nu"]["MnO"] = 1
    bad_path = tmp_path / "unknown-nu.json"
    bad_path.write_text(json.dumps(data))

    with pytest.raises(ImccMalformedDatapackError) as exc:
        load_datapack(bad_path)
    assert "canonical hash mismatch" in str(exc.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("complex", "FeS"),
        (
            "nu",
            {
                "SiO2": 1,
                "MgO": 3,
                "FeO": 0,
                "CaO": 0,
                "Al2O3": 0,
                "TiO2": 0,
                "Na2O": 0,
                "K2O": 0,
            },
        ),
    ],
)
def test_refusal_published_core_manifest_deviation(
    tmp_path: Path, field: str, value: object
) -> None:
    data = json.loads(DATAPACK_PATH.read_text())
    data["rows"][0][field] = value
    bad_path = tmp_path / "manifest-deviation.json"
    bad_path.write_text(json.dumps(data))

    with pytest.raises(ImccMalformedDatapackError) as exc:
        load_datapack(bad_path)
    assert "canonical hash mismatch" in str(exc.value)


@pytest.mark.parametrize(("field", "delta"), [("A", 0.5), ("B", -12.0)])
def test_refusal_published_core_altered_thermochemistry(
    tmp_path: Path, field: str, delta: float
) -> None:
    data = json.loads(DATAPACK_PATH.read_text())
    data["rows"][0][field] += delta
    bad_path = tmp_path / f"altered-{field}.json"
    bad_path.write_text(json.dumps(data))

    with pytest.raises(ImccMalformedDatapackError) as exc:
        load_datapack(bad_path)
    assert "canonical hash mismatch" in str(exc.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provenance_class", "extension-compound-thermo"),
        ("T_domain_K", [1.0, 1.0e9]),
        ("T_domain_basis", "forged-domain"),
        ("source", "forged-source"),
        ("future_identity_field", "forged-future-value"),
    ],
)
def test_refusal_published_core_any_row_field_hash_deviation(
    tmp_path: Path, field: str, value: object
) -> None:
    data = json.loads(DATAPACK_PATH.read_text())
    data["rows"][0][field] = value
    bad_path = tmp_path / f"altered-{field}.json"
    bad_path.write_text(json.dumps(data))

    with pytest.raises(ImccMalformedDatapackError) as exc:
        load_datapack(bad_path)
    assert exc.value.code == "imcc_malformed_datapack"
    assert "canonical hash mismatch" in str(exc.value)


def test_refusal_published_core_extra_row_hash_deviation(tmp_path: Path) -> None:
    data = json.loads(DATAPACK_PATH.read_text())
    extra = dict(data["rows"][0])
    extra["complex"] = "forged-row-39"
    data["rows"].append(extra)
    bad_path = tmp_path / "extra-row.json"
    bad_path.write_text(json.dumps(data))

    with pytest.raises(ImccMalformedDatapackError) as exc:
        load_datapack(bad_path)
    assert "exactly 38 published-core rows" in str(exc.value)


def test_refusal_published_datapack_version_hash_deviation(tmp_path: Path) -> None:
    data = json.loads(DATAPACK_PATH.read_text())
    data["imcc_sf04_datapack_version"] = "1.0.3"
    bad_path = tmp_path / "altered-version.json"
    bad_path.write_text(json.dumps(data))

    with pytest.raises(ImccMalformedDatapackError) as exc:
        load_datapack(bad_path)
    assert "canonical hash mismatch" in str(exc.value)


def test_canonical_manifest_normalizes_equivalent_json_numbers(
    tmp_path: Path,
) -> None:
    data = json.loads(DATAPACK_PATH.read_text())
    data["rows"][0]["row"] = 1.0
    data["rows"][0]["nu"]["SiO2"] = 1.0
    data["rows"][0]["T_domain_K"] = [1700.0, 3000.0]
    equivalent_path = tmp_path / "equivalent-numbers.json"
    equivalent_path.write_text(json.dumps(data))

    pack = load_datapack(equivalent_path)
    assert pack.version == "1.0.2"
    assert pack.model_id == "IMCC-SF04"
    assert set(pack.kernel_datapack.coverage.values()) == {"A-published-imcc"}


def test_refusal_raw_datapack_through_adapter() -> None:
    loaded = load_datapack(DATAPACK_PATH)
    source = loaded.kernel_datapack
    raw = ImccDatapack(
        reactions=source.reactions,
        nu=source.nu,
        A=source.A,
        B=source.B,
        domains=source.domains,
        version=source.version,
        parent_oxides=source.parent_oxides,
    )

    with pytest.raises(ImccUnprovenDatapackError) as exc:
        evaluate(_make_uniform_composition(loaded), 2500.0, raw)
    assert exc.value.code == "imcc_unproven_datapack"


def test_explicit_research_datapack_labels_survive_adapter() -> None:
    loaded = load_datapack(DATAPACK_PATH)
    source = loaded.kernel_datapack
    raw = ImccDatapack(
        reactions=source.reactions,
        nu=source.nu,
        A=source.A,
        B=source.B,
        domains=source.domains,
        version="research-test",
        parent_oxides=source.parent_oxides,
    )
    labelled = label_research_datapack(
        raw,
        model_id="IMCC-SF04-RE",
        coverage="RE-research",
    )

    result = evaluate(_make_uniform_composition(loaded), 2500.0, labelled)
    assert result.labels.identity["model_id"] == "IMCC-SF04-RE"
    assert set(result.labels.coverage.values()) == {"RE-research"}
    assert result.labels.trust == "internal-analytical"


@pytest.mark.parametrize(
    ("model_id", "coverage"),
    [
        ("IMCC-SF04", "RE-research"),
        ("IMCC-SF04-RE", "A-published-imcc"),
    ],
)
def test_research_labelling_helper_cannot_claim_published_identity(
    model_id: str, coverage: str
) -> None:
    loaded = load_datapack(DATAPACK_PATH)
    with pytest.raises(ValueError, match="cannot claim published"):
        label_research_datapack(
            loaded.kernel_datapack,
            model_id=model_id,
            coverage=coverage,
        )


def test_loaded_published_pack_carries_identity_into_kernel() -> None:
    pack = load_datapack(DATAPACK_PATH)
    result = solve_imcc_sf04(
        [0.125] * len(pack.parent_oxides),
        2500.0,
        pack.kernel_datapack,
    )
    assert result.labels.model_id == "IMCC-SF04"
    assert set(result.labels.coverage.values()) == {"A-published-imcc"}
    assert result.labels.evidence_class == "internal-analytical"


def test_loaded_pack_identity_inputs_are_immutable() -> None:
    kernel = load_datapack(DATAPACK_PATH).kernel_datapack
    assert isinstance(kernel.reactions, tuple)
    for array in (kernel.nu, kernel.A, kernel.B):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        kernel.A[0] += 0.5


def test_kernel_identity_issuer_refuses_altered_published_pack() -> None:
    source = load_datapack(DATAPACK_PATH).kernel_datapack
    altered_A = np.array(source.A, copy=True)
    altered_A[0] += 0.5
    altered = ImccDatapack(
        reactions=source.reactions,
        nu=source.nu,
        A=altered_A,
        B=source.B,
        domains=source.domains,
        version=source.version,
        parent_oxides=source.parent_oxides,
    )
    coverage = {
        name: "A-published-imcc"
        for name in (*altered.parent_oxides, *altered.reactions)
    }

    with pytest.raises(ValueError, match="complete canonical datapack hash"):
        _label_loaded_datapack(
            altered,
            model_id="IMCC-SF04",
            coverage=coverage,
        )


def test_kernel_identity_issuer_refuses_extension_nu_in_published_core() -> None:
    source = load_datapack(DATAPACK_PATH).kernel_datapack
    extended_nu = np.vstack([source.nu, np.zeros(source.n_complexes)])
    extended_nu[-1, 0] = 1.0
    altered = ImccDatapack(
        reactions=source.reactions,
        nu=extended_nu,
        A=source.A,
        B=source.B,
        domains=source.domains,
        version=f"{source.version}-forged-ext",
        parent_oxides=(*source.parent_oxides, "S"),
    )
    coverage = {
        name: ("EXT-SP" if name == "S" else "A-published-imcc")
        for name in (*altered.parent_oxides, *altered.reactions)
    }

    with pytest.raises(ValueError, match="complete canonical datapack hash"):
        _label_loaded_datapack(
            altered,
            model_id="IMCC-SF04-EXT",
            coverage=coverage,
        )


def test_kernel_identity_issuer_refuses_duplicate_extension_name() -> None:
    source = load_datapack(DATAPACK_PATH).kernel_datapack
    duplicate = ImccDatapack(
        reactions=source.reactions,
        nu=np.vstack([source.nu, np.zeros(source.n_complexes)]),
        A=source.A,
        B=source.B,
        domains=source.domains,
        version=f"{source.version}-duplicate",
        parent_oxides=(*source.parent_oxides, "SiO2"),
    )
    coverage = {
        name: "A-published-imcc"
        for name in (*duplicate.parent_oxides, *duplicate.reactions)
    }

    with pytest.raises(ValueError, match="globally unique"):
        _label_loaded_datapack(
            duplicate,
            model_id="IMCC-SF04-EXT",
            coverage=coverage,
        )


def test_refusal_classes_inherit_from_imcc_refusal() -> None:
    for cls in (
        ImccComponentOutsideDomainError,
        ImccCompositionOutsideValidatedEnvelopeError,
        ImccCompositionIncompleteError,
        ImccFerricInputUnsupportedError,
        ImccNonconvergenceError,
        ImccTOutsideDatapackDomainError,
        ImccMalformedDatapackError,
        ImccUnprovenDatapackError,
    ):
        assert issubclass(cls, ImccRefusal)


def test_label_block_fields_and_denylist() -> None:
    pack = load_datapack(DATAPACK_PATH)
    composition = _make_uniform_composition(pack)
    result = evaluate(composition, 2500.0, pack)
    labels = result.labels
    assert isinstance(labels, ImccAdapterLabels)
    assert labels.identity["model_id"] == "IMCC-SF04"
    assert labels.identity["datapack_version"] == "1.0.2"
    assert labels.trust == "internal-analytical"
    for name in result.species_names:
        assert labels.coverage[name] == "A-published-imcc"


def test_package_exports_only_denied_adapter_solve_path() -> None:
    assert not any(name.startswith("solve_") for name in imcc_sf04.__all__)
    assert not any(name.startswith("solve_") for name in dir(imcc_sf04))
    assert not hasattr(imcc_sf04, "solve_imcc_sf04")

    pack = load_datapack(DATAPACK_PATH)
    result = imcc_sf04.evaluate(_make_uniform_composition(pack), 2500.0, pack)
    assert result.labels.trust == "internal-analytical"


def test_extrapolation_flag() -> None:
    pack = load_datapack(DATAPACK_PATH)
    # Binary MgO-SiO2 at 1600 K: only Mg-silicate complexes are active, and
    # 1600 K is just below the SF04-exercised domain [1700, 3000] K.
    composition = {
        "SiO2": 0.5,
        "MgO": 0.5,
        "FeO": 0.0,
        "CaO": 0.0,
        "Al2O3": 0.0,
        "TiO2": 0.0,
        "Na2O": 0.0,
        "K2O": 0.0,
    }
    # Default: refusal.
    with pytest.raises(ImccTOutsideDatapackDomainError):
        evaluate(composition, 1600.0, pack)
    # Extrapolation flag: evaluate and mark.
    result = evaluate(composition, 1600.0, pack, allow_extrapolation=True)
    assert result.extrapolated is True
    assert isinstance(result.labels, ImccAdapterLabels)
