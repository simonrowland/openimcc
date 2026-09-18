"""Owner-directed IMCC-SF04 S/P extension gates."""

from __future__ import annotations

import json
import csv
import math
from pathlib import Path

import pytest

from openimcc import (
    ImccCompositionOutsideValidatedEnvelopeError,
    ImccMalformedDatapackError,
    ImccSPComponentRequiresExtensionError,
    evaluate,
    load_datapack,
)
# Inlined from the simulator's sulfide-liquid module, which does not travel
# with this package. Standard chemical potential of FeS(l) at 1300 K.
FES_MU0_1300K_J_PER_MOL = -257599.6112
from openimcc.kernel import solve_imcc_sf04


BASE_DATAPACK = Path(
    "src/openimcc/data/packs/imcc-sf04-v1.0.2.json"
)
EXT4 = Path(
    "docs-private/research/2026-08-09-upstream-mission/IMCC-impl/ext4"
)
_MISSING = object()


def _ext4() -> Path:
    """EXT4 is gitignored research working set; skip rather than FileNotFoundError."""
    if not EXT4.is_dir():
        pytest.skip(
            "gitignored IMCC ext4 working set is absent from this checkout"
        )
    return EXT4


def _write_ext_pack(tmp_path: Path, **row_overrides: object) -> Path:
    data = json.loads(BASE_DATAPACK.read_text())
    row = {
        "complex": "FeS",
        "nu": {"FeO": 1, "S": 1},
        "A": -1.0,
        "B": 1000.0,
        "T_domain_K": [400.0, 3000.0],
        "T_domain_basis": "JANAF Fe-024 liquid table",
        "provenance_class": "extension-compound-thermo",
        "tier": "EXT-SP",
        "certification": "denied",
        "provenance": {
            "source": "NIST-JANAF",
            "table_id": "Fe-024",
        },
    }
    for field, value in row_overrides.items():
        if value is _MISSING:
            row.pop(field, None)
        else:
            row[field] = value
    data.update(
        {
            "model_id": "IMCC-SF04-EXT",
            "imcc_sf04_datapack_version": "1.0.2-ext-sp-test",
            "sp_extension": {
                "enable_flag": "enable_sp_extension",
                "parents": ["S", "P2O5"],
                "tier": "EXT-SP",
                "certification": "denied",
                "authority": "screening-only; SulfLiq remains matte authority",
                "rows": [row],
            },
        }
    )
    path = tmp_path / "ext.json"
    path.write_text(json.dumps(data))
    return path


def _write_migrated_delivered_ext_pack(tmp_path: Path) -> Path:
    data = json.loads((_ext4() / "ext-pack.json").read_text())
    for row in data["sp_extension"]["rows"]:
        row["provenance_class"] = "extension-compound-thermo"
    path = tmp_path / "delivered-ext-pack.json"
    path.write_text(json.dumps(data))
    return path


def test_ext_pack_keeps_published_core_distinct(tmp_path: Path) -> None:
    pack = load_datapack(_write_ext_pack(tmp_path))
    assert pack.model_id == "IMCC-SF04-EXT"
    assert pack.parent_oxides[-2:] == ("S", "P2O5")
    assert pack.kernel_datapack.n_parents == 10
    assert pack.kernel_datapack.n_complexes == 39
    assert pack.extension_species == ("FeS",)


@pytest.mark.parametrize("component", ["S", "P2O5"])
def test_plain_pack_sp_input_has_explanatory_typed_refusal(component: str) -> None:
    pack = load_datapack(BASE_DATAPACK)
    with pytest.raises(ImccSPComponentRequiresExtensionError) as exc:
        evaluate({"SiO2": 1.0, component: 0.01}, 2500.0, pack)
    assert exc.value.code == "imcc_sp_extension_required"
    assert component in str(exc.value)
    assert "S/P EXT component class" in str(exc.value)
    assert "model_id='IMCC-SF04-EXT'" in str(exc.value)
    assert "enable_sp_extension=True" in str(exc.value)


def test_extension_requires_explicit_flag(tmp_path: Path) -> None:
    pack = load_datapack(_write_ext_pack(tmp_path))
    with pytest.raises(ImccSPComponentRequiresExtensionError) as exc:
        evaluate({"SiO2": 0.9, "FeO": 0.09, "S": 0.01}, 2500.0, pack)
    assert "enable_sp_extension=True" in str(exc.value)


def test_flag_does_not_widen_plain_model() -> None:
    pack = load_datapack(BASE_DATAPACK)
    with pytest.raises(ImccSPComponentRequiresExtensionError):
        evaluate(
            {"SiO2": 0.99, "S": 0.01},
            2500.0,
            pack,
            enable_sp_extension=True,
        )


def test_extension_flag_solves_and_labels_every_sp_output(tmp_path: Path) -> None:
    pack = load_datapack(_write_ext_pack(tmp_path))
    result = evaluate(
        {"SiO2": 0.60, "MgO": 0.19, "FeO": 0.20, "S": 0.01},
        2500.0,
        pack,
        enable_sp_extension=True,
    )
    assert result.labels.identity["model_id"] == "IMCC-SF04-EXT"
    for name in ("S", "P2O5", "FeS"):
        assert result.labels.coverage[name] == "EXT-SP"
    assert result.labels.coverage["SiO2"] == "A-published-imcc"


def test_extension_pack_carries_identity_through_direct_kernel_path(
    tmp_path: Path,
) -> None:
    pack = load_datapack(_write_migrated_delivered_ext_pack(tmp_path))
    parent_mol = [
        1.0 if name in {"SiO2", "MgO", "FeO", "S", "P2O5"} else 0.0
        for name in pack.parent_oxides
    ]
    result = solve_imcc_sf04(
        parent_mol,
        1500.0,
        pack.kernel_datapack,
        allow_extrapolation=True,
    )

    assert result.labels.model_id == "IMCC-SF04-EXT"
    for name in (*pack.extension_parents, *pack.extension_species):
        assert result.labels.coverage[name] == "EXT-SP"
    assert result.labels.coverage["SiO2"] == "A-published-imcc"


def test_extension_parents_do_not_dilute_oxide_envelope(tmp_path: Path) -> None:
    pack = load_datapack(_write_ext_pack(tmp_path))
    with pytest.raises(ImccCompositionOutsideValidatedEnvelopeError):
        evaluate(
            {"SiO2": 0.49, "Na2O": 0.51, "S": 100.0},
            2500.0,
            pack,
            enable_sp_extension=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param(
            "provenance_class",
            _MISSING,
            id="missing-provenance-class",
        ),
        pytest.param(
            "provenance_class",
            "published-imcc",
            id="wrong-provenance-class",
        ),
        ("tier", "A-published-imcc"),
        ("certification", "allowed"),
        ("provenance", {}),
        ("nu", {"FeO": 1}),
    ],
)
def test_extension_row_requires_denied_ext_sp_labels(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(ImccMalformedDatapackError) as exc:
        load_datapack(_write_ext_pack(tmp_path, **{field: value}))
    if field == "provenance_class":
        assert "sp_extension row 0" in str(exc.value)
        assert "provenance_class must be 'extension-compound-thermo'" in str(
            exc.value
        )


def test_delivered_ext_pack_has_five_janaf_rows(tmp_path: Path) -> None:
    migrated_pack = _write_migrated_delivered_ext_pack(tmp_path)
    pack = load_datapack(migrated_pack)
    assert pack.model_id == "IMCC-SF04-EXT"
    assert pack.kernel_datapack.n_parents == 10
    assert pack.kernel_datapack.n_complexes == 43
    assert set(pack.extension_species) == {
        "FeS",
        "Na2S",
        "MgSO4",
        "Na2SO4",
        "Mg3P2O8",
    }
    raw = json.loads(migrated_pack.read_text())
    for row in raw["sp_extension"]["rows"]:
        assert row["provenance_class"] == "extension-compound-thermo"
        assert row["tier"] == "EXT-SP"
        assert row["certification"] == "denied"
        assert row["provenance"]["table_id"]
        assert row["provenance"]["source_url"].startswith(
            "https://janaf.nist.gov/tables/"
        )


def test_delivered_ext_pack_executes_through_flagged_adapter(tmp_path: Path) -> None:
    pack = load_datapack(_write_migrated_delivered_ext_pack(tmp_path))
    result = evaluate(
        {
            "SiO2": 0.55,
            "MgO": 0.18,
            "FeO": 0.18,
            "CaO": 0.04,
            "Al2O3": 0.03,
            "Na2O": 0.005,
            "K2O": 0.005,
            "S": 0.005,
            "P2O5": 0.005,
        },
        1500.0,
        pack,
        enable_sp_extension=True,
        allow_extrapolation=True,
    )
    assert result.convergence.status == "converged"
    assert result.labels.identity["model_id"] == "IMCC-SF04-EXT"
    assert all(
        result.labels.coverage[name] == "EXT-SP"
        for name in (*pack.extension_parents, *pack.extension_species)
    )


@pytest.mark.parametrize(
    ("filename", "expected_rows"),
    [("compound-fit-workings.csv", 15), ("gas-fit-workings.csv", 39)],
)
def test_delivered_fits_have_three_independent_spots_per_species(
    filename: str, expected_rows: int
) -> None:
    rows = list(csv.DictReader((_ext4() / filename).open()))
    assert len(rows) == expected_rows
    by_species: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_species.setdefault(row["species"], []).append(row)
        assert abs(float(row["printed_minus_direct"])) < 0.01
        assert "1000[J/kJ]" in row["unit_check"]
        assert row["printed_vs_direct_attribution"] in {
            "printed JANAF rounding",
            "none at shown precision",
        }
        assert row["fit_vs_direct_attribution"].startswith("A+B/T compression")
        assert "fit-vs-direct: A+B/T form compression" in row[
            "discrepancy_attribution"
        ]
    assert {tuple(int(item["T_K"]) for item in items) for items in by_species.values()} == {
        (500, 900, 1500)
    }


def test_delivered_fit_sources_are_phase_verified_against_local_mirror() -> None:
    compounds = json.loads((_ext4() / "compound-fits.json").read_text())
    for row in compounds:
        source_lines = Path(row["local_mirror_path"]).read_text().splitlines()
        assert source_lines[0].split("\t")[-1].endswith("(l)")
        transition_lines = Path(row["local_transition_mirror_path"]).read_text().splitlines()
        assert transition_lines[0].split("\t")[-1].endswith("(cr,l)")
        fusion_prefix = f"{float(row['fusion_K']):.3f}\t"
        assert any(
            line.startswith(fusion_prefix) and "LIQUID" in line
            for line in transition_lines
        )

    gases = json.loads((_ext4() / "gas-channels.json").read_text())
    for row in gases:
        source_lines = Path(row["local_mirror_path"]).read_text().splitlines()
        assert source_lines[0].split("\t")[-1].endswith("(g)")
        assert row["table_T_range_K"] == [100, 6000]


def test_delivered_gas_set_and_so2_s2_hand_anchor() -> None:
    channels = json.loads((_ext4() / "gas-channels.json").read_text())
    assert {row["species"] for row in channels} == {
        "S2",
        "SO",
        "SO2",
        "SO3",
        "PS",
        "PO",
        "PO2",
        "P2",
        "P4",
        "P4O6",
        "P4O10(g)",
        "Na2SO4(g)",
        "K2SO4(g)",
    }
    anchors = json.loads((_ext4() / "anchors.json").read_text())
    # S2 + 2 O2 -> 2 SO2 at 1300 K. JANAF ΔrG° = -533.612 kJ/mol,
    # so log10(Kp) = 533612/(R*1300*ln(10)).
    hand_log10_kp = 533_612 / (8.31446261815324 * 1300 * math.log(10))
    assert anchors["SO2_S2"]["hand_log10_Kp"] == pytest.approx(
        hand_log10_kp, rel=1e-12
    )
    assert anchors["SO2_S2"]["test_composition"] == {
        "a_FeO": 0.35,
        "a_S": 0.01,
    }
    assert anchors["SO2_S2"]["test_quotient_log10_Kp"] == pytest.approx(
        anchors["SO2_S2"]["fitted_channel_log10_Kp"], abs=1e-12
    )
    assert abs(anchors["SO2_S2"]["fit_minus_hand_log10"]) < 0.001


def test_fes_anchor_reports_delta_without_retune() -> None:
    anchors = json.loads((_ext4() / "anchors.json").read_text())["FeS"]
    janaf_mu0_kj_mol = -68.811 - 1300 * 132.203 / 1000
    assert anchors["JANAF_Fe_024_mu0_1300K_kJ_mol"] == pytest.approx(
        janaf_mu0_kj_mol
    )
    assert anchors["SulfLiq_mu0_1300K_kJ_mol"] == pytest.approx(
        FES_MU0_1300K_J_PER_MOL / 1000
    )
    assert anchors["JANAF_minus_SulfLiq_kJ_mol"] == pytest.approx(
        16.9247112
    )


def test_screening_surface_is_complete_and_explicitly_non_authoritative() -> None:
    rows = list(csv.DictReader((_ext4() / "screening-surface.csv").open()))
    assert len(rows) == 24
    assert {row["feedstock_id"] for row in rows} == {
        "lunar_mare_low_ti",
        "lunar_pkt_kreep_average",
    }
    assert {int(row["T_K"]) for row in rows} == set(range(400, 1501, 100))
    for row in rows:
        assert row["tier"] == "EXT-SP"
        assert row["certification"] == "denied"
        assert row["published_base_imcc_rows_executed"] == "False"
        assert "screening-only" in row["authority"]
