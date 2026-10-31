"""Independent coefficient checks against the printed FC87 and SF04 tables."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "src" / "openimcc" / "data" / "packs"
FC87_PATH = ROOT / "data-src" / "fc87" / "fc87.yaml"
SF04_PATH = ROOT / "data-src" / "sf04" / "sf04.yaml"
PUBLISHED_PATH = PACK_DIR / "imcc-sf04-v1.0.2.json"
MANIFEST_PATH = PACK_DIR / "MANIFEST.json"
SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _yaml_rows(path: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["rows"]


def _source_rows() -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    fc87 = {int(row["row"][1:]): row for row in _yaml_rows(FC87_PATH)}
    sf04 = {
        _normalize_formula(row["product"]): row
        for row in _yaml_rows(SF04_PATH)
        if row["kind"] == "melt_association"
    }
    return fc87, sf04


def _normalize_formula(formula: str) -> str:
    return re.sub(r"\s+", "", formula.translate(SUBSCRIPT_DIGITS))


def _at_printed_precision(actual: Any, coefficient: dict[str, Any]) -> bool:
    """Compare by exact decimal rounding at the source's printed precision."""
    text = str(coefficient["as_published"]).replace("−", "-").replace(" ", "")
    precision = len(text.partition(".")[2]) if "." in text else 0
    quantum = Decimal(1).scaleb(-precision)
    return Decimal(str(actual)).quantize(quantum) == Decimal(text)


def _matches_source_coefficients(
    pack_row: dict[str, Any], source_row: dict[str, Any]
) -> bool:
    return _at_printed_precision(pack_row["A"], source_row["A"]) and _at_printed_precision(
        pack_row["B"], source_row["B"]
    )


def _source_row_for_pack_row(
    row_number: int,
    fc87_rows: dict[int, dict[str, Any]],
    sf04_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if row_number in (31, 32, 37, 38):
        published = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))
        product = _normalize_formula(published["rows"][row_number - 1]["complex"])
        return sf04_rows[product]
    return fc87_rows[row_number]


def _filtered_stoichiometry(values: dict[str, Any]) -> dict[str, float]:
    return {name: float(amount) for name, amount in values.items() if float(amount) != 0}


def _assert_pack_reaction_matches_source(
    pack_row: dict[str, Any], source_row: dict[str, Any]
) -> None:
    assert _normalize_formula(pack_row["complex"]) == _normalize_formula(
        source_row["product"]
    )
    assert _filtered_stoichiometry(pack_row["nu"]) == {
        name: float(amount)
        for name, amount in source_row["parent_stoichiometry"].items()
    }


def test_transcription_tables_are_complete_and_located() -> None:
    fc87 = yaml.safe_load(FC87_PATH.read_text(encoding="utf-8"))
    sf04 = yaml.safe_load(SF04_PATH.read_text(encoding="utf-8"))

    assert len(fc87["rows"]) == 36
    assert [row["row"] for row in fc87["rows"]] == [f"R{i}" for i in range(1, 37)]
    assert all(row["table"] == 3 and row["printed_page"] == 211 for row in fc87["rows"])
    assert len(sf04["rows"]) == 6
    assert [row["row"] for row in sf04["rows"]] == list(range(1, 7))
    assert all(row["table"] == 2 and row["printed_page"] == 218 for row in sf04["rows"])
    assert fc87["source"]["pdf_sha256"] == (
        "4b4f1f3792bec5d32179580df3b27c051ce19b58985be88544f71536909d7bee"
    )
    assert sf04["source"]["pdf_sha256"] == (
        "c50765341a3be2cdd0e305cda582f996f4f3a7046469e327043a70493f46c46c"
    )


def test_recorded_sf04_pack_doi_erratum_matches_immutable_pack() -> None:
    transcription = yaml.safe_load(SF04_PATH.read_text(encoding="utf-8"))
    erratum = transcription["verification"]["published_pack_doi_erratum"]
    assert erratum["pack"] == "src/openimcc/data/packs/imcc-sf04-v1.0.2.json"
    assert erratum["field"] == "sources.SF04.citation"
    assert erratum["correct_doi"] == transcription["source"]["doi"]

    pack = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))
    citation = pack["sources"]["SF04"]["citation"]
    assert f"DOI {erratum['incorrect_doi']}" in citation


def test_fc87_r35_preserves_printed_operator_and_interpretation() -> None:
    fc87_rows, _ = _source_rows()
    row = fc87_rows[35]
    assert row["pseudoreaction_as_printed"] == (
        "½K₂O(liq) + ½Al₂O₃(liq) + KAlO₂(liq)"
    )
    assert row["pseudoreaction_interpreted"] == (
        "½K₂O(liq) + ½Al₂O₃(liq) = KAlO₂(liq)"
    )
    assert "printing error" in row["interpretation_note"]


def test_published_profile_reproduces_both_printed_tables_row_for_row() -> None:
    # Premise: both tables define log10 K for the displayed formation reaction,
    # and each pack row uses the same oxide stoichiometry in its mass-action
    # denominator. Algebra: K_pack = a(product) / prod(a(oxide)**nu) = K_print,
    # so A_pack=A_print and B_pack=B_print; no standard-state conversion is
    # applied to this published-profile pack. Check: compare exact decimal
    # values after rounding only to each printed coefficient's displayed digits.
    fc87_rows, sf04_rows = _source_rows()
    pack = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))
    rows = pack["rows"]
    assert len(rows) == 38

    for pack_row in rows:
        source_row = _source_row_for_pack_row(pack_row["row"], fc87_rows, sf04_rows)
        _assert_pack_reaction_matches_source(pack_row, source_row)
        assert _matches_source_coefficients(pack_row, source_row), (
            pack_row["row"],
            pack_row["complex"],
            pack_row["A"],
            pack_row["B"],
            source_row["A"]["as_published"],
            source_row["B"]["as_published"],
        )


# Explicit exceptions in manifest-listed D066 packs. For each listed row, the
# corresponding pack metadata records the reason: `standard_state_refit`
# stores the parent set, fit interval and least-squares method; `k_translation`
# stores the reference change and preserved quantity; the Hess rows record an
# independent calorimetric formation-function fit; the row-3 fusion refit uses
# the selected Shornikov ΔfusH=55.81 kJ/mol; and the FeO rows record the
# Darken–Gurry/Masson standard-state conversion. These rows must differ from
# the source coefficients at printed precision and are never tolerance-matched.
_K_TRANSLATED_ROWS = {33, 34, 35, 36}
_STANDARD_STATE_REFITS = {
    1, 2, 7, 11, 12, 13, 14, 15, 17, 18, 20, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 36, 37, 38,
}
EXPECTED_D066_TRANSFORMS: dict[str, dict[str, set[int]]] = {
    "imcc-sf04-d066-ext-v1.json": {"k_translation": _K_TRANSLATED_ROWS},
    "imcc-sf04-d066-feo-hess-candidate.json": {
        "k_translation": _K_TRANSLATED_ROWS,
        "standard_state_refit": {
            1, 2, 7, 11, 12, 13, 14, 15, 17, 18, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 36, 37, 38,
        },
        "hess_refit": {1, 3, 11, 12, 14, 19, 20},
        "feo_conversion": {21},
    },
    "imcc-sf04-d066-feo-liquid-candidate.json": {
        "k_translation": _K_TRANSLATED_ROWS,
        "feo_conversion": {19, 20, 21},
    },
    "imcc-sf04-d066-sio2-na2o-candidate.json": {
        "k_translation": _K_TRANSLATED_ROWS,
        "standard_state_refit": _STANDARD_STATE_REFITS,
        "fusion_refit": {3},
        "feo_conversion": {19, 20, 21},
    },
    "imcc-sf04-d066-v1-kcaalsi2o7.json": {"k_translation": _K_TRANSLATED_ROWS},
    "imcc-sf04-d066-v1.json": {"k_translation": _K_TRANSLATED_ROWS},
    "imcc-sf04-d066-v2.json": {
        "k_translation": _K_TRANSLATED_ROWS,
        "feo_conversion": {19, 20, 21},
        "fusion_refit": {3},
    },
    "imcc-sf04-d066-v3.json": {
        "k_translation": _K_TRANSLATED_ROWS,
        "feo_conversion": {19, 20, 21},
        "fusion_refit": {3},
    },
    "imcc-sf04-d066-v4.json": {
        "k_translation": _K_TRANSLATED_ROWS,
        "standard_state_refit": _STANDARD_STATE_REFITS,
        "feo_conversion": {19, 20, 21},
        "fusion_refit": {3},
    },
    "imcc-sf04-d066-v5.json": {
        "k_translation": _K_TRANSLATED_ROWS,
        "standard_state_refit": _STANDARD_STATE_REFITS,
        "hess_refit": {1, 3, 11, 12, 14},
        "feo_conversion": {19, 20, 21},
    },
}


def _manifest_d066_entries() -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [entry for entry in manifest["packs"] if "d066" in entry["file"].lower()]


def _recorded_transforms(pack_row: dict[str, Any]) -> set[str]:
    transforms = set()
    if pack_row.get("standard_state_refit"):
        transforms.add("standard_state_refit")
    if pack_row.get("k_translation"):
        transforms.add("k_translation")
    if "Independent" in pack_row.get("source", "") and "Hess" in pack_row.get("source", ""):
        transforms.add("hess_refit")
    if "fusion enthalpy corrected from Shornikov" in pack_row.get("source", ""):
        transforms.add("fusion_refit")
    if (
        "parent standard transformed" in pack_row.get("source", "")
        or "FeO candidate transform" in pack_row.get("derivation", "")
    ):
        transforms.add("feo_conversion")
    return transforms


def _d066_direct_cases() -> list[tuple[str, int]]:
    cases = []
    for entry in _manifest_d066_entries():
        pack = json.loads((PACK_DIR / entry["file"]).read_text(encoding="utf-8"))
        transformed = set().union(*EXPECTED_D066_TRANSFORMS[entry["file"]].values())
        cases.extend(
            (entry["file"], row["row"])
            for row in pack["rows"]
            if row["row"] not in transformed
        )
    return cases


@pytest.mark.parametrize(("pack_name", "row_number"), _d066_direct_cases())
def test_d066_rows_claimed_unchanged_match_printed_coefficients(
    pack_name: str, row_number: int
) -> None:
    """Only source rows without an explicit refit/conversion are compared."""
    fc87_rows, sf04_rows = _source_rows()
    pack = json.loads((PACK_DIR / pack_name).read_text(encoding="utf-8"))
    pack_row = next(row for row in pack["rows"] if row["row"] == row_number)
    source_row = _source_row_for_pack_row(row_number, fc87_rows, sf04_rows)

    assert _recorded_transforms(pack_row) == set()
    assert "FC87 Table 3" in pack_row["source"] or "SF04 Table 2" in pack_row["source"]
    _assert_pack_reaction_matches_source(pack_row, source_row)
    assert _matches_source_coefficients(pack_row, source_row), (
        pack_name,
        row_number,
        pack_row["complex"],
        pack_row["A"],
        pack_row["B"],
        source_row["A"]["as_published"],
        source_row["B"]["as_published"],
    )


def _assert_transform_reason(
    pack_row: dict[str, Any], transform: str
) -> None:
    if transform == "standard_state_refit":
        metadata = pack_row["standard_state_refit"]
        assert metadata["parents"]
        assert metadata["temperature_range_K"]
        assert metadata["fit_method"]
        assert pack_row["derivation"]
    elif transform == "k_translation":
        metadata = pack_row["k_translation"]
        assert metadata["basis"]
        assert metadata["new_reference"]
        assert metadata["operation"]
        assert metadata["preserved_quantity"]
        assert pack_row["derivation"]
    elif transform == "hess_refit":
        assert "Independent" in pack_row["source"]
        assert "Hess" in pack_row["source"]
        assert "fit" in pack_row["derivation"].lower()
    elif transform == "fusion_refit":
        assert "fusion enthalpy corrected from Shornikov" in pack_row["source"]
        assert "Shornikov (2017)" in pack_row["derivation"]
        assert "55.81" in pack_row["derivation"]
    elif transform == "feo_conversion":
        assert (
            "parent standard transformed" in pack_row["source"]
            or "FeO candidate transform" in pack_row["derivation"]
        )
        assert "Δlog10K" in pack_row["derivation"]
        assert "c0=27778.965828" in pack_row["derivation"]
        assert "c1=-10.961554232" in pack_row["derivation"]
    else:  # pragma: no cover - the explicit map limits the accepted categories
        raise AssertionError(transform)


def test_d066_recorded_refits_and_standard_state_conversions_are_explicit() -> None:
    fc87_rows, sf04_rows = _source_rows()
    entries = _manifest_d066_entries()
    assert {entry["file"] for entry in entries} == set(EXPECTED_D066_TRANSFORMS)

    for entry in entries:
        pack = json.loads((PACK_DIR / entry["file"]).read_text(encoding="utf-8"))
        assert len(pack["rows"]) == entry["n_rows"]
        categories = {
            "standard_state_refit",
            "k_translation",
            "hess_refit",
            "feo_conversion",
            "fusion_refit",
        }
        expected = {
            category: EXPECTED_D066_TRANSFORMS[entry["file"]].get(category, set())
            for category in categories
        }
        observed = {
            category: {
                row["row"]
                for row in pack["rows"]
                if category in _recorded_transforms(row)
            }
            for category in categories
        }
        assert observed == expected

        for pack_row in pack["rows"]:
            source_row = _source_row_for_pack_row(pack_row["row"], fc87_rows, sf04_rows)
            _assert_pack_reaction_matches_source(pack_row, source_row)
            transforms = _recorded_transforms(pack_row)
            if not transforms:
                continue
            for transform in transforms:
                _assert_transform_reason(pack_row, transform)
            assert not _matches_source_coefficients(pack_row, source_row), (
                entry["file"],
                pack_row["row"],
                sorted(transforms),
                "recorded transformation unexpectedly reproduces source coefficients",
            )
