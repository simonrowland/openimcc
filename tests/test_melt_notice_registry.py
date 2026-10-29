from __future__ import annotations

from pathlib import Path

import pytest

from openimcc import evaluate, evaluate_sulfur, load_datapack
from openimcc.model import _melt_notice_score


ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "src/openimcc/data/packs"
MORB = {
    "SiO2": 50.42,
    "TiO2": 1.53,
    "Al2O3": 15.13,
    "FeO": 9.81,
    "MgO": 7.76,
    "CaO": 11.35,
    "Na2O": 2.83,
    "K2O": 0.14,
}
DEFAULT_ALKALI_NOTICE = (
    "K predictions from IMCC-SF04-D066 remain low against Hastie 1981 KEMS "
    "pressures (case 4: −1.17 to −1.22 dex); "
    "see https://github.com/simonrowland/openimcc"
)
DEFAULT_NA2O_NOTICE = (
    "na2o-activity-bias: Na2O activity and Na(g) pressure are over-predicted. "
    "At 1673 K only (temperature dependence untested), held-out Mathieu "
    "(2009) CMAS rows show: x(CaO+MgO) 0.30–<0.40, median +0.77 dex "
    "(range +0.37 to +1.20; n=12); x(CaO+MgO) >= 0.40 with Na2O/SiO2 < 1, "
    "median +1.53 dex (range +0.81 to +2.67; n=17); Na2O/SiO2 >= 1, "
    "median +4.29 dex (range +4.03 to +4.46; n=6). Pak et al. (1989) "
    "Na-rich binary points at Na2O/SiO2 ratios 1.5 and 1 are +2.76 and "
    "+1.74 dex; its Ca-bearing points at x(CaO) 0.32 and 0.42 are +0.49 and "
    "+0.67 dex; higher-silica Ca-free points are slightly negative. Evidence: "
    "R. Mathieu, PhD thesis, INPL Nancy (2009), "
    "https://docnum.univ-lorraine.fr/public/INPL/2009_MATHIEU_R.pdf (method: "
    "Mathieu et al. 2008, DOI 10.1016/j.jnoncrysol.2008.07.004); Pak et al. "
    "1989, DOI 10.2355/isijinternational.29.318. The cause remains unresolved."
)
DEFAULT_MULTICOMPONENT_NOTICE = (
    "multicomponent-activity-pattern: on default D066 v5, Kume et al. (2000) "
    "synthetic CAS/CMS/CMAS slags, after solid-to-liquid standard conversion, "
    "were measured at 1823–1873 K. SiO2 residuals "
    "(log10(predicted/measured)) decrease with NBO/T in CAS (Table 2: slope "
    "−0.238 dex/NBO/T, n=47) and CMS (Table 3: slope −0.362 dex/NBO/T, "
    "n=27), not in the four-oxide CMAS family that triggers this notice "
    "(Table 4 has no SiO2 rows). Across CAS+CMS+CMAS, RMSEs are 0.47 dex "
    "(SiO2, n=72), 0.59 (CaO, n=87), and 0.50 (MgO, n=43); four-oxide CMAS "
    "RMSEs are 0.81 dex (CaO, n=15) and 0.57 (MgO, n=15). CaO and MgO "
    "residuals become more positive with NBO/T. Al-bearing association, "
    "temperature, and source method are confounded; no cause is established. "
    "Effectively binary CaO–SiO2 Kume slopes are near zero. A regression of "
    "binary CaO–SiO2 KEMS activities from Stolyarova et al. (1991, 1933 K) "
    "gives a CaO slope of about +0.2 to +0.3 dex/NBO/T; their silica table "
    "conflicts with the paper's own closure, and this binary result does not "
    "establish the multicomponent pattern."
)
DEFAULT_CAO_SIO2_MODERATE_NOTICE = (
    "cao-sio2-activity-bias: moderate — effectively binary CaO–SiO2 melt "
    "with x(CaO) in [0.42, 0.56) on the CaO–SiO2 sub-basis at 1873–1933 K; "
    "Kume 2000 and Omori–Sanbongi 1961 show low a(SiO2); this pack raises "
    "predicted log10(a(SiO2)) by 0.045–0.410 dex over x(CaO)=[0.42, 0.56)"
)
DEFAULT_CAO_SIO2_STRONG_NOTICE = (
    "cao-sio2-activity-bias: strong — effectively binary CaO–SiO2 melt "
    "with x(CaO) in [0.56, 0.70] on the CaO–SiO2 sub-basis at 1873–1933 K, "
    "near Ca2SiO4; no Ca3SiO5 associate. Stolyarova 1991 Table IIb's "
    "0.6–1.0 dex low scale conflicts with Table Ib. That comparison's prior "
    "default residuals are not carried as v5 scores; this pack shifts predicted "
    "log10(a(SiO2)) non-monotonically by 0.076–0.479 dex, peaking near "
    "x(CaO)=0.614"
)
DEFAULT_SILICATE_PRODUCT_NOTICE = (
    "silicate-vapour-product-low-near-liquidus: diopside SI at its nominal "
    "1668 K incongruent-melting onset, diagnostic rather than congruent "
    "coexistence is −0.53 using crystal data extended to that temperature; "
    "the combined vapour product over the melt is low by 0.53 log10 units, "
    "with the split between melt activities and liquid standards unresolved"
)
TEMPERATURE_DOMAIN_NOTICE = (
    "temperature-domain-extension: active SF04 association rows are evaluated "
    "below their declared low-temperature domain using ΔrCp = 0; the active "
    "rows have declared domains 1700–3000 K; uncertainty band unavailable"
)
DEFAULT_LOW_TEMPERATURE_NOTICE = (
    "low-temperature-uncertain: the Na2Si2O5 association is evaluated below "
    "its declared domain; Piacente and Matoušek (1973) MgO-cell p(Na) is about "
    "0.54 dex low at 1278–1530 K, and Richet, Bottinga and Téqui (1984) "
    "sodium-silicate calorimetric curvature misses its envelope by 1.5–3x at "
    "1323–1700 K; treat Na-silicate activities below 1700 K as uncertain"
)
V2_ALKALI_NOTICE = (
    "K predictions from IMCC-SF04-D066 remain low against Hastie 1981 KEMS "
    "pressures (case 4: −1.18 to −1.24 dex); "
    "see https://github.com/simonrowland/openimcc"
)
V2_NA2O_UNSCORED_NOTICE = (
    "na2o-activity-bias-unscored: Na2O activity and Na(g) pressure bias "
    "evidence was measured on the default D066 v5 pack; this pack has not "
    "been scored."
)
V2_CAO_SIO2_UNSCORED_NOTICE = (
    "cao-sio2-activity-bias-unscored: the CaO–SiO2 silica-activity bias was "
    "measured on IMCC-SF04-D066 (binding digest "
    "13a185563870ec1a9c20d51bd25fb58df08c9bbbbdf7b2212b0f43ead64738cf) "
    "and has not been re-scored for this IMCC-SF04-D066 pack"
)
V2_MULTICOMPONENT_UNSCORED_NOTICE = (
    "multicomponent-activity-pattern-unscored: Kume 2000 multicomponent "
    "activity-pattern evidence was measured on the default D066 v5 pack; "
    "this pack has not been scored."
)
V2_SILICATE_PRODUCT_UNSCORED_NOTICE = (
    "silicate-vapour-product-low-near-liquidus-unscored: crystal–melt closure "
    "evidence is unscored for this pack"
)
V2_LOW_TEMPERATURE_UNSCORED_NOTICE = (
    "low-temperature-uncertain-unscored: Na-silicate low-temperature "
    "evidence is unscored for this pack"
)
FEO_HESS_CANDIDATE_NOTICE = (
    "feo-hess-candidate: opt-in Fe2SiO4 and FeTiO3 A-only Hess refits; "
    "open validation items are listed in docs/feo-liquid-candidate.md"
)
FE_HESS_ALKALI_UNSCORED_NOTICE = (
    "K predictions against Hastie 1981 KEMS pressures (case 4) have not been "
    "evaluated for this IMCC-SF04-D066 pack; "
    "see https://github.com/simonrowland/openimcc"
)


def _identity(pack) -> tuple[str, str]:
    kernel_pack = pack.kernel_datapack
    assert kernel_pack.binding_digest is not None
    return kernel_pack.model_id, kernel_pack.binding_digest


def _evaluate_notices(composition, temperature_K, pack, basis_type="mol"):
    return evaluate(
        composition,
        temperature_K,
        pack=pack,
        basis_type=basis_type,
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    ).labels.notices


@pytest.fixture(scope="module")
def default_pack():
    return load_datapack()


@pytest.mark.parametrize(
    ("composition", "temperature_K", "basis_type", "expected"),
    [
        (
            MORB,
            1800.0,
            "wt",
            (DEFAULT_ALKALI_NOTICE, DEFAULT_MULTICOMPONENT_NOTICE),
        ),
        (
            {"Na2O": 1.0, "SiO2": 1.0},
            1600.0,
            "mol",
            (
                DEFAULT_ALKALI_NOTICE,
                DEFAULT_NA2O_NOTICE,
                TEMPERATURE_DOMAIN_NOTICE,
                DEFAULT_LOW_TEMPERATURE_NOTICE,
            ),
        ),
        (
            {"CaO": 1.0, "SiO2": 1.0},
            1873.0,
            "mol",
            (DEFAULT_CAO_SIO2_MODERATE_NOTICE,),
        ),
        (
            {"CaO": 1.5, "SiO2": 1.0},
            1873.0,
            "mol",
            (DEFAULT_CAO_SIO2_STRONG_NOTICE,),
        ),
        (
            {"CaO": 1.0, "MgO": 1.0, "SiO2": 2.0},
            1668.0,
            "mol",
            (DEFAULT_SILICATE_PRODUCT_NOTICE, TEMPERATURE_DOMAIN_NOTICE),
        ),
    ],
)
def test_default_pack_notice_texts_are_pinned(
    default_pack, composition, temperature_K, basis_type, expected
) -> None:
    assert _evaluate_notices(
        composition, temperature_K, default_pack, basis_type
    ) == expected


def test_loaded_default_identity_scores_every_default_notice_family(default_pack) -> None:
    identity = _identity(default_pack)
    families = (
        "alkali-bias",
        "na2o-activity-bias",
        "cao-sio2-activity-bias",
        "multicomponent-activity-pattern",
        "silicate-vapour-product-low-near-liquidus",
        "low-temperature-uncertain",
    )
    assert all(_melt_notice_score(family, identity) == "scored" for family in families)


def test_candidate_and_d066_v2_keep_family_scoring_scoped() -> None:
    v2 = load_datapack(PACKS / "imcc-sf04-d066-v2.json")
    feo_candidate = load_datapack(PACKS / "imcc-sf04-d066-feo-hess-candidate.json")
    default_only_families = (
        "na2o-activity-bias",
        "cao-sio2-activity-bias",
        "multicomponent-activity-pattern",
        "silicate-vapour-product-low-near-liquidus",
        "low-temperature-uncertain",
    )

    assert _melt_notice_score("alkali-bias", _identity(v2)) == "scored"
    assert all(
        _melt_notice_score(family, _identity(v2)) == "unscored"
        for family in default_only_families
    )
    assert all(
        _melt_notice_score(family, _identity(feo_candidate)) == "unscored"
        for family in ("alkali-bias", *default_only_families)
    )
    assert _melt_notice_score("feo-hess-candidate", _identity(feo_candidate)) == "scored"


def test_d066_v2_unscored_notice_texts_are_pinned() -> None:
    v2 = load_datapack(PACKS / "imcc-sf04-d066-v2.json")
    assert _evaluate_notices(MORB, 1800.0, v2, "wt") == (
        V2_ALKALI_NOTICE,
        V2_MULTICOMPONENT_UNSCORED_NOTICE,
    )
    assert _evaluate_notices({"Na2O": 1.0, "SiO2": 1.0}, 1600.0, v2) == (
        V2_ALKALI_NOTICE,
        V2_NA2O_UNSCORED_NOTICE,
        TEMPERATURE_DOMAIN_NOTICE,
        V2_LOW_TEMPERATURE_UNSCORED_NOTICE,
    )
    assert _evaluate_notices({"CaO": 1.0, "SiO2": 1.0}, 1873.0, v2) == (
        V2_CAO_SIO2_UNSCORED_NOTICE,
    )
    assert _evaluate_notices(
        {"CaO": 1.0, "MgO": 1.0, "SiO2": 2.0}, 1668.0, v2
    ) == (V2_SILICATE_PRODUCT_UNSCORED_NOTICE, TEMPERATURE_DOMAIN_NOTICE)


def test_fe_hess_candidate_notice_and_unscored_activity_pattern() -> None:
    candidate = load_datapack(
        PACKS / "imcc-sf04-d066-feo-hess-candidate.json"
    )
    assert _evaluate_notices(MORB, 1800.0, candidate, "wt") == (
        FEO_HESS_CANDIDATE_NOTICE,
        FE_HESS_ALKALI_UNSCORED_NOTICE,
        V2_MULTICOMPONENT_UNSCORED_NOTICE,
    )


def test_sulfur_results_carry_melt_composition_bias_notices() -> None:
    multicomponent = {
        "SiO2": 30.0,
        "TiO2": 2.0,
        "Al2O3": 15.0,
        "FeO": 4.0,
        "MgO": 15.0,
        "CaO": 10.0,
        "Na2O": 20.0,
        "K2O": 4.0,
    }
    result = evaluate_sulfur(
        multicomponent,
        1773.0,
        log10_fO2=-8.0,
        log10_fS2=-10.0,
        melt_pack=load_datapack(),
    )
    forwarded = tuple(
        (notice.code, notice.message)
        for notice in result.notices
        if notice.model == "openimcc-melt-pack"
    )
    assert forwarded == (
        ("alkali-bias", DEFAULT_ALKALI_NOTICE),
        ("na2o-activity-bias", DEFAULT_NA2O_NOTICE),
        ("multicomponent-activity-pattern", DEFAULT_MULTICOMPONENT_NOTICE),
    )
    assert len({code for code, _message in forwarded}) == len(forwarded)

    cao_sio2 = evaluate_sulfur(
        {"CaO": 50.0, "SiO2": 49.99, "FeO": 0.01},
        1873.0,
        log10_fO2=-8.0,
        log10_fS2=-10.0,
        melt_pack=load_datapack(),
    )
    cao_forwarded = tuple(
        (notice.code, notice.message)
        for notice in cao_sio2.notices
        if notice.model == "openimcc-melt-pack"
    )
    assert cao_forwarded == (
        ("cao-sio2-activity-bias", DEFAULT_CAO_SIO2_MODERATE_NOTICE),
    )

    v2 = load_datapack(PACKS / "imcc-sf04-d066-v2.json")
    unscored = evaluate_sulfur(
        multicomponent,
        1773.0,
        log10_fO2=-8.0,
        log10_fS2=-10.0,
        melt_pack=v2,
    )
    unscored_forwarded = tuple(
        (notice.code, notice.message)
        for notice in unscored.notices
        if notice.model == "openimcc-melt-pack"
    )
    assert unscored_forwarded == (
        ("alkali-bias", V2_ALKALI_NOTICE),
        ("na2o-activity-bias-unscored", V2_NA2O_UNSCORED_NOTICE),
        ("multicomponent-activity-pattern-unscored", V2_MULTICOMPONENT_UNSCORED_NOTICE),
    )

    unscored_cao = evaluate_sulfur(
        {"CaO": 50.0, "SiO2": 49.99, "FeO": 0.01},
        1873.0,
        log10_fO2=-8.0,
        log10_fS2=-10.0,
        melt_pack=v2,
    )
    unscored_cao_forwarded = tuple(
        (notice.code, notice.message)
        for notice in unscored_cao.notices
        if notice.model == "openimcc-melt-pack"
    )
    assert unscored_cao_forwarded == (
        ("cao-sio2-activity-bias-unscored", V2_CAO_SIO2_UNSCORED_NOTICE),
    )

    feo_candidate = load_datapack(PACKS / "imcc-sf04-d066-feo-hess-candidate.json")
    candidate = evaluate_sulfur(
        multicomponent,
        1773.0,
        log10_fO2=-8.0,
        log10_fS2=-10.0,
        melt_pack=feo_candidate,
    )
    candidate_forwarded = tuple(
        (notice.code, notice.message)
        for notice in candidate.notices
        if notice.model == "openimcc-melt-pack"
    )
    assert candidate_forwarded == (
        ("alkali-bias-unscored", FE_HESS_ALKALI_UNSCORED_NOTICE),
        ("na2o-activity-bias-unscored", V2_NA2O_UNSCORED_NOTICE),
        (
            "multicomponent-activity-pattern-unscored",
            V2_MULTICOMPONENT_UNSCORED_NOTICE,
        ),
    )


def test_sulfur_keeps_closure_and_temperature_notices_on_melt_result() -> None:
    pack = load_datapack()
    cases = (
        ({"CaO": 1.0, "MgO": 1.0, "SiO2": 2.0}, 1668.0),
        ({"Na2O": 1.0, "SiO2": 1.0}, 1600.0),
    )
    melt_only_families = (
        "silicate-vapour-product-low-near-liquidus",
        "temperature-domain-extension",
        "low-temperature-uncertain",
    )
    for composition, temperature_K in cases:
        melt = evaluate(
            composition,
            temperature_K,
            pack=pack,
            allow_extrapolation=True,
            allow_out_of_envelope=True,
        )
        sulfur = evaluate_sulfur(
            composition,
            temperature_K,
            log10_fO2=-8.0,
            log10_fS2=-10.0,
            melt_pack=pack,
        )
        assert any(
            notice.startswith(melt_only_families)
            for notice in melt.labels.notices
        )
        assert not any(
            notice.code in melt_only_families for notice in sulfur.notices
        )

    fe_candidate = load_datapack(
        PACKS / "imcc-sf04-d066-feo-hess-candidate.json"
    )
    fe_melt = evaluate(MORB, 1800.0, pack=fe_candidate)
    fe_sulfur = evaluate_sulfur(
        MORB,
        1800.0,
        log10_fO2=-8.0,
        log10_fS2=-10.0,
        melt_pack=fe_candidate,
    )
    assert any(
        notice.startswith("feo-hess-candidate:")
        for notice in fe_melt.labels.notices
    )
    assert not any(notice.code == "feo-hess-candidate" for notice in fe_sulfur.notices)
