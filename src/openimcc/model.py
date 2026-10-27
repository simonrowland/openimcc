"""IMCC-SF04 domain adapter: JSON datapack loader and caller-facing API.

Wraps the kernel from ``openimcc.kernel`` with a JSON datapack loader, a
wt-to-mol basis converter, and a caller-facing label block.

This module is the MODEL half and is deliberately free of simulator policy:
its only non-model dependency is ``openimcc.scalar_boundary`` (a stdlib-only
leaf). The ``MeltBackend`` surface used by ``resolve_backend`` lives in
the simulator's backend glue.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from importlib import resources
from pathlib import Path
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from openimcc.gas import ImccGasDatapack

from openimcc.kernel import (
    ImccComponentOutsideDomainError,
    ImccCompositionIncompleteError,
    ImccDatapack,
    ImccFerricInputUnsupportedError,
    ImccInvalidWarmStartError,
    ImccNonconvergenceError,
    ImccRefusal,
    ImccResult,
    ImccTOutsideDatapackDomainError,
    _PUBLISHED_DATAPACK_SHA256,
    _PUBLISHED_MODEL_ID,
    _label_loaded_datapack,
    _published_datapack_manifest_hash,
    _solve_imcc_sf04_with_seed,
    label_research_datapack,
    solve_imcc_sf04,
)
from openimcc.scalar_boundary import is_declared_real_scalar



# Stable molar masses (g/mol) for the IMCC-SF04 parent components.
# Values from the project physical-constants table (CIAAW/NIST derived).
_OXIDE_MOLAR_MASS_G_MOL = MappingProxyType(
    {
        "SiO2": 60.0843,
        "MgO": 40.3044,
        "FeO": 71.844,
        "CaO": 56.0774,
        "Al2O3": 101.9613,
        "TiO2": 79.866,
        "Na2O": 61.9789,
        "K2O": 94.196,
        "S": 32.06,
        "P2O5": 141.9445,
    }
)

_EXPECTED_PARENT_OXIDES = (
    "SiO2",
    "MgO",
    "FeO",
    "CaO",
    "Al2O3",
    "TiO2",
    "Na2O",
    "K2O",
)

_PUBLISHED_CORE_ROWS = 38

_D066_MODEL_ID = "IMCC-SF04-D066"
_EXCESS_GIBBS_MODEL_ID = "IMCC-SF04-D066-EXCESS-CANDIDATE"
_D066_CORE_ROWS = (
    "Mg2SiO4", "MgSiO3", "MgAl2O4", "MgTiO3", "MgTi2O5", "Mg2TiO4",
    "Al6Si2O13", "CaAl2O4", "CaAl4O7", "Ca12Al14O33", "CaSiO3",
    "CaAl2Si2O8", "CaMgSi2O6", "Ca2MgSi2O7", "Ca2Al2SiO7", "CaTiO3",
    "Ca2SiO4", "CaTiSiO5", "FeTiO3", "Fe2SiO4", "FeAl2O4", "CaAl12O19",
    "Mg2Al4Si5O18", "Na2SiO3", "Na2Si2O5", "NaAlSiO4", "NaAlSi3O8",
    "NaAlO2", "Na2TiO3", "NaAlSi2O6", "K2SiO3", "K2Si2O5", "KAlSiO4",
    "KAlSi3O8", "KAlO2", "KAlSi2O6", "K2Si4O9", "KCaAlSi2O7",
)
_D066_TRANSLATED_ROWS = frozenset(
    {"KAlSiO4", "KAlSi3O8", "KAlO2", "KAlSi2O6"}
)
_D066_REFERENCE_UNKNOWN_ROWS = frozenset(
    {"K2SiO3", "K2Si2O5", "K2Si4O9"}
)
_D066_E25_ROWS = frozenset({"KCaAlSi2O7"})
_D066_CARRIED_PUBLISHED_ROWS = frozenset(
    name
    for name in _D066_CORE_ROWS
    if name not in (
        _D066_TRANSLATED_ROWS
        | _D066_REFERENCE_UNKNOWN_ROWS
        | _D066_E25_ROWS
    )
)
_D066_ROW_COVERAGE = MappingProxyType(
    {
        name: label
        for rows, label in (
            (_D066_TRANSLATED_ROWS, "D-translated-d066"),
            (_D066_REFERENCE_UNKNOWN_ROWS, "D-reference-unknown"),
            (_D066_E25_ROWS, "D-e25-kcaalsi2o7"),
            (_D066_CARRIED_PUBLISHED_ROWS, "D-carried-published"),
        )
        for name in rows
    }
)
_D066_E25_INACTIVE_REASON = "E25: out of liquid domain; owner-confirmed"

_SP_EXTENSION_MODEL_ID = "IMCC-SF04-EXT"
_SP_EXTENSION_PARENTS = ("S", "P2O5")
_SP_EXTENSION_TIER = "EXT-SP"
_SP_EXTENSION_FLAG = "enable_sp_extension"
_SP_EXTENSION_PROVENANCE_CLASS = "extension-compound-thermo"
_R_J_MOL_K = 8.31446261815324
_ENVELOPE_RELATIVE_SLACK = 1.0e-5
_ACID_SINK_FAMILY_SHARE = 0.20
_PARENT_CATION_SYMBOL = MappingProxyType(
    {
        "SiO2": "Si",
        "MgO": "Mg",
        "FeO": "Fe",
        "CaO": "Ca",
        "Al2O3": "Al",
        "TiO2": "Ti",
        "Na2O": "Na",
        "K2O": "K",
        "S": "S",
        "P2O5": "P",
    }
)
# Data derivation (2026-09-25; no solver retune):
# (a) The smallest strict-path validated ratio is 1.95684635346e-3
#     (kume2000_s145, 1823 K), and every SF04 Table 5 rock also remains
#     unflagged.
# (b) At 1473 K, evaluate both binaries with both overrides and scan
#     X_Me2O=.450,.451,...,.500. For each point, residual is
#     log10(a_model(Me2O)) minus the linear bridge between the published
#     anchors. The relevant boundary rows are:
#
#       binary       X       ratio             bridge residual   below floor  flag
#       K2O-SiO2     .493    2.183642e-3       +1.33              no           no
#       K2O-SiO2     .494    1.864382e-3       +1.40              yes          yes
#       K2O-SiO2     .495    1.547655e-3       +1.47              yes          yes
#       K2O-SiO2     .496    1.233415e-3       +1.57              yes          yes
#       Na2O-SiO2    .497    2.472783e-3       +0.47              no           no
#       Na2O-SiO2    .498    1.644304e-3       +0.64              yes          yes
#       Na2O-SiO2    .499    8.225617e-4       +0.93              yes          yes
#       Na2O-SiO2    .500    5.783685e-5       +2.07              yes          yes
#
# The K bridge is -7.46 at X=.457 to -7.27 at X=.500; the Na bridge is
# -6.220 at X=.450 to -5.622 at X=.500. The largest (b) ratio below the
# floor is therefore 1.86438232260e-3 at K2O-SiO2 X=.494. The same 0.001
# scan at the other temperatures gives this compact stability table (the
# first sub-floor point is also the largest sub-floor point in each scan):
#
#       T (K)   K2O-SiO2 X / ratio       Na2O-SiO2 X / ratio
#       1373    .493 / 1.762926e-3       .498 / 1.410644e-3
#       1473    .494 / 1.864382e-3       .498 / 1.644304e-3
#       1573    .495 / 1.863152e-3       .498 / 1.879865e-3
#
# Every stability-table ratio stays below the final cut; the 1473 K
# published-anchor evidence sets the lower bound.
# The admissible window is (1.86438232260e-3, 1.95684635346e-3), so use
# their geometric mean, sqrt(product) = 1.91005490744e-3. This is a
# predict-and-flag threshold, not a coefficient change.
_SPECIES_COVERAGE_EDGE_RATIO = 1.9100549074388355e-3
# The benchmark range uses the tracked Hastie case-4 K points and the bench
# runner's default gas pack. Key by the complete melt-pack identity so a new
# or external pack cannot inherit another pack's measured bias.
_ALKALI_BIAS_BENCHMARKS = MappingProxyType(
    {
        (
            "IMCC-SF04",
            "ae979b68295f6e653fb1c37d6b381f06cb0be8d4c5b421e673420a6e4596357c",
        ): "−1.20 to −1.26 dex",
        (
            "IMCC-SF04-EXT",
            "de51a9bcf52724fd4b06e9cb639779f31189d86ecf304073fd72fc3019f29770",
        ): "−1.20 to −1.26 dex",
        (
            "IMCC-SF04-D066",
            "02c88c04fe09f8f494adda8a76e1f95fee536d11b2b6d66639aaea097fce5e75",
        ): "−1.17 to −1.22 dex",
        (
            "IMCC-SF04-D066",
            "45e12de67ee8effa56fe3935e12e66e0dda2ea73ce877dc598ccda3faaea453f",
        ): "−1.17 to −1.22 dex",
        (
            "IMCC-SF04-D066",
            "25125d24b9a7f3907773811c245b8137f40e46794caba5ce12f9ea26b7a49ce8",
        ): "−1.17 to −1.22 dex",
        (
            "IMCC-SF04-D066",
            "efd2330f57d366b660e100d821806054b59134cb95b6404e1f9c27b5e468893e",
        ): "−1.18 to −1.24 dex",
        (
            "IMCC-SF04-D066",
            "252f8705a3d04799ab9131836785992ffbedb6e01c98780c849498f4836804f2",
        ): "−1.18 to −1.24 dex",
        (
            "IMCC-SF04-D066",
            "0879f5a8e94dfd844ed5cfcd900677b496a89e717a85f320b3e0b374ebcdc3b0",
        ): "−1.18 to −1.24 dex",
        (
            "IMCC-SF04-D066",
            "13a185563870ec1a9c20d51bd25fb58df08c9bbbbdf7b2212b0f43ead64738cf",
        ): "−1.17 to −1.22 dex",
        (
            "IMCC-SF04-D066",
            "ac88aa2272c50d1cbf300634737c5d0071bddee0cd0f0483a6530c4a4992c393",
        ): "−1.18 to −1.24 dex",
    }
)
_ALKALI_BIAS_BENCH_URL = "https://github.com/simonrowland/openimcc"

# At 1673 K the 38 held-out Mathieu rows with x(CaO+MgO) in [0.30, 0.40)
# have median +0.77 dex (range +0.37 to +1.20); below 0.30, the three rows
# are mixed (−0.26 to +1.37). This makes 0.30 the lowest observed basicity
# boundary with a consistently positive residual. Basicity is the CaO+MgO
# mole fraction on the complete parent-oxide basis. Independently, Na2O/SiO2
# >= 1 catches the severe low-silica Mathieu and Na-rich Pak binary errors.
_NA2O_ACTIVITY_BIAS_RATIO_MIN = 1.0
_NA2O_ACTIVITY_BIAS_BASICITY_MIN = 0.30
_NA2O_ACTIVITY_BIAS_NOTICE = (
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


def _na2o_activity_bias_notice(
    parent_oxides: Sequence[str], moles: np.ndarray
) -> str | None:
    """Flag Na-bearing melts in either held-out composition region."""
    try:
        na_index = parent_oxides.index("Na2O")
        silica_index = parent_oxides.index("SiO2")
    except ValueError:
        return None
    na = float(moles[na_index])
    silica = float(moles[silica_index])
    if na <= 0.0:
        return None
    if silica > 0.0 and na / silica >= _NA2O_ACTIVITY_BIAS_RATIO_MIN:
        return _NA2O_ACTIVITY_BIAS_NOTICE
    basicity = 0.0
    total = float(moles.sum())
    if total <= 0.0:
        return None
    for name in ("CaO", "MgO"):
        if name in parent_oxides:
            basicity += float(moles[parent_oxides.index(name)])
    if basicity / total >= _NA2O_ACTIVITY_BIAS_BASICITY_MIN:
        return _NA2O_ACTIVITY_BIAS_NOTICE
    return None


def _alkali_bias_notice(model_id: str, binding_digest: str) -> str:
    measured_bias = _ALKALI_BIAS_BENCHMARKS.get((model_id, binding_digest))
    if measured_bias is None:
        return (
            "K predictions against Hastie 1981 KEMS pressures (case 4) have "
            f"not been evaluated for this {model_id} pack; see "
            f"{_ALKALI_BIAS_BENCH_URL}"
        )
    return (
        f"K predictions from {model_id} remain low against Hastie 1981 KEMS "
        f"pressures (case 4: {measured_bias}); see {_ALKALI_BIAS_BENCH_URL}"
    )


# CaO–SiO2 activity-bias notice. Derivation from Kume 2000, Omori and
# Sanbongi 1961, and Stolyarova 1991. The residuals support a notice, not
# a coefficient change. The measured text is only for the typed default
# identity below (model_id + binding_digest). Any other loaded identity
# inside the same composition and temperature gate gets the unscored
# code, so a later pack cannot inherit this evidence.
#
# Independent silica activities support a low bias from roughly x(CaO)=0.42.
# On the prior D066 default, Kume 2000
# Table 1 at 1873 K had mean residual −0.507 dex (range −0.641 to −0.334),
# and Omori–Sanbongi 1961 silica activities, converted from β-cristobalite to
# liquid, had mean residual −0.441 dex (range −0.722 to −0.192). On the new
# default, predicted log10(a(SiO2)) rises 0.045–0.410 dex over
# x(CaO)=[0.42, 0.56) at 1873–1933 K. In the strong interval the shift is
# non-monotonic, spanning 0.076–0.479 dex and peaking near x(CaO)=0.614.
# The cited measurements still show a low
# silica activity, but prior residual magnitudes are not carried as current
# pack scores. The basic side
# approaches Ca2SiO4. There the direct Stolyarova 1991 Table IIb
# comparison reaches about 0.6–1.0 dex low for x(CaO)≈0.56–0.67, but
# Table IIb's absolute scale conflicts with the same paper's Table Ib
# pSiO and atomic-oxygen pressures. At x(CaO)=0.62 the prior-default
# pressure-implied residual is 0.25–0.29 dex low. The pack has no Ca3SiO5 associate; the
# notice continues through x(CaO)=0.70. No target data were available at
# 1973 K, so that map is outside the notice window.
#
# Gates use the loaded parent mole vector after wt→mol conversion,
# including extension parents when they are present. Let sum(n) be that
# vector's total. Effectively binary when both of the following hold:
#   (n_CaO + n_SiO2) / sum(n) >= 0.98   (inclusive)
#   every other parent n_i / sum(n) < 0.01
# Sub-basis x(CaO) = n_CaO / (n_CaO + n_SiO2), and both amounts must be
# positive:
#   moderate: 0.42 <= x(CaO) < 0.56
#   strong:   0.56 <= x(CaO) <= 0.70
# Temperature is inclusive on [1873, 1933] K. Below 1873 K, above 1933 K,
# and at the unvalidated 1973 K map, the notice is absent. A miss on any
# gate emits nothing. The notice does not alter activities.
_CAO_SIO2_BINARY_SHARE = 0.98
_CAO_SIO2_OTHER_PARENT_MAX = 0.01
_CAO_SIO2_XCAO_MODERATE_MIN = 0.42
_CAO_SIO2_XCAO_STRONG_MIN = 0.56
_CAO_SIO2_XCAO_MAX = 0.70
_CAO_SIO2_T_MIN_K = 1873.0
_CAO_SIO2_T_MAX_K = 1933.0
# Scored on the current default D066 melt pack only. A different
# model_id or binding_digest does not inherit these residuals.
_CAO_SIO2_MEASURED_MODEL_ID = "IMCC-SF04-D066"
_CAO_SIO2_MEASURED_BINDING_DIGEST = (
    "13a185563870ec1a9c20d51bd25fb58df08c9bbbbdf7b2212b0f43ead64738cf"
)
_CAO_SIO2_NOTICE_CODE = "cao-sio2-activity-bias"
_CAO_SIO2_UNSCORED_CODE = "cao-sio2-activity-bias-unscored"
_CAO_SIO2_MODERATE_NOTICE = (
    f"{_CAO_SIO2_NOTICE_CODE}: moderate — effectively binary CaO–SiO2 melt "
    "with x(CaO) in [0.42, 0.56) on the CaO–SiO2 sub-basis at 1873–1933 K; "
    "Kume 2000 and Omori–Sanbongi 1961 show low a(SiO2); this pack raises "
    "predicted log10(a(SiO2)) by 0.045–0.410 dex over x(CaO)=[0.42, 0.56)"
)
_CAO_SIO2_STRONG_NOTICE = (
    f"{_CAO_SIO2_NOTICE_CODE}: strong — effectively binary CaO–SiO2 melt "
    "with x(CaO) in [0.56, 0.70] on the CaO–SiO2 sub-basis at 1873–1933 K, "
    "near Ca2SiO4; no Ca3SiO5 associate. Stolyarova 1991 Table IIb's "
    "0.6–1.0 dex low scale conflicts with Table Ib. That comparison's prior "
    "default residuals are not carried as v5 scores; this pack shifts predicted "
    "log10(a(SiO2)) non-monotonically by 0.076–0.479 dex, peaking near "
    "x(CaO)=0.614"
)

_MELT_EVIDENCE_MODEL_ID = "IMCC-SF04-D066"
_MELT_EVIDENCE_BINDING_DIGEST = (
    "13a185563870ec1a9c20d51bd25fb58df08c9bbbbdf7b2212b0f43ead64738cf"
)
_SILICATE_PRODUCT_ROWS = (
    ("diopside", {"CaO": 1, "MgO": 1, "SiO2": 2}, 1668.0, "−0.53"),
    ("akermanite", {"CaO": 2, "MgO": 1, "SiO2": 2}, 1727.0, "+0.11"),
    ("gehlenite", {"CaO": 2, "Al2O3": 1, "SiO2": 1}, 1823.0, "−0.51"),
    ("anorthite", {"CaO": 1, "Al2O3": 1, "SiO2": 2}, 1826.0, "+0.044"),
    ("wollastonite", {"CaO": 1, "SiO2": 1}, 1817.0, "+0.0028"),
)


def _silicate_vapour_product_notice(
    parent_oxides: Sequence[str],
    moles: np.ndarray,
    temperature_K: float,
    model_id: str,
    binding_digest: str,
) -> str | None:
    if not 1618.0 <= float(temperature_K) <= 1876.0:
        return None
    total = float(moles.sum())
    if total <= 0:
        return None
    for phase, formula, melting_K, miss in _SILICATE_PRODUCT_ROWS:
        if abs(float(temperature_K) - melting_K) > 50.0:
            continue
        indices = {
            name: parent_oxides.index(name)
            for name in formula
            if name in parent_oxides
        }
        if len(indices) != len(formula):
            continue
        formula_sum = sum(formula.values())
        basis_total = sum(float(moles[indices[name]]) for name in formula)
        if basis_total <= 0.0 or basis_total / total < 0.98:
            continue
        distance = sum(
            abs(float(moles[indices[name]]) / basis_total - amount / formula_sum)
            for name, amount in formula.items()
        )
        if distance > 0.02 + 1e-12 or any(
            float(moles[i]) / total >= 0.01
            for i, name in enumerate(parent_oxides)
            if name not in formula
        ):
            continue
        if (
            model_id != _MELT_EVIDENCE_MODEL_ID
            or binding_digest != _MELT_EVIDENCE_BINDING_DIGEST
        ):
            return (
                "silicate-vapour-product-low-near-liquidus-unscored: crystal–melt "
                "closure evidence is unscored for this pack"
            )
        melting_description = (
            f"nominal {melting_K:g} K incongruent-melting onset, diagnostic "
            "rather than congruent coexistence"
            if phase == "diopside"
            else f"nominal melting point {melting_K:g} K"
        )
        signed_miss = float(miss.replace("−", "-"))
        direction = "low" if signed_miss < 0.0 else "high"
        return (
            f"silicate-vapour-product-low-near-liquidus: {phase} SI at its "
            f"{melting_description} is {miss} using crystal data extended to "
            f"that temperature; the combined vapour product over the melt is "
            f"{direction} by {abs(signed_miss):g} log10 units, with the split between "
            "melt activities and liquid standards unresolved"
        )
    return None


def _cao_sio2_unscored_notice(model_id: str) -> str:
    return (
        f"{_CAO_SIO2_UNSCORED_CODE}: the CaO–SiO2 silica-activity bias was "
        f"measured on {_CAO_SIO2_MEASURED_MODEL_ID} "
        f"(binding digest {_CAO_SIO2_MEASURED_BINDING_DIGEST}) and has not "
        f"been re-scored for this {model_id} pack"
    )


def _cao_sio2_activity_bias_notice(
    parent_oxides: Sequence[str],
    moles: np.ndarray,
    temperature_K: float,
    model_id: str,
    binding_digest: str,
) -> str | None:
    """Return the binary CaO–SiO2 activity-bias notice, or None."""
    temperature_K = float(temperature_K)
    if temperature_K < _CAO_SIO2_T_MIN_K or temperature_K > _CAO_SIO2_T_MAX_K:
        return None
    try:
        cao_index = parent_oxides.index("CaO")
        silica_index = parent_oxides.index("SiO2")
    except ValueError:
        return None
    total = float(moles.sum())
    if total <= 0.0:
        return None
    cao = float(moles[cao_index])
    silica = float(moles[silica_index])
    if cao <= 0.0 or silica <= 0.0:
        return None
    binary = cao + silica
    if binary / total < _CAO_SIO2_BINARY_SHARE:
        return None
    for index, name in enumerate(parent_oxides):
        if name in ("CaO", "SiO2"):
            continue
        if float(moles[index]) / total >= _CAO_SIO2_OTHER_PARENT_MAX:
            return None
    x_cao = cao / binary
    if x_cao < _CAO_SIO2_XCAO_MODERATE_MIN or x_cao > _CAO_SIO2_XCAO_MAX:
        return None
    if (
        model_id != _CAO_SIO2_MEASURED_MODEL_ID
        or binding_digest != _CAO_SIO2_MEASURED_BINDING_DIGEST
    ):
        return _cao_sio2_unscored_notice(model_id)
    if x_cao < _CAO_SIO2_XCAO_STRONG_MIN:
        return _CAO_SIO2_MODERATE_NOTICE
    return _CAO_SIO2_STRONG_NOTICE


class ImccMalformedDatapackError(ImccRefusal):
    """Raised when a datapack JSON file is malformed or schema-invalid."""

    code = "imcc_malformed_datapack"


def _validate_band_schema_rows(
    data: Mapping[str, Any], *, row_label: str = "published core row"
) -> None:
    allowed_basis = {
        "source_uncertainty", "measured_construction", "independent_measurement",
        "fit_only", "none",
    }
    required = {"fit_residual", "accuracy_band", "band_basis", "independent_check"}
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ImccMalformedDatapackError("datapack rows must be a list")
    labeled_rows = [(f"{row_label} {index}", row) for index, row in enumerate(rows)]
    for label, row in labeled_rows:
        if not isinstance(row, dict):
            raise ImccMalformedDatapackError(f"{label} must be an object")
        legacy = {
            "band_kj_mol", "band_method", "uncertainty_band",
            "max_residual_J_per_mol", "max_residual_J_per_mol_by_interval",
            "max_residual_log10_K", "additional_uncertainty_bands",
            "max_abs_heldout_log10K_residual",
        } & row.keys()
        if legacy:
            raise ImccMalformedDatapackError(f"{label} contains legacy fields {sorted(legacy)}")
        if required - row.keys():
            raise ImccMalformedDatapackError(f"{label} is missing fields {sorted(required - row.keys())}")
        if row["accuracy_band"] is not None:
            raise ImccMalformedDatapackError(f"{label} complex rows must have accuracy_band null")
        basis = row["band_basis"]
        if not isinstance(basis, dict) or basis.get("kind") not in allowed_basis or not basis.get("explanation"):
            raise ImccMalformedDatapackError(f"{label} has an invalid band_basis")
        if basis["kind"] != "none":
            raise ImccMalformedDatapackError(f"{label} complex rows require band_basis none")
        check = row["independent_check"]
        check_fields = {
            "status", "equilibrium", "primary_source", "conditions", "measured_value",
            "predicted_value", "miss", "miss_to_band_ratio", "reason",
        }
        if not isinstance(check, dict) or check_fields - check.keys():
            raise ImccMalformedDatapackError(f"{label} has an incomplete independent_check")
        if check.get("status") not in {"not_found", "not_applicable", "lead", "tested"}:
            raise ImccMalformedDatapackError(f"{label} has an invalid independent_check status")

    extension = data.get("sp_extension")
    extension_rows = extension.get("rows") if isinstance(extension, dict) else None
    if isinstance(extension_rows, list):
        for index, row in enumerate(extension_rows):
            if isinstance(row, dict):
                legacy = {
                    "band_kj_mol", "band_method", "uncertainty_band",
                    "max_residual_J_per_mol", "max_residual_J_per_mol_by_interval",
                    "max_residual_log10_K", "additional_uncertainty_bands",
                    "max_abs_heldout_log10K_residual",
                } & row.keys()
                if legacy:
                    raise ImccMalformedDatapackError(
                        f"sp_extension row {index} contains legacy fields {sorted(legacy)}"
                    )


class ImccUnprovenDatapackError(ImccRefusal):
    """Raised when adapter evaluation receives an unlabelled raw datapack."""

    code = "imcc_unproven_datapack"


class ImccCompositionOutsideValidatedEnvelopeError(ImccRefusal):
    """Raised when the melt lies outside the validated Tier-A envelope."""

    code = "imcc_composition_outside_validated_envelope"


class ImccSPComponentRequiresExtensionError(ImccComponentOutsideDomainError):
    """Raised when S/P input lacks the EXT model plus explicit enable flag."""

    code = "imcc_sp_extension_required"


@dataclass(frozen=True)
class ExcessGibbsTerm:
    """One disclosed analytical-parent interaction from a candidate pack."""

    basis: str
    components: tuple[str, str]
    W_H_J_mol: float
    W_S_J_mol_K: float
    provenance: str | Mapping[str, Any]
    calibration_role: str
    valid_composition: Mapping[str, tuple[float, float]]
    T_range_K: tuple[float, float]
    fit_status: str

    def W_J_mol(self, temperature_K: float) -> float:
        return self.W_H_J_mol - temperature_K * self.W_S_J_mol_K

    @property
    def active(self) -> bool:
        return self.W_H_J_mol != 0.0 or self.W_S_J_mol_K != 0.0


@dataclass(frozen=True)
class ImccLoadedDatapack:
    """Adapter-level loaded datapack: kernel datapack plus provenance metadata."""

    kernel_datapack: ImccDatapack
    version: str
    parent_oxides: Sequence[str]
    domain_basis: Sequence[str]
    extension_parents: Sequence[str] = ()
    extension_species: Sequence[str] = ()
    inactive_rows: tuple[tuple[str, str], ...] = ()
    excess_gibbs_terms: tuple[ExcessGibbsTerm, ...] = ()

    @property
    def model_id(self) -> str:
        return self.kernel_datapack.model_id

    @property
    def paper_domains(self) -> Sequence[tuple[float, float] | None]:
        return self.kernel_datapack.paper_domains

    @property
    def binding_digest(self) -> str | None:
        """Canonical identity of the complete manifest used to load this pack."""
        return self.kernel_datapack.binding_digest


@dataclass(frozen=True)
class ImccAdapterLabels:
    """Caller-facing identity, coverage, edge metric, and flags."""

    identity: Mapping[str, str]
    coverage: Mapping[str, str]
    envelope_status: str
    flags: tuple[str, ...] = ()
    notices: tuple[str, ...] = ()
    acid_sink_ratio: float | None = None
    inactive_rows: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EngineBindingIdentity:
    """Content identity for the melt association constants and both gas tables."""

    digest: str
    melt_binding_digest: str
    condensate_table_digest: str
    gas_table_digest: str


def _parse_excess_gibbs_terms(
    section: Any,
    parent_oxides: Sequence[str],
) -> tuple[ExcessGibbsTerm, ...]:
    if not isinstance(section, dict):
        raise ImccMalformedDatapackError("excess_gibbs must be an object")
    raw_terms = section.get("terms")
    if not isinstance(raw_terms, list):
        raise ImccMalformedDatapackError("excess_gibbs.terms must be an array")

    terms: list[ExcessGibbsTerm] = []
    for index, raw in enumerate(raw_terms):
        label = f"excess_gibbs term {index}"
        if not isinstance(raw, dict):
            raise ImccMalformedDatapackError(f"{label} must be an object")
        basis = raw.get("basis")
        if basis == "species_pair":
            raise ImccMalformedDatapackError(
                f"{label} basis 'species_pair' is reserved for a later format extension"
            )
        if basis != "analytical_parent":
            raise ImccMalformedDatapackError(
                f"{label} basis must be 'analytical_parent'"
            )
        components = raw.get("components")
        if (
            not isinstance(components, list)
            or len(components) != 2
            or not all(isinstance(name, str) for name in components)
            or components[0] == components[1]
            or any(name not in parent_oxides for name in components)
        ):
            raise ImccMalformedDatapackError(
                f"{label} components must name two distinct parent oxides"
            )

        coefficients: list[float] = []
        for field_name in ("W_H_J_mol", "W_S_J_mol_K"):
            value = raw.get(field_name)
            if not is_declared_real_scalar(value):
                raise ImccMalformedDatapackError(
                    f"{label} {field_name} must be a finite number"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ImccMalformedDatapackError(
                    f"{label} {field_name} must be a finite number"
                )
            coefficients.append(number)

        provenance = raw.get("provenance")
        if isinstance(provenance, str) and provenance.strip():
            parsed_provenance: str | Mapping[str, Any] = provenance
        elif isinstance(provenance, dict) and provenance:
            parsed_provenance = MappingProxyType(dict(provenance))
        else:
            raise ImccMalformedDatapackError(
                f"{label} provenance must be a non-empty string or object"
            )

        calibration_role = raw.get("calibration_role")
        fit_status = raw.get("fit_status")
        if not isinstance(calibration_role, str) or not calibration_role.strip():
            raise ImccMalformedDatapackError(
                f"{label} calibration_role must be a non-empty string"
            )
        if not isinstance(fit_status, str) or not fit_status.strip():
            raise ImccMalformedDatapackError(
                f"{label} fit_status must be a non-empty string"
            )

        raw_composition = raw.get("valid_composition")
        if not isinstance(raw_composition, dict) or not raw_composition:
            raise ImccMalformedDatapackError(
                f"{label} valid_composition must map parent oxides to ranges"
            )
        composition_ranges: dict[str, tuple[float, float]] = {}
        for name, bounds in raw_composition.items():
            if name not in parent_oxides or not isinstance(bounds, list) or len(bounds) != 2:
                raise ImccMalformedDatapackError(
                    f"{label} valid_composition ranges must name parent oxides"
                )
            if not all(is_declared_real_scalar(bound) for bound in bounds):
                raise ImccMalformedDatapackError(
                    f"{label} valid_composition bounds must be finite numbers"
                )
            low, high = (float(bound) for bound in bounds)
            if not (math.isfinite(low) and math.isfinite(high)) or not (
                0.0 <= low <= high <= 1.0
            ):
                raise ImccMalformedDatapackError(
                    f"{label} valid_composition bounds must lie in [0, 1]"
                )
            composition_ranges[name] = (low, high)

        raw_temperature_range = raw.get("T_range_K")
        if (
            not isinstance(raw_temperature_range, list)
            or len(raw_temperature_range) != 2
            or not all(is_declared_real_scalar(bound) for bound in raw_temperature_range)
        ):
            raise ImccMalformedDatapackError(
                f"{label} T_range_K must be a finite [low, high] pair"
            )
        t_low, t_high = (float(bound) for bound in raw_temperature_range)
        if not (math.isfinite(t_low) and math.isfinite(t_high)) or not (
            0.0 < t_low <= t_high
        ):
            raise ImccMalformedDatapackError(
                f"{label} T_range_K must be a positive finite [low, high] pair"
            )

        terms.append(
            ExcessGibbsTerm(
                basis=basis,
                components=(components[0], components[1]),
                W_H_J_mol=coefficients[0],
                W_S_J_mol_K=coefficients[1],
                provenance=parsed_provenance,
                calibration_role=calibration_role,
                valid_composition=MappingProxyType(composition_ranges),
                T_range_K=(t_low, t_high),
                fit_status=fit_status,
            )
        )
    return tuple(terms)


def _analytical_parent_excess_state(
    parent_oxides: Sequence[str],
    parent_mol: Sequence[float],
    temperature_K: float,
    terms: Sequence[ExcessGibbsTerm],
) -> tuple[float, float, np.ndarray]:
    """Return extensive Gex, Hex, and every analytical-parent μex."""
    amounts = np.asarray(parent_mol, dtype=float)
    total = float(np.sum(amounts))
    fractions = amounts / total
    parent_index = {name: index for index, name in enumerate(parent_oxides)}
    g_terms: list[float] = []
    h_terms: list[float] = []
    mu_excess = np.zeros(len(parent_oxides), dtype=float)
    for term in terms:
        first, second = (parent_index[name] for name in term.components)
        x_first = float(fractions[first])
        x_second = float(fractions[second])
        product = x_first * x_second
        interaction_w = term.W_J_mol(temperature_K)
        g_terms.append(interaction_w * product)
        h_terms.append(term.W_H_J_mol * product)
        # Differentiate N0*W*X_first*X_second with respect to every
        # analytical inventory. The -W*X_first*X_second normalization term
        # belongs to every parent, including spectators.
        mu_excess -= interaction_w * product
        mu_excess[first] += interaction_w * x_second
        mu_excess[second] += interaction_w * x_first

    return (
        total * math.fsum(g_terms),
        total * math.fsum(h_terms),
        mu_excess,
    )


def list_excess_gibbs_interactions(
    pack: ImccLoadedDatapack | None = None,
) -> tuple[ExcessGibbsTerm, ...]:
    """List the interaction terms disclosed by a loaded candidate pack."""
    loaded = load_datapack() if pack is None else pack
    if not isinstance(loaded, ImccLoadedDatapack):
        raise TypeError("pack must be an ImccLoadedDatapack")
    return loaded.excess_gibbs_terms


def engine_binding_identity(
    melt_pack: ImccLoadedDatapack | ImccDatapack,
    gas_pack: "ImccGasDatapack",
) -> EngineBindingIdentity:
    """Hash the three content digests that define one thermodynamic engine."""
    melt_digest = getattr(melt_pack, "binding_digest", None)
    condensate_digest = getattr(gas_pack, "condensate_table_digest", None)
    gas_digest = getattr(gas_pack, "gas_table_digest", None)
    if not all(
        isinstance(value, str) and value
        for value in (melt_digest, condensate_digest, gas_digest)
    ):
        raise ImccUnprovenDatapackError(
            "engine binding identity requires a labelled melt pack and both "
            "gas table content digests"
        )

    components = {
        "melt_binding_digest": melt_digest,
        "condensate_table_digest": condensate_digest,
        "gas_table_digest": gas_digest,
    }
    return EngineBindingIdentity(
        digest=_published_datapack_manifest_hash(components),
        **components,
    )


def _as_fraction(value: Any) -> Fraction:
    """Parse a JSON numeric value as an exact rational."""
    if is_declared_real_scalar(value, allow_numeric_str=True):
        return Fraction(str(value))
    raise TypeError(f"cannot parse {value!r} as a rational")


def _published_core_manifest_payload(
    data: Mapping[str, Any],
    *,
    model_id: str,
    version: str,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in data.items()
        if key not in {"model_id", "sp_extension"}
    }
    if model_id == _SP_EXTENSION_MODEL_ID:
        version = version.partition("-ext-sp-")[0]
    payload["model_id"] = _PUBLISHED_MODEL_ID
    payload["imcc_sf04_datapack_version"] = version
    return payload


def _validate_published_core(
    data: Mapping[str, Any],
    *,
    model_id: str,
    version: str,
) -> tuple[list[dict[str, Any]], str]:
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != _PUBLISHED_CORE_ROWS:
        raise ImccMalformedDatapackError(
            "datapack must contain exactly 38 published-core rows, got "
            f"{len(rows) if isinstance(rows, list) else None}"
        )
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ImccMalformedDatapackError(
                f"published core row {idx} is not an object"
            )
    _validate_active_row_metadata(rows, label="published core")
    try:
        content_hash = _published_datapack_manifest_hash(
            _published_core_manifest_payload(
                data,
                model_id=model_id,
                version=version,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ImccMalformedDatapackError(
            "published IMCC datapack cannot be canonically serialized"
        ) from exc
    if content_hash != _PUBLISHED_DATAPACK_SHA256:
        raise ImccMalformedDatapackError(
            "published IMCC datapack canonical hash mismatch: "
            f"expected {_PUBLISHED_DATAPACK_SHA256}, got {content_hash}"
        )
    return rows, content_hash


def _validate_active_row_metadata(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> None:
    for idx, row in enumerate(rows):
        active = row.get("active", True)
        if "active" in row and not isinstance(active, bool):
            raise ImccMalformedDatapackError(
                f"{label} row {idx} active must be a boolean"
            )
        inactive_reason = row.get("inactive_reason")
        if active is False:
            if not isinstance(inactive_reason, str) or not inactive_reason.strip():
                raise ImccMalformedDatapackError(
                    f"{label} row {idx} inactive_reason must be a non-empty string when active is false"
                )
        elif "inactive_reason" in row:
            raise ImccMalformedDatapackError(
                f"{label} row {idx} inactive_reason is only allowed when active is false"
            )


def _validate_d066_core(
    data: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != len(_D066_CORE_ROWS):
        raise ImccMalformedDatapackError(
            f"D066 core must contain exactly {len(_D066_CORE_ROWS)} rows"
        )
    if any(not isinstance(row, dict) for row in rows):
        raise ImccMalformedDatapackError("D066 core rows must be objects")
    _validate_active_row_metadata(rows, label="D066 core")

    names = tuple(row.get("complex") for row in rows)
    if names != _D066_CORE_ROWS:
        raise ImccMalformedDatapackError(
            "D066 core row set or order does not match the frozen SF04 row set"
        )
    for idx, row in enumerate(rows, start=1):
        name = row["complex"]
        source_row = row.get("row")
        if type(source_row) is not int or source_row != idx:
            raise ImccMalformedDatapackError(
                f"D066 row {name!r} must retain source row number {idx}"
            )
        if row.get("coverage") != _D066_ROW_COVERAGE[name]:
            raise ImccMalformedDatapackError(
                f"D066 row {name!r} has invalid coverage label"
            )
        lineage = row.get("d066_source_row")
        if (
            not isinstance(lineage, dict)
            or set(lineage) != {"pack", "row"}
            or lineage.get("pack") != "imcc-sf04-v1.0.2.json"
            or type(lineage.get("row")) is not int
            or lineage["row"] != idx
        ):
            raise ImccMalformedDatapackError(
                f"D066 row {name!r} must identify its v1.0.2 source row"
            )
        flags = row.get("flags", [])
        if not isinstance(flags, list) or not all(
            isinstance(flag, str) for flag in flags
        ):
            raise ImccMalformedDatapackError(
                f"D066 row {name!r} flags must be a list of strings"
            )
        if name in _D066_REFERENCE_UNKNOWN_ROWS:
            if row.get("active", True) is not True or "reference_unknown" not in flags:
                raise ImccMalformedDatapackError(
                    f"D066 row {name!r} must remain active and carry reference_unknown"
                )
        elif name not in _D066_E25_ROWS and row.get("active", True) is not True:
            raise ImccMalformedDatapackError(
                f"D066 row {name!r} cannot be inactive"
            )

    e25 = next(row for row in rows if row["complex"] in _D066_E25_ROWS)
    if e25.get("active", True) is False:
        if e25.get("inactive_reason") != _D066_E25_INACTIVE_REASON:
            raise ImccMalformedDatapackError(
                "D066 inactive KCaAlSi2O7 must identify the E25 ruling"
            )
        if "kcaalsi2o7_out_of_liquid_domain" in e25.get("flags", []):
            raise ImccMalformedDatapackError(
                "inactive D066 KCaAlSi2O7 cannot carry the sensitivity flag"
            )
    elif "kcaalsi2o7_out_of_liquid_domain" not in e25.get("flags", []):
        raise ImccMalformedDatapackError(
            "active D066 KCaAlSi2O7 must carry the sensitivity flag"
        )

    try:
        content_hash = _published_datapack_manifest_hash(data)
    except (TypeError, ValueError) as exc:
        raise ImccMalformedDatapackError(
            "D066 datapack cannot be canonically serialized"
        ) from exc
    return rows, content_hash


def _build_nu_vector(
    parent_oxides: Sequence[str],
    nu: Mapping[str, Any],
    *,
    row_label: str,
) -> list[float]:
    """Return a nu column aligned to parent_oxides, preserving exact rationals."""
    unknown_nu = sorted(set(nu) - set(parent_oxides))
    if unknown_nu:
        raise ImccMalformedDatapackError(
            f"{row_label} nu has unknown key {unknown_nu[0]!r}"
        )
    return [float(_as_fraction(nu.get(name, 0))) for name in parent_oxides]


def _wt_to_mol(vector: np.ndarray, parent_oxides: Sequence[str]) -> np.ndarray:
    """Convert a mass-fraction vector (g) to a mole vector (mol)."""
    molar_masses = np.array(
        [_OXIDE_MOLAR_MASS_G_MOL[name] for name in parent_oxides],
        dtype=float,
    )
    return vector / molar_masses


def load_datapack(path: str | Path | None = None) -> ImccLoadedDatapack:
    """Load an IMCC-SF04 datapack JSON into the kernel datapack object.

    Published packs validate against the frozen published-core hash. D066 packs
    validate their own frozen row set and carry a non-published identity.
    ``sp_extension`` may accompany either identity. With no path, the packaged
    D066 primary resource is loaded.
    """
    source = (
        resources.files("openimcc")
        / "data"
        / "packs"
        / "imcc-sf04-d066-v5.json"
        if path is None
        else Path(path)
    )
    try:
        with source.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ImccMalformedDatapackError(
            f"datapack JSON at {source} is not valid JSON"
        ) from exc
    except FileNotFoundError as exc:
        raise ImccMalformedDatapackError(
            f"datapack file not found: {source}"
        ) from exc

    if not isinstance(data, dict):
        raise ImccMalformedDatapackError("datapack JSON root must be an object")

    version = data.get("imcc_sf04_datapack_version")
    if not isinstance(version, str) or not version:
        raise ImccMalformedDatapackError(
            "datapack missing 'imcc_sf04_datapack_version' string"
        )

    base_parents = data.get("parents")
    if not isinstance(base_parents, list) or base_parents != list(_EXPECTED_PARENT_OXIDES):
        raise ImccMalformedDatapackError(
            f"datapack parents {base_parents!r} do not match expected "
            f"{_EXPECTED_PARENT_OXIDES!r}"
        )
    base_parents = tuple(base_parents)

    model_id = data.get("model_id", "IMCC-SF04")
    has_excess_gibbs = "excess_gibbs" in data
    if has_excess_gibbs and model_id != _EXCESS_GIBBS_MODEL_ID:
        raise ImccMalformedDatapackError(
            "excess_gibbs requires the new candidate identity "
            f"{_EXCESS_GIBBS_MODEL_ID!r}"
        )
    if model_id == _EXCESS_GIBBS_MODEL_ID and not has_excess_gibbs:
        raise ImccMalformedDatapackError(
            "the excess-Gibbs candidate identity requires an excess_gibbs section"
        )
    sp_extension = data.get("sp_extension")
    extension_parents: tuple[str, ...] = ()
    extension_rows: list[dict[str, Any]] = []
    if sp_extension is None:
        if model_id not in {"IMCC-SF04", _D066_MODEL_ID, _EXCESS_GIBBS_MODEL_ID}:
            raise ImccMalformedDatapackError(
                f"model_id {model_id!r} requires a recognized extension section"
            )
    else:
        if model_id not in {
            _SP_EXTENSION_MODEL_ID,
            _D066_MODEL_ID,
            _EXCESS_GIBBS_MODEL_ID,
        }:
            raise ImccMalformedDatapackError(
                "sp_extension requires model_id='IMCC-SF04-EXT' or "
                "model_id='IMCC-SF04-D066'"
            )

    if model_id in {_D066_MODEL_ID, _EXCESS_GIBBS_MODEL_ID}:
        rows, _d066_manifest_sha256 = _validate_d066_core(data)
        published_manifest_sha256 = None
    else:
        rows, published_manifest_sha256 = _validate_published_core(
            data,
            model_id=model_id,
            version=version,
        )
    # Field-level band checks run only after the published-identity gate
    # above: an unpublished or modified pack must refuse on its canonical
    # hash and never reach field validation.
    _validate_band_schema_rows(data)

    if sp_extension is not None:
        if not isinstance(sp_extension, dict):
            raise ImccMalformedDatapackError("sp_extension must be an object")
        if sp_extension.get("enable_flag") != _SP_EXTENSION_FLAG:
            raise ImccMalformedDatapackError(
                "sp_extension enable_flag must be 'enable_sp_extension'"
            )
        if sp_extension.get("tier") != _SP_EXTENSION_TIER:
            raise ImccMalformedDatapackError("sp_extension tier must be 'EXT-SP'")
        if sp_extension.get("certification") != "denied":
            raise ImccMalformedDatapackError(
                "sp_extension certification must be explicitly denied"
            )
        raw_extension_parents = sp_extension.get("parents")
        if raw_extension_parents != list(_SP_EXTENSION_PARENTS):
            raise ImccMalformedDatapackError(
                f"sp_extension parents must be {list(_SP_EXTENSION_PARENTS)!r}"
            )
        raw_extension_rows = sp_extension.get("rows")
        if not isinstance(raw_extension_rows, list) or not raw_extension_rows:
            raise ImccMalformedDatapackError(
                "sp_extension rows must be a non-empty list"
            )
        for idx, row in enumerate(raw_extension_rows):
            if not isinstance(row, dict):
                raise ImccMalformedDatapackError(
                    f"sp_extension row {idx} is not an object"
                )
            if row.get("provenance_class") != _SP_EXTENSION_PROVENANCE_CLASS:
                raise ImccMalformedDatapackError(
                    f"sp_extension row {idx} provenance_class must be "
                    f"{_SP_EXTENSION_PROVENANCE_CLASS!r}"
                )
            if row.get("tier") != _SP_EXTENSION_TIER:
                raise ImccMalformedDatapackError(
                    f"sp_extension row {idx} tier must be 'EXT-SP'"
                )
            if row.get("certification") != "denied":
                raise ImccMalformedDatapackError(
                    f"sp_extension row {idx} certification must be denied"
                )
            provenance = row.get("provenance")
            if not isinstance(provenance, dict):
                raise ImccMalformedDatapackError(
                    f"sp_extension row {idx} missing provenance object"
                )
            if not all(
                isinstance(provenance.get(field), str) and provenance[field]
                for field in ("source", "table_id")
            ):
                raise ImccMalformedDatapackError(
                    f"sp_extension row {idx} provenance requires source and table_id"
                )
            extension_nu = row.get("nu")
            if isinstance(extension_nu, dict):
                unknown_nu = set(extension_nu) - set(
                    base_parents + _SP_EXTENSION_PARENTS
                )
                if unknown_nu:
                    raise ImccMalformedDatapackError(
                        f"sp_extension row {idx} nu has unknown parents "
                        f"{sorted(unknown_nu)}"
                    )
                if not any(
                    _as_fraction(extension_nu.get(parent, 0)) > 0
                    for parent in _SP_EXTENSION_PARENTS
                ):
                    raise ImccMalformedDatapackError(
                        f"sp_extension row {idx} must consume S or P2O5"
                    )
            extension_rows.append(row)
        extension_parents = _SP_EXTENSION_PARENTS
        _validate_band_schema_rows(
            {"rows": extension_rows}, row_label="sp_extension row"
        )

    parents = base_parents + extension_parents
    excess_gibbs_terms = (
        _parse_excess_gibbs_terms(data["excess_gibbs"], parents)
        if has_excess_gibbs
        else ()
    )
    all_rows = rows + extension_rows

    reactions: list[str] = []
    nu_cols: list[list[float]] = []
    A: list[float] = []
    B: list[float] = []
    domains: list[tuple[float, float]] = []
    paper_domains: list[tuple[float, float] | None] = []
    domain_basis: list[str] = []

    for idx, row in enumerate(all_rows):
        if not isinstance(row, dict):
            raise ImccMalformedDatapackError(f"row {idx} is not an object")

        complex_name = row.get("complex")
        if not isinstance(complex_name, str):
            raise ImccMalformedDatapackError(
                f"row {idx} missing 'complex' string"
            )
        reactions.append(complex_name)

        nu = row.get("nu")
        if not isinstance(nu, dict):
            raise ImccMalformedDatapackError(
                f"row {idx} missing 'nu' object"
            )
        row_label = (
            f"published core row {idx}"
            if idx < len(rows)
            else f"sp_extension row {idx - len(rows)}"
        )
        nu_cols.append(_build_nu_vector(parents, nu, row_label=row_label))

        A_val = row.get("A")
        B_val = row.get("B")
        if (
            not is_declared_real_scalar(A_val)
            or not isinstance(A_val, (int, float))
            or not is_declared_real_scalar(B_val)
            or not isinstance(B_val, (int, float))
        ):
            raise ImccMalformedDatapackError(
                f"row {idx} A/B must be numeric"
            )
        A_float = float(A_val)
        B_float = float(B_val)
        if idx >= len(rows) and (
            not np.isfinite(A_float) or not np.isfinite(B_float)
        ):
            raise ImccMalformedDatapackError(
                f"{row_label} A and B must be finite"
            )
        A.append(A_float)
        B.append(B_float)

        t_domain = row.get("T_domain_K")
        if not isinstance(t_domain, list) or len(t_domain) != 2:
            raise ImccMalformedDatapackError(
                f"row {idx} missing valid T_domain_K [low, high]"
            )
        if not all(
            is_declared_real_scalar(value, allow_numeric_str=True)
            for value in t_domain
        ):
            raise ImccMalformedDatapackError(
                f"row {idx} T_domain_K values must be numeric"
            )
        domain_values = (float(t_domain[0]), float(t_domain[1]))
        if idx >= len(rows) and not all(
            np.isfinite(value) for value in domain_values
        ):
            raise ImccMalformedDatapackError(
                f"{row_label} endpoints must be finite Kelvin values, "
                f"got ({t_domain[0]!r}, {t_domain[1]!r})"
            )
        domains.append(domain_values)

        paper_domain = row.get("T_domain_paper_demonstrated_K")
        if paper_domain is None:
            paper_domains.append(None)
        else:
            if not isinstance(paper_domain, list) or len(paper_domain) != 2:
                raise ImccMalformedDatapackError(
                    f"row {idx} T_domain_paper_demonstrated_K must be a pair"
                )
            if not all(
                is_declared_real_scalar(value, allow_numeric_str=True)
                for value in paper_domain
            ):
                raise ImccMalformedDatapackError(
                    f"row {idx} T_domain_paper_demonstrated_K values must be numeric"
                )
            paper_domain_values = (float(paper_domain[0]), float(paper_domain[1]))
            if not all(np.isfinite(value) for value in paper_domain_values):
                raise ImccMalformedDatapackError(
                    f"row {idx} T_domain_paper_demonstrated_K values must be finite"
                )
            paper_domains.append(paper_domain_values)

        basis = row.get("T_domain_basis")
        if not isinstance(basis, str):
            raise ImccMalformedDatapackError(
                f"row {idx} missing T_domain_basis string"
            )
        domain_basis.append(basis)

    if len(set(reactions)) != len(reactions):
        raise ImccMalformedDatapackError("datapack complex names must be unique")

    inactive_rows = tuple(
        (str(row["complex"]), row["inactive_reason"])
        for row in rows
        if row.get("active", True) is False
    )

    binding_manifest = dict(data)
    binding_manifest["model_id"] = model_id
    binding_manifest["imcc_sf04_datapack_version"] = version
    try:
        binding_digest = _published_datapack_manifest_hash(binding_manifest)
    except (TypeError, ValueError) as exc:
        raise ImccMalformedDatapackError(
            "datapack cannot be canonically serialized for binding identity"
        ) from exc

    # nu_cols is (n_complexes, n_parents); transpose to kernel shape.
    nu_array = np.array(nu_cols, dtype=float).T

    kernel_datapack = ImccDatapack(
        reactions=reactions,
        nu=nu_array,
        A=np.array(A, dtype=float),
        B=np.array(B, dtype=float),
        domains=domains,
        paper_domains=paper_domains,
        version=version,
        parent_oxides=parents,
    )
    extension_species = tuple(str(row["complex"]) for row in extension_rows)
    extension_names = set(extension_parents) | set(extension_species)
    if model_id in {_D066_MODEL_ID, _EXCESS_GIBBS_MODEL_ID}:
        coverage = {
            name: (
                _SP_EXTENSION_TIER if name in extension_names else
                "D-carried-published"
            )
            for name in parents
        }
        coverage.update(
            {
                str(row["complex"]): str(row["coverage"])
                for row in rows
            }
        )
        coverage.update(
            {
                name: _SP_EXTENSION_TIER
                for name in extension_species
            }
        )
    else:
        coverage = {
            name: (_SP_EXTENSION_TIER if name in extension_names else "A-published-imcc")
            for name in (*parents, *reactions)
        }
    kernel_datapack = _label_loaded_datapack(
        kernel_datapack,
        model_id=model_id,
        coverage=coverage,
        published_manifest_sha256=published_manifest_sha256,
        binding_digest=binding_digest,
    )

    active_indices = list(range(len(all_rows)))
    if inactive_rows:
        active_indices = [
            idx
            for idx, row in enumerate(all_rows)
            if row.get("active", True) is True
        ]
        labelled_identity = kernel_datapack._identity
        kernel_datapack = replace(
            kernel_datapack,
            reactions=tuple(kernel_datapack.reactions[idx] for idx in active_indices),
            nu=kernel_datapack.nu[:, active_indices],
            A=kernel_datapack.A[active_indices],
            B=kernel_datapack.B[active_indices],
            domains=tuple(kernel_datapack.domains[idx] for idx in active_indices),
            paper_domains=tuple(
                kernel_datapack.paper_domains[idx] for idx in active_indices
            ),
        )
        for array_name in ("nu", "A", "B"):
            getattr(kernel_datapack, array_name).setflags(write=False)
        # The digest identifies the complete source manifest; the loaded
        # coverage map must describe the same active species as the arrays.
        active_species = set(kernel_datapack.parent_oxides) | set(
            kernel_datapack.reactions
        )
        active_coverage = {
            name: label
            for name, label in labelled_identity.coverage.items()
            if name in active_species
        }
        kernel_datapack = _label_loaded_datapack(
            kernel_datapack,
            model_id=model_id,
            coverage=active_coverage,
            published_manifest_sha256=published_manifest_sha256,
            binding_digest=binding_digest,
        )

    return ImccLoadedDatapack(
        kernel_datapack=kernel_datapack,
        version=version,
        parent_oxides=parents,
        domain_basis=tuple(domain_basis[idx] for idx in active_indices),
        extension_parents=extension_parents,
        extension_species=extension_species,
        inactive_rows=inactive_rows,
        excess_gibbs_terms=excess_gibbs_terms,
    )


def _amount_as_float(value: Any, label: str) -> float:
    """Convert one supplied amount, refusing a non-numeric value as invalid input.

    A bare float() here would raise an untyped ValueError or TypeError for
    "abc" or None, so a caller would get a crash instead of a refusal that
    names the bad entry.
    """
    if not is_declared_real_scalar(value, allow_numeric_str=True):
        raise ImccCompositionIncompleteError(
            f"{label} is not a number: {value!r}"
        )
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ImccCompositionIncompleteError(
            f"{label} is not a number: {value!r}"
        ) from None
    except OverflowError:
        # An int beyond float range (e.g. 10**400). The value is not echoed:
        # its repr can run to hundreds of digits.
        raise ImccCompositionIncompleteError(
            f"{label} is too large to represent as a float"
        ) from None


def _sp_extension_refusal(component_names: Sequence[str]) -> ImccSPComponentRequiresExtensionError:
    names = ", ".join(sorted(component_names)) or "S/P EXT component(s)"
    return ImccSPComponentRequiresExtensionError(
        f"{names} belong to the S/P EXT component class; unlock only with "
        "an extension pack with model_id='IMCC-SF04-EXT' or "
        "model_id='IMCC-SF04-D066' and enable_sp_extension=True. Plain core "
        "packs intentionally exclude S/P speciation."
    )


def evaluate(
    composition: Mapping[str, float] | Sequence[float],
    T_K: float,
    pack: ImccLoadedDatapack | ImccDatapack | None = None,
    *,
    basis: float | None = None,
    basis_type: str = "mol",
    extra_mol: Mapping[str, float] | None = None,
    enable_sp_extension: bool = False,
    allow_extrapolation: bool = False,
    allow_out_of_envelope: bool = False,
    tol: float = 1.0e-12,
    max_iter: int = 100,
    y_init: Sequence[float] | None = None,
) -> ImccResult:
    """Caller-facing IMCC-SF04 evaluation.

    Accepts a composition on either a mol or wt basis with a declared basis.
    Converts wt-to-mol before any cache boundary, validates the FeO-equivalent
    contract, and delegates to the kernel.

    Parameters
    ----------
    composition:
        Dict keyed by parent-oxide name, or a numeric vector aligned with the
        pack's parent oxides.
    T_K:
        Temperature in Kelvin.
    pack:
        Loaded datapack, or a raw kernel datapack labelled by
        ``label_research_datapack()``. If omitted, the packaged
        D066 primary datapack is used. Unlabelled raw packs are refused.
    basis:
        Declared normalization basis in the same units as ``basis_type``. If
        ``None``, the composition sum is used.
    basis_type:
        ``"mol"`` or ``"wt"``.
    extra_mol:
        Additional components in mol (e.g. Tier-B/C screens). Positive Fe2O3
        raises ``ImccFerricInputUnsupportedError``; other positives outside the
        parent basis raise ``ImccComponentOutsideDomainError``.
    enable_sp_extension:
        Explicitly enable S and P2O5 parents in an extension pack.
        The flag alone never widens a plain ``IMCC-SF04`` pack.
    allow_extrapolation:
        If ``True``, evaluate outside declared T domains and mark the result
        as extrapolated.
    allow_out_of_envelope:
        If ``True``, evaluate outside the validated Tier-A composition
        envelope and mark the result ``outside_validated``.
    tol:
        Infinity-norm residual convergence tolerance passed to the kernel.
    max_iter:
        Newton iteration ceiling passed to the kernel.
    y_init:
        Optional finite vector of natural-log free-parent fractions in the
        order of positive-mole parents in the input and datapack. This is the
        kernel's `y` vector; pass the previous result's active entries after
        taking their natural logs to continue a time-stepped solve.

    Returns
    -------
    ImccResult
        Kernel result with the adapter's label block installed.
    """
    if pack is None:
        pack = load_datapack()

    # Resolve the kernel datapack and metadata.
    if isinstance(pack, ImccLoadedDatapack):
        kernel_pack = pack.kernel_datapack
        parent_oxides = pack.parent_oxides
        extension_parents = tuple(pack.extension_parents)
        inactive_rows = pack.inactive_rows
        excess_gibbs_terms = pack.excess_gibbs_terms
    else:
        kernel_pack = pack
        parent_oxides = pack.parent_oxides
        extension_parents = tuple(
            name for name in _SP_EXTENSION_PARENTS if name in parent_oxides
        )
        inactive_rows = ()
        excess_gibbs_terms = ()

    binding_digest = kernel_pack.binding_digest
    if binding_digest is None:
        raise ImccUnprovenDatapackError(
            "raw ImccDatapack has no proven identity; load a frozen JSON pack "
            "with load_datapack() or apply explicit non-published provenance "
            "with label_research_datapack()"
        )

    pack_version = kernel_pack.version
    model_id = kernel_pack.model_id

    if isinstance(composition, Mapping):
        for name, raw in composition.items():
            if not np.isfinite(_amount_as_float(raw, f"composition value for {name}")):
                raise ImccCompositionIncompleteError(
                    "composition contains non-finite values"
                )
    if extra_mol:
        for species, mol in extra_mol.items():
            value = _amount_as_float(mol, f"extra component {species}")
            if not np.isfinite(value):
                raise ImccCompositionIncompleteError(
                    f"extra component {species} has non-finite moles {value}"
                )
    supplied_sp_names: set[str] = set()
    if isinstance(composition, Mapping):
        supplied_sp_names.update(
            name for name in _SP_EXTENSION_PARENTS
            if float(composition.get(name, 0.0)) != 0.0
        )
    elif len(composition) > len(_EXPECTED_PARENT_OXIDES):
        supplied_sp_names.update(_SP_EXTENSION_PARENTS)
    if extra_mol:
        supplied_sp_names.update(
            name for name in _SP_EXTENSION_PARENTS
            if float(extra_mol.get(name, 0.0)) != 0.0
        )

    if extension_parents:
        if not enable_sp_extension:
            raise _sp_extension_refusal(extension_parents)
    elif enable_sp_extension or supplied_sp_names:
        raise _sp_extension_refusal(supplied_sp_names)

    if basis_type not in ("mol", "wt"):
        raise ImccCompositionIncompleteError(
            f"basis_type must be 'mol' or 'wt', got {basis_type!r}"
        )

    # Normalize composition to a numeric vector aligned with parent_oxides.
    if isinstance(composition, Mapping):
        # Ferric refusal at the adapter boundary (FeO-equivalent contract).
        if "Fe2O3" in composition and composition["Fe2O3"] != 0:
            raise ImccFerricInputUnsupportedError(
                "Fe2O3 input is unsupported; convert to FeO under the "
                "caller's redox model before calling IMCC-SF04"
            )
        unknown = set(composition.keys()) - set(parent_oxides)
        if unknown:
            raise ImccComponentOutsideDomainError(
                f"component(s) outside IMCC-SF04 domain: {sorted(unknown)}"
            )
        vector = np.array(
            [float(composition.get(name, 0.0)) for name in parent_oxides],
            dtype=float,
        )
    else:
        raw_vector = np.asarray(composition, dtype=object)
        if raw_vector.ndim != 1:
            raise ImccCompositionIncompleteError(
                "composition vector must be 1-D"
            )
        if raw_vector.shape[0] != len(parent_oxides):
            if raw_vector.shape[0] > len(parent_oxides):
                raise ImccComponentOutsideDomainError(
                    f"composition vector length {raw_vector.shape[0]} exceeds "
                    f"the {len(parent_oxides)}-oxide basis"
                )
            raise ImccCompositionIncompleteError(
                f"composition vector length {raw_vector.shape[0]} does not match "
                f"the {len(parent_oxides)}-oxide basis"
            )
        vector = np.asarray(
            [
                _amount_as_float(value, f"composition value at index {index}")
                for index, value in enumerate(raw_vector)
            ],
            dtype=float,
        )

    if not np.all(np.isfinite(vector)):
        raise ImccCompositionIncompleteError(
            "composition contains non-finite values"
        )
    if np.any(vector < 0.0):
        raise ImccCompositionIncompleteError(
            "composition contains negative values"
        )

    # Finite entries can overflow this preliminary sum; <= 0 below does not
    # reject +inf. Mole-basis overflow is rejected later by
    # kernel.solve_imcc_sf04's finite parent-mole-total check; wt input replaces
    # this total with the converted mole sum below.
    with np.errstate(over="ignore"):
        total = float(vector.sum())
    if total <= 0.0:
        raise ImccCompositionIncompleteError("composition total is zero")

    if basis is None:
        basis = total
    else:
        basis = _amount_as_float(basis, "declared basis")
        # nan and inf both pass a bare `<= 0.0` (IEEE 754); see the matching
        # guard in kernel.solve_imcc_sf04.
        if not np.isfinite(basis) or basis <= 0.0:
            raise ImccCompositionIncompleteError(
                f"declared basis must be a positive finite number, got {basis}"
            )
        if abs(total - basis) > 1.0e-6 * basis:
            raise ImccCompositionIncompleteError(
                f"composition sum {total:.12g} does not match declared basis "
                f"{basis:.12g} within 1e-6 relative"
            )

    # Convert wt to mol before the cache boundary (E8).
    if basis_type == "wt":
        vector = _wt_to_mol(vector, parent_oxides)
        basis = float(vector.sum())

    # X_Me2O = (n_Na2O + n_K2O) / sum(n_oxide) over the canonical
    # 8-oxide mol vector. Treat the boundary as inclusive within a 1e-5
    # relative comparison slack; decimal wt% input for a printed X=0.5 can
    # arrive a few ppm above the ideal ratio after conversion. The 1e-5
    # cutoff accepts the measured 3.947e-6 rounding residual but still
    # refuses a composition at X=0.50001.
    alkali_mol = sum(
        float(vector[parent_oxides.index(name)]) for name in ("Na2O", "K2O")
    )
    canonical_oxide_mol = sum(
        float(vector[parent_oxides.index(name)])
        for name in _EXPECTED_PARENT_OXIDES
    )
    if canonical_oxide_mol <= 0.0:
        raise ImccCompositionIncompleteError(
            "canonical 8-oxide composition total is zero"
        )
    x_me2o = alkali_mol / canonical_oxide_mol
    outside_validated_envelope = x_me2o > 0.5 * (1.0 + _ENVELOPE_RELATIVE_SLACK)
    if outside_validated_envelope and not allow_out_of_envelope:
        raise ImccCompositionOutsideValidatedEnvelopeError(
            f"X_Me2O={x_me2o:.12g} exceeds validated bound 0.5"
        )

    try:
        result, _continuation_y = _solve_imcc_sf04_with_seed(
            vector,
            T_K,
            kernel_pack,
            basis=basis,
            extra_mol=extra_mol,
            allow_extrapolation=allow_extrapolation,
            tol=tol,
            max_iter=max_iter,
            y_init=y_init,
        )
    except ImccNonconvergenceError as cold_error:
        positive = np.flatnonzero(vector > 0.0)
        if positive.size == 0:
            raise
        trace_index = int(positive[np.argmin(vector[positive])])
        total_moles = float(vector.sum())
        trace_fraction = float(vector[trace_index] / total_moles)
        floor_fraction = 1.0e-3
        recovered = False
        trace_stages = 0
        recovery_attempts: list[dict[str, Any]] = []

        def record_recovery_failure(
            route: str, stage: int | str, error: ImccNonconvergenceError
        ) -> None:
            recovery_attempts.append(
                {
                    "route": route,
                    "stage": stage,
                    "message": str(error),
                    "diagnostics": {
                        key: value
                        for key, value in error.diagnostics.items()
                        if key != "final_y"
                    },
                }
            )

        if trace_fraction < floor_fraction:
            # The depletion sweep found failures down to 1e-12 of total; a
            # 1e-3 floor clears the observed ill-conditioned 1e-4 boundary.
            # Seven log-spaced stages span at most 1.5 decades per step.
            floor_moles = floor_fraction * total_moles
            stage_moles = np.geomspace(floor_moles, vector[trace_index], 7)
            y_stage = None
            last_success_y = None
            for stage_number, trace_moles in enumerate(stage_moles, start=1):
                trace_stages = stage_number
                staged_vector = vector.copy()
                final_stage = stage_number == len(stage_moles)
                staged_vector[trace_index] = (
                    vector[trace_index] if final_stage else float(trace_moles)
                )
                stage_basis = basis if final_stage else float(staged_vector.sum())
                try:
                    result, continuation_y = _solve_imcc_sf04_with_seed(
                        staged_vector,
                        T_K,
                        kernel_pack,
                        basis=stage_basis,
                        extra_mol=extra_mol,
                        allow_extrapolation=allow_extrapolation,
                        tol=tol,
                        max_iter=max_iter,
                        y_init=y_stage,
                    )
                except ImccNonconvergenceError as stage_error:
                    record_recovery_failure("trace", stage_number, stage_error)
                    candidate = np.asarray(
                        stage_error.diagnostics.get("final_y", ()), dtype=float
                    )
                    active_count = int(np.count_nonzero(vector > 0.0))
                    if (
                        candidate.ndim == 1
                        and candidate.size == active_count
                        and np.all(np.isfinite(candidate))
                    ):
                        y_stage = candidate
                        continue
                    break
                # Keep the pre-certification solver iterate as the walk seed:
                # using the refined root made one low-temperature trace refuse
                # and changed another case's solver path.
                y_stage = np.log(np.exp(continuation_y))
                last_success_y = y_stage.copy()
                if final_stage:
                    result = replace(
                        result,
                        convergence=replace(
                            result.convergence,
                            solver_path="composition-continuation",
                            continuation_stages=stage_number,
                        ),
                    )
                    recovered = True
                    break

            if not recovered and last_success_y is not None:
                # A failed later stage can leave a poor `final_y` even though
                # an earlier stage converged on the connected branch. Insert
                # a trace anchor at 1e-11 before returning to the true vector;
                # the fraction is dimensionless and sits one decade above the
                # sweep's 1e-12 lower end. At or below the anchor, retry the
                # true vector directly from the last continuation start.
                bridge_fraction = min(trace_fraction, 1.0e-11)
                bridge_vector = vector.copy()
                bridge_vector[trace_index] = bridge_fraction * total_moles
                bridge_is_input = bridge_fraction == trace_fraction
                try:
                    bridge_result, bridge_y = _solve_imcc_sf04_with_seed(
                        bridge_vector,
                        T_K,
                        kernel_pack,
                        basis=(
                            basis if bridge_is_input
                            else float(bridge_vector.sum())
                        ),
                        extra_mol=extra_mol,
                        allow_extrapolation=allow_extrapolation,
                        tol=tol,
                        max_iter=max_iter,
                        y_init=last_success_y,
                    )
                except ImccNonconvergenceError as bridge_error:
                    record_recovery_failure("trace_bridge", "anchor", bridge_error)
                else:
                    result = bridge_result
                    if not bridge_is_input:
                        # Preserve the pre-certification bridge seed for the
                        # same established continuation basin.
                        bridge_y = np.log(np.exp(bridge_y))
                        try:
                            result, _continuation_y = _solve_imcc_sf04_with_seed(
                                vector,
                                T_K,
                                kernel_pack,
                                basis=basis,
                                extra_mol=extra_mol,
                                allow_extrapolation=allow_extrapolation,
                                tol=tol,
                                max_iter=max_iter,
                                y_init=bridge_y,
                            )
                        except ImccNonconvergenceError as retry_error:
                            record_recovery_failure(
                                "trace_bridge", "input_retry", retry_error
                            )
                            result = None
                    if result is not None:
                        retry_stages = 1 if bridge_is_input else 2
                        result = replace(
                            result,
                            convergence=replace(
                                result.convergence,
                                solver_path="composition-continuation",
                                continuation_stages=trace_stages + retry_stages,
                            ),
                        )
                        recovered = True

        if not recovered:
            # A failed trace walk can still start outside the selected basin.
            # Raising SiO2 moves the analytical composition toward a nearby
            # well-conditioned silicate root; geometric factors interpolate
            # that root back to the requested vector. The factors are
            # dimensionless, the basis is recomputed at each intermediate
            # composition, and the final factor 1 restores the exact input.
            silica_index = parent_oxides.index("SiO2")
            if vector[silica_index] > 0.0:
                y_stage = None
                silica_factors = (10.0, 8.0, 6.0, 4.0, 2.0, 1.0)
                for silica_stage, factor in enumerate(silica_factors, start=1):
                    silica_vector = vector.copy()
                    silica_vector[silica_index] *= factor
                    final_stage = silica_stage == len(silica_factors)
                    stage_basis = basis if final_stage else float(silica_vector.sum())
                    try:
                        result, continuation_y = _solve_imcc_sf04_with_seed(
                            silica_vector,
                            T_K,
                            kernel_pack,
                            basis=stage_basis,
                            extra_mol=extra_mol,
                            allow_extrapolation=allow_extrapolation,
                            tol=tol,
                            max_iter=max_iter,
                            y_init=y_stage,
                        )
                    except ImccNonconvergenceError as silica_error:
                        record_recovery_failure(
                            "silica", silica_stage, silica_error
                        )
                        candidate = np.asarray(
                            silica_error.diagnostics.get("final_y", ()), dtype=float
                        )
                        active_count = int(np.count_nonzero(vector > 0.0))
                        if (
                            candidate.ndim == 1
                            and candidate.size == active_count
                            and np.all(np.isfinite(candidate))
                        ):
                            y_stage = candidate
                            continue
                        break
                    # Match the trace walk above to retain its established basin.
                    y_stage = np.log(np.exp(continuation_y))
                    if final_stage:
                        result = replace(
                            result,
                            convergence=replace(
                                result.convergence,
                                solver_path="composition-continuation",
                                continuation_stages=trace_stages + silica_stage,
                            ),
                        )
                        recovered = True
                        break

        if not recovered:
            active_parent = vector > 0.0
            inactive_complex = np.any(
                (kernel_pack.nu > 0.0) & (~active_parent[:, None]), axis=0
            )
            active_domains = [
                domain
                for domain, inactive in zip(
                    kernel_pack.domains, inactive_complex, strict=True
                )
                if not inactive
            ]
            if active_domains:
                shared_lower = max(domain[0] for domain in active_domains)
                shared_upper = min(domain[1] for domain in active_domains)
                # Start 10 K above the highest active-row lower edge. This
                # margin keeps the start strictly inside the common domain;
                # the one linear interval to T_K avoids extra failed solves.
                start_temperature = shared_lower + 10.0
                if (
                    shared_lower < start_temperature <= shared_upper
                    and T_K < start_temperature
                ):
                    try:
                        warm_result = evaluate(
                            vector,
                            start_temperature,
                            pack,
                            basis=basis,
                            basis_type="mol",
                            extra_mol=extra_mol,
                            enable_sp_extension=enable_sp_extension,
                            allow_extrapolation=allow_extrapolation,
                            allow_out_of_envelope=allow_out_of_envelope,
                            tol=tol,
                            max_iter=max_iter,
                            y_init=y_init,
                        )
                        warm_y = np.log(
                            warm_result.parent_x_star[active_parent]
                        )
                        result, _continuation_y = _solve_imcc_sf04_with_seed(
                            vector,
                            T_K,
                            kernel_pack,
                            basis=basis,
                            extra_mol=extra_mol,
                            allow_extrapolation=allow_extrapolation,
                            tol=tol,
                            max_iter=max_iter,
                            y_init=warm_y,
                            relative_recovery=True,
                        )
                    except ImccRefusal as temperature_error:
                        if isinstance(temperature_error, ImccNonconvergenceError):
                            record_recovery_failure(
                                "temperature", "warm_start", temperature_error
                            )
                    else:
                        result = replace(
                            result,
                            convergence=replace(
                                result.convergence,
                                solver_path="temperature-continuation",
                                continuation_stages=(
                                    warm_result.convergence.continuation_stages
                                    + 1
                                ),
                            ),
                        )
                        recovered = True

        if not recovered:
            # Earlier composition and temperature continuation stages preserve
            # their warm-started roots. If those all stall on a trace balance,
            # solve once from log(x) with parent-scaled residuals; the kernel's
            # certificate still decides whether the candidate is returned.
            try:
                result, _continuation_y = _solve_imcc_sf04_with_seed(
                    vector,
                    T_K,
                    kernel_pack,
                    allow_extrapolation=allow_extrapolation,
                    basis=basis,
                    extra_mol=extra_mol,
                    tol=tol,
                    max_iter=max_iter,
                    relative_log_x_recovery=True,
                )
            except ImccRefusal as relative_error:
                if isinstance(relative_error, ImccNonconvergenceError):
                    record_recovery_failure(
                        "relative_log_x", "requested_composition", relative_error
                    )
            else:
                recovered = True

        if not recovered:
            if recovery_attempts:
                cold_error.diagnostics["recovery_attempts"] = recovery_attempts
            raise cold_error from None

    active_excess_terms = tuple(term for term in excess_gibbs_terms if term.active)
    if excess_gibbs_terms:
        excess_gibbs_J, excess_enthalpy_J, parent_mu_excess = (
            _analytical_parent_excess_state(
                parent_oxides,
                vector,
                T_K,
                excess_gibbs_terms,
            )
        )
        # The correction is a function of conserved analytical inventories.
        # For each complex j, μex_j = Σ_i ν_ij μex_i, so its reaction
        # correction μex_j - Σ_i ν_ij μex_i is exactly zero. The ideal
        # species solve and its mass-amount certificate therefore remain
        # unchanged; only parent activities and the extensive liquid G move.
        if np.any(parent_mu_excess[vector > 0.0] != 0.0):
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                parent_activity = result.parent_x_star * np.exp(
                    parent_mu_excess / (_R_J_MOL_K * T_K)
                )
            parent_gamma = np.divide(
                parent_activity,
                result.parent_x,
                out=np.zeros_like(parent_activity),
                where=result.parent_x > 0.0,
            )
        else:
            parent_activity = result.parent_activity
            parent_gamma = result.parent_gamma
        result = replace(
            result,
            parent_activity=parent_activity,
            parent_gamma=parent_gamma,
            excess_gibbs_J=excess_gibbs_J,
            excess_enthalpy_J=excess_enthalpy_J,
        )

    # Build the caller-facing label block. The paper-window flag is computed
    # by the kernel from row metadata; the edge flag is computed from the
    # solved free-parent fraction, so neither condition is a hardcoded Na bound.
    identity: Mapping[str, str] = {
        "model_id": model_id,
        "datapack_version": pack_version,
        "binding_digest": binding_digest,
    }
    coverage: Mapping[str, str] = result.labels.coverage
    flags = list(result.labels.flags)
    # Infer network-forming parent oxides from the loaded pack: their declared
    # formula implies a formal cation charge of at least +3, and their nu row
    # participates in a complex with another parent. This selects acid sinks
    # from pack stoichiometry rather than from a species-name list.
    acid_sink_indices: list[int] = []
    for parent_index, parent_name in enumerate(parent_oxides):
        oxide_formula = re.fullmatch(r"([A-Z][a-z]?)(\d*)O(\d*)", parent_name)
        if oxide_formula is None:
            continue
        cation_count = int(oxide_formula.group(2) or 1)
        oxygen_count = int(oxide_formula.group(3) or 1)
        if 2 * oxygen_count < 3 * cation_count:
            continue
        if any(
            kernel_pack.nu[parent_index, complex_index] > 0.0
            and np.count_nonzero(kernel_pack.nu[:, complex_index] > 0.0) > 1
            for complex_index in range(kernel_pack.n_complexes)
        ):
            acid_sink_indices.append(parent_index)

    si_index = parent_oxides.index("SiO2") if "SiO2" in parent_oxides else None
    acid_sink_ratio = (
        float(result.parent_x_star[si_index] / result.parent_x[si_index])
        if si_index is not None and result.parent_x[si_index] > 0.0
        else None
    )
    for sink_index in acid_sink_indices:
        sink_name = parent_oxides[sink_index]
        nominal_sink = float(result.parent_x[sink_index])
        if nominal_sink <= 0.0:
            continue
        sink_ratio = float(result.parent_x_star[sink_index] / nominal_sink)
        if sink_ratio >= _SPECIES_COVERAGE_EDGE_RATIO:
            continue

        # Partition each sink over solved complexes, not input parents. A
        # complex contributes nu(sink) * x(complex); only families holding at
        # least 20% of that solved amount are named. The silica branch keeps
        # its original cation grouping and wording byte-for-byte.
        family_amounts: dict[tuple[str, ...], float] = {}
        bound_sink = 0.0
        for complex_index, complex_name in enumerate(kernel_pack.reactions):
            sink_amount = float(
                kernel_pack.nu[sink_index, complex_index]
                * result.complex_x[complex_index]
            )
            if sink_amount <= 0.0:
                continue
            cations = tuple(
                sorted(
                    (
                        _PARENT_CATION_SYMBOL.get(name, name)
                        for name in parent_oxides
                        if (sink_name != "SiO2" or name != "SiO2")
                        and kernel_pack.nu[
                            parent_oxides.index(name), complex_index
                        ]
                        > 0.0
                    ),
                    key=complex_name.find,
                )
            )
            family_amounts[cations] = (
                family_amounts.get(cations, 0.0) + sink_amount
            )
            bound_sink += sink_amount
        dominant_families = [
            "–".join(cations)
            for cations, family_amount in sorted(
                family_amounts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if bound_sink > 0.0
            and family_amount / bound_sink >= _ACID_SINK_FAMILY_SHARE
        ]
        if sink_name == "SiO2":
            if dominant_families:
                family = f"{' / '.join(dominant_families)} silicate"
            else:
                family = "solution silicate"
        else:
            family = (
                f"{' / '.join(dominant_families)} complex"
                if dominant_families
                else "solution complex"
            )
        flags.append(
            f"species-coverage-edge: free x*({sink_name}) is below "
            f"{_SPECIES_COVERAGE_EDGE_RATIO:g} of nominal x({sink_name}); "
            f"the {family} ladder has exhausted its acidic sink"
        )

    notices = []
    if active_excess_terms:
        interactions = ", ".join(
            f"{'–'.join(term.components)} ({term.basis}; {term.fit_status})"
            for term in active_excess_terms
        )
        notices.append(
            "excess-gibbs-candidate: empirical interaction terms are active: "
            f"{interactions}"
        )
    if any(
        vector[parent_oxides.index(name)] > 0.0
        for name in ("Na2O", "K2O")
        if name in parent_oxides
    ):
        notices.append(_alkali_bias_notice(model_id, binding_digest))

    na2o_activity_bias_notice = _na2o_activity_bias_notice(parent_oxides, vector)
    if na2o_activity_bias_notice is not None:
        notices.append(na2o_activity_bias_notice)

    cao_sio2_notice = _cao_sio2_activity_bias_notice(
        parent_oxides, vector, T_K, model_id, binding_digest
    )
    if cao_sio2_notice is not None:
        notices.append(cao_sio2_notice)

    silicate_product_notice = _silicate_vapour_product_notice(
        parent_oxides, vector, T_K, model_id, binding_digest
    )
    if silicate_product_notice is not None:
        notices.append(silicate_product_notice)

    if result.extrapolated:
        active_parent = vector > 0.0
        active_complex = ~np.any(
            (kernel_pack.nu > 0.0) & (~active_parent[:, None]), axis=0
        )
        below_domain_rows = np.flatnonzero(
            active_complex
            & (T_K < np.asarray(kernel_pack.domains, dtype=float)[:, 0])
        )
        if below_domain_rows.size:
            active_domains = sorted(
                {
                    tuple(float(bound) for bound in kernel_pack.domains[index])
                    for index in below_domain_rows
                }
            )
            domain_text = " and ".join(
                f"{low:g}–{high:g} K" for low, high in active_domains
            )
            notices.append(
                "temperature-domain-extension: active SF04 association rows "
                "are evaluated below their declared low-temperature domain "
                f"using ΔrCp = 0; the active rows have declared domains "
                f"{domain_text}; uncertainty band unavailable"
            )
            # Held-out check (no coefficient change): Zaitsev et al. Table 4
            # (2000, p. 535) compares measured Na2O and SiO2 activities over
            # x(Na2O)=0.195-0.618 and 1173-1673 K. Against the default SF04
            # pack, 55 of 56 tabulated oxide activities differ by more than
            # 3%; the paper reports 1-3% experimental error for these KEMS
            # determinations (p. 531). Thus the activity residual exceeds even
            # the stated upper error in 55/56 cases through 1673 K. The tested
            # subset does not locate a higher-temperature recovery, so retain
            # the conservative 1700 K row-domain floor. Independently, below
            # 1700 K the Na2Si2O5 row misses Kracek's Na2O-SiO2 liquidus at
            # 1079-1147 K by about 17 kJ/mol in crystal-minus-melt Gibbs energy
            # (~0.6-0.7 dex), several times the Na2SiO3 miss (~4 kJ/mol).
            reaction_names = list(kernel_pack.reactions)
            if any(
                reaction_names[index] == "Na2Si2O5" for index in below_domain_rows
            ):
                if (model_id, binding_digest) == (
                    _MELT_EVIDENCE_MODEL_ID,
                    _MELT_EVIDENCE_BINDING_DIGEST,
                ):
                    notices.append(
                        "low-temperature-uncertain: the Na2Si2O5 association is "
                        "evaluated below its declared domain; Piacente and Matoušek "
                        "(1973) MgO-cell p(Na) is about 0.54 dex low at 1278–1530 K, "
                        "and Richet, Bottinga and Téqui (1984) sodium-silicate "
                        "calorimetric curvature misses its envelope by 1.5–3x at "
                        "1323–1700 K; treat Na-silicate activities below 1700 K "
                        "as uncertain"
                    )
                else:
                    notices.append(
                        "low-temperature-uncertain-unscored: Na-silicate low-temperature "
                        "evidence is unscored for this pack"
                    )

    adapter_labels = ImccAdapterLabels(
        identity=identity,
        coverage=coverage,
        envelope_status=(
            "outside_validated"
            if outside_validated_envelope
            else "inside"
        ),
        acid_sink_ratio=acid_sink_ratio,
        flags=tuple(flags),
        notices=tuple(notices),
        inactive_rows=inactive_rows,
    )

    return replace(result, labels=adapter_labels)
