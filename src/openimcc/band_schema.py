"""Shared structural validation for band-schema records."""

from collections.abc import Mapping
from typing import Any


BAND_SCHEMA_FIELDS = frozenset({
    "fit_residual", "accuracy_band", "band_basis", "independent_check",
})
LEGACY_BAND_FIELDS = frozenset({
    "band_kj_mol", "band_method", "uncertainty_band",
    "max_residual_J_per_mol", "max_residual_J_per_mol_by_interval",
    "max_residual_log10_K", "additional_uncertainty_bands",
    "max_abs_heldout_log10K_residual",
})
BAND_BASIS_KINDS = frozenset({
    "source_uncertainty", "measured_construction", "independent_measurement",
    "fit_only", "none",
})
INDEPENDENT_CHECK_FIELDS = frozenset({
    "status", "equilibrium", "primary_source", "conditions", "measured_value",
    "predicted_value", "miss", "miss_to_band_ratio", "reason",
})
INDEPENDENT_CHECK_STATUSES = frozenset({
    "tested", "lead", "not_found", "not_applicable",
})


def validate_band_schema(
    record: Mapping[str, Any], label: str, error_type: type[Exception]
) -> None:
    """Validate fields shared by melt and redox band records."""
    legacy = LEGACY_BAND_FIELDS & record.keys()
    if legacy:
        raise error_type(f"{label} contains legacy band fields: {sorted(legacy)}")
    missing = BAND_SCHEMA_FIELDS - record.keys()
    if missing:
        raise error_type(f"{label} is missing band schema fields: {sorted(missing)}")

    basis = record["band_basis"]
    if not isinstance(basis, dict) or basis.get("kind") not in BAND_BASIS_KINDS:
        raise error_type(f"{label}/band_basis/kind is not an allowed value")
    if (
        "provenance" not in basis
        or not isinstance(basis.get("explanation"), str)
        or not basis["explanation"]
    ):
        raise error_type(f"{label}/band_basis needs explanation and provenance")

    check = record["independent_check"]
    if not isinstance(check, dict) or INDEPENDENT_CHECK_FIELDS - check.keys():
        raise error_type(f"{label}/independent_check has an incomplete structure")
    if check.get("status") not in INDEPENDENT_CHECK_STATUSES:
        raise error_type(f"{label}/independent_check/status is invalid")
    if check.get("status") == "not_found" and not isinstance(check.get("reason"), str):
        raise error_type(f"{label}/independent_check/reason must explain not_found")
