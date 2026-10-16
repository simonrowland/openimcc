# Thermochemistry provenance fields

Gas provenance records in `src/openimcc/data/gas/PROVENANCE.yaml`, liquid
construction inputs, and redox records use the same four fields to distinguish
fit quality from evidence about thermodynamic accuracy.

| Field | Shape | Meaning and units |
|---|---|---|
| `fit_residual` | `null` or object | Source-fit residual only. The object records `value`, `unit`, `target_source`, `comparison_method`, `temperature_range`, and `held_out_or_in_sample`. Units follow the compared function: J mol⁻¹, kJ mol⁻¹, or log10 K. |
| `accuracy_band` | `null` or object | Source uncertainty or an empirical construction envelope. The object records `value`, `unit`, `temperature_points_or_range`, and `coverage`. Gas continuation envelopes may retain a temperature-keyed value map; redox bands use kJ mol⁻¹ of the named species or reaction. |
| `band_basis` | object | `kind` is one of `source_uncertainty`, `measured_construction`, `independent_measurement`, `fit_only`, or `none`; `explanation` states the basis and `provenance` points to its source. |
| `independent_check` | object | Records `status`, `equilibrium`, `primary_source`, `conditions`, `measured_value`, `predicted_value`, `miss`, `miss_to_band_ratio`, and `reason`. `status` is `tested`, `lead`, `not_found`, or `not_applicable`. Unknown values stay null; a `not_found` record explains what was not located. |

A fit residual is never copied into `accuracy_band`. If an independent study
provides a lead but no OpenIMCC prediction was computed, record the source and
comparison limits and leave `miss` and `miss_to_band_ratio` null. Existing
`provenance_class` values remain unchanged because downstream consumers pin
those strings.

Melt complex rows have `band_basis.kind: none`, `accuracy_band: null`, and an
explicit `independent_check.status: not_found` unless a row-specific check is
identified. Existing fit diagnostics are retained in `fit_residual` with their
original source-specific units and target description.
