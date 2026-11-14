# Pure-oxide standard checks

`pure_oxide_standard_check(oxide, temperature_K, *, condensate_path=None,
crystal_source="auto")` reports three independent values at the requested
temperature:

1. **JANAF-pair closure** is `G_JANAF,l − G_JANAF,cr` from the vendored pair.
   It uses apparent Gibbs energies on the shared 298.15 K element reference.
   At each pair's transition, the absolute residual should be at most
   `JANAF_PAIR_TOLERANCE_KJ_MOL = 0.01 kJ/mol`, the table/interpolation target.
   K₂O has no JANAF liquid pair and is returned as a flagged constructed
   standard case.
2. **Packaged closure** is `G_pack,l − G_source,cr`. It is the measured
   residual and is never adjusted to zero. When `crystal_source="auto"`, a
   crystal row carried in the selected condensate table is used; otherwise the
   JANAF crystal row is used. This makes the selected pack and crystal source
   explicit in the result provenance.
3. **Switch prediction** is the bracketed root of packaged closure over the
   shared in-range temperature window. It is not moved to a JANAF transition.
   If the sampled function has no sign-changing root, the result is
   `no_switch_in_range`. A root below 1700 K is labelled as liquid-extrapolated
   on `temperature_K`.

The closure uses `species_thermo()` from the selected condensate table. JANAF
apparent Gibbs energy is reconstructed as
`Hf(298.15 K) + [H(T) − H(298.15 K)] − T*S(T)`: enthalpy terms are converted
from kJ/mol to J/mol and entropy is J/(mol K), so Gibbs energy is J/mol.
Transition rows retained as source-parser ambiguities are used directly.
Source table ranges are enforced. A requested JANAF temperature outside the
row range raises `StandardCheckRefusal` with `phase`, `axis`, and the available
range. No source table, condensate pack, or existing calculation is modified.

The legacy silica residual is reported to `±0.002 kJ/mol`, because the cited
`+2.902 kJ/mol` value is rounded to 0.001 kJ/mol and the tolerance allows for
that rounding plus arithmetic representation. Switch brackets target
`0.1 K`, independently of the JANAF interpolation tolerance. The root
solver's internal `SWITCH_SOLVER_XTOL_K` is `1e-8 K`; reported root
uncertainty is the larger of the final bracket width and the residual-derived
temperature error.

```python
from openimcc.standard_checks import (
    pure_oxide_standard_check,
    pure_oxide_standard_checks,
)

# The runtime's packaged default, including its O-035 SiO2(cr) row.
default_rows = pure_oxide_standard_checks()

# Reproduce the documented legacy residual with that pack's LAM1987 crystal row.
legacy_silica = pure_oxide_standard_check(
    "SiO2", 1995.9,
    condensate_path="src/openimcc/data/gas/condensate-feo-janaf-legacy.csv",
)
```

`openimcc diagnose standard-checks --condensate <table.csv>` writes
`pure-oxide-standard-checks.csv`. Without `--condensate`, it uses the packaged
`condensate-default-v1.csv` table.

These controls test reference bookkeeping and range handling. They do not
validate multicomponent activities or thermochemical accuracy.
