# High temperature candidate tables

These opt-in tables add rough gas and liquid-parent thermochemistry through 6000 K. The packaged defaults are unchanged. Load the candidates explicitly:

```python
from openimcc.gas import load_gas_datapack, evaluate_gas

candidate = load_gas_datapack(
    gas_path="src/openimcc/data/gas/gas-shomate-hight-candidate.csv",
    oxide_path="src/openimcc/data/gas/condensate-hight-candidate.csv",
)
result = evaluate_gas(activities, 3500.0, 1.0e-10, candidate, allow_extrapolation=True)
```

For melt activities above 3000 K, call `evaluate(..., allow_extrapolation=True)` as well. The association constants retain their existing A + B/T form and are extrapolated beyond their declared 1700–3000 K interval. Gas results that use a candidate interval carry a `rough high-temperature extension` notice. These Cp shapes are source-card shapes anchored to the existing row at 3000 K; they are not validated mixture thermochemistry.

CaO(l) and MgO(l) use the current row’s Cp continuation from 3000 K to their CEA liquid-card starts at 3172 K and 3100 K, then follow the CEA card Cp. FeO(l) has no matching CEA liquid card; its current row is continued at constant Cp = 59.837249 J mol⁻¹ K⁻¹ above 3000 K. The current FeO standard itself remains under separate review.

At 3500 K and 4500 K the basalt melt solve requires extrapolation permission for the association rows. Many melts are fully volatilised before 6000 K; expect the existing `P > 1 bar` notice and treat pressures beyond that regime cautiously.

The offset table reports absolute differences between the anchored H, S, G values and the raw NASA CEA card H, S, G at the stated temperatures. Each cell is `|ΔH| J/mol / |ΔS| J/(mol K) / |ΔG| J/mol`. CaO/MgO raw values at 3000 K use the liquid-card polynomial extended below its stated liquid range. FeO has no source card, so offsets are not applicable. The large offsets for some species (for example MgO(g), NaO(g) and AlO2(g)) are not a reference-zero conversion: the CEA cards follow Gurvich assessments and the shipped rows follow NIST-JANAF, and their assessed formation enthalpies differ (MgO(g) at 298.15 K is 32.3 kJ/mol in the CEA card versus 58.2 kJ/mol in the shipped JANAF record). The candidate keeps the shipped values at 3000 K and imports only the CEA heat-capacity shape above it.

| Species | 3000 K | 4000 K | 6000 K |
|---|---:|---:|---:|
| `SiO(g)` | 1656.938265 / 0.062369 / 1469.829936 | 1657.009051 / 0.062406 / 1407.383491 | 1656.935365 / 0.062394 / 1282.570723 |
| `O(g)` | 0.010790 / 0.001121 / 3.372625 | 0.001157 / 0.001125 / 4.502027 | 0.013313 / 0.001123 / 6.751280 |
| `O2(g)` | 97.854301 / 0.051392 / 56.320217 | 97.825820 / 0.051378 / 107.686168 | 97.862307 / 0.051385 / 210.445962 |
| `Mg(g)` | 0.953992 / 0.000951 / 3.807784 | 1.288950 / 0.000777 / 4.398921 | 0.934922 / 0.000837 / 5.956927 |
| `Fe(g)` | 9.133380 / 0.001476 / 4.705196 | 9.464336 / 0.001645 / 2.884451 | 9.100498 / 0.001583 / 0.394715 |
| `Na(g)` | 189.262642 / 0.046187 / 50.700943 | 189.285981 / 0.046198 / 4.494404 | 189.254022 / 0.046192 / 87.897127 |
| `K(g)` | 25.381805 / 0.015300 / 20.517050 | 24.108528 / 0.014641 / 34.455510 | 25.464125 / 0.014870 / 63.754466 |
| `Ca(g)` | 51.618108 / 0.027911 / 32.113590 | 51.863472 / 0.028051 / 60.341328 | 51.671390 / 0.028025 / 116.478842 |
| `Al(g)` | 298.908037 / 0.001357 / 294.835830 | 298.930616 / 0.001369 / 293.453570 | 298.907484 / 0.001365 / 290.714844 |
| `AlO(g)` | 250.904889 / 0.109527 / 579.486756 | 249.915106 / 0.110042 / 690.082804 | 250.955447 / 0.109868 / 910.160943 |
| `Ti(g)` | 639.327304 / 0.006292 / 620.452647 | 639.195227 / 0.006222 / 614.307407 | 639.329392 / 0.006244 / 601.865321 |
| `TiO(g)` | 4455.462893 / 1.102796 / 1147.073478 | 4455.236074 / 1.102682 / 44.506721 | 4455.493635 / 1.102727 / 2160.869896 |
| `SiO2(g)` | 16236.984567 / 0.212492 / 16874.459201 | 16236.977005 / 0.212495 / 17086.957522 | 16236.986703 / 0.212493 / 17511.946766 |
| `MgO(g)` | 23242.228140 / 0.357634 / 24315.129873 | 23242.252550 / 0.357641 / 24672.816810 | 23242.126862 / 0.357671 / 25388.154600 |
| `FeO(g)` | 2.542055 / 0.001831 / 8.035799 | 2.383303 / 0.001914 / 10.039062 | 2.549414 / 0.001886 / 13.866443 |
| `CaO(g)` | 4672.480451 / 0.601869 / 6478.087834 | 4678.529925 / 0.598797 / 7073.718087 | 4671.801875 / 0.599958 / 8271.552190 |
| `AlO2(g)` | 48102.531979 / 18.580292 / 7638.344460 | 48102.513857 / 18.580282 / 26218.614933 | 48102.530253 / 18.580285 / 63379.177961 |
| `TiO2(g)` | 56.295896 / 0.104897 / 258.395267 | 56.408874 / 0.104957 / 363.417773 | 56.294484 / 0.104938 / 573.332865 |
| `NaO(g)` | 27525.493776 / 2.484163 / 20073.005972 | 27526.114203 / 2.484485 / 17588.173096 | 27525.462518 / 2.484376 / 12619.205881 |
| `KO(g)` | 5193.908424 / 3.352915 / 15252.654375 | 5193.132110 / 3.353319 / 18606.407296 | 5193.948901 / 3.353182 / 25313.039953 |
| `Na2O(g)` | 15201.512807 / 10.554180 / 16461.026071 | 15201.513132 / 10.554180 / 27015.205992 | 15201.512719 / 10.554180 / 48123.565512 |
| `K2O(g)` | 15469.517822 / 2.331980 / 22465.458161 | 15469.517632 / 2.331980 / 24797.438450 | 15469.517870 / 2.331980 / 29461.398837 |
| `Al2O(g)` | 3500.575604 / 0.700132 / 5600.972672 | 3500.573414 / 0.700133 / 6301.106999 | 3500.576214 / 0.700133 / 7701.373502 |
| `Al2O2(g)` | 596.565017 / 11.964335 / 36489.570472 | 596.559367 / 11.964338 / 48453.910703 | 596.566593 / 11.964337 / 72382.585623 |
| `Al2(g)` | 13101.314119 / 6.249889 / 5648.353221 | 13100.358894 / 6.249396 / 11897.223686 | 13101.379033 / 6.249568 / 24396.029436 |
| `Na2(g)` | 1246.045123 / 0.607396 / 576.142155 | 1243.159486 / 0.605925 / 1180.541775 | 1246.343408 / 0.606473 / 2392.492800 |
| `K2(g)` | 20148.009722 / 7.521820 / 2417.449254 | 20151.882712 / 7.523880 / 9943.638769 | 20148.048583 / 7.523260 / 24991.508534 |
| `Si(g)` | 4.754998 / 0.000574 / 6.475879 | 4.830838 / 0.000535 / 6.972141 | 4.745419 / 0.000550 / 8.046397 |
| `Si2(g)` | 15519.332197 / 2.976174 / 24447.855357 | 15519.048428 / 2.976320 / 27424.328936 | 15519.355788 / 2.976268 / 33376.962505 |
| `Si3(g)` | 10389.743033 / 8.376427 / 35519.022965 | 10389.701220 / 8.376451 / 43895.506072 | 10389.730915 / 8.376448 / 60648.416168 |
| `SiO2(l)` | 2607.624387 / 0.935801 / 199.779395 | 3612.141334 / 0.520374 / 1530.644017 | 72229.691810 / 11.741770 / 1779.072347 |
| `Al2O3(l)` | 16999.575656 / 6.077601 / 1233.228252 | 27830.357415 / 8.613384 / 6623.180100 | 147244.129313 / 28.141291 / 21603.618293 |
| `CaO(l)` | 8982.733018 / 4.595681 / 4804.311105 | 7432.635150 / 3.881117 / 8091.831677 | 36697.632587 / 8.716610 / 15602.027010 |
| `MgO(l)` | 599.330225 / 0.050554 / 447.668166 | 1769.014820 / 0.160828 / 1125.702057 | 37052.126089 / 5.974681 / 1204.037488 |
| `TiO2(l)` | 3254.346041 / 0.255940 / 4022.165964 | 2897.520672 / 1.696292 / 3887.647572 | 70693.037739 / 12.782792 / 6003.711460 |
| `Na2O(l)` | 14516.148313 / 15.015778 / 30531.185086 | 16749.522781 / 15.538951 / 45406.279866 | 41095.110104 / 19.518447 / 76015.570312 |
| `K2O(l)` | 1531.976750 / 4.892844 / 13146.554598 | 3532.445273 / 5.361502 / 17913.561420 | 25295.471147 / 8.918574 / 28215.974116 |
| `FeO(l)` | — | — | — |

The CEA source is NASA Glenn CEA `thermo.inp` (NASA/TP-2002-211556; file retrieved 2026-09-06, MD5 `709338513d1d6dfe0bc6eb2387995d96`). The liquid cards are Gurvich cards. The source uses 1 atm condensed-phase standards; no pressure correction is applied. See `src/openimcc/data/gas/PROVENANCE.yaml` for the candidate file records and SHA-256 digests.
