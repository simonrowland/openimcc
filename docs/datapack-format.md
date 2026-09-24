# IMCC datapack format

This is the JSON format used by the melt model in openimcc.model and
openimcc.kernel. It is derived from the five JSON files in
src/openimcc/data/packs/ (excluding MANIFEST.json) and from the loader and
solver code. The loader is deliberately stricter than the descriptive
metadata: a file can be a useful research record and still not be a pack that
the current production loader accepts.

## 1. What a pack contains

An IMCC datapack contains an ordered set of parent-oxide formula units, one
active reaction row per complex, fitted equilibrium-constant coefficients, and
the evidence/provenance record for those coefficients. For complex j, the
pack stores the fit

~~~text
log10 K_j(T) = A_j + B_j / T_K
~~~

where T_K is temperature in kelvin. The kernel evaluates the mass-action
equation exactly as

~~~text
x_j = K_j(T) * prod_i (x_i*)^nu_ij
ln K_j = ln(10) * (A_j + B_j / T_K)
~~~

x_i* is the unbound parent-oxide mole fraction on the total-species basis,
x_j is the complex mole fraction on that same basis, and nu_ij is the
amount of parent i consumed by one mole of complex j. The ordered parents
array defines the order of the i index. Fractional values in nu are allowed.

For completeness, the parent-balance system solved by kernel.py is:

~~~text
n_i = N_total * x_i* + N_total * sum_j(nu_ij * x_j)
S_j = sum_i nu_ij
D = 1 + sum_j((S_j - 1) * x_j)
x_i = (x_i* + sum_j(nu_ij * x_j)) / D
f_i = x_i* + sum_j(nu_ij * x_j) - x_i * D = 0
~~~

In Equation (3), the implementation supplies the solver with
`x_i = parent_mol_i / basis`. It does not renormalise that vector when the
declared basis is within the accepted `1e-6` relative slack, so the analytical
fraction in the equation and the solver input can differ slightly when
`sum(parent_mol)` is not exactly `basis`.

The solver works in y_i = ln(x_i*), with x_j computed from the logarithmic
mass-action form above. The returned parent activity is x_i*, and the
returned activity coefficient is gamma_i = x_i* / x_i. A complex is inactive
on a particular solve when any parent with positive nu_ij has zero analytical
input; inactive complexes are assigned zero and do not impose a temperature
domain check.

The JSON is not a generic thermodynamic database. In particular, the runtime
model consumes parents, the active rows fields described below, and the
identity/provenance information installed by the loader. The other fields are
audit, source, merge, or screening metadata unless stated otherwise.

## 2. Top-level object

The Required? column means required by the current load_datapack() path,
not merely present in every historical file. “Conditional” describes a field
whose presence changes the validation path. Metadata fields are not rejected
when absent, although they are present in the shipped packs as noted.

| Key | JSON type in shipped packs | Required? | Meaning |
|---|---|---|---|
| created | string | No | Pack creation date, represented as an ISO-like date such as 2026-08-11. The loader does not parse it. |
| errata | array of strings or objects | No | Pack-specific corrections, omissions, or caveats. Ext-v1 through ext-v3 use strings; v1.0.2 and ext-v4 use objects with `change`, `date`, and `version`. The loader does not inspect the entries. |
| extension_parent_datapack_version | string | No | Version of the parent/earlier extension pack used by the extension record. Present in ext-v3; the loader treats it as metadata. |
| extension_wave | string | No | Label for the extension work wave, for example EXT3-JANAF-native-endmember-screen. Its exact release semantics are not established by the code. |
| gas_only_inventory | array of objects | No | Species considered for the gas inventory but deliberately refused as melt components. In the shipped ext-v3 record each object has cation, species, table_id, T_range_K, melt_disposition, and certification. |
| imcc_sf04_datapack_version | non-empty string | Yes | Datapack version. load_datapack() stores it as the loaded version and uses it in the canonical published-identity projection. |
| merge_semantics | object | No | Describes how an extension overlays or carries forward published rows: mode, row counts, replacement counts, and rationale. It is not interpreted by load_datapack(). |
| model | string | No | Human-readable model description. Older extension files use IMCC-SF04-EXT; v1.0.2 and ext-v4 use a longer equation/species-count description. Runtime identity comes from model_id, not this field. |
| model_id | string | Conditional | Canonical runtime identity. If absent, the loader defaults it to IMCC-SF04. A present extension identity must be exactly IMCC-SF04-EXT and must have a valid sp_extension object. |
| pack_arithmetic | string | No | Prose describing how the pack's rows and coefficient revisions were assembled. It does not alter arithmetic. |
| parents | array of strings | Yes | Ordered parent-oxide basis. The current loader requires exactly ["SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O", "K2O"], in that order. sp_extension adds S and P2O5 only on its own extension path. |
| provenance_split | object | No | Pack-level provenance summary. v1.0.2 and ext-v4 use numeric class counts plus a note; ext-v1 through ext-v3 use prose `published`/`extension` mappings. It is descriptive; row-level provenance_class remains the audit record. |
| rows | array of objects | Yes | Active published-core reaction rows. The current loader requires exactly 38 object rows. Historical ext-v2 has 39 rows and is therefore a research record, not loadable by the current fixed production path. |
| screens | array of objects | No | Candidate components considered for a deliberately gated, non-active screen. The loader neither activates nor validates these candidates. |
| sources | object | No | Bibliographic and table-source inventory. In the base and ext-v4 packs it includes FC87/SF04 citation metadata; older extensions use central/local inventory labels. |
| sp_extension | object | Conditional | Separate S/P extension declaration. If present, the loader validates its identity, certification, parents, and non-empty extension rows, then appends those rows to the published core. |
| spec | string | No | Path and, in v1.0.2 and ext-v4, revision label for the model specification used by the pack. It is not opened by the loader. |
| transcription | string or object | No | Record of how source values were transcribed and checked. Older packs use a string; v1.0.2/ext-v4 use an object with `method` and `diff_result` fields. |

### sp_extension structure

The shipped ext-v4 sp_extension is separate from the 38-row published core.
Its enable_flag is enable_sp_extension, its tier is EXT-SP, its
certification is denied, and its parents are exactly ["S", "P2O5"]. It also
carries parent_reference_states, authority, redox_scope, and a provenance
object. The loader requires the provenance object at each extension row to
contain non-empty string fields source and table_id.

Each nested extension row has the common coefficient fields (complex, nu, A,
B, T_domain_K, and T_domain_basis) plus the shipped extension metadata such
as reaction, phase_or_construction, tier, certification, authority,
external_oxygen_stoich_product_positive, and the nested provenance record.
The loader requires every extension row to have provenance_class equal to
extension-compound-thermo, tier equal to EXT-SP, and certification equal to
denied. A nested row's nu object, when present, may name only the eight base
parents plus S and P2O5, and it must consume at least one of the two extension
parents.

The nested fields below are present in the shipped extension, screen, gas-only,
merge, and provenance records. “Not read” means that the production loader and
kernel do not inspect the field; it remains descriptive audit metadata. Common
row fields already have their runtime meanings in the active-row table above.

### Nested extension fields

| Key | JSON type in shipped packs | Runtime use | Meaning |
|---|---|---|---|
| authority | string | Not read | Authority or source family recorded for an extension declaration, extension row, or screen candidate. |
| certification | string | Loader for `sp_extension` and its rows; not read for screens or gas inventory | Certification state. The loadable S/P extension requires `denied`; the other occurrences remain metadata. |
| enable_flag | string | Loader | Literal opt-in flag `enable_sp_extension` required by the S/P extension declaration. |
| parent_reference_states | object | Not read | Reference-state descriptions for the S and P2O5 extension parents. |
| provenance | object | Loader for extension-row shape; contents otherwise metadata | Extension-wide or row-level evidence record. Row provenance must contain non-empty `source` and `table_id`; other entries are not interpreted. |
| redox_scope | string | Not read | Declared redox scope or limitation of the S/P extension. |
| rows | array of objects | Loader | Non-empty extension-row list appended to the published core. |
| tier | string | Loader for `sp_extension` and its rows; not read for screens | Extension or screen tier label. The loadable S/P extension and each of its rows require `EXT-SP`. |
| external_oxygen_stoich_product_positive | number | Not read | Signed oxygen stoichiometry associated with the extension reaction construction. |
| phase_or_construction | string | Not read | Phase and thermodynamic construction used for an extension row. |

The extension-wide `sp_extension.provenance` object adds these audit fields:

| Key | JSON type in shipped packs | Runtime use | Meaning |
|---|---|---|---|
| correction_policy | string | Not read | Policy used to decide which source corrections were applied. |
| corrections_ledger | string | Not read | Identifier or description of the corrections ledger. |
| corrections_ledger_report | string | Not read | Report associated with the corrections ledger. |
| fit_form | string | Not read | Functional form used for the extension fit evidence. |
| fit_temperatures_K | array of numbers | Not read | Temperatures used for the extension fit checks, in kelvin. |
| independent_spot_temperatures_K | array of numbers | Not read | Independent spot-check temperatures, in kelvin. |
| quality_mandate | string | Not read | Quality or review requirement recorded for the extension evidence. |

The S/P extension row provenance objects add these fields. `source` and
`table_id` are required by the loader; the remaining fields are not read.

| Key | JSON type in shipped packs | Runtime use | Meaning |
|---|---|---|---|
| caveat | string | Not read | Caveat attached to the row's source evidence. |
| correction_rows_applied | array | Not read | Source-correction row identifiers applied to the fit. |
| fit_T_K | array of numbers | Not read | Temperatures used for row fit evidence, in kelvin. |
| local_mirror_path | string | Not read | Local mirror or access-copy path for the source. |
| max_abs_heldout_log10K_residual | number | Not read | Maximum held-out residual in log10 K units. |
| parent_basis | string | Not read | Parent basis used by the row's evidence record. |
| source_url | string | Not read | URL for the row's source or access copy. |
| spot_T_K | array of numbers | Not read | Independent row spot-check temperatures, in kelvin. |
| table_T_range_K | array of numbers | Not read | Temperature range of the cited source table, in kelvin. |
| table_id | string | Loader for extension-row provenance; not read in gas inventory | Source table identifier. The loader requires it alongside a non-empty `source` on each S/P extension row. |

## 3. Active row objects

The following are the 28 keys found across the active rows arrays in all
five shipped packs. The Required? column again describes the current
loader. Every shipped active row has the metadata fields marked “No” in at
least some pack, but those fields are not required to construct the kernel
datapack.

| Key | JSON type in shipped packs | Required? | Meaning |
|---|---|---|---|
| A | number | Yes | Intercept in log10 K_j(T) = A_j + B_j/T_K. |
| B | number | Yes | Kelvin-scaled inverse-temperature coefficient in the same fit. |
| T_domain_K | array of two numbers | Yes | Declared inclusive [low, high] Kelvin interval used by the solver for this active complex. |
| T_domain_basis | string | Yes | Evidence or policy basis for the declared runtime interval, such as the SF04-exercised interval or a JANAF regression interval. The loader stores the string but does not interpret its wording. |
| T_domain_paper_demonstrated_K | array of two numbers | No | Temperature interval demonstrated by the cited paper. It is audit metadata and is not the kernel's runtime gate. |
| complex | string | Yes | Unique complex/species name. It becomes the kernel reaction name and must be unique across all published and extension rows. |
| delta_H_fusion_kJ_mol | number | No | Fusion enthalpy used in a thermo construction, in kJ/mol, when the row was built from a liquid/metastable-liquid construction. |
| derivation | string | No | Reaction, reference-state, and coefficient-derivation narrative. The loader does not parse it. |
| fit_metadata | object | No | Regression grid, point count, fit residuals, direct spot checks, or related fit diagnostics. |
| fusion_transition_K | number | No | Fusion transition temperature used by the row's thermodynamic construction, in kelvin. |
| janaf_liquid_table_range_K | array of two numbers | No | Temperature range represented by the cited JANAF liquid table, in kelvin. |
| janaf_range_caveat | string or null | No | Caveat about liquid-table range, metastable/supercooled liquid values, or related table interpretation. |
| merge_action | string | No | Extension-overlay action such as replace-same-name-published-fit. It records merge intent; the production loader does not apply it. |
| metastable_liquid_construction | string | No | Details of how a liquid standard state was constructed below a fusion transition. |
| nu | object mapping parent names to numbers | Yes | Stoichiometric vector for the reaction. Missing base-parent names are treated as zero; unknown names are refused. Values are converted through exact rational parsing before the kernel matrix is built. |
| phase_basis | string | No | Phase or standard-state basis used for the row, for example an explicit liquid or metastable liquid construction. |
| provenance_class | string | No for base rows; fixed for S/P extension rows | Row-level provenance label. Base rows are not required by load_datapack() to use a particular label, but extension rows are required to use extension-compound-thermo. |
| reaction | string | No | Human-readable balanced reaction corresponding to nu, normally using liquid parent and complex reference states. |
| replaces_published_fit | object | No | Prior published coefficient record replaced by an extension overlay. Shipped objects contain prior A, B, source, and source_provenance_class. |
| row | number | No | Stable/display row number from the source or pack record. It is not used as the array index by the loader. |
| source | string | No | Primary source URL or source identifier for the row. |
| source_access_copy | string | No | Access-copy URL for a supporting source, used by one extension row. Its semantics are provenance metadata only. |
| source_compilation | string | No | Name/edition of the supporting thermochemical compilation, used by one extension row. |
| source_provenance_class | string | No | Provenance class of the source being replaced or compared, distinct from this row's provenance_class. |
| source_ref_cited_by_paper | string | No | Source named by the cited paper for the row or fit. |
| source_tables | string | No | Table/page identifiers used for a compound and its parents, used by one extension row. |
| supersedes | object | No | Earlier row/fit record superseded by this row; shipped objects contain prior A, B, and source. |
| transcription_note | string | No | Note about OCR, visual transcription, or a field-level source-reading decision. |

The loader uses only the coefficient, stoichiometry, name, and declared-domain
fields needed to construct the kernel object. It does not validate that
reaction balances nu, that a URL resolves, that a provenance description
matches its class, or that optional metadata objects have a particular shape.
That is intentional: these records are audit evidence, not a second runtime
schema.

## 4. Provenance values and splits

The active rows use these five provenance_class values:

| Value | Meaning evidenced by the shipped packs |
|---|---|
| authority_by_publication | Coefficients retained from the published authority (the FC87/SF04 publication record), rather than recomputed from the local JANAF exercise. The base and ext-v4 packs contain 28 such rows. |
| janaf_direct_liquid | Coefficients regressed directly from explicit JANAF liquid-standard-state tables. The base and ext-v4 packs contain 9 such rows: rows 1–6, 24, 25, and 31. |
| partial_non_janaf_direct_liquid | The KAlO2 row (row 35): a direct liquid compilation through Glushko/Gurvich as cited by FC87, not a complete modern JANAF direct-liquid row. The pack records the limitation rather than silently upgrading the class. |
| published-imcc | Coarse label used by ext-v1, ext-v2, and ext-v3 for the 30 carried-forward published rows. It intentionally collapses the finer authority/JANAF/partial distinction. |
| extension-compound-thermo | Thermochemical coefficients added or overlaid by the extension work. Ext-v1/v2/v3 use it for their 8/9/8 extension rows; ext-v4 uses it in its nested S/P extension rows. |

For v1.0.2 and ext-v4, provenance_split reports
authority_by_publication: 28, janaf_direct_liquid: 9, and
partial_non_janaf_direct_liquid: 1, with a note identifying the JANAF
consistency-check rows and the KAlO2 limitation. For ext-v1 through ext-v3,
`provenance_split` is two prose mappings: `published` describes the 30
carried-forward rows as `published-imcc`, and `extension` describes the
extension rows as `extension-compound-thermo`. The 30/8, 30/9, and 30/8
counts are carried by the row arrays and `merge_semantics`, not by this
object. The inconsistency is documented as shipped; it is not a reason to
rewrite the packs.

The loader does not compare provenance_split to row counts and does not
normalize these labels. A consumer auditing provenance should read both the
row-level class and the pack-level split.

## 5. Base packs, overlays, screens, and gas-only inventory

The base imcc-sf04-v1.0.2.json has eight parent oxides and 38 active
published-core rows. Ext-v1 and ext-v3 describe an overlay in which 30
published rows are carried forward and 8 same-name fits are replaced by
extension rows; their merge_semantics.mode is
replace-by-complex-name, with no novel active complex. Ext-v2 adds a
research-only CoO parent and Co2SiO4 complex, giving 39 rows; its own
metadata says that the research loader constructs it directly while the
production load_datapack() remains fixed at eight parents and 38 rows.

Ext-v3 adds 13 screens and a ten-entry gas_only_inventory without making
those candidates active reaction rows. Each v3 screen is marked
certification: denied, has an explicit opt-in activation_gate, and is
no_complexes: true and one_at_a_time: true. The candidates include B, Ba,
Co, Cr, Cu, Li, Mo, Nb, Pb, Sr, V, W, and Zr oxide scenarios. They are
considered, recorded, and deliberately not activated by the current IMCC
solver.

The v3 gas_only_inventory records P and S gas species such as P, PO, P4O6,
PO2, P4O10, S, SO, S2O, SO2, and SO3. Each is marked
melt_disposition: hard-refused-by-spec; gas inventory only and
certification: denied. Listing one in a pack does not add it to parents,
does not add a complex row, and does not make it acceptable to evaluate().

Ext-v4 is the current S/P extension shape. It retains the frozen published
core and adds S and P2O5 through sp_extension. The loader accepts it only
with the explicit extension identity and flag; evaluate() also requires
enable_sp_extension=True. The extension is explicitly uncertified (denied)
and records that redox and sulfur-solubility behavior remain out of scope.

The older ext-v1–v3 JSON files are valuable research/overlay records, but they
do not all satisfy the current production loader. On the shipped files,
v1/v3 fail the frozen canonical hash and v2 fails the fixed parent-basis check;
base v1.0.2 and ext-v4 load.

The nested screen and gas-inventory fields are descriptive only; the loader
does not read any of these fields or activate the candidates.

| Key | JSON type in shipped packs | Runtime use | Meaning |
|---|---|---|---|
| T_range_K | array of numbers | Not read | Temperature range recorded for a gas-only inventory species, in kelvin. |
| activity_basis | string | Not read | Basis on which a screen activity value is stated. |
| activity_semantics | string | Not read | Interpretation of the screen activity or activity coefficient. |
| alternative_endmembers_not_activated | array | Not read | Alternative endmembers considered but deliberately left inactive. |
| activation_gate | object | Not read | Explicit opt-in gate recorded for a screen candidate; it does not activate the candidate in the current solver. |
| cation | string | Not read | Cation represented by a screen or gas-only inventory record. |
| companion_crl_table_id | string | Not read | Identifier for the companion condensed/reaction-limit table. |
| component_atoms | object | Not read | Element-count mapping for the screened component formula. |
| component_formula | string | Not read | Formula of the screened component. |
| condensed_formation_gibbs_grid | array of objects | Not read | Condensed-phase formation-Gibbs values and source checks over a temperature grid. |
| condensed_phase_contract | string | Not read | Contract or policy for the screened condensed phase. |
| condensed_table_id | string | Not read | Identifier for the condensed-phase source table. |
| condensed_table_range_K | array of numbers | Not read | Temperature range of the condensed-phase source table, in kelvin. |
| datapack_version | string | Not read | Datapack version associated with the screen record. |
| evidence_class | string | Not read | Evidence classification assigned to the screen candidate. |
| fusion | object | Not read | Fusion-transition evidence record for the screen candidate. |
| fusion_construction_used | boolean | Not read | Whether a fusion construction was used in the screen calculation. |
| fusion_derivation_role | string | Not read | Role of fusion data in deriving the screen candidate's values. |
| fusion_stability_boundary_K | number or string | Not read | Fusion or stability boundary recorded for the screen, in kelvin when numeric. |
| gamma | number | Not read | Screen activity-coefficient value or proxy recorded by the research screen. |
| gas_channels | array of objects | Not read | Candidate gas-channel records associated with a screen. |
| input_basis | string | Not read | Composition or activity basis used by the screen calculation. |
| melt_disposition | string | Not read | Disposition of the gas-only species in the melt specification; shipped entries say gas inventory only. |
| model_id | string | Not read in screen entries | Identity label copied into the screen record; it does not label or load the runtime model. |
| no_complexes | boolean | Not read | Screen declaration that no melt complex rows are activated. |
| one_at_a_time | boolean | Not read | Screen declaration that the candidate was evaluated one at a time. |
| oxide_hosting_assumption | string | Not read | Assumption about which oxide hosts the screened cation. |
| phase_mapping | string | Not read | Mapping from the screened component to its modeled phase. |
| range_caveat | string | Not read | Caveat about the temperature or source range of the screen. |
| redox_warning | string | Not read | Redox limitation recorded for the screen candidate. |
| screen_id | string | Not read | Stable identifier for a screen candidate. |
| source_identity | string | Not read | Source identity label for the screen record. |
| species | string | Not read | Gas species recorded in the gas-only inventory. |
| T_domain_K | array of numbers | Not read in screen entries | Candidate temperature interval recorded for the screen; unlike an active row's field, it is not a solver domain gate. |
| uncertainty | string | Not read | Uncertainty statement for the screen candidate. |

The extension-overlay bookkeeping fields are also metadata only; no loader or
kernel path applies their merge instructions.

| Key | JSON type in shipped packs | Runtime use | Meaning |
|---|---|---|---|
| extension | string | Not read | Prose `provenance_split` description of the extension rows in ext-v1 through ext-v3. |
| extension_rows_active | integer | Not read | Number of extension rows described as active by the overlay record. |
| mode | string | Not read | Overlay mode, such as `replace-by-complex-name`. |
| note | string | Not read | Note attached to the numeric `provenance_split` in v1.0.2 and ext-v4. |
| novel_complexes_added | integer | Not read | Count of novel complexes added by an overlay. |
| published | string | Not read | Prose `provenance_split` description of the carried-forward published rows in ext-v1 through ext-v3. |
| published_fits_replaced | integer | Not read | Count of published fits replaced by an overlay. |
| published_rows_carried_forward | integer | Not read | Count of published rows carried forward by an overlay. |
| reason | string | Not read | Rationale for the overlay bookkeeping. |
| research_loader | string | Not read | Loader or construction path named by the ext-v2 research record. |
| runtime_pack_rows | integer | Not read | Runtime row count recorded by the overlay metadata. |

## 6. Domains and extrapolation

T_domain_K and T_domain_paper_demonstrated_K are intentionally different:

* T_domain_K is the declared runtime domain. In the published core it is
  [1700, 3000] K. Extension thermo rows may declare a different interval,
  such as [400, 1500] K.
* T_domain_paper_demonstrated_K records the span demonstrated by the cited
  paper. For most base rows it is [2500, 3500] K; some rows do not carry the
  field. It is not loaded into ImccDatapack.domains and is not a runtime
  restriction.

For each active complex, solve_imcc_sf04() compares T_K with its T_domain_K,
inclusively. Outside that declared interval it raises
ImccTOutsideDatapackDomainError unless allow_extrapolation=True; with the
flag, it evaluates the same A + B/T_K fit and sets the result's extrapolated
flag. A temperature outside the paper-demonstrated interval but inside the
declared interval is not refused and is not marked extrapolated, because the
solver has no runtime use for that metadata field. A row whose required parent
is absent is inactive and does not impose its domain check.

## 7. What load_datapack() validates

The loader returns ImccMalformedDatapackError (code
imcc_malformed_datapack) for the normal file/schema refusals below. It does
not silently repair a pack.

### JSON, identity, and published core

1. The path must exist and contain valid JSON. A missing file, JSON decode
   failure, or JSON root that is not an object is a malformed-datapack refusal.
2. imcc_sf04_datapack_version must be a non-empty string.
3. parents must be a list exactly equal to the eight canonical parent names
   and order shown in the top-level table.
4. If sp_extension is absent, the effective model_id must be IMCC-SF04. If it
   is present, model_id must be exactly IMCC-SF04-EXT.
5. rows must be a list of exactly 38 published-core objects. Every one of
   those first 38 entries must be a dictionary.
6. The loader builds a canonical published-core payload by removing model_id
   and sp_extension, forcing model_id to IMCC-SF04, and (for the S/P extension
   identity) using the version prefix before -ext-sp-. It sorts object keys
   and normalizes JSON numbers before hashing. The result must equal the
   frozen published identity SHA-256:

~~~text
f2b479cd54e3c82704a5863fcc06836f72045375d9a8c7f8d2fad19e98f75d05
~~~

Canonicalization failure or a mismatch is a malformed-datapack refusal.
This is a semantic identity gate; it is distinct from the raw-byte hashes
in MANIFEST.json described below.

### S/P extension checks

When sp_extension is present, it must be an object with:

* enable_flag == "enable_sp_extension";
* tier == "EXT-SP";
* certification == "denied";
* parents == ["S", "P2O5"];
* a non-empty list in rows whose entries are objects;
* for every extension row, provenance_class == "extension-compound-thermo",
  tier == "EXT-SP", certification == "denied", and a dictionary provenance
  with non-empty string source and table_id fields.

If an extension row has a dictionary nu, its keys must be among the eight
base parents plus S and P2O5, and at least one of S/P2O5 must have a positive
exact-rational value. The common row checks below then apply to the extension
rows as well. Each failed check is an ImccMalformedDatapackError.

### Common row and kernel-object checks

Every published or extension row must have:

* a string complex, and unique complex names across the complete loaded pack;
* an object nu; unknown parent names produce ImccMalformedDatapackError,
  while omitted known parent names mean zero;
* numeric Python JSON numbers for A and B;
* a two-element numeric T_domain_K list; and
* a string T_domain_basis.

The loader converts the row vectors into a parent-by-complex matrix and
constructs ImccDatapack. Its constructor additionally refuses with ordinary
ValueError if the matrix is not 2-D, shapes do not agree, a nu coefficient
is negative, a complex column is all zero, A/B is non-finite, or a domain
endpoint is non-finite. Inverted but finite domain endpoints are retained and
will fail the later temperature gate rather than being rejected here. An
invalid nu scalar can also surface the built-in TypeError/ValueError from
exact-rational parsing; it is not converted to the custom refusal.

Finally, published identity labeling checks that parent and reaction names are
globally unique, coverage labels exactly match all species, and every coverage
label is a non-empty string. A published claim requires the frozen hash, the
exact frozen published species set, and—only for plain IMCC-SF04—exactly
eight parents and 38 complexes. These internal identity failures surface as
ValueError; normal JSON failures have already been converted to
ImccMalformedDatapackError.

No loader validation is performed for created, errata, model,
pack_arithmetic, provenance_split, screens, sources, spec, transcription,
merge_semantics, or gas_only_inventory. Unknown top-level metadata can still
change the canonical published hash and therefore make a published pack fail
the identity check.

## 8. Evaluation-time refusals after loading

Temperature and composition are not load-time properties. The following
checks happen when evaluate() delegates to the kernel:

| Condition | Refusal or result |
|---|---|
| A raw ImccDatapack has no proven identity | ImccUnprovenDatapackError; use load_datapack() or explicitly label a research pack. |
| S/P names with nonzero values are supplied without an ext pack and explicit enablement, or an ext pack is used without enable_sp_extension=True | ImccSPComponentRequiresExtensionError, code imcc_sp_extension_required. Only nonzero S/P values count as supplied; on a plain pack, a zero-valued S/P mapping key remains an unknown component and raises ImccComponentOutsideDomainError. The flag does not widen a plain pack. |
| basis_type is not mol or wt, vector is not 1-D, has the wrong length, is non-finite, negative, or sums to zero | ImccCompositionIncompleteError. An overlong vector on a plain pack is intercepted earlier by the S/P gate and raises ImccSPComponentRequiresExtensionError; an overlong vector on an enabled extension pack raises ImccComponentOutsideDomainError. |
| An enabled S/P pack receives only S and P2O5 with positive values | ImccCompositionIncompleteError because the total supplied moles are positive but the canonical eight-oxide composition total is zero. |
| Mapping contains Fe2O3 with a nonzero value | ImccFerricInputUnsupportedError; the caller must convert to FeO under its redox model. |
| Mapping or extra_mol contains another component outside the loaded parent basis | ImccComponentOutsideDomainError. |
| Declared basis is non-positive or differs from the input sum by more than 1e-6 relative | ImccCompositionIncompleteError. If no basis is supplied, the input sum is used. Weight input is converted to moles before the kernel call. |
| The canonical eight-oxide alkali fraction X_Me2O = (n_Na2O+n_K2O)/sum(n_canonical_oxide) exceeds 0.5 | ImccCompositionOutsideValidatedEnvelopeError, code imcc_composition_outside_validated_envelope, unless allow_out_of_envelope=True; then the result is marked outside_validated. |
| T_K is not finite or is not positive | ImccTOutsideDatapackDomainError. |
| An active row is outside its T_domain_K | ImccTOutsideDatapackDomainError, unless allow_extrapolation=True; then the result is marked extrapolated. |
| extra_mol has negative nonzero moles | ImccCompositionIncompleteError from the sign check, including negative `Fe2O3`. A positive `Fe2O3` extra is the ferric refusal; another positive extra is a component-outside-domain refusal. |
| tol is not convertible to float | built-in TypeError, because `float(tol)` is called outside the guarded `max_iter` conversion. |
| tol is non-positive or non-finite, or max_iter is non-finite/non-numeric | built-in ValueError. A finite max_iter <= 0 reaches the solver and produces ImccNonconvergenceError. |
| The parent-balance solve produces non-finite residuals, misses the tolerance within its evaluation budget, or cannot complete continuation | ImccNonconvergenceError with diagnostics. |

The composition envelope is therefore a validated model-use boundary, not a
property of the JSON schema. T_domain_paper_demonstrated_K is likewise an
audit statement, not a second solver gate.

## 9. Integrity and versioning

tools/packmanifest.py is the raw-file drift detector. It scans every JSON
pack except MANIFEST.json, computes SHA-256 over the exact file bytes (not a
parsed/reformatted representation), and records the file name, hash, byte
count, model/version, active-row count, screened count, provenance counts, and
provenance_split in src/openimcc/data/packs/MANIFEST.json.

Run:

~~~bash
python tools/packmanifest.py --check
~~~

--check reports a nonzero result for a new, changed, or removed pack. Use
--write only after an intentional, reviewed new version or pack addition.
The raw-byte hash catches changes that canonical semantic hashing could hide,
including key order, whitespace, and float representation. The loader's
frozen canonical hash catches drift in the published identity independently.

Never edit a shipped pack in place while keeping its version unchanged. A
coefficient edit moves every mass-action constant, activity, and downstream
residual while leaving consumers that pin only the version believing they are
running the old data. Create a new version, update the manifest deliberately,
and explain the provenance/change in the same change set.

## 10. Minimal worked example

The smallest loadable published pack is not a tiny JSON object: the current
loader requires the complete frozen 38-row core and its canonical hash. The
following is an annotated excerpt from
src/openimcc/data/packs/imcc-sf04-v1.0.2.json; the real file contains all
top-level metadata keys carried by that file and all 38 rows.

~~~json
{
  "imcc_sf04_datapack_version": "1.0.2",
  "parents": ["SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O", "K2O"],
  "rows": [
    {
      "row": 1,
      "complex": "Mg2SiO4",
      "reaction": "2 MgO(liq) + SiO2(liq) = Mg2SiO4(liq)",
      "nu": {"SiO2": 1, "MgO": 2},
      "A": -0.94,
      "B": 7434,
      "T_domain_K": [1700, 3000],
      "T_domain_basis": "sf04-exercised-ADOPTED ..."
    }
    // 37 more complete active rows follow in the real file.
  ]
}
~~~

Load and evaluate the complete file:

~~~python
from openimcc import evaluate, load_datapack

pack = load_datapack("src/openimcc/data/packs/imcc-sf04-v1.0.2.json")
result = evaluate(
    {"SiO2": 0.45, "MgO": 0.10, "CaO": 0.15, "Al2O3": 0.15, "FeO": 0.15},
    1800.0,
    pack,
)
print(pack.model_id, pack.version, result.D)
~~~

This loads as IMCC-SF04, version 1.0.2, and evaluates in the declared
temperature interval. The reference run returned D = 1.983671832299221.

## Evidence limits

The loader gives no runtime semantics to descriptive fields such as
extension_wave, merge_semantics, or transcription, beyond the explicit
values and structures recorded in the shipped files. Their prose meanings
are therefore documented as pack metadata, not promoted into an invented
schema contract. No unresolved metadata meaning prevents a reader from
predicting whether the current loader accepts a pack.
