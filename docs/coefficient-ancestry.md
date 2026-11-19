# Coefficient ancestry

`openimcc.coefficient_ancestry()` returns the ancestry manifest for the
packaged default melt and redox packs, gas and condensate tables, liquid
standards registry, JANAF crystal resources, and review-γ activity and redox
fragments under `data/fragments/<element>/rows.json`. Pass an artifact's
identity, file digest, or runtime digest to return just that entry.

Each artifact records its identity, SHA-256 file digest, runtime-loaded digest,
rows, and the union of its source works. A row's `sources` list gives the
source-store work ID, citation text, and provenance role (`derived`, `fitted`,
`selected`, `anchor`, or `diagnostic`). The artifact-level `work_ids` field is
the union across its rows.

Roles follow explicit evidence: a source named by a fit is `fitted`, a source
named in a selection decision is `selected`, construction inputs are
`derived` or `anchor` as recorded, and validation-only datasets are
`diagnostic`. A selection-rationale field by itself does not make every source
selected, and held-out data are never counted as fit evidence.

The generator reads structured citation fields directly. Citations that appear
only in derivation or construction prose are curated by row in
`tools/coefficient-ancestry-prose-citations.json`; rows without an additional
prose citation carry a short `none` decision. This keeps prose parsing out of
regeneration and makes each added citation reviewable.

Every citation without a curated mapping to the recorded work-ID vocabulary
appears in `unmapped` with its row and reason. An artifact is `complete` only
when that list is empty. Unknown work IDs are never guessed; unresolved
citations make the affected artifact incomplete.

Fragment artifacts retain each row's citation, DOI, and locator in its source
record. For example, the MoO3 activity row maps O'Neill and Eggins (2002) to
`oneill-2002-feo-activity-coefficients-cmas`; citations absent from the pinned
source index remain in `unmapped` with a reason.

The vocabulary snapshot records its source commit and checksum. The sorted
work-ID list referenced by the manifest is in
`tests/data/coefficient-ancestry-vocabulary.json`. Regenerate the packaged
manifest with `python tools/build_coefficient_ancestry.py`; pass `--check` to
verify byte-identical regeneration.

To also verify the snapshot against the simulator's source index, pass its
repository path with `--vocabulary-from-repo`. The builder reads
`data/literature/INDEX.yaml` from the snapshot's recorded commit and checks
each work ID and recorded DOI.
This optional check requires the repository and PyYAML; plain `--check` does
not.

```sh
python tools/build_coefficient_ancestry.py --check --vocabulary-from-repo /path/to/simulator
```
