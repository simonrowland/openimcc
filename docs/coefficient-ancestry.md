# Coefficient ancestry

`openimcc.coefficient_ancestry()` returns the ancestry manifest for the
packaged default melt and redox packs, gas and condensate tables, liquid
standards registry, and JANAF crystal resources. Pass an artifact's identity,
file digest, or runtime digest to return just that entry.

Each artifact records its identity, SHA-256 file digest, runtime-loaded digest,
rows, and the union of its source works. A row's `sources` list gives the
source-store work ID, citation text, and provenance role (`derived`, `fitted`,
`selected`, `anchor`, or `diagnostic`). The artifact-level `work_ids` field is
the union across its rows.

Every citation without a curated mapping to the recorded work-ID vocabulary
appears in `unmapped` with its row and reason. An artifact is `complete` only
when that list is empty. Unknown work IDs are never guessed; unresolved
citations make the affected artifact incomplete.

The vocabulary snapshot records its source commit and checksum. The sorted
work-ID list used by the focused validation is in
`tests/data/coefficient-ancestry-vocabulary.json`. Regenerate the packaged
manifest with `python tools/build_coefficient_ancestry.py`; pass `--check` to
verify byte-identical regeneration.
