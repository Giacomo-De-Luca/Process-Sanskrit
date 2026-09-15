# Dictionary result components

Dictionary tables expose three fields: the IAST headword, an optional component
analysis, and the cleaned definition body.  The component field is useful for
`process(..., mode="parts")`, but it is not populated by every source.  In
particular, CPED contains many real entries whose `components` column is SQL
`NULL`.

## Normalized lookup contract

`functions/dictionaryLookup.py::multidict()` returns the first non-empty
component analysis supplied by any consulted dictionary.  If matching rows
exist but none supplies one, it returns the matched headword in that slot.  The
dictionary definitions are unchanged.  This normalization belongs at the
lookup boundary so downstream consumers never need to understand each source's
null conventions.

For a bare-word lookup, the resulting three-field entry therefore has this
shape:

```python
[headword, components_or_headword, definitions_by_dictionary]
```

Falling back to the headword means “one known part”; it does not invent an
etymological decomposition.

## Defensive `parts` formatting

`functions/cleanResults.py::roots_splitted()` consumes both three-field
dictionary entries and seven-field inflection-plus-dictionary entries.  It also
falls back to `entry[0]` if a caller supplies an absent or non-string component
value.  This second check protects the public output formatter even when an
entry was assembled outside `multidict()`.

The behavior is pinned in `tests/test_null_dictionary_components.py`, including
Yoga Sutra 53 (`samādhibhāvanārthaḥ kleśatanūkaraṇārthaś ca`), whose full-line
analysis reaches a CPED-only `samādhibhāvanā` row with a null component field.

## Inflection before dictionary lookup

`dict_search()` looks up the supplied headword; it does not stem inflected
strings. The morphology stage must therefore resolve compound pieces before
passing them to dictionary lookup. `Inflector` shares one fallback for ordinary
words and unresolved prefixes, calling `root_compounds(..., inflection=True)`.
Analysable pieces produce five-field
morphology entries, while dictionary-only pieces such as `niṣyanda` remain
strings and still receive their definitions.

For example, `sarvatragāminīpratipajjñānabalam` previously returned empty stubs
for `pratipaj` and `balam`, despite both resolving through `root_any_word()` in
isolation. The route to the missing entries was:

1. Preprocessing inserted a space into every `jj`, producing
   `sarvatragāminīpratipaj jñānabalam`. That artificial boundary prevented the
   statistical splitter from analysing the original compound together.
2. The morphology fallback split those chunks into bare strings, then passed
   `pratipaj` and `balam` directly to dictionary lookup without resolving their
   stems.

Enabling inflection in the compound fallback resolves the missing entries.
The reported compound returns `sarvatragāmin`, `pratipad`, `jñāna`, and `bala`,
with morphology and dictionary definitions. In particular, `balam` retains its
neuter nominative/accusative singular analysis under `bala` (alongside the
database's masculine accusative singular alternative). The existing `jj`
preprocessing is retained: removing it changes statistical candidate ranking
and regresses `tajjñānam` from `tad` + `jñāna` to `tajjña` + `ana`. That broader
segmentation issue needs separate work. The compound fallback also still
reduces `sarvatragāminī` to `sarvatragāmin` before morphology, so the first
component's feminine analysis is not preserved by this fix.

`tests/test_compound_inflection.py` covers the reported compound, fallback
morphology independently of statistical splitting, dictionary-only components,
standalone `balam`, and the existing `tajjñānam` analysis. The hybrid cache signature
was bumped to `hybrid-morphology-v5` for this fix; the subsequent audit uses
`hybrid-morphology-v6`. Old uninflected fallback results are not reused; see
[local-cache.md](local-cache.md).

## Missing definitions and request context

`consult_references()` always returns two fields for the caller to append:
`[components, definitions]`. An unresolved lookup returns `[word, [word]]`.
A dictionary-name key with an empty mapping, such as `{"mw": {}}`, is a miss.
This gives these stable public shapes:

- Bare-word miss: `[word, word, [word]]`.
- Morphology with missing definitions: the original five fields followed by
  `[word, [word]]`. The lemma, grammatical tags, paradigm, and matched surface
  remain available even though no definition exists.

SQL wildcard patterns (`_` and `%`) use the same miss shape and retain the
literal pattern. `process()` formats that result directly; retrying the same
pattern would recurse indefinitely. A trailing `*` retains its existing
fallback to ordinary processing of the word without the star.

Dictionary-name membership checks ignore case (`AP90` and `ap90` select the
same source), while result payload keys retain the requested spelling. The
existing fallback to other dictionaries remains available when none of the
requested sources indexes the word.

`process()` passes its dictionary selection and active session to
`clean_results()`. Both prefix rejoining and whole-word replacements reuse
that context, avoiding an implicit switch to the default dictionary.

See [analysis-audit.md](analysis-audit.md) for the regression cases, corpus
comparison, and remaining parsing issues.
