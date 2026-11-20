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

## Component marks in `parts` output

Monier-Williams' `components` field uses three marks that mean different things:

| Mark | Meaning | Example |
|---|---|---|
| `—` | boundary between the members of a compound | `citta—vṛtti` |
| `-` | prefix or suffix boundary inside one word | `ni-rodha`, `anu-gam` |
| `/`, `^` | accent, written after the accented vowel | `u/pa` (úpa), `agni/`, `akṣitavya^` |

The marks are read in one place, `ComponentMarks` (`utils/componentMarks.py`).
`ComponentMarks.parts`, which `roots_splitted()` uses, cuts on `—`, `-` and `,`
(which separates alternative accentuations of one headword, as in
`a/-ghnya,a-ghnya/`). `ComponentMarks.members`, which lemma mode uses, cuts on
`—` only.
It used to cut on `/` as well, so every headword accented anywhere but on its
last vowel came back in pieces — `upa` as `['u', 'pa']`, `tad` as `['ta', 'd']`,
`sarva` as `['sa', 'rva']`. A final accent hid the bug, because the piece after
it is empty and discarded, which is why `agni` always looked right. The accent
marks are not boundaries; the letters-only clean-up that follows the cut removes
them.

About 20,000 Monier-Williams headwords carry `/`. Across the Yoga Sūtra and the
1,500 held-out tokens of `tests/datasets/split_gold/`, 491 distinct headwords
changed, every one by having its pieces joined back together.

`parts` still reports a prefix boundary and a compound boundary the same way
(`nirodha` → `['ni', 'rodha']`); `mode="lemmas"` reports compound members only,
and only those of the reading most of a headword's entries give
(`ComponentMarks.prevailing_members`; see rule 9 of [lemma-mode.md](lemma-mode.md)).
`parts` does not vote: it reads one entry's `components`, so a rare homograph
can still decide it (`sukha` → `['su', 'kha']`, `ayoga` → `['ayo', 'ga']`); see
[known-errors.md](known-errors.md).

Pinned by `tests/test_parts_components.py`. `parts` is rendered after the cache
is read, so this fix needs no cache-version bump.

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
database's masculine accusative singular alternative). The `jj`
preprocessing was retained here, because removing it outright regresses
`tajjñānam` from `tad` + `jñāna` to `tajjña` + `ana`. It has since been made a
per-word decision that keeps that analysis; see [double-j.md](double-j.md). The compound fallback also still
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
