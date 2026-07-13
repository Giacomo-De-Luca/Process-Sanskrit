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

