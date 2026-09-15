# Analysis functions

This package implements the stages behind the public `process()` and
`dict_search()` functions.

- `process.py`: input normalization, pre-split/wildcard handling, pipeline
  selection, morphology caching, and output rendering.
- `rootAnyWord.py`: direct noun/verb morphology, final-sandhi retries, and prefix
  stripping with a request-local memo.
- `SQLiteFind.py`: indexed nominal/verbal lookups and paradigm access.
- `compoundAnalysis.py`: dictionary-based compound candidates and scanning.
- `sandhiSplitter.py`: the statistical splitter adapter and `SplitResult`.
- `hybridSplitter.py`: `HybridAnalysis` and selection between statistical and
  dictionary-based compound results.
- `sandhiSplitScorer.py`: `SandhiSplitScorer` ranks candidate token sequences.
- `inflect.py`: `Inflector` resolves each token, tries prefix combinations,
  inflects fallback compound pieces, and retains completely unresolved tokens.
  `inflect()` is its compatibility entry point.
- `dictionaryLookup.py`: dictionary selection, lookup, and definition attachment.
- `cleanResults.py`: prefix rejoining and detailed, roots, or parts output.
- `taddhitaDerivation.py`: productive abstract-noun derivations after whole-word
  morphology and dictionary lookup fail.
- `model_inference.py` and `processBYT5.py`: optional BYT5 segmentation adapter.

## Data passed between stages

Split results are lists of strings. Morphology entries have five fields:
`[lemma, grammatical_type, tags, paradigm, matched_surface]`. Dictionary lookup
appends components and definitions, producing a seven-field entry. A bare
dictionary lookup has three fields: `[lemma, components, definitions]`.
Missing definitions use `[lemma]` in the payload slot and preserve any existing
morphology. `parts` output maps lemmas to component lists; `roots` output is a
list of lemmas, with tuples for alternative stems of the same matched form.

See [dictionary results](../../documentation/dictionary-results.md),
[prefix segmentation](../../documentation/prefix-segmentation.md), and the
[analysis audit](../../documentation/analysis-audit.md) for contracts, regression
cases, and known limitations.
