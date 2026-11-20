# Analysis functions

This package implements the stages behind the public `process()` and
`dict_search()` functions.

- `process.py`: input normalization, pre-split/wildcard handling, pipeline
  selection, morphology caching, and output rendering. `preprocess` is
  `to_iast` followed by `clean_iast`. It never cuts its input: a text over
  `max_length` is analysed word by word (`_process_each`).
- `wordCascade.py`: `WordCascade` is the single-word cascade (forms table,
  dictionary, `-tā`/`-tva` deriver) and `WordResolution` its result; `process()`
  and lemma mode both resolve a word through it. `read_double_j` cuts a `jj` as
  `j j` only in a word the cascade cannot resolve whole
  (`documentation/double-j.md`).
- `lemmaAnalysis.py`: `mode="lemmas"`. `LemmaAnalyzer` analyses a text token
  by token, `LemmaBuilder` turns an analysis into `LemmaPart`s and finishes
  them (`finished`: the typed headword, then the preverb compounds such as
  `pratī`), `ScannerPolicy` decides when to believe the fallback scanner, and
  `LemmaLexicon` is the builder's only window on the dictionaries.
- `rootAnyWord.py`: direct noun/verb morphology, final-sandhi retries, and prefix
  stripping with a request-local memo. `_prefix_entries` looks a stripped prefix
  up under the spelling the forms table knows (`vy` as `vi`), so it is not lost.
  `every_verb_root()` makes the lookups report every root of a shared verb form;
  lemma mode holds it, the other modes read the first root.
- `SQLiteFind.py`: indexed nominal/verbal lookups and paradigm access. A verb
  form of several roots gives one entry per root, the old last-match answer
  first; `paradigm_headword` names the headword (for a verb, the root) a
  paradigm row is filed under, for both lookups.
- `compoundAnalysis.py`: dictionary-based compound candidates and scanning.
  `root_compounds` reports the pieces it matched and the characters it skipped
  to an optional `ScanTrace`.
- `sandhiSplitter.py`: the statistical splitter adapter and `SplitResult`.
- `hybridSplitter.py`: `HybridAnalysis` and selection between statistical and
  dictionary-based compound results.
- `sandhiSplitScorer.py`: `SandhiSplitScorer` ranks candidate token sequences.
- `inflect.py`: `Inflector` resolves each token, tries prefix combinations,
  inflects fallback compound pieces, and retains completely unresolved tokens.
  It withholds one-letter verb roots the scanner cuts out of an unknown token
  (`_fragments`), which is what lets `process()` keep `i` and `ṛ` when it
  filters one-letter lemmas, and it reads the splitter's `niḥ` token as the
  prefix `nis` (`SPLIT_TOKEN_SPELLINGS`). `Inflector.resolve` returns one
  `TokenAnalysis` per split token (its entries, and whether they came from the
  lexicon or the scanner); `analyze` is its flattening, and `inflect()` the
  compatibility entry point.
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
[prefix segmentation](../../documentation/prefix-segmentation.md),
[one-letter roots](../../documentation/one-letter-roots.md),
[lemma mode](../../documentation/lemma-mode.md),
[long input](../../documentation/long-input.md), and the
[analysis audit](../../documentation/analysis-audit.md) for contracts, regression
cases, and known limitations.
