# Test suite

Process-Sanskrit uses `unittest`. Run Python through `uv` from the repository
root, for example:

```bash
uv run python -m unittest tests.test_splitter_parity
uv run python -m unittest tests.test_splitter_backends
uv run python -m unittest tests.test_splitter_benchmark
uv run python -m unittest tests.test_publish_workflow
uv run python -m unittest tests.test_optimizations
uv run python -m unittest tests.test_reference_comparison
uv run python -m unittest tests.test_analysis_cache tests.test_database_lifecycle
uv run python -m unittest tests.test_prefix_merge
uv run python -m unittest tests.test_prefix_segmentation
uv run python -m unittest tests.test_one_letter_roots
uv run python -m unittest tests.test_lemma_builder tests.test_word_cascade tests.test_lemma_mode
uv run python -m unittest tests.test_shared_verb_forms
uv run python -m unittest tests.test_double_j
uv run python -m unittest tests.test_preverb_sandhi tests.test_compound_verbs
uv run python -m unittest tests.test_final_av
uv run python -m unittest tests.test_long_input
uv run python -m unittest tests.test_parts_components
uv run python -m unittest tests.test_null_dictionary_components
uv run python -m unittest tests.test_compound_inflection
uv run python -m unittest tests.test_analysis_result_contracts
uv run python -m unittest tests.test_verb_persons
uv run python -m unittest tests.test_input_scheme
uv run python -m unittest tests.test_update_database tests.test_database_path_configuration
uv run python -m unittest tests.test_huet_assessment
```

Most pipeline tests require `process_sanskrit/resources/SQliteDB.sqlite`. The
always-on half of `test_splitter_parity.py` uses only packaged splitter assets;
its upstream comparison classes are skipped unless `sanskrit-parser==0.2.6`
and gensim are installed.

`test_splitter_backends.py` exercises strict Rust/Python selection and uses the
native extension when one has been built. Its full comparison of the 699 unique
complex compounds is opt-in so ordinary discovery remains fast:

```bash
PROCESS_SANSKRIT_FULL_NATIVE_PARITY=1 \
  uv run python -m unittest \
  tests.test_splitter_backends.NativeBackendTests.test_all_complex_compounds_match
```

## Structure

- `test_*.py` contains automated regression suites. In particular,
  `VerbPersonTests` checks all nine person/number cells without the lexicon,
  while `PublicVerbPersonTests` checks the installed lexicon through `process()`
  across present, imperfect, future, and Devanagari input; the cache suite also
  checks eviction of v6 results containing reversed person tags;
  `HuetAssessmentTests` checks the read-only inventory assessment, normalization,
  deduplication, and canonical lemma matching;
  `SplitterTests` pins public candidates, scoring, resource failures, and
  concurrency; `SandhiSplitterWrapperTests` pins the pipeline wrapper;
  `NativeBackendTests` compares Rust with the Python oracle and verifies native
  asset/concurrency behavior; `BenchmarkConfigurationTests` pins the
  config-driven runner; `NativePublishWorkflowTests` prevents the release
  workflow from losing its native build, smoke, collection, or authentication
  gates; `SamPrefixRejoinTests` pins the prefix re-joining block in
  `clean_results` in both directions (an unattested join such as `samupekṣa`
  stays split, an attested one such as `samādhi` still collapses), with
  `DictSearchStubShapeTests` pinning the `dict_search` miss/hit shapes that
  guard depends on — see `documentation/prefix-rejoin.md`;
  `PrefixIsItsOwnWordTests` and its collateral-regression classes pin prefix
  segmentation and matched-surface handling, `SandhiSpellingOfPrefixTests`
  that a prefix stripped in a sandhi spelling (`vy`, `praty`, `ud`) keeps an
  entry of its own, and `NirPrefixTests` that `nir-` is read as `nis` from the
  split token without pre-empting the `-tā`/`-tva` deriver — see
  `documentation/prefix-segmentation.md`;
  `RootSurvivesTheSplitPathTests` and `NoiseIsStillFilteredTests` pin both
  clauses of the one-letter filter (the verb roots `i` and `ṛ` are kept when
  they resolve a whole token, scanner fragments and one-letter nominal lemmas
  are not), `ConsequencesOfKeepingTheRootTests` what follows from keeping the
  root, and `CacheRoundTripTests` that a cache hit keeps it too, with
  `VerbalParadigmTests` and `OneLetterRootPredicateTests` covering the
  predicates without the lexicon — see `documentation/one-letter-roots.md`;
  `ScannerPolicyTests`, `LookupPartTests`, `ResolutionPartTests` and
  `TokenPartTests` pin the lemma-mode rules on hand-built entries with a fake
  lexicon, and `TypedHeadwordTests`, `JoinedPreverbKeyTests`,
  `FinishedWordTests` and `SuppletiveRootTests` of `test_lemma_builder.py` the
  typed headword, the joined preverb keys, their order and `SUPPLETIVE_ROOTS`;
  `CascadeStageTests` pins which stage of the single-word cascade
  resolves a word, and `TracesOnlyObserveTests` that the analysis functions
  return the same with and without a trace; `SpecificationExampleTests`,
  `PreverbTests`, `UnknownWordTests`, `WordBoundaryTests` and the other classes
  of `test_lemma_mode.py` pin the `mode="lemmas"` contract through `process()`
  (`TypedHeadwordTests` with a regression pin over every root of the verb
  table, `CommonFormsReachTheirRootTests` a floor for sixteen common forms of
  four roots), and `LexiconTests` pins `LemmaLexicon` against the installed
  dictionaries — see `documentation/lemma-mode.md`;
  `test_preverb_sandhi.py` pins the sandhi `PreverbSandhi` applies between a
  preverb and a stem, and `test_compound_verbs.py` pins `CompoundVerbIndex`'s
  spelling rules and cache on an in-memory `mw` table (neither needs the
  database);
  `test_double_j.py` pins that a word keeps its `jj` when it resolves whole and
  is cut as `j j` otherwise (`lajjā`, `tajjapas`), in every mode — see
  `documentation/double-j.md`;
  `LookupTests` of `test_shared_verb_forms.py` pin that the lookup returns
  every root of a shared verb form, the old answer first, each tagged from its
  own paradigm; `ReadingPolicyTests` that `every_verb_root` decides how many
  are reported; `LemmaModeTests` that lemma mode reads them all, and
  `OtherModesTests` that `roots`, `parts` and detailed output are unchanged —
  see `documentation/lemma-mode.md` ("Verb forms of several roots");
  `LongTextFallsBackToWordByWordTests`, `MaxLengthMeaningTests` and
  `WildcardsStayOutOfRunningTextTests` pin that no mode truncates its input and
  that the word-by-word fallback does not turn stray pattern characters into
  dictionary patterns — see `documentation/long-input.md`;
  `RestoreByTokenTests` and `ClosingQuoteTests` (in `test_avagraha.py`) pin
  avagraha restoration over a text already cut into tokens, and that a closing
  quotation mark is never an avagraha — see
  `documentation/avagraha-normalization.md`;
  `ComponentMarkTests` and `PublicPartsTests` pin which marks of a dictionary
  `components` field are boundaries in `parts` output (accent marks are not) —
  see `documentation/dictionary-results.md`;
  `NullDictionaryComponentsTests` pins the headword fallback for dictionary
  rows without component metadata, while `YogaSutraNullComponentsTests` pins
  Yoga Sutra 53 end to end — see
  `documentation/dictionary-results.md`;
  `CompoundInflectionTests` pins stem analysis and dictionary payloads for
  compound fallback pieces, including preservation of the existing
  `tajjñānam` analysis;
  `AnalysisResultContractTests` covers missing definitions, unresolved tokens,
  `api` morphology, wildcard misses, `parts` mappings, and dictionary/session
  forwarding — see `documentation/analysis-audit.md`;
  `AtomicDownloadTests`, `ExternalDatabaseUpdateTests`, `PackagedUpdateTests`
  and `LibraryEntryPointTests` pin the lexicon installer: no partial
  `SQliteDB.sqlite` survives an interrupted download or rebuild,
  `process_sanskrit.update_database()` raises `DatabaseUpdateError` instead of
  exiting, and the console script wraps it — see
  `documentation/database-setup.md`; and
  `UpstreamParityTests` is the optional upstream reference comparison.
- `datasets/` contains reusable Sanskrit corpora and compound fixtures. Dataset
  modules expose lists/dictionaries consumed by benchmarks; the two JSON files
  contain compound records used by reference comparisons.
- `results/` contains historical benchmark and comparison output. These files
  are reports, not automatically authoritative golden fixtures unless a test
  explicitly reads one.
- `runBenchmarks.py`, `gretilTesting.py`, `processCompounds.py`, `ysTest.py`, and
  `BYT5test.py` are manual or long-running evaluation scripts rather than unit
  test discovery targets.
- `Removable/` contains legacy diagnostic scripts retained for reference.

Keep small deterministic assertions in `test_*.py`. Put shared corpora in
`datasets/`, and write newly generated reports to `results/` only when the
associated documentation identifies how they were produced.
