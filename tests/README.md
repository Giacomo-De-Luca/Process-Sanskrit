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
uv run python -m unittest tests.test_null_dictionary_components
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
  `NullDictionaryComponentsTests` pins the headword fallback for dictionary
  rows without component metadata, including Yoga Sutra 53 — see
  `documentation/dictionary-results.md`; and
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
