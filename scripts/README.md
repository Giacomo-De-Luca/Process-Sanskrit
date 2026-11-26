# Maintenance and benchmark scripts

- `benchmark_optimizations.py` measures the complete processing pipeline,
  database calls, and persistent-cache behavior.
- `benchmark_splitter.py` is the database-free sandhi splitter benchmark. It
  reads a JSON configuration, starts isolated backend workers, and writes a
  machine-readable correctness and performance report.
- `assess_huet_merge.py` compares the local Huet exports with the installed
  SQLite morphology inventory without changing either source. Its
  `AssessmentConfiguration` holds paths, `Coverage` defines the spelling and
  set-comparison rules, and `HuetMergeAssessment` validates the exports and
  writes counts, examples, and source hashes to
  `build/huet-merge-assessment.json`. Run it with
  `uv run --no-sync python scripts/assess_huet_merge.py`; interpretation is in
  [`documentation/huet-merge-assessment.md`](../documentation/huet-merge-assessment.md).
- `evaluate_compound_members.py` compares ways of reading compound `members`
  for lemma search against the Digital Corpus of Sanskrit. It reads
  `benchmarks/compound-members.json`; its classes live in
  [`compound_members/`](compound_members/README.md), and the results are in
  [`documentation/compound-members-evaluation.md`](../documentation/compound-members-evaluation.md).


The splitter runner is organized around a few reusable responsibilities:

- `BenchmarkConfiguration` validates all corpus, backend, repetition, and
  output settings before work starts.
- `CorpusLoader` deduplicates inputs while retaining their source/category
  memberships.
- `WarmBenchmarkWorker` and `ColdBenchmarkWorker` own the two measurement
  lifecycles; `BackendCoordinator` isolates them in backend-specific processes.
- `CandidateSnapshot`, `Distribution`, and `CorrectnessComparator` define the
  stable result/digest and Python-versus-Rust parity contracts.
- `BackendRuntimeInspector` records the exact module, build profile, dependency
  versions, and resource hashes; `CorrectnessGate` rejects errors,
  nondeterminism, parity drift, and debug native binaries.
- `BenchmarkReport` owns the detailed JSON shape, while `BenchmarkSummary`
  keeps terminal output small.

Run scripts from the repository root through `uv run python`. Generated
benchmark output should go under the ignored `build/` directory unless a
specific historical report is deliberately being curated.

For example:

```bash
uv run python scripts/benchmark_splitter.py \
  --config benchmarks/splitter-benchmark.json
```

The backend worker contract and interpretation of parity/performance results
are documented in
[`documentation/rust-splitter-benchmark.md`](../documentation/rust-splitter-benchmark.md).
