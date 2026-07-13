# Maintenance and benchmark scripts

- `benchmark_optimizations.py` measures the complete processing pipeline,
  database calls, and persistent-cache behavior.
- `benchmark_splitter.py` is the database-free sandhi splitter benchmark. It
  reads a JSON configuration, starts isolated backend workers, and writes a
  machine-readable correctness and performance report.

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
