# Benchmark configurations

This directory holds committed, reviewable configurations for performance and
correctness measurements. Generated reports belong under `build/benchmarks/`
and are intentionally not committed.

`splitter-benchmark.json` defines the splitter-only Python baseline and
Python/Rust comparison: both compound corpora, the requested complex focus
case, length buckets, repetitions, backend selection, and report destination.
The committed configuration already uses `["python", "rust"]` as `backends`
and keeps `python` as `reference_backend`. It observes every corpus case twice
for determinism and requires a release-built native module. A worker selects
its backend before importing the splitter; there is no runtime fallback between
measurements.

The runner and complete output schema are documented in
[`documentation/rust-splitter-benchmark.md`](../documentation/rust-splitter-benchmark.md).
