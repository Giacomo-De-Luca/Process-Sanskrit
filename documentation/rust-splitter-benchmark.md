# Rust splitter benchmark

The splitter port is measured separately from dictionary and morphology work by
[`scripts/benchmark_splitter.py`](../scripts/benchmark_splitter.py). The runner
calls `process_sanskrit.splitter.Parser.split()` directly, so the SQLite database
is not required and persistent analysis caches cannot influence the result.

## Running the benchmark

From the repository root:

```bash
uv run python scripts/benchmark_splitter.py \
  --config benchmarks/splitter-benchmark.json
```

The committed configuration runs `python` and `rust` in isolated workers, with
Python as the reference backend. It reads both compound datasets, which contain
700 records and 699 unique strings. It also adds the requested
`pṛthagjanatvamityevamādibhedasamādānāḥ` focus case, producing 701 loaded
records and 700 unique measured inputs. The focus case receives 20 extra warm
samples and is also used for seven fresh-process cold samples.

The complete report is written to
`build/benchmarks/splitter-benchmark.json`; a compact JSON summary is printed to
standard output. The output path is ignored by Git. Corpus-wide timings use two
warm observations per unique input (1,400 total samples), making the
determinism check meaningful for every case. The focus repetitions are reported
separately and are not mixed into the aggregate or bucket distributions.

## Validated release-mode comparison

The final uncontended same-run comparison on 2026-07-13 loaded the integrated
wheel's release-built `_native.abi3.so`, not the editable source-tree extension.
The safety-fixed loader retains verified owned bytes rather than file-backed
maps. The two backends achieved exact ordered parity on all 700 measured inputs:
candidate multisets, candidate order, and winners matched 700/700. Both reported
zero errors, 40 no-split cases, and no nondeterministic repeated result.

| Warm aggregate (1,400 samples) | Python | Rust | Speedup |
|---|---:|---:|---:|
| Mean | 21.158360 ms | 2.260618 ms | 9.36× |
| p50 | 11.035208 ms | 1.170605 ms | 9.43× |
| p95 | 70.283685 ms | 6.918915 ms | 10.16× |

Every code-point-length bucket was faster by mean latency:

| Code-point length | Mean speedup |
|---|---:|
| 0–9 | 7.76× |
| 10–19 | 11.00× |
| 20–39 | 8.41× |
| 40–79 | 9.47× |
| 80+ | 10.41× |

Bucket p95 speedups ranged from 9.98× to 15.63×. The requested complex focus
case had these 20-sample warm results:

| Focus case | Python | Rust | Speedup |
|---|---:|---:|---:|
| Mean | 57.072673 ms | 4.483233 ms | 12.73× |
| p50 | 55.863854 ms | 4.364479 ms | 12.80× |
| p95 | 67.824573 ms | 5.235865 ms | 12.95× |

Its winning split remained:

```text
pṛthak + jana + tvam + iti + evam + ādi + bheda + samā + dānāḥ
```

That is the configured `limit: 10` result. The retained Python implementation
first takes `limit` edge-weighted paths and only then applies full-sequence
reranking, so `limit` can intentionally change the winner. At `limit: 1`, both
backends instead return `… + bheda + samāḥ + da + anāḥ`; this legacy behavior
is pinned alongside the standard result.

Cold worker time includes import, parser/model initialization, and the split:

| Cold worker (7 processes) | Python | Rust | Speedup |
|---|---:|---:|---:|
| Mean | 678.147214 ms | 227.036976 ms | 2.99× |
| p50 | 677.548166 ms | 224.743500 ms | 3.01× |
| p95 | 832.899359 ms | 275.473838 ms | 3.02× |

This is not steady-state split latency, so compare it only between
same-harness, same-host, uncontended runs.

Warm peak RSS fell from 294.281250 MiB for Python to 97.109375 MiB for Rust, a
3.03× reduction. The local arm64 ABI3 wheel was 21.5 MiB, below the 50 MiB
packaging criterion. Exact parity, aggregate latency, focus-case latency,
cold-start, every length bucket, memory, and local wheel-size criteria pass.
Cross-platform wheel CI and installation smoke tests remain publication work,
but there is no remaining splitter parity or performance blocker.

## Comparing Python and Rust

The checked-in configuration already contains `backends: ["python", "rust"]`
and `reference_backend: "python"`; do not edit it for the standard comparison.
The scored differential now matches candidate multiset, winner, and order on
all 700 measured inputs, and the figures above come from the final
integrated release-wheel run. Each worker sets
`PROCESS_SANSKRIT_SPLITTER_BACKEND` before importing Process-Sanskrit. Cold
samples run in separate processes. Each backend's warm sweep uses one parser
instance after the configured warmup inputs.

The comparison reports candidate-multiset, order, and winner parity separately.
`behavioral_parity` requires the same candidate multiset (including
multiplicity) and the same winner. The command's correctness gate additionally
requires every backend's cold, warm, and focus runs to be error-free and
deterministic; matching exceptions therefore never count as parity. The
standard configuration also rejects a native binary built with debug
assertions.
`ordered_parity` and the combined `exact_parity` remain diagnostic because
tied-score traversal can reorder lower candidates without changing behavior. Set
`correctness.fail_on_mismatch` to `false` only for an exploratory run where a
complete report is more useful than a failing exit status.

That gate describes this benchmark's configured `score: true` mode. Explicit
unscored diagnostics use a narrower tie contract, documented in
[`rust-splitter.md`](rust-splitter.md#correctness-and-parity-gates), because the
Python reference has no stable ordering or winner inside an equal-weight cutoff.

Do not quote a speedup from an unvalidated exploratory report. A publishable
result must compare both backends in the same invocation, load the intended
release wheel rather than a source-tree debug extension, pass the correctness
gate, and manually meet the acceptance criteria in
[`rust-splitter.md`](rust-splitter.md#manual-performance-acceptance-criteria).

## Output contract

Timing distributions report sample count, mean, p50, p95, and maximum in
milliseconds. Percentiles use linear interpolation. Warm results are grouped in
three independent ways:

- configured Unicode-code-point length ranges;
- source categories such as `long` and `medium`;
- dataset/category pairs, so the two files' `long` groups remain distinguishable.

Memory output includes current RSS when the platform can provide it and peak RSS
from the operating system. Unsupported current-RSS measurements are `null`; they
do not abort a run. Cold results include both the worker's import/initialization/
split time and parent-observed process wall time.

Each worker records the measured module path and SHA-256, build profile,
Process-Sanskrit/NumPy/SentencePiece versions, and resource identity. The Rust
record embeds the verified native manifest; the Python record hashes its trie,
rules, scorer, and tokenizer assets. This makes an installed-wheel report
distinguishable from an editable or resource-modified run.

Every case records SHA-256 digests for:

- candidates in returned order;
- the sorted candidate multiset, retaining duplicates;
- the first candidate (the parser's winner).

`None`, an empty candidate list, and an exception have distinct digests.
Aggregate suite digests make reports easy to compare, while optional per-case
candidates explain a mismatch. Warnings, no-split cases, exceptions, and
nondeterministic repeated results are reported explicitly. Exceptions and
nondeterminism fail the correctness gate even when both backends fail in the
same way. Ordered digests remain in the report even though
tie-only order drift does not fail `behavioral_parity`.

## Tests

The harness's configuration, corpus deduplication, percentile, digest, and parity
contracts have a fast database-free suite:

```bash
uv run python -m unittest tests.test_splitter_benchmark
```

Do not put the full 700-case sweep in normal unit-test discovery. Run it for
benchmarks, release validation, and explicit Python/Rust comparisons.

The opt-in backend test provides a full-corpus candidate-multiset and winner
check without mixing performance into unit-test discovery:

```bash
PROCESS_SANSKRIT_FULL_NATIVE_PARITY=1 \
  uv run python -m unittest \
  tests.test_splitter_backends.NativeBackendTests.test_all_complex_compounds_match
```
