# Native Rust sandhi splitter

Process-Sanskrit now has a native implementation of sandhi graph construction,
valid-form lookup, candidate search, and DCS scoring. The public
`process_sanskrit.splitter.Parser` API remains Python and selects the Rust
backend by default. The vendored Python splitter remains available as an
explicit reference backend during the migration.

> **Release validation status (2026-07-13): correctness and performance green.**
> The release-wheel differential has exact ordered parity on all
> 700 benchmark inputs and passes every latency, length-bucket, RSS, and local
> wheel-size criterion below. Cross-platform wheel CI and smoke tests remain required
> before publication, but there is no remaining splitter parity or performance
> blocker.

The history and invariants of the Python reference implementation are in
[`sandhi-splitter.md`](sandhi-splitter.md). Benchmark methodology and the
compound corpus are in
[`rust-splitter-benchmark.md`](rust-splitter-benchmark.md).

## Architecture and compatibility boundary

The extension boundary deliberately accepts canonical SLP1 only:

```text
Parser (Python)
  -> input normalization and transliteration to SLP1
  -> RustBackend or explicit PythonBackend
  -> candidate paths in SLP1
  -> public Python Split/SanskritObject values in the requested encoding
```

Python continues to own normalization, transliteration, pre-segmented input,
warnings, and the public `Split` representation. Consequently, `Parser`,
`Parser.split()`, `Split`, `process()`, and `sandhi_splitter()` retain their
existing signatures and output shapes. The private native methods reject
non-ASCII SLP1, while the public facade preserves the Python reference's
ordinary no-split warning for malformed non-ASCII text that survives public
normalization; pre-segmented malformed input also keeps its legacy failure
shape.

The native implementation is divided into three Rust crates:

| Component | Responsibility |
|---|---|
| `rust/splitter-core` | Immutable resources, validity lookup, inverse-sandhi graph construction, deterministic top-path search, and the DCS scorer |
| `rust/python` | Private PyO3 module `process_sanskrit.splitter._native`, GIL release, exception translation, and statically linked SentencePiece inference |
| `rust/resource-builder` | Offline conversion of neutral source data into deterministic native assets |

`process_sanskrit/splitter/backends.py` is the Python-side selection layer. It
shares one native splitter per process. Native resources are immutable and
shared through `Arc`; graph nodes, memoization, validity caches, and path state
are local to each request. Native splitting and scoring detach from the GIL, so
one parser can serve concurrent requests without sharing mutable graph state.

Resource immutability is enforced by ownership, not by assuming installed files
remain unchanged. Initialization reads each Rust lookup/scoring asset into
process-owned bytes and checks the manifest size and SHA-256 against those exact
bytes. The FST indexes query their owned buffers without copying keys, the flat
rule table retains its owned byte buffer, and scorer sections are decoded into
typed owned vectors. Changing or truncating a package file after initialization
therefore cannot alter a live splitter or invalidate memory it is reading. The
SentencePiece C++ adapter receives the already verified model bytes and loads
them as a serialized proto; it does not reopen the package path or expose
file-backed memory to Rust.

The scorer intentionally preserves the original gensim behavior rather than
using mathematically cleaner approximations. In particular, it uses scalar
`float32` accumulation, drops OOV SentencePiece pieces, skips saturated Huffman
terms, and uses the integer-scaled 1,000-entry log-sigmoid table. Changing any
of these details can change the winning split. It consumes SentencePiece's
surface-piece strings, not only numeric IDs: an unknown ID can still carry a
runtime surface such as `J` or `Q` that exists in the DCS vocabulary. Runtime
looks that surface up in `scorer_vocab.fst`; a matching surface is scored even
when its SentencePiece ID is the generic unknown ID, while a surface absent
from the DCS vocabulary is dropped exactly as in the Python reference.

## Backend selection and fail-loud behavior

Set `PROCESS_SANSKRIT_SPLITTER_BACKEND` before constructing the first `Parser`:

| Value | Behavior |
|---|---|
| unset or `rust` | Use the native backend |
| `python` | Use the retained Python reference backend |
| anything else, including `auto` | Raise `ValueError` |

Selection is captured on first use and remains process-wide. Changing the
environment variable later does not affect existing or subsequently created
parsers; start a new process to change backend.

Examples:

```bash
# Production/default path.
uv run python -c "from process_sanskrit.splitter import Parser; print(Parser(output_encoding='iast').split('yogaścittavṛttinirodhaḥ', limit=1))"

# Explicit reference path for differential testing.
PROCESS_SANSKRIT_SPLITTER_BACKEND=python \
  uv run python -c "from process_sanskrit.splitter import Parser; print(Parser(output_encoding='iast').split('yogaścittavṛttinirodhaḥ', limit=1))"
```

There is deliberately no automatic native-to-Python fallback:

- an absent or unloadable `_native` module raises `RuntimeError`;
- a missing, corrupt, or schema-incompatible native asset raises `RuntimeError`;
- a missing scorer or tokenizer is fatal instead of degrading to length-based
  ranking;
- invalid SLP1 passed directly to the private native API raises `ValueError`;
- a pathological sandhi graph deeper than 512 recursive states raises
  `RuntimeError` instead of risking a native stack overflow.

The explicit `python` setting is a reference and diagnostic mode, not an
installation-recovery mechanism. The extension is marked required in
`pyproject.toml`, so a source installation must successfully build it even if
the Python backend will be selected at runtime. Release wheels are currently
configured to contain both implementations and both asset sets.

The pipeline wrapper in `functions/sandhiSplitter.py` follows the same
fail-loud contract: parser exceptions propagate instead of being converted to
the original unsplit text. Its `attempts=1` branch now selects the first item
from the parser's returned list; the previous `next(list)` call raised a
`TypeError`. Focused tests pin both observable behaviors.

## Building and installing

Supported source builds require:

- CPython 3.9 or newer;
- Rust 1.83 or newer;
- a C++17 compiler for the statically linked SentencePiece processor;
- the ordinary Python build tooling declared in `pyproject.toml`.

For an editable development installation:

```bash
uv pip install -e .
uv run python -c "from process_sanskrit.splitter import Parser; assert Parser().split('astyuttarasyAMdiSi', limit=1)"
```

The compiled `_native` library is a generated build artifact. Do not commit a
local `.so`, `.dylib`, or `.pyd` from an editable build.

Run the Rust checks from the repository root:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

Build a local wheel through the declared setuptools/setuptools-rust backend:

```bash
uv build --wheel
```

The validated local artifacts used Rust 1.83: all 32 Rust tests passed, and the
final arm64 wheel was 21.5 MiB with 75 files. The source distribution was
23.2 MiB with 397 entries. Licence, archive, and dynamic-library audits passed.
These are local release checks, not a statement that artifacts have been
published.

Only release-mode native code is valid for performance measurement. A benchmark
must load an extension built with Cargo's release profile; an editable debug
extension may establish correctness, but its timings must be discarded.
Benchmarking the installed wheel is the least ambiguous route.

The packaging configuration targets one `cp39-abi3` extension for manylinux
x86_64, macOS x86_64/arm64, and Windows x86_64. musllinux and PyPy are excluded.
`cibuildwheel` settings are present in `pyproject.toml`, but a configured matrix
is not proof that release wheels have been built, audited, and smoke-tested.

SentencePiece v0.2.1 at commit
`31646a467d2051eb904e0b45de3a73e91fe1c1e3` is compiled statically; a system
SentencePiece library is neither required nor used. Its Apache-2.0 license and
the licenses for the vendored protobuf-lite and darts-clone sources must ship
with source and binary distributions alongside the existing upstream splitter
notices.

### Rust dependency notices

`THIRD_PARTY_NOTICES.md` is the checked-in license bundle for third-party Cargo
packages used to compile the native extension. The current bundle covers 36
locked packages: 32 runtime or code-generation dependencies and four build-only
dependencies. It follows the union of target-specific normal and build edges
reachable from `process-sanskrit-python`; development-only dependencies, local
workspace crates, and the resource-builder-only graph are excluded. Each
inventory row records its Cargo license expression and references the complete,
deduplicated license or notice text taken from the downloaded crate.

After changing `Cargo.lock`, regenerate and verify the bundle with:

```bash
cargo fetch --locked
uv run --no-project python tools/generate_rust_third_party_notices.py
uv run --no-project python tools/generate_rust_third_party_notices.py --check
```

The generator resolves metadata with `--locked --offline` and reads license
files from Cargo's local source cache. Only the optional preparatory
`cargo fetch` can use the network. `pyproject.toml` declares the bundle as a
PEP 639 license file, and `MANIFEST.in` includes it in source distributions.
Wheel smoke CI checks its `License-File` metadata, archive presence, lockfile
digest, and representative runtime/build dependency rows. The vendored C++ and
Python splitter notices remain separately maintained in
`process_sanskrit/splitter/NOTICE.md` and its adjacent `LICENSE.*` files.

## Native resource contract and regeneration

Native assets are package data under
`process_sanskrit/splitter/data/native/`. They are unrelated to the 583 MB
lexical SQLite database and are not installed or updated by
`update-ps-database`.

| Asset | Contents |
|---|---|
| `forms.fst` | Process-owned FST set of 5,474,379 accepted forms |
| `sandhi_after.fst` | Map from matched surface strings to rule-group IDs |
| `sandhi_variants.bin` | 350,937 deterministic left/right variants for 197,792 rule keys |
| `scorer.bin` | CBOW matrices, Huffman data, compatibility tokenizer map, and the exact log table |
| `scorer_vocab.fst` | Map from all 7,995 DCS surface pieces to scorer matrix rows |
| `sentencepiece.model` | Tokenizer model used to build the scorer mapping |
| `native-assets.json` | Schema, formats, byte sizes, SHA-256 hashes, counts, and scorer dimensions |

At first native initialization the loader checks manifest schema, expected
format, size, and SHA-256 for every required asset. Verification and runtime
parsing use the same owned buffers, avoiding a check-then-reopen window. This
eager ownership increases private resident memory and cold initialization work
relative to file-backed mappings; release benchmarks must include these loader
costs. The builder installs the manifest last, making it the completion marker
for a generated asset set.

Regeneration has two config-driven stages. The normal path derives neutral
inputs from the committed Python assets without rewriting them:

```bash
uv run python tools/build_splitter_data.py
cargo run --release --locked -p process-sanskrit-resource-builder
```

Both commands read `rust/resources.toml`. Intermediate text, JSON, and NPY files
go under ignored `build/splitter-native-input/`; native outputs replace the
contents of `process_sanskrit/splitter/data/native/` through a staging
directory. The builder rejects unsorted, duplicate, malformed, or
shape-inconsistent inputs rather than silently normalizing them.

Only when deliberately refreshing from `sanskrit_parser==0.2.6` should the
legacy assets also be regenerated:

```bash
uv pip install sanskrit-parser==0.2.6 gensim sentencepiece marisa-trie
uv run python tools/build_splitter_data.py --upstream
cargo run --release --locked -p process-sanskrit-resource-builder
```

Do not hand-edit generated assets or update SentencePiece independently of the
tokenizer/scorer parity fixtures. See `rust/resource-builder/README.md` for the
binary layouts and neutral-input contract.

## Correctness and parity gates

Correctness precedes performance. The release candidate must pass:

```bash
cargo test --workspace --locked
uv run python -m unittest tests.test_splitter_backends tests.test_splitter_parity
PROCESS_SANSKRIT_FULL_NATIVE_PARITY=1 \
  uv run python -m unittest tests.test_splitter_backends.NativeBackendTests.test_all_complex_compounds_match
```

The gates cover:

- public signatures, encodings, `None`, warnings, pre-segmented input, scoring,
  and limit behavior;
- in the default scored mode, candidate multiplicity/content and an identical
  winning split;
- native/Python scorer agreement within `1e-3`, including OOV and saturated
  terms;
- deterministic native ties, duplicate paths, two-stage ranking, and complete
  candidate/output-shape parity for `limit > 1000`;
- concurrent use of one native parser;
- fatal behavior for corrupt native assets;
- controlled failure beyond the 512-state traversal limit, while the deepest
  packaged benchmark compound remains accepted;
- all 699 unique compounds in the two benchmark datasets.

The default benchmark has `score: true`. Its behavioral gate requires the same
candidate multiset and winner. It permits lower candidates in a genuinely tied
score group to have a different order, while recording `ordered_parity`,
`exact_parity`, and ordered digests as diagnostics. Winner or candidate-content
drift remains fatal in scored mode.

Large-limit requests have a separate upstream contract. For either scoring
setting, `limit > 1000` enumerates every simple path and sorts the returned paths
only by token count. With `score=True`, graph scoring still runs first, so a
missing or broken scorer remains fatal, but those scores do not determine the
all-path return order. Tests therefore require the complete candidate multiset
and public `Split` output shape to agree, but do not prescribe an order or
winner for this branch.

For finite `score=False` requests, NetworkX traversal in the Python reference
selects among equal-weight paths using process-dependent hash and insertion
order; eight seed probes produced seven different limited candidate
fingerprints and changed the shortest tied winner. Rust deliberately orders
these ties by token count and then lexical SLP1 path. If a finite limit cuts
through an equal-length tie group, tests require the same candidate count and
path-length cutoff, but do not prescribe Python's arbitrary subset, ordering,
or tie winner.

Validated result: the release-mode scored harness has candidate-multiset,
winner, and ordered parity on all 700 measured inputs (the 699 unique dataset
compounds plus the requested focus case). In other words,
`candidate_multiset_matches`, `winner_matches`, and
`ordered_candidate_matches` are each 700/700, with no mismatches.

## Manual performance acceptance criteria

Use the committed splitter-only harness so dictionary, morphology, and cache
work do not contaminate measurements:

```bash
uv run python scripts/benchmark_splitter.py \
  --config benchmarks/splitter-benchmark.json
```

The committed configuration already runs `python` and `rust` with Python as the
reference. Both backends run in the same invocation on the same host. The
report is written under ignored `build/benchmarks/` and includes candidate
digests, errors, warm/cold latency, length/category buckets, and RSS. The
runner machine-enforces parity, error-free deterministic execution, and a
release-built native binary. The performance criteria below are reviewed from
the report; they are not an automatic CLI pass/fail gate.

The release acceptance thresholds are:

- at least 3× aggregate warm speedup and 2× p95 speedup over the untouched
  Python baseline;
- at least 1.75× aggregate speedup over the shared-scorer Python reference;
- no input-length bucket more than 10% slower;
- no more than 10% regression in cold initialization or peak RSS;
- compressed wheel below 50 MB.

All thresholds pass in the validated arm64 macOS release-mode run:

| Measure | Python | Rust | Result |
|---|---:|---:|---:|
| Warm mean | 21.158360 ms | 2.260618 ms | 9.36× faster |
| Warm p50 | 11.035208 ms | 1.170605 ms | 9.43× faster |
| Warm p95 | 70.283685 ms | 6.918915 ms | 10.16× faster |
| Requested compound, warm p50 | 55.863854 ms | 4.364479 ms | 12.80× faster |
| Requested compound, cold-worker p50 | 677.548166 ms | 224.743500 ms | 3.01× faster |
| Peak RSS | 294.281250 MiB | 97.109375 MiB | 3.03× lower |

Every length bucket is faster by mean latency: 7.76× for 0–9 characters,
11.00× for 10–19, 8.41× for 20–39, 9.47× for 40–79, and 10.41× for 80+.
The same-run Python backend already includes the shared-scorer optimization, so
the 9.36× aggregate result clears the 1.75× optimized-reference criterion.
The local 21.5 MiB `cp39-abi3` arm64 wheel clears the 50 MB compressed-size
criterion.

See [`rust-splitter-benchmark.md`](rust-splitter-benchmark.md) for the complete
measurement contract. Future numbers are comparable only when both backends
run together with the same corpus and an installed release-built extension.

## Migration and Python reference lifetime

The current staged package keeps:

- the Rust backend as the default;
- the Python backend only by explicit selection;
- native and legacy assets together;
- Python-only splitter dependencies needed by the reference backend.

There is no automatic fallback and no scheduled removal date yet. Remove the
Python runtime backend, its assets, or dependencies from production wheels only
after at least one stable release cycle with all parity, wheel, performance, and
field-validation gates green. Retain reference source and differential fixtures
in the repository even after they leave production wheels.
