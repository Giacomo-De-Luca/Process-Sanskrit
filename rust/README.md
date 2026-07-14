# Rust sandhi splitter

The Rust code is split into three crates so resource generation, the reusable
splitter, and Python packaging stay independent:

- `splitter-core/` owns immutable native-resource loading, inverse-sandhi graph
  construction, deterministic path search, and the DCS scorer. `Resources` is
  shared between threads; `Splitter` creates all graph and memoization state per
  request.
- `python/` provides the private PyO3 `_native` extension and the statically
  linked SentencePiece adapter. Transliteration and public `Split` objects
  remain in Python.
- `resource-builder/` compiles deterministic neutral exports into the indexed
  assets described in its own README.
- `vendor/sentencepiece/` records the pinned SentencePiece source and licence
  used by the extension build.

`resources.toml` is the checked-in configuration for resource generation. The
runtime asset manifest and binaries live under
`process_sanskrit/splitter/data/native/`; they are package data, not database
downloads.

The main Rust types are `splitter_core::Resources` (validated immutable data),
`splitter_core::Splitter` (thread-safe facade with request-local graph state),
`splitter_core::DcsScorer` (gensim-compatible inference),
`resource_builder::ResourceBuilder` (deterministic asset compiler), and the
private PyO3 `NativeSplitter` class. Python owns transliteration and public
result objects; Rust accepts and returns canonical SLP1.

The complete architecture, strict backend-selection contract, source build
requirements, regeneration workflow, and validation procedure are documented
in [`documentation/rust-splitter.md`](../documentation/rust-splitter.md).

Source builds require Rust 1.87 or newer and a C++17 compiler. The Python crate
uses PyO3's `abi3-py39` interface and statically compiles the vendored
SentencePiece v0.2.1 processor. Build with the locked dependency graph:

```bash
uv build --wheel
```

Run Rust checks from the repository root:

```bash
cargo test --workspace --locked
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```
