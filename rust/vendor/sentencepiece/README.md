# Vendored SentencePiece processor

This directory contains the C++ processor sources from Google SentencePiece
v0.2.1, commit `31646a467d2051eb904e0b45de3a73e91fe1c1e3`.

Only deterministic model loading and `Encode(..., pieces)` inference are exposed
to Rust. Training and command-line entry points are not linked into the Python
extension. The sources remain under their upstream Apache-2.0 license; see
`LICENSE` in this directory.

Bundled dependencies retain their own terms under `third_party/*/LICENSE`.
The inference build uses Abseil (Apache-2.0), protobuf-lite (BSD-3-Clause),
and darts-clone (BSD-3-Clause); their binary-distribution notices are copied
into `process_sanskrit/splitter/` for inclusion in wheels. The retained esaxx
training-only source is not compiled into the extension, but its license stays
beside it in source distributions.

Do not update these files independently of the tokenizer parity fixtures and
the native asset manifest. A tokenizer change can alter the winning split even
when the graph and word2vec weights are unchanged.
