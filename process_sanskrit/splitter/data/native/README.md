# Native sandhi splitter assets

These generated files are the immutable runtime data for the Rust splitter.
They coexist with the Python differential-reference assets in the parent
directory; both formats are deliberately retained.

- `forms.fst` is the exact valid-form set.
- `sandhi_after.fst` maps matched surface strings to groups in
  `sandhi_variants.bin`.
- `sandhi_variants.bin` stores sorted, length-prefixed left/right rule pairs.
- `scorer.bin` stores the DCS CBOW matrices, Huffman paths, tokenizer mapping,
  and gensim-compatible log-sigmoid table.
- `scorer_vocab.fst` maps all DCS word2vec surface pieces to their scorer rows.
  This preserves pieces whose runtime surface is meaningful even when
  SentencePiece assigns the generic unknown ID. A mapped surface is scored;
  an absent surface is dropped, matching the Python reference. The unknown ID
  alone therefore never decides whether a piece is out of vocabulary.
- `sentencepiece.model` is a byte-identical copy of the tokenizer model used to
  build the scorer mapping.
- `native-assets.json` records the format schema, cardinalities, dimensions,
  source hashes, and output hashes. Runtime initialization must reject a
  missing or incompatible manifest rather than falling back silently.

Do not edit these files manually. Regenerate them with the config-driven steps
in `rust/resource-builder/README.md`. Intermediate text/NPY files belong under
the ignored `build/splitter-native-input/` directory and are not committed.

At runtime, the process-wide native splitter checks the manifest schema,
declared format, byte size, and SHA-256 hash for every required asset. The exact
verified bytes become process-owned runtime storage: FSTs index those buffers
directly, rules retain their flat byte table, and scorer sections are decoded
into typed vectors. Later changes to package files cannot affect a live
splitter. The SentencePiece C++ adapter loads its model from the same verified
bytes as a serialized proto; it does not reopen the package path or expose
file-backed memory to Rust. Any
verification failure is fatal; the backend never falls back to the Python asset
set. See
[`documentation/rust-splitter.md`](../../../../documentation/rust-splitter.md)
for the runtime and regeneration contracts.
