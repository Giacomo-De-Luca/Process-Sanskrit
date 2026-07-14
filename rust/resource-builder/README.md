# Native splitter resource builder

This crate converts deterministic, language-neutral splitter inputs into the
compact indexed resources consumed by `splitter-core`. It is an offline build
tool, not part of the installed Python runtime.

Run `uv run python tools/build_splitter_data.py` first. By default it reads the
packaged legacy assets without modifying them or requiring upstream packages,
and writes neutral inputs under `build/splitter-native-input/`. Use
`--upstream` only when deliberately regenerating both legacy and neutral inputs
from an installed `sanskrit_parser==0.2.6` and gensim. Then run:

```console
cargo run --release --locked -p process-sanskrit-resource-builder
```

The Python exporter and Rust builder share the default paths in
`rust/resources.toml`. Both accept `--config PATH`, which is useful for release
verification without changing the checked-in configuration. Relative paths
are resolved from the config file, not from the current directory.

## Neutral inputs

- `forms.txt`: one non-empty UTF-8 form per line, strictly sorted and unique.
- `sandhi-rules.jsonl`: strictly sorted `after` keys. Each record has
  `{"after": "...", "variants": [{"left": "...", "right": "..."}]}`;
  variants are also strictly sorted and unique. Rule annotations are omitted
  because splitting never observes them.
- `scorer.json`: schema version, a nonzero window, CBOW-mean flag, and the
  word2vec vocabulary in matrix-row order.
- NPY files: little-endian-neutral scorer matrices, Huffman data and offsets,
  and the exact 1,000-entry gensim-compatible log-sigmoid table. Matrix and
  table values must all be finite, and each vocabulary row must have equal
  Huffman code and point spans. The exporter loads the lookup values from the
  committed `process_sanskrit/splitter/data/log-table.npy` source rather than
  recomputing platform-dependent transcendental functions.
- `sentencepiece.model`: the authoritative tokenizer protobuf. The builder
  reads its piece order to create the tokenizer-ID-to-word2vec-row map.

The builder rejects unsorted, duplicated, malformed, or shape-inconsistent
inputs instead of normalizing them silently.

## Native outputs

- `forms.fst`: standard `fst::Set` bytes.
- `sandhi_after.fst`: standard `fst::Map` bytes mapping an after-string to its
  zero-based group number. Group numbers follow sorted-key order.
- `sandhi_variants.bin`: `PSSV0001`, a little-endian header, absolute group
  offsets, and length-prefixed UTF-8 `(left, right)` records.
- `scorer.bin`: `PSSC0001`, a fixed 128-byte little-endian header and 64-byte
  aligned arrays. It contains syn0/syn1, Huffman offsets/data, the
  SentencePiece-to-word2vec map, and the exact lookup table.
- `scorer_vocab.fst`: standard `fst::Map` bytes mapping every word2vec surface
  piece to its matrix row. Runtime scoring uses this map because an unknown
  SentencePiece ID can retain a surface string that exists in the DCS
  vocabulary. A mapped surface is scored even under the generic unknown ID;
  an absent surface is dropped, matching the Python reference.
- `sentencepiece.model`: byte-for-byte copy of the source model.
- `native-assets.json`: schema, generator, SHA-256 and byte size for every
  input/output, cardinalities, and scorer dimensions. It is installed last and
  serves as the commit marker for a complete asset set.

For each existing output, the builder first renames the old entry into its
same-filesystem staging directory and then renames the generated entry into the
vacant destination. This avoids mutating a file that another process already
has open and permits rollback if installing the replacement fails. A new open
during the brief two-rename gap can fail; initialized native splitters are
unaffected because they own their verified bytes. The manifest is installed
only after all other assets, so an interrupted build leaves a missing or
hash-mismatched set that loaders reject loudly. Rebuilding the same output
directory is supported.

Outputs are built in a staging directory under `splitter/data/native/` so the
legacy and native contracts cannot be confused. Loaders must verify the
manifest schema and hashes and must fail loudly on missing or incompatible
resources.

After regeneration, run the core resource/scorer contracts and the Python/Rust
differential suite described in
[`documentation/rust-splitter.md`](../../documentation/rust-splitter.md#correctness-and-parity-gates).
Do not treat a successful build alone as parity evidence.
