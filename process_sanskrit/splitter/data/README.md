# Splitter data

This directory contains the committed, database-free resources used by the
Python reference splitter and the native Rust backend.

- `forms.trie` is the Python reference validity set.
- `sandhi_rules.zip` contains the retained backward sandhi rule table.
- `sentencepiece.model` tokenizes candidate splits for scoring.
- `w2v.npz` stores the DCS CBOW matrices, Huffman paths, vocabulary, and scorer
  configuration used by the Python reference.
- `log-table.npy` is the canonical 1,000-entry float32 log-sigmoid lookup table.
  It is committed because recomputing transcendental functions produces
  different low-order bits across platforms.
- `native/` contains the manifest-verified FST and binary forms consumed by the
  Rust backend; its own README describes those structures.

The normal regeneration path reads these files and writes neutral intermediate
inputs under the ignored `build/splitter-native-input/` directory. Do not edit
the resources directly. See
[`documentation/rust-splitter.md`](../../../documentation/rust-splitter.md) for
the config-driven regeneration and parity procedure.
