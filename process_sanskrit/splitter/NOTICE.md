# Vendored sandhi splitter

This package is a reduced copy of [kmadathil/sanskrit_parser](https://github.com/kmadathil/sanskrit_parser)
v0.2.6 (MIT — see `LICENSE.upstream`), by Karthik Madathil and contributors.
Process-Sanskrit uses it for sandhi splitting only.

## Why vendor

Process-Sanskrit called exactly one thing in that library: `Parser.split()`, from
`functions/sandhiSplitter.py`. Depending on the published package to get it cost:

- **`werkzeug==2.1.2`**, a hard pin from 2022 that drags Flask down to 2.1.3.
  It is needed only by upstream's `rest_api/` module, which we never import.
- **~42 transitive packages** (flask, flask-restx, flask-cors, jsonpickle, pydot,
  lxml, xlrd, pandas, tinydb, sanskrit_util, sqlalchemy…) for a call that needs
  about four.
- **88.9 MB of data**, most of it serving `parse()` and morphological tagging.
- **gensim** for scoring, which has no Python 3.14 wheel and so caps the whole
  project's Python version.

## What was taken

| module | from upstream | change |
|---|---|---|
| `sanskrit_base.py` | `base/sanskrit_base.py` | dropped `six` |
| `normalization.py` | `util/normalization.py` | verbatim |
| `sandhi.py` | `parser/sandhi.py` | imports only |
| `sandhi_analyzer.py` | `parser/sandhi_analyzer.py` | dropped tagging (`getMorphologicalTags`, `hasTag`, `tagSandhiGraph`) |
| `datastructures.py` | `parser/datastructures.py` | **`SandhiGraph` only**; dropped `VakyaGraph` (1,373 → 187 lines) |
| `lookup.py` | replaces `util/lexical_lookup*.py`, `util/inria*.py`, `util/sanskrit_data_wrapper.py` | trie instead of 2 sqlite DBs + ORM |
| `scorer.py` | replaces `util/lexical_scorer.py` | numpy instead of gensim |

Not vendored: `rest_api/`, `generator/`, `app_engine/`, `web/`, `cmd_line.py`,
`DhatuWrapper.py`, `disjoint_set.py`.

`VakyaGraph` is the load-bearing omission. It implements `Parser.parse()`
(dependency parsing), is the sole consumer of `DhatuWrapper` (tinydb) and
`DisjointSet`, and is the reason upstream needs the stems-and-tags pickle. We
never call `parse()` — morphology is `process_sanskrit.functions.inflect`'s job —
so dropping it removes most of the code *and* most of the data.

## Data

`tools/build_splitter_data.py` regenerates `data/` from upstream. 88.9 MB → 22.7 MB:

| file | size | replaces |
|---|---|---|
| `forms.trie` | 13.2 MB | `inria_forms_pos.db` (28 MB) + `inria_stems_tags_buf.pkl` (33 MB) + `sanskrit_data.db` (17 MB) |
| `w2v.npz` | 6.1 MB | `word2vec_model.dat` (7.6 MB, a gensim pickle) |
| `log-table.npy` | 4.1 KB | canonical float32 scorer lookup table |
| `sandhi_rules.zip` | 3.0 MB | verbatim |
| `sentencepiece.model` | 0.4 MB | verbatim |

Dropped entirely: `dhAtu-pATha-kRShNAchArya.json` (only `DhatuWrapper` read it).

Two subtleties, both covered by `tests/test_splitter_parity.py`:

1. **The validity oracle is precomputed, not reimplemented.** Upstream's
   `valid()` is partly *generative*: `SimpleAnalyzer._analyze_as_stem` strips a
   nominal ending and looks for a matching stem, so it accepts forms that appear
   in no table — about 40% of accepted forms on the Yoga Sutra. `forms.trie`
   contains that accept set enumerated in full (5.47M forms), not just the form
   tables. A trie of the tables alone would silently degrade splits.

2. **The scorer reproduces gensim's quirks on purpose.** It skips saturated terms
   (`|f| >= 6`) and indexes a 1000-entry sigmoid table with an *integer*-division
   scale (83, not 83.33). These are faithfully ported; correcting them shifts
   every score by ~0.07/token and would change which split wins. The table's
   exact float32 bytes are committed because recomputing `exp` and `log` changes
   low-order bits between operating systems.

## Equivalence

`tests/test_splitter_parity.py` pins this against upstream (skipped unless
`sanskrit-parser` is installed):

- `valid()` agrees on **100%** of queries a real corpus generates (24,309/24,309).
- Splits are **identical** on every word of the Yoga Sutra test corpus.
- Scores match gensim to within float32 rounding (max |Δ| ~1e-4).

## Native Rust implementation and SentencePiece

The native splitter in `rust/` is a Process-Sanskrit implementation of the
same split, validity, and scoring contract described above. The Python
reference remains available for differential testing, and the original
`sanskrit_parser` attribution and MIT license continue to apply to the
vendored reference implementation and its derived data.

The native Python extension statically compiles the deterministic inference
portion of [Google SentencePiece](https://github.com/google/sentencepiece)
v0.2.1, commit `31646a467d2051eb904e0b45de3a73e91fe1c1e3`. SentencePiece is
Copyright 2016 Google Inc. and is distributed under the Apache License 2.0;
see `LICENSE.sentencepiece`. Only model loading and deterministic piece-string
encoding are exposed by Process-Sanskrit; training and SentencePiece
command-line tools are not linked into the extension.

That SentencePiece source snapshot includes three third-party components used by
the compiled inference path:

- Abseil compatibility headers/source, Copyright 2016 Google Inc., under
  Apache-2.0 (`LICENSE.sentencepiece` contains the applicable license text).
- Protocol Buffers lite runtime, Copyright 2008 Google Inc., under the
  three-clause BSD license; see `LICENSE.protobuf-lite`.
- Darts-clone trie headers, Copyright 2008–2011 Susumu Yata, under the
  three-clause BSD license; see `LICENSE.darts-clone`.

Source distributions retain the complete vendored SentencePiece tree and its
original component license files. Binary wheels carry this notice and the
three license files named above next to the splitter package.
