# The sandhi splitter

Process-Sanskrit splits sandhi with a vendored, reduced copy of
[kmadathil/sanskrit_parser](https://github.com/kmadathil/sanskrit_parser) v0.2.6
(MIT), living in [`process_sanskrit/splitter/`](../process_sanskrit/splitter/).
Before v1.1.0 we depended on the published `sanskrit-parser` package instead.

This document explains what changed, why, and — most importantly — the two places
where the vendored copy could silently drift from the original and produce worse
splits without anything appearing to break.

## Why we stopped depending on the package

Process-Sanskrit called exactly one thing in `sanskrit_parser`: `Parser.split()`,
from a single file ([`functions/sandhiSplitter.py`](../process_sanskrit/functions/sandhiSplitter.py)).
We never called `Parser.parse()`, its dependency parser, because morphology is
handled by our own [`functions/inflect.py`](../process_sanskrit/functions/inflect.py).

Depending on the package to reach that one method cost:

- **`werkzeug==2.1.2`** — a hard pin from 2022, which drags Flask down to 2.1.3.
  It exists only for upstream's `rest_api/` module, which we never import.
- **~40 transitive packages** — flask, flask-restx, flask-cors, jsonpickle, pydot,
  lxml, xlrd, pandas, tinydb, sanskrit_util … for a call needing about four.
- **88.9 MB of data**, nearly all of it serving `parse()` and morphological tagging.
- **gensim**, for split scoring. gensim publishes no wheel for Python 3.14, so it
  capped the Python version of this entire project.

`Parser.parse()` is the load-bearing thing we don't use. It is the only consumer of
`DhatuWrapper` (tinydb) and of the 33 MB stems-and-tags pickle. Dropping it removes
most of the code *and* most of the data.

## What the splitter actually needs

Three things, and nothing else:

| need | how upstream does it | how we do it |
|---|---|---|
| sandhi rules | `sandhi_rules.zip` | same file, verbatim |
| a **validity oracle** — "is this string a real Sanskrit word form?" | 2 sqlite DBs + a 33 MB pickle, behind the `sanskrit_util` ORM | one marisa-trie |
| a **scorer** — to rank candidate splits | word2vec via gensim | the same word2vec, via numpy |

Splitting only ever asks *is this a word*. It never asks *what are its tags* — that
is `parse()`'s question. Upstream answers the first by way of the second, which is
why it carries so much machinery.

Result: **88.9 MB → 22.7 MB**, **44 → 27 packages**, and splitting is ~1.5× faster.

## The two things that could silently break

Both substitutions are exact, and both are the kind of thing that fails *quietly* —
the splitter keeps working and just gets worse. Both are pinned by
[`tests/test_splitter_parity.py`](../tests/test_splitter_parity.py).

### 1. The validity oracle is *enumerated*, not reimplemented

The obvious way to build the trie is to dump upstream's form tables. **That would be
wrong**, and it took measurement to notice.

Upstream's `valid()` is partly **generative**. `SimpleAnalyzer._analyze_as_stem`
strips a nominal ending off the candidate and looks for a matching stem, so it
accepts forms that appear in *no table at all*. On the Yoga Sutra, these are **~40%
of all accepted forms** (605 of 1,517). A trie built from the form tables alone
would reject them, and splits would quietly degrade.

So [`tools/build_splitter_data.py`](../tools/build_splitter_data.py) runs that rule
*forwards* over every stem × ending pair and bakes the entire accept set — 5.47M
forms — into `forms.trie`. It reproduces upstream's compatibility rules exactly,
including a genuine upstream bug (a tautological guard that drops every feminine
ending for participles) which is faithfully preserved, because "fixing" it would
change which splits are legal.

### 2. The scorer reproduces gensim's *quirks* on purpose

Scoring is a forward pass over pretrained weights — no training, no optimiser. gensim
was carried purely to unpickle a matrix and run ~20 lines of arithmetic.
`build_splitter_data.py` exports the weights to a plain `w2v.npz`, and
[`splitter/scorer.py`](../process_sanskrit/splitter/scorer.py) reimplements the pass
in numpy.

It is a **faithful port, not an improvement**. Two details in gensim's
`word2vec_inner.pyx` are not what a from-scratch implementation would produce:

- It **skips** any term with `|f| >= 6` rather than computing it, so very confident
  predictions contribute exactly `0`, not `~0`.
- The sigmoid is a 1000-entry lookup table whose index scale is
  `EXP_TABLE_SIZE / MAX_EXP / 2` evaluated with **integer** division: **83**, not
  83.33.

Writing the mathematically-correct version instead shifts every score by
~0.07 per token. That is more than enough to reorder near-tied candidates and change
which split wins. If you touch `scorer.py`, run the parity test.

## Guarantees, and how to check them

Verified against the real `sanskrit_parser` **with gensim enabled** (the correctly
configured upstream, not the degraded one):

- `valid()` agrees on **24,309 / 24,309** queries generated by a real corpus — 100%.
- Candidate split sets are **identical on all 572** Yoga Sutra words.
- `sandhi_splitter()` chooses the **same final split 572 / 572** — 100%.
- Scores match gensim to within float32 rounding (max |Δ| ≈ 1e-4).

The parity tests are skipped by default — the whole point is not to depend on
upstream — so to actually run them:

```bash
pip install sanskrit-parser==0.2.6 gensim sentencepiece
python -m unittest tests.test_splitter_parity
```

## Scoring is no longer optional

This is a **behaviour change**, and it is a fix.

Upstream degrades gracefully: with gensim absent it sets `gensim_enabled = False`,
logs a warning, and ranks splits by *length* instead of likelihood. Splitting keeps
working — it just gets quietly worse. Because scoring used to live behind our
`[gensim]` extra, a plain `pip install process-sanskrit` landed you in exactly that
state, and `process_sanskrit/__init__.py` silences the splitter's logger, so the
warning never reached anyone.

The degradation is not subtle:

| | best split for `astyuttarasyAMdiSi` |
|---|---|
| unscored (the old default) | `['astī', 'uttarasyām', 'di', 'ṣi']` ✗ |
| scored | `['asti', 'uttarasyām', 'di', 'ṣi']` ✓ |

`numpy` and `sentencepiece` are now hard dependencies and scoring is unconditional.
A scorer that cannot load is a **broken install, not a valid configuration**, so it
raises `RuntimeError` rather than falling back.

## Regenerating the data

`splitter/data/` is committed. Rebuild it only if upstream's data changes:

```bash
pip install sanskrit-parser==0.2.6 gensim sentencepiece marisa-trie
python tools/build_splitter_data.py
```

| file | size | replaces |
|---|---|---|
| `forms.trie` | 13.2 MB | `inria_forms_pos.db` (28 MB) + `inria_stems_tags_buf.pkl` (33 MB) + `sanskrit_data.db` (17 MB) |
| `w2v.npz` | 6.1 MB | `word2vec_model.dat` (7.6 MB, a gensim pickle) |
| `sandhi_rules.zip` | 3.0 MB | verbatim |
| `sentencepiece.model` | 0.4 MB | verbatim |

Then re-run the parity suite above. Do not hand-edit these files.

## Layout

```
process_sanskrit/splitter/
├── NOTICE.md          provenance, module-by-module
├── LICENSE.upstream   sanskrit_parser's MIT licence
├── api.py             Parser.split() — the only public entry point
├── sandhi_analyzer.py drives splitting (tagging removed)
├── datastructures.py  SandhiGraph  (VakyaGraph dropped: 1,373 → 187 lines)
├── lookup.py          the trie validity oracle
├── scorer.py          the numpy word2vec scorer
├── sandhi.py          sandhi rule application
├── sanskrit_base.py   SanskritString / transliteration
├── normalization.py   input normalisation
└── data/              forms.trie, w2v.npz, sandhi_rules.zip, sentencepiece.model
```

Upstream credit remains with Karthik Madathil and the sanskrit_parser contributors;
see `NOTICE.md` and `LICENSE.upstream`.
