# Explicit input scheme for `process`

## The problem

`process()` converts its input to IAST before anything else, and by default it
guesses the input scheme with `detect()` (`utils/detectTransliteration.py`).
The guess is reliable for scripts with their own Unicode block (Devanagari,
Kannada, Tamil...) and for IAST/Kolkata, which have distinctive diacritics. It
is not reliable for the ASCII schemes, because short words carry no
distinguishing characters:

- **Plain ASCII falls back to Harvard-Kyoto.** SLP1 `Darma` (*dharma*) becomes
  `ḍarma`, and ITRANS `chakra` (*cakra*) stays `chakra`.
- **WX has no rule in `detect()` at all.** WX `Xarma` (*dharma*) matches the
  SLP1 pattern and becomes `ḹarma`.

Measured on 5,000 distinct words from the Sanskrit Voyager books (shown in each
scheme, auto-detected, converted back to IAST), the lookups that came back
wrong were:

| Scheme | Wrong |
|---|---|
| Devanagari, Kannada, Kolkata | 0% |
| HK | 0.5% |
| ITRANS | 5% |
| SLP1 | 22% |
| WX | 82% |

## The argument

```python
process(text, *dict_names, ..., input_scheme=None)
```

- `input_scheme=None` (the default) keeps auto-detection, so existing callers
  see no change.
- Any sanscript scheme name is accepted, case-insensitively: `"SLP1"`, `"wx"`,
  `"Tamil"`, `"DEVANAGARI"`. Unknown names raise `ValueError`.
- `preprocess()` takes the same argument and passes it to `transliterate()`.

```python
process("Darma", mode="roots")                       # ['ḍa', 'ma']
process("Darma", input_scheme="SLP1", mode="roots")  # [('dharma', 'dharman')]
```

## Scheme validation

`transliterate()` resolves both scheme names through
`resolve_scheme()` (`utils/transliterationUtils.py`), which raises
`ValueError("Unknown transliteration scheme: ...")` for a name that is not a
sanscript scheme. Before this change an unknown name raised `AttributeError`
from a bare `getattr` on the sanscript module, and a module attribute that is
not a scheme (`"schemes"`) was passed through. The two `getattr` branches of
`transliterate()` were also collapsed into one path.

## The recursion contract

`handle_special_characters` re-enters `process()` for wildcard (`*`) misses
and for each segment of a pre-split compound (`-`, `+`). Every other option is
forwarded verbatim (see `pre-split-compounds.md`). `input_scheme` is the
exception: it is pinned to `"IAST"` in `forwarded`, because `text` has already
been converted by `preprocess()`. Forwarding the caller's scheme would read IAST
text in that scheme a second time. IAST `dharma` read as WX is `ḍharma`, so
`Xarma-kAya` would analyse as `ḍha + ma + kāya`.

## Cache

The analysis cache keys on the text *after* `preprocess()` (normalized IAST),
so the scheme is already reflected in the key. Default behaviour is unchanged,
so `ANALYSIS_ALGORITHM_VERSION` was not bumped.

## Limits

An explicit scheme fixes detection, not lossy scripts. Tamil writes *k/kh/g/gh*
with one letter and Bengali writes *b* and *v* with one letter, so a word
displayed in those scripts cannot be converted back reliably (88% and 5% wrong
in the measurement above, with or without the argument). A caller that
displays text in another script should send the word in its source scheme.

## Tests

`tests/test_input_scheme.py`. The two recursion tests build their expected
value from plain words, which never reach the recursion. A comparison against
the same compound in IAST would send both sides through the same recursion,
and a wrongly forwarded scheme would corrupt both identically and still pass.
This was confirmed by mutation: forwarding `"WX"` in `forwarded` fails exactly
those two tests.
