# Avagraha normalization

## The problem

Sanskrit elides a word-initial *a-* after a preceding *e* or *o* and marks the
elision with an **avagraha**. In Devanagari that is `ऽ` (U+093D); in romanised
text it is written with an apostrophe:

```
saḥ + anupalambhena  ->  so 'nupalambhena
```

But "an apostrophe" is not one character. Editions, OCR passes and PDF
copy-paste each reach for a different glyph, and the library used to recognise
only the ASCII `'` (U+0027). Every other glyph broke the word in one of two
ways, both of which end with the elided *a-* never being restored:

| Glyph | Codepoint | Unicode category | Old failure mode |
|---|---|---|---|
| `’` | U+2019 | Pf (punctuation) | deleted by the `\p{L}` filter in `preprocess` |
| `‘` | U+2018 | Pi (punctuation) | deleted |
| `` ` `` | U+0060 | Sk (symbol) | deleted |
| `´` | U+00B4 | Sk (symbol) | deleted |
| `′` | U+2032 | Po (punctuation) | deleted |
| `ʼ` | U+02BC | **Lm (letter)** | **survived** the filter and reached the splitter as a bogus consonant |
| `ʻ` | U+02BB | **Lm (letter)** | **survived** |

The U+02BC/U+02BB cases are the nastiest: because they are *letters*, they
passed the letter-only filter untouched.

Either way the token degraded from `'nupalambhena` to `nupalambhena`, which is
not a word, so the splitter shattered it into fragments:

```
'nupalambhena  ->  ['anupalambha']                  correct
’nupalambhena  ->  ['nu', 'pa', 'pa', 'lambha']     garbage (pre-fix)
```

Those fragments are not dictionary headwords, so they came back with no
grammatical analysis and no dictionary entries attached — **a result with an
empty payload is the visible symptom of this class of bug.**

A second, independent bug had the same effect: the restoration was
`if text[0] == "'"`, which only looks at index 0 of the **whole input string**.
An avagraha on any word other than the first was therefore never restored, so
passing a phrase rather than a single token corrupted the elided word.

## Position, not glyph, identifies an avagraha

The naive fix — fold every apostrophe-like glyph onto `'`, then treat any
apostrophe at the head of a word as an avagraha — **silently corrupts quoted
text**, because `‘` and `’` are ordinary quotation marks far more often than
they are avagrahas:

```
iti ‘yoga’ ucyate   ->   iti ayoga ucyate     # WRONG: ayoga ("non-union") is a real lemma
```

There is no exception and no empty result to warn anyone: the analysis just
comes back confidently wrong. So the restoration is keyed on **position**, which
is exactly what the sandhi rule constrains: an *a-* is elided only after a
preceding **e** or **o**, or at the head of a bare single-token input. An
apostrophe anywhere else is not an avagraha.

## The rules

`restore_avagraha()` in [`utils/transliterationUtils.py`](../process_sanskrit/utils/transliterationUtils.py)
applies, in order:

1. **Fold the glyphs** (`normalize_avagraha()`, a single `str.translate()` over
   `AVAGRAHA_VARIANTS`) so the rules below only ever match one character.
2. **Strip balanced quotes** — `^'(.+)'$`. A leading apostrophe is ambiguous
   (`'nupalambhena` is an avagraha, `'tapas'` is a quotation); what tells them
   apart is that a quotation gets *closed*.
3. **`o` + avagraha** — `(\p{L}*o)\s*'`. That *o* normally came from `-aḥ`/`-as`,
   so both it and the elision are undone at once (`so 'nupalambhena` →
   `saḥ anupalambhena`). The `\s*` is what makes the spaced form (how printed
   text sets it) behave like the unspaced form (what Devanagari transliterates
   to). **Exception:** the indeclinables in `O_NOT_FROM_VISARGA`
   (`o`, `aho`, `bho`, `ho`) have an *original* `-o`, so only the *a-* is
   restored: `aho 'yam` → `aho ayam`, never `ahaḥ ayam`.
4. **`e` + avagraha** — `(\p{L}*e)\s*'`. The *e* is original and stays:
   `te 'pi` → `te api`, `vane 'smin` → `vane asmin`.
5. **Token-initial** — `^'`. Covers bare single-word input, `process`'s
   documented use case: `'nupalambhena` → `anupalambhena`.
6. **Drop what is left.** Any apostrophe still standing is not an avagraha — a
   quote, an OCR artefact, an English genitive — and is removed rather than
   passed on. This is what neutralises U+02BC/U+02BB, which would otherwise
   survive the `\p{L}` filter as bogus consonants.

`preprocess()` calls `normalize_avagraha()` as its **first** step — before
`transliterate()`, so scheme detection never sees a stray glyph — and
`restore_avagraha()` after transliteration, once the text is in IAST and the
Devanagari `ऽ` has already become `'`.

## Result

```
'nupalambhena  ’nupalambhena  ʼnupalambhena  `nupalambhena  ‘nupalambhena  ´nupalambhena
    -> ['anupalambha']            (Inst. Sg., full inflection table + MW entry)
so ’nupalambhena / so’nupalambhena / सोऽनुपलम्भेन
    -> ['sa', 'tad', 'anupalambha']
te ’pi / te'pi          -> te api
aho ’yam                -> aho ayam
iti ‘yoga’ ucyate       -> iti yoga ucyate       (quotes dropped, not read as elision)
```

## Adding a glyph

Add it to `AVAGRAHA_VARIANTS`. The tests iterate over that table directly, so
coverage follows automatically — there is no second list to keep in sync.

## Tests

```bash
python -m unittest tests.test_avagraha    # 20 tests
```

They cover the glyph fold and the restoration rules in isolation, `preprocess`
output for every glyph (token-initial, mid-sentence, the *o*-contraction, the
indeclinable exception, the *e* half of the rule), idempotency (`preprocess`
re-runs on the wildcard and pre-split recursion paths), the Devanagari `ऽ` path,
negative cases (quoted and ordinary text must survive unmangled), and end-to-end
`process()` assertions that the resolved entry carries a full payload and that
none of the old fragments (`nu`, `pa`, `pala`, `lambha`) reappear.

## Known duplication (not fixed)

[`compoundAnalysis.py:268`](../process_sanskrit/functions/compoundAnalysis.py#L268)
carries its own `if word.startswith("'"): word = 'a' + word[1:]`. On the
`process()` path it is now dead — `preprocess` guarantees no word reaches it with
a leading apostrophe — but `root_compounds()` is importable and callable
directly, so it still functions as a defensive guard. It handles only the ASCII
glyph. Left in place deliberately; flagged here rather than silently deleted.

## Note on `references/app-root`

The legacy Flask parser has the same limitation — it keeps only
`c.isalpha() or c == "'"` and tests `text[0] == "'"` — so OCR text with a
typographic apostrophe degrades there in exactly the same way, which is the
origin of the payload-less `anupala` result seen in that version. That code is a
frozen reference and has not been changed.
