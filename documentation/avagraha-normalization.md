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

The U+02BC case is the nastiest: because it is a *letter*, it passed the
letter-only filter untouched.

Either way the token degraded from `'nupalambhena` to `nupalambhena`, which is
not a word, so the splitter shattered it into fragments:

```
'nupalambhena  ->  ['anupalambha']                  correct
’nupalambhena  ->  ['nu', 'pa', 'pa', 'lambha']     garbage (pre-fix)
```

Those fragments are not dictionary headwords, so they came back with no
grammatical analysis and no dictionary entries attached — a result with an
empty payload is the visible symptom of this class of bug.

A second, independent bug had the same effect: the restoration was
`if text[0] == "'"`, which only looks at index 0 of the **whole input string**.
An avagraha on any word other than the first was therefore never restored, so
passing a phrase rather than a single token corrupted the elided word.

## The fix

`normalize_avagraha()` in [`utils/transliterationUtils.py`](../process_sanskrit/utils/transliterationUtils.py)
folds every glyph in `AVAGRAHA_VARIANTS` onto the ASCII apostrophe with a single
`str.translate()`. `preprocess()` in
[`functions/process.py`](../process_sanskrit/functions/process.py) calls it as
its **first** step — before `transliterate()`, so scheme detection never sees a
stray glyph, and before the `\p{L}` filter, which is what used to destroy them.

Downstream of the normalisation, `preprocess` then:

1. `re.sub(r"o\s*'", "aḥ a", text)` — an avagraha after a word-final *o* means
   that *o* itself came from `-aḥ`/`-as`, so both are undone at once. The `\s*`
   is what lets the spaced form (`so 'nupalambhena`, how printed text normally
   sets it) behave the same as the unspaced form (`so'nupalambhena`, what
   Devanagari transliterates to).
2. `re.sub(r"(^|\s)'", r"\1a", text)` — restores the elided *a-* at the start of
   **any** word, not just the first.

Result:

```
'nupalambhena / ’nupalambhena / ʼnupalambhena / `nupalambhena / ‘nupalambhena
    -> ['anupalambha']                       (Inst. Sg., full inflection table + MW entry)
so ’nupalambhena / so’nupalambhena / सोऽनुपलम्भेन
    -> ['sa', 'tad', 'anupalambha']
```

## Adding a glyph

Add it to `AVAGRAHA_VARIANTS` and to `MARKS` in
[`tests/test_avagraha.py`](../tests/test_avagraha.py); the tests iterate over
that list, so coverage follows automatically.

## Tests

```bash
python -m unittest tests.test_avagraha    # 13 tests
```

They cover the normalisation utility in isolation, `preprocess` output for each
glyph (word-initial, mid-sentence, and the *o*-contraction), the Devanagari
`ऽ` path, and end-to-end `process()` assertions that the resolved entry carries
a full payload (grammar tags, inflection table, dictionary entries) and that
none of the old fragments (`nu`, `pa`, `pala`, `lambha`) reappear.

## Note on `references/app-root`

The legacy Flask parser has the same limitation — it keeps only
`c.isalpha() or c == "'"` and tests `text[0] == "'"` — so OCR text with a
typographic apostrophe degrades there in exactly the same way. That code is a
frozen reference and has not been changed.
