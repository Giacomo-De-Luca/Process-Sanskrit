# Long input: `process()` never cuts its text

## What was wrong

`preprocess` truncated the text at `max_length` — 100 characters by default — and
said nothing. Everything past the limit was simply absent from the result:

```
sūtras 1–15 of the Yoga Sūtra, one call (519 characters, 40 words)
    mode="roots"   ->  12 lemmas          # 28 words never analysed
```

A single word was cut too: a 112-character compound was analysed as its first
100 characters. In the WSMP evaluation 4 of 1,000 sentences were affected
([sentence-vs-word-evaluation.md](sentence-vs-word-evaluation.md)).

The limit bought no speed: the splitter takes about 2 ms on those 519
characters. What it did hide is one failure, described under "A word the
splitter cannot handle" below.

## What happens now

Nothing is truncated, in any mode.

- **A text longer than `max_length` is analysed word by word**, and the results
  are combined: lists are concatenated, and the mappings of `mode="parts"` are
  merged. This is the same combination a pre-split compound has always used, and
  both go through one helper, `_process_each` (`functions/process.py`).
- **A single word is analysed whole**, whatever its length. It has no smaller
  unit to fall back to.
- **`mode="lemmas"`** is always word by word and ignores `max_length`.

```
sūtras 1–15, one call
    mode="roots"   ->  75 lemmas
    mode="lemmas"  ->  40 items, one per word
```

## What `max_length` means now

It no longer limits *what* is analysed, only *how*: it is the longest text handed
to the splitter as one unit. A text within the limit is analysed as a sentence,
as before; a longer one is analysed word by word.

**The length is measured before cleaning**, on the transliterated text, which is
what the old cut measured. Cleaning lengthens a text (`ko 'pi` becomes `kaḥ api`),
and measuring the cleaned text instead would send a 99-character input word by
word that has always been analysed whole.

Word by word is not the same analysis as a sentence, so a text just over the
limit reads differently from one just under it. Each word goes through the
single-word path, which:

- does not fold a repeated word (`tatra tatra`);
- also offers the whole-word dictionary headword.

```
process("tatra tatra", mode="roots")                   ->  ['tatra']
process("tatra tatra", mode="roots", max_length=10)    ->  ['tatra', 'tatra']
process("tatra gacchati", mode="roots")                ->  ['tatra', 'gam']
process("tatra gacchati", mode="roots", max_length=5)  ->  ['tatra', 'gacchati', 'gam']
```

**Pattern characters are removed from the words of a long text.** `*`, `_` and
`%` are dictionary patterns in a one-word query and nothing in running text,
where the sentence path never treated them as patterns. Analysed one word at a
time they would become patterns, and a stray `%` returns every headword.

`preprocess` lost its `max_length` parameter along with the cut, and its
remaining options are keyword-only, so an old positional `preprocess(text, 150)`
fails instead of being read as something else. It is now the composition of
`to_iast` (the text `max_length` measures) and `clean_iast`.

## What changed for existing callers

Checked on 2,378 inputs (the Yoga Sūtra line by line and word by word, 1,500
held-out corpus tokens, a list of prefixed verbs, and three passages):

- every input of 100 characters or fewer gives identical `roots`, `parts` and
  detailed output;
- the six longer inputs each gain the words they were losing — two single
  compounds of 109 and 112 characters, one 124-character sūtra, and the three
  passages (sūtras 21–60, 1,705 characters, go from 13 lemmas to 254).

No cache-version bump was needed: a text over the limit is no longer looked up as
a whole, so no stored row can answer for the new behaviour.

## A word the splitter cannot handle

The splitter raises on a single token of about a thousand characters
(`RuntimeError: the sandhi graph exceeds the safe traversal depth`). Truncation
used to make such a token unreachable; now it reaches the splitter.

- `mode="lemmas"` catches this: the word is analysed unsplit and still comes
  back, so one bad token does not take the rest of the text with it.
- The other modes raise. The failure is loud, and it needs a single unbroken
  token of that length; text without spaces of up to 2,000 characters of real
  Sanskrit was analysed without error.

## Not covered

The optional BYT5 adapter (`functions/model_inference.py`) still truncates at the
model's token limit (`truncation=True`). It is outside `process()`, and removing
that cut needs chunking designed for the model.

## Invariants to preserve

- **`preprocess` returns the whole cleaned text.** A length limit belongs in the
  choice of *how* to analyse, never in what is kept.
- **`max_length` is compared with the text before cleaning.**
- **Every recursive `process()` call takes its options from
  `_forwarded_options`**, which pins `input_scheme` to `"IAST"`. The pre-split
  branch and the long-text fallback both use it; an option left out there is
  silently demoted to its default.

Pinned by `tests/test_long_input.py`.
