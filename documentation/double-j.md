# `jj`: the word's own, or the sandhi of t/d + j

A `jj` in a word is one of two things:

- a double consonant of the word itself: `lajjā`, `rajju`, `ujjvala`,
  `sajjana`, the roots `lajj`, `majj`, `bhrajj`, `ujjh`;
- the sandhi of a final `t`/`d` before `j`: `tat` + `japaḥ` → `tajjapaḥ`,
  `tat` + `jñānam` → `tajjñānam`. The splitter does not undo it.

## Before

`clean_iast` (`functions/process.py`) cut every `jj` as `j j`, in every mode and
before any lookup. That was right for `tajjapas` (`taj` reaches `tad` through
the final-`j` sandhi variant in `root_any_word`) and wrong for every word
spelled with `jj`: `lajjā` → `laj` + `ja`, `majjati` → `asmad` + `ja` + `ti`,
`rajjau sarpaḥ` → `raj` + `ja` + `sarpa`.

## Now

`WordCascade.read_double_j` (`functions/wordCascade.py`) decides it word by
word. A word, or a segment of a pre-split word (`lajjā-vismaya`), keeps its
`jj` when the single-word cascade resolves it as it stands: the forms table,
the whole-word dictionary lookup, or the `-tā`/`-tva` deriver. A word that does
not resolve is cut exactly as before. A one-word pattern (`_`, `%`, `*`) is a
headword query and keeps its `jj`. In running text a pattern character makes no
pattern, so the word is read like any other (`tajjapas% tadartha` is still cut).
Lemma mode's `*` query that finds no headword falls back to the word and
decides then. Text without `jj` is returned unread: no lookup, no change.

The decision is taken on a memo of its own, not the request's. The request memo
hands a word it already holds back as a copy whose tags are lists, where a fresh
lookup gives tuples. Deciding on it made `process("lajjā")` report `[['Nom',
'Sg']]` where `process("rāmaḥ")` reports `[('Nom', 'Sg')]`.

It runs where the old cut ran, on the cleaned text: in `process()` right after
`clean_iast`, and in lemma mode in `LemmaAnalyzer._words`. So both the
single-word path and the sentence path see the decision, and every later stage
sees the same text it saw before for a word that does not resolve.

## What changed

Measured over 794 inputs: every verb form of the verb table spelled with `jj`,
300 noun forms of the forms table, the `jj` words of the Yoga Sūtra, and six
sentences. `roots`, `parts` and lemma output changed for 783 of them; none
raises.

| input | before | after |
|---|---|---|
| `lajjā` | `laj`, `ja` | `lajjā` |
| `lajjate` | `laj`, `ja`, `ta`… | `lajj` |
| `rajjau sarpaḥ` | `raj`, `ja`, `sarpa` | `rajju`, `sarpa` |
| `ujjvalam`, `sajjanaḥ` | `ut`+`jvala`, `sat`+`jana` | `ujjvala`, `sajjana` |
| `abhṛjjata` | `bhṛt`, `ja`, `ta` | `bhrajj` |
| `lajj`, `ujjh` | `laj`, `ut` | `lajj`, `ujjh` |
| `tajjaḥ` | `tad`, `ja` | `tajja` |
| `tajjayāt` | `tad`, `jaya` | `tajjaya` (members `taj`, `jaya`) |
| `tajjapas`, `tajjñānam` | `tad` + `japa`, `tad` + `jñāna` | unchanged |

The 11 unchanged inputs are the ones that do not resolve whole: compound tokens
(`tajjapas`, `lajjāvismayamanmathaiḥ`, `sarvatragāminīpratipajjñānabalam`), and
four precatives the verb table files with a final `-ur` (`bhṛjjyāsur`) while
the cascade looks them up as `-uḥ`.

The last rows show the trade. A `tad` + `j` compound that a dictionary lists as
a word (`tajja` "born of that", `tajjaya` "mastery of that") now comes back as
that word, not as `tad` + its second member. That is the cascade's rule for
every word, not only for `jj`. The members of `tajjaya` keep the sandhi spelling
of the dictionary's components field (`taj`, not `tad`).

## No cache bump

`ANALYSIS_ALGORITHM_VERSION` stays as it is. The persistent caches key on the
text after this decision. A word that does not resolve is cut as before, so it
reaches the same key and the same computation. A word that keeps its `jj`
reaches a key that never existed, since no text with `jj` ever reached the
caches before. No stored row can be stale.

## Known limits

- A compound token that contains a `jj` word is still cut, because the token
  does not resolve whole: `lajjāvismayamanmathaiḥ` → `laj`, `ja`, … . Doing
  better needs the splitter to rank both readings.
- The check costs one cascade lookup per word with `jj`, outside the request
  memo. A `jj` word is therefore looked up twice on the single-word path (to
  decide, then to analyse), and three times on the word-by-word path for long
  text, where the recursive `process()` decides again.

Pinned by `tests/test_double_j.py`, by `TypedHeadwordTests` in
`tests/test_lemma_mode.py` (the `jj` roots come back as themselves), and by
`test_existing_jj_sandhi_analysis_survives` in
`tests/test_compound_inflection.py`.
