# One-letter verb roots on the split path

`process()` drops every one-letter lemma on its split path. Two verb roots are
one letter long, `i` (to go) and `ṛ`, so a sentence used to lose its verb:

```
process("eti")            -> ['eti', 'i']     # single-word path: no filter
process("tatra eti")      -> ['tatra']        # split path: `i` filtered away
process("paryeti")        -> ['pari']
```

`i` is among the commonest roots, alone and after a preverb (`paryeti`,
`adhyeti`, `atyeti`), so this was not a corner case.

## Why the filter exists

It guards against the compound scanner. `root_compounds`
(`functions/compoundAnalysis.py`) walks a token the lexicon does not know one
character at a time, and every single vowel is a dictionary headword, so its
output is full of one-letter "words". The filter removes them — but it tested the
*lemma's* length, which cannot tell that noise from a real root.

## The two clauses

A one-letter lemma is kept only when both hold:

1. **It is a verbal analysis.** The forms table also files `i` and `ṛ` as nouns
   (the letter names), and those are still noise. The test is the table's
   layout, not a label: a verbal paradigm is three persons by three numbers
   (`utils/paradigm.py::is_verbal_paradigm`). In the current database the only
   one-letter verb roots are `i` and `ṛ`.
2. **It resolved a whole token.** The scanner cuts an unknown token wherever
   something matches, and `i` has short forms that match the tail of almost any
   causative: `arcayet` is not in the lexicon, and the scanner leaves `ayet`, an
   optative of `i`. A root the scanner cut out of a larger token is withheld.

The clauses live where the knowledge is:

- `Inflector._fragments` (`functions/inflect.py`) withholds one-letter verb
  roots from the scanner's pieces. It is the only place that knows whether an
  entry came from a whole-token lookup or from the scanner.
- The filter in `process()` (`functions/process.py`) keeps a one-letter lemma
  when it is a verbal analysis. By then the scanner's ones are already gone.

Both use one predicate, `utils/paradigm.py::is_one_letter_root`.

Both halves are needed. Relaxing the filter alone readmits the scanner's
fragments; withholding fragments alone keeps no real root, because the filter
still drops them.

## Effect

On the 1,500 held-out tokens in `tests/datasets/split_gold/`, the old filter
dropped ten verbal analyses with a one-letter root:

| | kept now | still dropped |
|---|---|---|
| real (`anveti`, `avehi`, `īyāt`) | 3 | — |
| the `ita` inside `adhītavantau`, reached through the scanner | — | 1 |
| scanner fragments (`ayet`, `ara`, `iyāt`, `īya`) | — | 5 |
| a mis-split whole token (`dhyāyīta` cut as `dhyāyī` + `ita`) | 1 | — |

The 88 one-letter nominal lemmas the filter removed on the same tokens (`a`,
`ā`, `ḷ`, `e`, ...) are still removed.

```
process("tatra eti")       -> ['tatra', 'i']
process("tatra ṛcchati")   -> ['tatra', 'ṛ']
process("paryeti")         -> ['pari', 'i']
process("prasādamanveti")  -> ['prasāda', 'anu', 'i']
process("śārṅgamarcayet")  -> ['śārṅga', 'arc']      # no `i` from `ayet`
```

## What else changes

Three consequences follow from keeping the root. None is a separate rule.

- **A word in a sentence reads as it does alone.** A token that is also a form
  of `i` or `ṛ` regains that reading on the split path, which the single-word
  path always gave it: `tatra itaḥ` → `['tatra', ('ita', 'i')]`, as `itaḥ` →
  `[('ita', 'i')]`. Other common forms gain a reading the same way (`agāt`,
  `etā`, `are`).
- **The kept root takes part in the prefix re-join.** `tatra sameti` →
  `['tatra', 'sami']`, the attested headword for `sam` + `i`, exactly as `sam` +
  `gam` gives `saṃgam` (see [prefix-rejoin.md](prefix-rejoin.md)).
- **A token left with no pieces is reported unresolved.** When the scanner finds
  nothing in an unknown token but a withheld root, the token comes back as
  itself: `tatra ñiyāt` → `['tatra', 'ñiyāt']`. It used to vanish from the
  result, because the root was emitted and then filtered.

## Not fixed by this

A verb that never reaches the filter is untouched: `anveti` alone is returned as
a Pali-dictionary headword with no root, and `sa udeti` comes back as `sa`,
`uda`, `ud`, `iti`. Both are separate problems (the headword early return and
the analysis of `udeti`).

## Invariants to preserve

- **The filter and `Inflector._fragments` change together.** See above.
- **Test the table, not the lemma.** An exception keyed on the strings `i` and
  `ṛ` readmits the letter-name nouns.
- The single-word path has no filter at all, which is why `process("eti")`
  always worked. A word that falls through to the split path (`paryeti`) is
  filtered like a sentence.

Pinned by `tests/test_one_letter_roots.py`. The change alters cached morphology,
so `ANALYSIS_ALGORITHM_VERSION` was bumped for it (`hybrid-morphology-v9`).
