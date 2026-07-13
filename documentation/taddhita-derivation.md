# Productive `-tā` / `-tva` derivation

## The problem

The taddhita (secondary) suffixes **`-tā`** (feminine, ā-stem) and **`-tva`**
(neuter, a-stem) build an abstract noun out of any nominal stem:

| base | | derivative |
|---|---|---|
| `śūnya` "empty" | → | `śūnyatā` "emptiness" |
| `niṣyanda` "flowing" | → | `niṣyandatā` "the state of flowing" |

They are **productive**: a text may coin one from any stem it likes, so no
lexicon can list them exhaustively. The database lexicalises the common ones —
roughly 2050 in `-tā` and 2340 in `-tva`, each with a hyphenated stem and a full
paradigm (`śūnya-tā`, model `f_A`) — but a coined derivative resolves nowhere.

Before this feature, such a word fell through every lookup to the compound
splitter, which cut it in two and then analysed the orphaned suffix as a verb:

```python
process("niṣyandatā", mode="roots")
['niṣyanda', ('tā', 'tṛ', 'tan')]     # lemma destroyed; tṛ / tan invented
```

Two errors compound: the lemma is lost, and a verbal root that is not in the word
is asserted to be. Now:

```python
process("niṣyandatā", mode="roots")
['niṣyandatā']
```

## How it works

`functions/taddhitaDerivation.py` exposes a `TaddhitaDeriver` (and the shared
`taddhita_deriver` instance). Given an unresolved word it:

1. strips a `-tā`/`-tva` paradigm ending (longest first);
2. checks the remaining **base** is a real nominal stem — a dictionary headword,
   or a stem that inflects in `lgtab2`. Both halves are needed: `niṣyanda` is a
   headword with no inflection table of its own, while many stems inflect without
   heading a dictionary entry;
3. regenerates the full 24-form paradigm and reads the case/number tags off it,
   exactly as a stored lookup would.

The endings are **not hardcoded**. They are lifted from an exemplar row the
database already stores (`śūnya-tā` / `f_A`, `a-karuṇa-tva` / `n_a`) by stripping
that exemplar's own base off each of its forms. So the forms generated for a
coined derivative are by construction the forms the database would have stored
for it. `tests/test_taddhita_derivation.py` pins this: it regenerates several
lexicalised tables from their bases and asserts they match the stored tables form
for form.

A missing exemplar row is **fatal**, not a silent skip — following the precedent
of commit `5bac481` (missing split scorer). Degrading silently here would look
exactly like the bug the module fixes.

## Ordering is load-bearing

The deriver is consulted **only after every other layer has missed**: after
`root_any_word` (inflection tables, sandhi variants, prefix stripping) *and*
after the whole-word dictionary lookup. It sits immediately before the splitter
in `process()`.

That ordering is what makes it safe. Words that merely *end* in `-tā` — feminine
past participles (`kṛtā`, `gatā`, `jātā`), instrumentals of `-vat` stems
(`bhagavatā`, `dhīmatā`), agent nouns (`pitā`, `kartā`), plain words (`sītā`,
`latā`) — all resolve earlier and never reach this code.

**Do not move the derivation into `root_any_word`.** That runs *before* the
dictionary, so a manufactured analysis would outrank an attested word: `vārtā`
(f., "news") would come back as `vār` + `tā` instead of as itself.

## The `-atā` ambiguity

`-atā` has a second, equally valid parse. A consonant stem in `-at` (typically a
present participle) makes its Inst. Sg. in `-atā`, its Gen. Pl. in `-atām` and
its Dat. Sg. in `-ate`, all colliding head-on with base(-a) + `tā`:

```
gacchatā  =  gacchat + ā    Inst. Sg., "by the one going"     <- gacchat is attested
gacchatā  =  gaccha  + tā   abstract noun, "going-ness"       <- spurious
```

Nothing in the surface form separates them, so the tie is broken on **evidence**:
if the competing `-at` stem is itself attested, it owns the word and no derivation
is offered. `gacchat`, `paśyat`, `jayat` and `vidvat` are all dictionary
headwords, so those keep their participle reading.

This is deliberately conservative. It also declines the genuinely ambiguous cases
(`jayatā` — "by the conquering one", or "victoriousness") rather than overriding
an attested stem with a manufactured one. Only `-tā` is guarded; `-tva` has no
consonant-stem competitor.

## Glossing

`niṣyandatā` heads no dictionary entry, but `niṣyanda` does — and that is where
its meaning lives. So `process()` glosses the **base**, not the derivative, and
appends the payload to the derived entry. The result has the same 7-slot shape
`dict_search` produces for an attested word, so nothing downstream can tell a
reconstructed analysis from a stored one.

This asymmetry (lemma from the derivation, gloss from the base) is why
`dict_search` cannot do the job itself and the entry is assembled in `process()`.

## Superseded

`functions/handleTvaEndings.py` attempted this for `-tva` alone. It was dead code
— not importable (its imports were missing, so it raised `NameError`) and its
call site in `rootAnyWord.py` was commented out (commit `d330e47`, "temporary
removed tva endings"). It hand-listed 14 endings, handled only bases in `-a`, and
returned a marker string rather than a stem. It was deleted; this module covers
every ending it enumerated, on any base, with a real paradigm.

## Known gap

`vidvatā` (Inst. Sg. of `vidvas`/`vidvat`) is correctly *declined* by the deriver
— `vidvat` is attested, so the guard fires — but nothing downstream then supplies
the participle reading, and the splitter still returns `['vid', 'vata']`. That is
a pre-existing splitter limitation, untouched by this change.
