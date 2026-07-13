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

## `-te` is generated but never licenses a derivation

A second collision is worse than the first, and is handled by
`NON_LICENSING_ENDINGS`.

`-te` is the ā-stem's Voc. Sg. and all three of its dual cells. It is *also* the
ending of every ātmanepada and every passive 3rd sg. in the language — `śocate`
"he grieves", `dṛśyate` "it is seen", `kriyate` "it is done". Such verbs routinely
miss the verb tables, and would then reach the deriver, which would strip the
`-te`, find the perfectly real nominal stem underneath, and bury a correct verbal
root under a fabricated noun:

```
śocate     deriver OFF: ['śuc', ...]      <- correct root
           deriver ON : ['śocatā']        <- fabricated: "grievingness (Voc. Sg.)"
```

That is precisely the damage this module exists to undo, running backwards. So
`-te` **is** generated into the paradigm (the table stays complete and still
matches the stored one form for form) but is **never accepted as the ending that
identifies a derivative**. The trade is lopsided: a vocative or dual of an
abstract noun ("O emptiness!", "two emptinesses") is vanishingly rare, while
ātmanepada and passive 3rd sg. are among the commonest forms in Sanskrit.

`-tām` is deliberately **not** de-licensed, even though it is also the 3rd person
imperative (`kurutām` "let him do"). It is the ā-stem's Acc. Sg. (`śūnyatām`),
which is common in the philosophical corpus, and the imperatives all resolve in
the verb tables long before the deriver is consulted — measured at **zero** changed
analyses across an imperative probe set, against two real gains (`niṣyandatām`,
`nirākāṅkṣatām`). De-licensing it would cost real coverage to prevent a collision
that never actually arrives.

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

## Known gaps

**The deriver never fires inside a compound.** It is called only on the
single-word path in `process()`, before the splitter, and the splitter path
(`analyze_hybrid` + `inflect`) never re-enters `process()`. So a derivative that
sits *inside* a compound is still shattered:

```python
process("niṣyandatāvāda", mode="roots")
['niṣyanda', ('ta', 'tā', 'tas', 'tad'), 'vāda']     # still broken
```

Given that `-tā` compounds (`śūnyatāvāda`, medial `-tā-`) are pervasive in exactly
the philosophical corpus this feature targets, this is arguably the more common
real-world case than the standalone word. Fixing it means running the deriver over
splitter segments, which is a larger design change and a separate decision.

**Pre-split input is not derived either.** `process("niṣyanda-tā")` routes through
`handle_special_characters`, which processes each segment independently, so the
suffix is analysed on its own.

**`vidvatā`** (Inst. Sg. of `vidvas`/`vidvat`) is correctly *declined* by the
deriver — `vidvat` is attested, so the guard fires — but nothing downstream then
supplies the participle reading, and the splitter still returns `['vid', 'vata']`.
A pre-existing splitter limitation, untouched by this change.

## A pre-existing bug this routes around

`dict_search` (`functions/dictionaryLookup.py`) has a latent bug: for a *list*
entry whose lemma is not a dictionary headword it runs
`entry = [entry, entry, [entry]]`, a fallback shape that is only correct for
*string* entries. For a 5-slot list entry it nests the entry inside itself, so
`mode="roots"` returns a list instead of the lemma and `mode="parts"` raises
`TypeError`. It fires for any word whose lemma is in the inflection tables but in
no dictionary (e.g. `process("kṣaṇeṇa", mode="parts")`).

This is **not** fixed here. The taddhita path assembles its entry directly in
`process()` rather than passing the 5-slot row through `dict_search`, so it never
touches the broken branch.
