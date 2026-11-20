# Prefix segmentation: a stripped upasarga is a word, not a rival reading

Sibling document to [`prefix-rejoin.md`](prefix-rejoin.md), which covers the
*opposite* move — collapsing a prefix split back into a lexicalised headword
(`sam` + `ādhi` → `samādhi`). This one is about what happens when the split is
correct and must be *reported* as a split.

## The bug

`śabdenopadiśyate` and `dṛṣṭānumitārthe` split correctly, but `upa` and `anu` came
back flagged as *alternative readings* of the word after them rather than as words
in their own right:

```
upadiśyate       ->  [('upa', 'diś')]                 # "upa OR diś"
dṛṣṭānumitārthe  ->  ['dṛṣṭa', ('anu', 'mitārtha')]   # "anu OR mitārtha"
```

A tuple in `mode='roots'` means *rival analyses of one span*. Two sequential words
must be two list items.

## Why the two collapse: `entry[4]` is read two incompatible ways

`extract_roots` (`functions/cleanResults.py`) folds a run of consecutive entries
that share a surface form (`entry[4]`) into one tuple:

```python
while list_of_entries[j][4] == original_word:   # same surface => same span
    stemmed_forms.append(list_of_entries[j][0])
```

That is sound only when the entries came from **one lookup of one span** — which
is how the database returns them (masculine *and* neuter `śabda` for `śabdena`).

The prefix branch of `root_any_word` (`functions/rootAnyWord.py`) broke the
invariant by stamping the whole word onto *every* match it was about to return,
prefix and root alike, so `upa` and `diś` both claimed the surface `upadiśyate`
and were folded together.

The stamp is not gratuitous. Two other consumers read `entry[4]` expecting the
**whole word**, and both look only at root entries:

| consumer | wants `entry[4]` to be | breaks if it sees the segment |
| --- | --- | --- |
| `extract_roots` (`cleanResults`) | the span the stem covers | prefix folds into the stem's readings |
| the `-n` lemma rule (`cleanResults`) | the whole word | `viṣayin` decays to `viṣayī` |
| the whole-word dictionary check (`process.py`) | the whole word, **as matched** | `virati`, `saṃskāro` lose their entry |

So the field is overloaded, and that is why the obvious fixes each trade one bug
for another. All three were measured against the Yoga Sutra corpus:

- **Drop the stamp.** Prefixes separate, but the `-n` rule now sees the *segment*
  surface `viṣayī` — which is itself a headword — and swaps the correct lemma
  `viṣayin` for it.
- **Stamp only the remainder's entries.** Nested prefixes re-collapse: resolving
  `avirati` strips `a`, and the remainder `virati` is itself resolved by stripping
  `vi`, so blindly re-stamping everything the recursive call returned flattens the
  inner split again.
- **Key the dictionary check on the raw input `text`.** Loses the *normalised*
  surface — `saṃskāro` is matched as `saṃskāraḥ` and `saṅgrahaḥ` as `saṃgrahaḥ`,
  and only the matched form is a headword — while inventing entries for `hānaṃ`
  and `kiṃ`.
- **Stamp the entries carrying the remainder *as requested*.** The remainder may
  resolve through a sandhi variant or `samMap` — `pratisaṃvedanā` reaches its stem
  as `samvedanā` — and then no entry carries the remainder that was asked for, so
  the stamp silently does nothing and the root entries keep a *sub-span* surface.
  The whole-word check then reads that fragment and drops the attested entry:
  `pratisaṃvedanā` came back as `['prati', 'samvedana']`, losing itself. Take the
  span from what the recursion actually matched (`matches[-1][4]`), never from the
  remainder as requested.

## The fix

**`rootAnyWord._stamp_whole_word(matches, word)`** stamps the whole word on exactly
the entries that resolved the remainder, identified as those sharing the surface of
the **last** match — what the recursion actually matched, rather than the remainder
as requested. Prefix entries keep their own surface, so they no longer share one
with the root; entries carried up from a *deeper* prefix decomposition carry a
surface of their own and so fall outside that group, which is what keeps `a` + `vi`
+ `rati` apart. The root entries still carry the whole word, so the `-n` rule and
the dictionary check see what they always saw.

**`process()`** now reads the whole word for its dictionary check from the **last**
entry rather than `result[0]`: once prefixes keep their own surface, `result[0]` is
the prefix (`vi` of `virati`) and keying the lookup on it drops the attested entry
for the whole word. The root entries carry the whole word *as matched*, which is
also why the raw input will not serve. The check additionally skips insertion when
some analysis already yields that stem — otherwise `pratyakṣam` is reported twice,
once as itself and once as the re-inserted headword — and it tests the result for a
real dictionary hit, since a miss comes back as a stub rather than as `None` (see
[`prefix-rejoin.md`](prefix-rejoin.md)).

## Result

```
upadiśyate       ->  ['upa', 'diś']
dṛṣṭānumitārthe  ->  ['dṛṣṭa', 'anu', 'mitārtha']
virati           ->  ['virati', 'vi', 'ratī']    # attested whole word still offered
avirati          ->  ['avirati', 'a', 'vi', 'ratī']
samupekṣa        ->  [('sam', 'sa'), 'upekṣa']   # sam/sa are rival readings *of the prefix*
```

Across the Yoga Sutra corpus (588 unique tokens) 10 change, all of them this fix.
`viṣayin`, `pratyakṣam`, `saṃskāro`, `saṅgrahaḥ` and the genuine-ambiguity tuples
(`tasya` → `('ta', 'tad')`) are unchanged.

## A prefix stripped in a sandhi spelling

A stripped prefix gets its own entry by being looked up in the forms table, as if
it were a word. Three spellings in `SANSKRIT_PREFIXES` are not words — `vy` and
`praty` are only what `vi` and `prati` look like before a vowel, and the table
files `ud` under `ut` — so that lookup found nothing. `ud` had a hand-written
special case. The other two were dropped from the result without a trace:

```
vyavaharati   ->  ['avahṛ']            # vi lost; avahṛ is a real verb, and the wrong one
pratyāharati  ->  [('ā', 'ḷ'), 'hṛ']   # prati lost
```

`PREFIX_LOOKUP_SPELLINGS` (`utils/lexicalResources.py`) now names the spelling
the forms table knows for each such prefix, and `rootAnyWord._prefix_entries`
is the one place a stripped prefix is looked up — the `ud` special case is an
entry in the table like the others.

```
vyavaharati      ->  ['vi', 'avahṛ']
pratyāharati     ->  ['prati', ('ā', 'ḷ'), 'hṛ']
pratyabhimṛśati  ->  ['prati', 'abhi', 'mṛś']
```

`vi` + `avahṛ` is not yet `vyavahṛ`: the re-join in `cleanResults.py` handles one
prefix, and only `sam`, `anu` and `ava`.

Across 2,356 inputs (the Yoga Sūtra, the 1,500 held-out tokens of
`tests/datasets/split_gold/`, and a list of about a hundred prefixed verbs) seven
changed through this fix, each by regaining `vi` or `prati`. All seven come from
the verb list: finite prefixed verbs are thinly covered in the two corpora.

Still unresolved by the stripper, because no prefix spelling can fix them: a
junction that changes the root (`praṇamati`, `niṣīdati`), fuses vowels
(`upekṣate`) or inserts a sibilant (`saṃskaroti`).

## The prefix `nis` written `nir-`

`nir` is not a strippable prefix, so `nirgacchati` reaches the splitter, which
cuts it `niḥ` + `gacchati`. `niḥ` looked up as a word is the noun `ni`, and the
prefix came back wrong:

```
nirgacchati     ->  ['ni', 'gam']
sa nirgacchati  ->  ['sa', 'ni', 'gam']
```

`Inflector.SPLIT_TOKEN_SPELLINGS` (`functions/inflect.py`) reads the split token
`niḥ` back as the prefix `nis`. Because it acts on the split, it covers a bare
word, a word inside a sentence and a stacked prefix alike:

```
nirgacchati       ->  ['nis', 'gam']
sa vinirgacchati  ->  ['sa', 'vi', 'nis', 'gam']
```

Of the 1,855 forms in `resources/split_frequencies.tsv` that contain `nir`,
`niḥ`, `niṣ` or `niś`, 239 change, every one by `ni` becoming `nis` and in no
other way.

**Do not make `nir` strippable instead.** That was the first implementation.
`root_any_word` runs before the whole-word dictionary lookup and the
`-tā`/`-tva` deriver, so a strippable `nir` pre-empts both:

```
nirvikāratvāt  ->  ['nis', 'vikāratva']   # was ['nirvikāratva']
nirbhayatām    ->  ['nis', 'bhī']         # was ['nirbhayatā']
```

On the same 1,855 forms it changed 36 beyond the `ni` → `nis` swap. Five lost an
abstract noun or adverb as above. Nine gained a prefix reading beside an attested
headword (`nirvṛti` → `nirvṛti`, `nis`, `vṛt`). The other 22 are mostly finite
verbs the stripper resolves and the splitter does not (`nirmathyate` → `nis`,
`manth`), which is what makes the rewrite tempting. Those gains cannot be had
without the losses until prefix stripping stops outranking the deriver — the
ordering problem already recorded for `adhimātratvāt` in
[sentence-vs-word-evaluation.md](sentence-vs-word-evaluation.md).

`niṣ` is handled nowhere, on purpose: it is `nis` only before `k`, `kh`, `p`,
`ph`, `ṭ`, and otherwise `ni` plus a root in `s-` (`niṣīdati`).

## Invariants to preserve

- **Every strippable prefix must resolve to its own entry**, as written or
  through `PREFIX_LOOKUP_SPELLINGS`. A sandhi spelling added to
  `SANSKRIT_PREFIXES` without one is stripped and then either dropped or read as
  an unrelated word that shares the spelling.
- **`nir` is read from the split token, never stripped.** See the section above;
  `NirPrefixTests` pins both the rule and the abstract nouns the alternative
  breaks.
- **A tuple means rival readings of one span.** Anything that makes two entries
  covering *different* spans share `entry[4]` will silently fold them together.
- **Root entries carry the whole word; prefix entries carry their own surface.**
  Both halves are load-bearing — see the table above.
- The whole-word dictionary check needs the **matched** surface, never the raw
  input.

Pinned by `tests/test_prefix_segmentation.py`, which asserts the fix *and* one test
per collateral failure above, so each of the four tempting rewrites fails loudly.
A change here alters morphology output and so must bump
`ANALYSIS_ALGORITHM_VERSION` (`utils/analysisCache.py`); this fix took it to
`hybrid-morphology-v4`, the sandhi-spelling fix (`SandhiSpellingOfPrefixTests`)
to `hybrid-morphology-v9`, and the `nir-` rule (`NirPrefixTests`) to
`hybrid-morphology-v11` (`v12` on main, which does not have the split ranker).
