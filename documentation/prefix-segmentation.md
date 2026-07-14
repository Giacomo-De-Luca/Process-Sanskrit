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

## Invariants to preserve

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
`hybrid-morphology-v4`.
