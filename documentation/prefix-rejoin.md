# Prefix re-joining in `clean_results`

## What the block is for

When the forms database does not know a word whole, `root_any_word` falls back to
stripping a leading upasarga (`SANSKRIT_PREFIXES`) and rooting the remainder, so
`samupekṣa` comes back as the entries `sam`, `sa`, `sa`, `upekṣa`.

That decomposition is right for a coined form but wrong for a *lexicalised* one:
`samādhi` is a dictionary headword, not `sam` + `ādhi`. The prefix blocks in
`clean_results` (`process_sanskrit/functions/cleanResults.py`) exist to undo the
split in exactly that case — they re-join prefix and stem, look the whole thing
up, and collapse the entries into the single dictionary entry when one exists.

## The trap: `dict_search` never returns `None`

This is the load-bearing fact for everything below. A miss is a *stub*, not
`None`, so a re-join guarded only by `if voc_entry is not None` fires anyway and
replaces a correct prefix analysis with a lookup failure for a headword that does
not exist.

`dict_search` returns two different entry shapes depending on what it is fed, and
conflating them is its own trap — **the slot the payload lands in is not the same
in both.**

*Fed a bare word* (`dict_search(["samādhi"])`) — this is what the prefix blocks do.
Both outcomes are 3 slots, and slot `[2]` discriminates:

| outcome | slots | `[2]` |
| --- | --- | --- |
| headword found | 3 | **dict** keyed by dictionary — `{'mw': {...}}` |
| not found | 3 | **list** holding the word itself — `['samupekṣa']` |

*Fed a 5-slot inflection entry* — this is what `list_of_entries` inside
`clean_results` is made of. A hit appends the word and its payload, giving 7
slots, and now slot `[2]` is the **grammatical tags list**, with the dictionary
payload at `[-1]`:

```
['upekṣa', 'masculine noun/adjective ending in -a', [('Voc','Sg')],
 ['upekṣaḥ', ...], 'samupekṣa', 'upekṣa', {'mw': {...}}]
   [0]stem       [1]type          [2]TAGS   [3]forms  [4]original [5]word [6]payload
```

So "a dict at `[2]` means a hit" holds **only for the bare-word path**. Applied to
a 7-slot entry it reports a miss for every real hit. The guard below is correct
because it only ever inspects the result of `dict_search(["sam" + stem])`, which
is the bare-word path.

That was the `samupekṣa` bug: `sam` + `upekṣa` (both with real Monier-Williams
entries) were deleted and overwritten with the stub for `samupekṣa`, which heads
no entry in any dictionary. The word came back as nothing but itself. (Strictly,
the broken guard fires whenever `dict_search` was *called* at all — the block
skips the lookup when the stem is itself a prefix.) Inside a
compound the damage was visible twice over, because the block deletes the entries
it consumed but the *sibling* stem of the same original word survives:
`saṃyojanānyāvaraṇamudvegasamupekṣayoḥ` ended in the stub `samupekṣa` followed by
an orphaned `upekṣā`.

**The guard is therefore: merge only on a real dictionary hit — `isinstance(voc_entry[0][2], dict)`.**
This is the same test `process()` uses at its own dictionary checkpoints, and the
`anu`/`ava` blocks express it as `not isinstance(voc_entry[0][2], list)`.

On the Yoga Sutra + compound-benchmark corpora (1519 unique words) adding the
guard changed 4 analyses, every one of them replacing an invented headword with a
real decomposition (`samupoṣya` → `sam` + `upoṣya`, `samanukrānta` → `sam` +
`anukrānta`, `samnidhimat` → `sam` + `nidhimat`, `samsag` → `sam` + `sag`). No
legitimate merge was lost; `samādhi` still re-joins.

Pinned by `tests/test_prefix_merge.py`, which asserts both directions — the
unattested join stays split, and the attested one still collapses.

## One walk, three prefixes: `REJOINABLE_PREFIXES` and `rejoin_prefix`

The three prefix blocks used to be near-duplicates carrying divergent guards and
dead code — the in-file `TODO` asked for "a generalised function for prefixes".
They are now one `rejoin_prefix()` helper driven by a table:

```python
REJOINABLE_PREFIXES = {
    "sam": ("sam", "sa"),
    "anu": ("anu",),
    "ava": ("ava", "av"),
}
```

### The absorbed set is the whole design

The value is **every stem `root_any_word` emits for the prefix itself**. Stripping
an upasarga leaves those homographs sitting between the prefix and the real stem,
and the walk has to step over them to reach it. They are not a guess:

| `root_any_word(prefix)` | emits | filler to absorb |
| --- | --- | --- |
| `sam` | `sam`, `sa`, `sa` | `sa` — the `-a` noun whose Acc.Sg is `sam` |
| `ava` | `ava`, `ava`, `av` | `av` — the verb root |
| `anu` | `anu`, `anu` | none |

Get this set wrong and the walk halts on a filler, looks up `prefix + filler`
(`ava` + `av` → `avaav`), misses, and the join is **silently lost**. Pinned by
`PrefixFillerContractTests`, which asserts the table against what `root_any_word`
actually emits, so a lexicon change that adds a homograph fails loudly here
instead of quietly degrading analyses.

### Why the old `ava` block looked like it worked

`ava` indexed `list_of_entries[j + 1]` where its siblings indexed `[j]`. That was
not a simple off-by-one: the `+1` **hopped the single `av` filler**, so `avaruhya`
re-joined as `avaruh` *by accident*. The compensation held only while exactly one
filler sat between prefix and stem. Otherwise the same `+1` read the word *after*
the stem — looking up `ava` + the next word entirely, losing the merge — or ran
off the end and raised `IndexError` outright.

It also deleted only up to the filler, so the real stem survived as a duplicate
beside the merged form: `avalokayāmi` → `['avalok', 'lok']`. Absorbing the filler
by name fixes the index, the crash, and the stranded duplicate together, and the
corpus shows all three: `avalokayāmi` → `['avalok']`, `vrīhīnavahanti` →
`['vrīhi', 'avahan']`, `sāmyenāvatiṣṭhamānāḥ` → `['sāmya', 'avasthā', …]`.

### No spelling retry belongs here

`dict_search` already folds `sam` → `saṃ` itself via `samMap`, so `samyoga` finds
the `saṃyoga` entry unaided. The `sam` block's `saṃ` retry was therefore redundant
*as well as* dead (it gated on `'MW'` while the payload is keyed `'mw'`, and would
have passed a bare string to a `dict_search` that wants a list). It is gone;
`test_join_relies_on_dict_search_to_fold_sam_into_saṃ` pins the behaviour it was
groping for.

`DICTIONARY_REFERENCES` looks like a free way to pre-check the join without a
database hit — **it is not.** `samyoga` is absent from that index yet `dict_search`
resolves it, so gating on membership would silently drop exactly the `sam`/`saṃ`
merges above. The helper makes the same single `dict_search` call the old blocks
made: on the 1529-word corpus, 129 calls before and after.

### Two deliberate choices worth knowing

**The stacked-prefix guard is now uniform.** Only the old `sam` block refused to
join when the stem was itself a prefix; `anu` and `ava` had no such guard. The
helper applies it to all three, so `ava` + `sam` + `graha` yields `ava`,
`samgraha` rather than re-joining the outer prefix. Lexicalised stacked prefixes
(`samavāya`, `anuvyavasāya`) therefore never re-join — out of scope, not a bug.

**The merged lemma is the dictionary's headword, not the query.** `dict_search`
folds `sam` → `saṃ` for the *lookup* but echoes the query back at slot `[0]`, so
the re-join used to emit `samvedana` while the rest of the pipeline said
`saṃvedana`. `_canonical_headword` takes the real headword out of the payload
already in hand — the inner key of `voc_entry[0][2]` — so it costs no extra
lookup (129 → 130 → **130** `dict_search` calls across the corpus, unchanged by
this).

This is not a new convention being imposed; it stops the re-join being the only
path that disagrees with the other two. Both other authorities already say `saṃ-`:
the forms DB (`root_any_word("samvedana")` → `saṃvedana`) and Monier-Williams,
which files the entry under `saṃvedana`. It is self-limiting — where no fold
happened the headword *is* the query, so `samādhi` (where `sam-` genuinely is
canonical before a vowel) and `avagraha` are returned untouched.

On the corpus it corrects 6 lemmas of 1529, all `sam-` → `saṃ-`: `saṃvedana`,
`saṃkramaṇa`, `saṃtāra`, `saṃdha`, `saṃnidhā`, `saṃha`. Pinned by
`CanonicalLemmaTests`.

### `duḥ` is in the table too

`duḥ` is not an upasarga — it is absent from `SANSKRIT_PREFIXES` and
`root_any_word("duḥ")` is `None`, so it is never *stripped*. It reaches the entry
list only when the compound splitter cuts a word like `duḥkha` in two. That gives
it no homographs and so an empty absorbed set, `"duḥ": ()`, but the re-join it
needs is the identical operation.

The hand-written `duḥ`/`kha` block it replaces carried three bugs at once: it read
`list_of_entries[1 + 2]` (a typo for `i + 2`) into a list the `del` above had
already shortened, so `clean_results([duḥ, kha])` raised `IndexError`; it compared
an entry *list* against the string `"kha"`, so that arm was dead anyway; and it
merged on `if replacement is not None`, the same never-`None` misuse as the
original `samupekṣa` bug. Folding it into the table deleted all three.

The generalisation is deliberate: `duḥ` + *any attested stem* now re-joins
(`duḥśīla`, `duḥkṛta`), not just `duḥkha`. It costs one extra `dict_search` per
`duḥ` cut — 129 → 130 lookups across the 1529-word corpus, with no analysis
changed.

## Known defects remaining in the same block

- **Spurious filler in prefix output.** A surviving prefix split still carries the
  homograph: `samupekṣa` → `[('sam', 'sa'), 'upekṣa']`. Harmless but noisy; the
  fillers are only *absorbed* when a merge actually happens.
- **One `is not None`-on-`dict_search` misuse survives**, in the `-n` stem
  replacement above the re-join. It is benign today because it is gated on
  `in DICTIONARY_REFERENCES` before the call, but it is the same latent class as
  the bug fixed here: if that gate is ever removed, the stub will overwrite the
  entry.
