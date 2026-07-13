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

## Known defects in the same block (not fixed)

The three prefix blocks are near-duplicates of one another and the `sam` one
carries dead code. These are pre-existing and deliberately left alone; the
in-file `TODO` ("replace this entirely with a generalised function for prefixes")
is the right eventual fix.

- **The `saṃ` retry is unreachable.** The `sam` block gates it on
  `'MW' in voc_entry[0][2]`, but `dict_search` keys its payload with the
  lowercase `'mw'`. The branch has never run. Were it to run it would call
  `dict_search("saṃ" + ...)` with a bare **string** rather than a list.
- **The `ava` block can crash.** It guards `j < len(list_of_entries)` and then
  indexes `list_of_entries[j + 1]`, where `sam` and `anu` index `[j]`. A entry
  list ending in `ava` + stem raises `IndexError`.
- **The `duḥkha` block has a typo:** `if list_of_entries[1+2] == "kha"` — `1+2`
  should be `i+2`, and it compares an entry *list* against a string, so it is
  always false.
- **Spurious `sa` in prefix output.** `root_any_word('sam')` returns the
  indeclinable `sam` plus two `sa` homographs (whose accusative singular is
  `sam`), so every surviving prefix split now carries a `sa` it does not need:
  `samupekṣa` → `('sam', 'sa', 'upekṣa')`. Harmless but noisy.
- **The same `is not None`-on-`dict_search` anti-pattern survives twice more**, in
  the `duḥkha` block and in the `-n` stem replacement above it. Both are benign
  today (`duḥkha` is attested; the other is gated on `in DICTIONARY_REFERENCES`),
  but they are the same latent class as the bug fixed here.
