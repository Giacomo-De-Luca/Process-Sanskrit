# The `word_list` dictionary index

`word_list` maps an IAST headword to the dictionaries that attest it. It backs
`DICTIONARY_REFERENCES` (`utils/dictionary_references.py`), which the compound
analyser, the sandhi scorer, `process()` and the `-tā`/`-tva` deriver all hit as
a membership test — it is the hottest predicate in the pipeline.

**`word_list` is derived data, not source data.** Everything in it is
recoverable from the dictionary tables in the same database. Treat it as an
index that can be rebuilt at will, never as something to patch.

## The v1.0.2 defect

The database ships seven dictionary tables: `ap90`, `bhs`, `cae`, `cped`,
`ddsa`, `gra`, `mw`. The `word_list` baked into the released artifact was built
from **only five of them** — `cae` and `ddsa` were omitted. That gave 242,379
rows instead of the correct 246,955, and under-reported the attesting
dictionaries for a further ~42,000 words.

Commit `07b6678` moved `DICTIONARY_REFERENCES` off a 247,000-entry Python
literal (which cost ~558 MB of transient RSS to compile) and onto this table.
Because the table was wrong, that commit shipped a 1.4 MB
`dictionary_references_overlay.json` — 42,371 `overrides` plus 4,576
`only_python` entries — loaded into memory and consulted before every lookup, to
paste `cae` and `ddsa` back in at runtime.

The overlay is gone. `word_list` is now rebuilt from all seven dictionary
tables, which reproduces the historical mapping **exactly**: 246,955 entries, no
missing keys, no value differences. Removing the overlay also made lookups
*faster*, because two dict probes no longer precede every query:

| | Python literal | word_list + overlay | word_list rebuilt |
| --- | --- | --- | --- |
| Import (peak RSS) | 558 MB | 51 MB | 34.5 MB |
| Import time | 2.4 s | 0.62 s | 0.24 s |
| Warm membership | 0.46 µs | 0.97 µs | 0.54 µs |

Two dead artifacts were removed with it: `dictionary_cross_references` (a
byte-identical duplicate table, referenced by no code) and
`resources/dictionary_references.json` (9.2 MB, never loaded by any Python file
in the project's history).

## Rebuilding

`utils/wordListBuilder.py` owns the rebuild. Dictionary tables are **discovered
from the schema** — any table carrying the `keys_iast` / `cleaned_body` /
`components` column signature — rather than hardcoded, so adding a dictionary to
the database and rebuilding is enough to index it. The inflection tables
(`lgtab*`, `vlgtab*`) and the split cache do not match the signature.

The swap runs in a single transaction, so an interrupted rebuild leaves the
previous index in place rather than a half-written one.

```bash
uv run python tools/build_word_list.py             # rebuild the packaged database
uv run python tools/build_word_list.py --vacuum    # + reclaim space, for a release build
```

`update-ps-database` calls the same builder (`ensure_word_list_index`) after
download and repairs a stale database in place, so existing installs converge in
about a second without re-downloading 596 MB.

When `PROCESS_SANSKRIT_DB_PATH` is set, the command rebuilds that existing
external database without downloading anything. It writes and validates a
sibling copy before atomically replacing the configured path, so immutable
readers already holding the old inode are not mutated underneath. The database
must exist and be readable, and its parent directory must be writable with room
for a temporary full-size copy. A missing configured path is an error rather
than a reason to create an empty SQLite file or fall back to the packaged
database. External repairs preserve legacy tables owned by the deployment.

### Releasing a corrected artifact

`--vacuum` is only worth running on the artifact you intend to publish: it
rewrites the whole file to reclaim the space freed by dropping
`dictionary_cross_references`, needs roughly twice the database size in free
disk, and is slow. Rebuild with `--vacuum`, gzip, upload as the release asset,
then bump `RELEASE_TAG` in `process_sanskrit/setup/updateDB.py`. Users on the
old artifact are repaired by `update-ps-database` regardless.

## `word_list_stubs`: bare variant-reading pointers

Being a dictionary key does not make a form a word. Monier-Williams records
manuscript spellings it does not endorse: the entry exists only to say *"this
reading is a variant of that one"*. `tanni` heads nothing but

> `tanni` `°nnī` variant reading (varia lectio) for `°nvī`, q.v.

Such a headword is an artifact of the critical apparatus rather than a lexical
definition — but it *is* a key, so the compound walk in
`functions/compoundAnalysis.py` could cut a word on it, and being **longer** than
the true cut it won the tie:

```
process("tannirodha", mode="roots")   # "the cessation of that"
  ->  ['tanni', ('rodha', 'rudh')]    # before: tad + nirodha destroyed
  ->  ['tad', 'nirodha']              # after
```

`WordListBuilder` flags these during the rebuild into a `word_list_stubs` table
(432 headwords in the current seven-dictionary database). The flag is *derived*,
like the index itself. The classifier is deliberately conservative: visible
definition prose before or after the pointer rejects the entry, and every
attestation across every dictionary must qualify. Rebuilding streams one boolean
per headword instead of retaining all dictionary bodies in memory.

**A headword qualifies only if every entry for it, in every dictionary, is such a
pointer.** That quantifier is the core safety rule: a real word may well carry a
cross-reference among its entries, and demanding that *all* entries be bare
pointers keeps it out of the set. Same-entry definitions such as `cakṣūroga`
("disease of the eye" plus a variant note) are also explicitly rejected.

Two properties are deliberate:

- **The stubs stay in the index.** They are ranked, not deleted, so
  `dict_search("tanni")` still finds it — someone may well look up a spelling
  they have actually read. The ordinary numeric score still decides whether a
  candidate clears `dict_word_iterative`'s 0.6 acceptance gate. Among eligible
  candidates, a genuine headword ranks before a stub; a stub therefore remains
  available when nothing genuine fits.
- **The signal is lexical, not morphological.** The tempting shortcut — demote
  a first member that heads no inflection table — *cannot work*, and the near miss
  is worth recording:

  | | longer cut | shorter cut | which inflects | who must win |
  | --- | --- | --- | --- | --- |
  | `tannirodha` | `tanni` (5) | `tan` (3) | only `tan` | **shorter** |
  | `gacchatā` | `gacchat` (7) | `gaccha` (6) | only `gaccha` | **longer** |

  The two are the same shape, so any weight large enough to beat `tanni` also
  beats `gacchat` and hands the participle `gacchatā` ("by the one going") to
  `gaccha` + `tā`. `gacchat` and `niṣyanda` are real words that merely lack a
  paradigm; `tanni` is not a word at all. Pinned by
  `tests/test_compound_preference.py` and `tests/test_taddhita_derivation.py`.

The index stores `STUB_CLASSIFIER_VERSION` in `word_list_metadata`. A missing
stub table, missing metadata, or outdated classifier version makes the derived
index stale and causes `update-ps-database` to rebuild it. Analysis against a
legacy database degrades to "no stubs known" rather than failing a lookup, and
logs an update warning. Changing this ranking changes hybrid/process results, so
it requires a bump of `ANALYSIS_ALGORITHM_VERSION` — currently
`hybrid-morphology-v4`; see [local-cache.md](local-cache.md).

## Staleness is reported, not hidden

`word_list_sources` records which dictionaries the current index was built from,
so `WordListBuilder.missing_dictionaries()` is an O(1) check rather than a scan
of a quarter-million JSON values. A database whose index does not cover every
dictionary present logs a warning once per connection and keeps serving.

This matters most for externally provisioned lexicons
(`PROCESS_SANSKRIT_DB_PATH`, see [database-location.md](database-location.md)),
which are opened read-only and immutable by analysis workers. The updater's
atomic replacement lets existing workers finish against the old inode; restart
them so all workers converge on the new database. A five-dictionary lexicon
returns wrong dictionary lists for roughly 19% of the vocabulary — exactly the
silent failure the overlay was concealing — so it must announce itself.

An index predating `word_list_sources` declares coverage of nothing and reads as
stale. That is deliberate: proving a legacy index really covers `mw` would
require scanning every row, and a rebuild is cheap enough that distinguishing
the two cases would not pay for itself.

Pinned by `tests/test_word_list_integrity.py`.
