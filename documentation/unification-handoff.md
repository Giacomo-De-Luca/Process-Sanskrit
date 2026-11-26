# Unification handoff: duplicated caches and tables

Work list for an agent that removes duplication in the lemma-mode code. Every
item here changes structure only: **no output of `process()` in any mode may
change.** Line numbers are as of 2026-11-25; search by name if they have moved.

**Status (2026-11-25):** items 1 (with its optional part), 3 and 4 are done; see
the "Done" note under each. Items 2 and 5 are open, and can start: the
member-repair change they waited for has landed (see the last section).

## Before you start

- **The member-repair change has landed (2026-11-25).** What it changed that
  items 2 and 5 touch:
  - `MemberRepair` reads a pair again with `JunctionSandhi` (no vendored
    rules); its `repair(members, elided)` takes the indexes an avagraha marks
    (`ComponentMarks.elisions`), and `unrepaired` is `keep` or `drop`;
  - `SandhiRules` is gone from the library: the evaluation's joiner is
    `SandhiJoiner` in `scripts/compound_members/joiner.py`;
  - `AffixFilter.members()` is `AffixFilter.default(is_word)`, and
    `AffixFilter.preverb_spellings()` derives the runs of two or more preverbs
    from `PreverbSandhi.spellings()` minus `spellings(depth=1)`;
  - `PreverbSandhi.spellings()` itself is unchanged.
- **Project rules** (from `CLAUDE.md`):
  - Never commit, never stash.
  - Run Python with `uv run python`.
  - Write the tests first, then change the code.
  - Do not edit the vendored splitter modules (`process_sanskrit/splitter/`).
  - Update the documentation you make wrong.
  - Finish with a code-quality review by a separate agent.
- **Tests to run after each item:**
  ```bash
  uv run python -m unittest tests.test_compound_verbs tests.test_compound_members \
      tests.test_lemma_builder tests.test_lemma_mode tests.test_preverb_sandhi \
      tests.test_prefix_segmentation tests.test_prefix_merge tests.test_shared_verb_forms \
      tests.test_word_list_integrity tests.test_database_path_configuration
  uv run python -m unittest discover -s tests/compound_members -t .
  uv run python -m unittest discover -s tests        # full suite at the end
  ```
  The full suite stood at 805 tests after the member-repair change: OK, 4
  skipped, 5 expected failures.

## 1. One per-engine cache (approved)

Three classes keep data read from the database once per SQLAlchemy engine.
Each has its own copy of the same code: a `weakref.WeakKeyDictionary` keyed on
`session.get_bind()`, a `threading.Lock`, double-checked locking, and a
`reset()` classmethod.

| class | file | what it stores per engine |
|---|---|---|
| `CompoundVerbIndex` | `process_sanskrit/utils/compoundVerbs.py:38-71` | one table, read once (`_read`) |
| `CompoundForms` | `process_sanskrit/utils/compoundMembers.py:262-315` | one table, read once (`_read`) |
| `PieceLemmas` | `process_sanskrit/functions/lemmaAnalysis.py:442-461` | a dict filled word by word (`_lemmas`) |

**Why weak keys on the engine:** every in-memory database has the URL
`sqlite://`, so a URL key would share one table between test databases (the
comment at `compoundVerbs.py:38`). Keep this.

**Defect this fixes:** `_reset_reference_state`
(`process_sanskrit/utils/dictionary_references.py:142-156`) resets
`CompoundVerbIndex` and `CompoundForms` by name but forgets `PieceLemmas`.
`PieceLemmas`' answers also depend on `DICTIONARY_REFERENCES`, which follows
`PROCESS_SANSKRIT_DB_PATH`. A test that points that variable at a scratch
database while the engine stays (`tests/test_word_list_integrity.py:236`)
therefore gets stale lemmas. In production nothing calls the reset:
`get_engine` refuses a second database path, and a new engine starts with an
empty cache entry.

**Target:**
- A new module `process_sanskrit/utils/engineCache.py` with one class, for
  example `EngineCache`, holding:
  - the weak-key store and the lock;
  - `table(session, read)`: the whole-table case, read once under the lock;
  - `memo(session)`: the per-word dict for `PieceLemmas`;
  - `clear()`;
  - a class-level registry of every instance, and `EngineCache.clear_all()`.
- Each of the three classes holds one `EngineCache` instance as a class
  attribute. It keeps `reset()` as a one-line wrapper, because
  `tests/test_compound_verbs.py:74,94` calls `CompoundVerbIndex.reset()`.
- `_reset_reference_state` calls `EngineCache.clear_all()` instead of naming
  the classes. Keep the import inside the function, as now, to avoid an import
  cycle.
- Add a short entry to `process_sanskrit/utils/README.md`.

**Tests first:**
- For `CompoundForms` and `PieceLemmas`, the two tests
  `tests/test_compound_verbs.py:72-95` (`CacheTests`) has for
  `CompoundVerbIndex`:
  - each in-memory database gets its own data;
  - the data is read once, and read again after a reset.
- `_reset_reference_state()` empties all three caches.
- An `EngineCache` created later is also emptied by `clear_all()`.
- A dropped engine drops its entry (weak key).

**Do not fold in:**
- `_lookup_for_path` and `_stubs_for_path` (`dictionary_references.py:114,129`):
  they are keyed by file path on a raw `sqlite3` connection, not by engine, and
  are already cleared with `cache_clear()` in the same function.
- `SandhiJoiner.shared()` (`scripts/compound_members/joiner.py`): the
  evaluation's process-wide singleton, not an engine cache.
- `process_sanskrit/utils/analysisCache.py`: the persistent cache, a separate
  design.

**Optional, measure first:** `LemmaLexicon._members`
(`lemmaAnalysis.py:131, 153-155`) memoises members per `LemmaLexicon`
instance, so they are worked out again on every `process()` call. Members
depend only on the database, so they could use `EngineCache.memo`.
- Compare a warm second pass over the Yoga Sūtra and benchmark texts before
  and after; the last measurement was 3.59 s.
- Keep the change only if it is faster, and report the numbers to the
  maintainer.

**Done.** `process_sanskrit/utils/engineCache.py` has `EngineCache`
(`table`, `memo`, `clear`, `clear_all`, a weak registry of instances).
`CompoundVerbIndex`, `CompoundForms` and `PieceLemmas` each hold one and keep
`reset()` as a one-line wrapper; `_reset_reference_state` calls
`EngineCache.clear_all()`. `engineCache.py` imports nothing from the package,
so it is imported at the top of `dictionary_references.py`, and that module's
now unused `CompoundVerbIndex` import is gone. Pinned by
`tests/test_engine_cache.py`. The defect was latent: the cited test never asks
`PieceLemmas` anything, so no test failed before.

The optional part is done too. `LemmaLexicon` keeps members in an
`EngineCache.memo`. Measured on a warm second pass of `process(text,
mode="lemmas")` over the 195 Yoga Sūtra lines, the compound benchmark and the
hand-picked cases (612 texts), with the persistent cache warm, in five
interleaved pairs of fresh processes: CPU time fell from 0.76–0.90 s (mean 0.81)
to 0.64–0.73 s (mean 0.67), lower in every pair, and the output is
byte-identical. Wall-clock time was too noisy on a loaded machine to compare.
Both subclasses (`_MemberlessLexicon`, the evaluation's `RecordingLexicon`)
override `members` entirely, so neither reads the shared memo. Like
`PieceLemmas`' answers, the memo is not bounded: it grows with the distinct
lemmas of a process's lifetime.

## 2. One table of preverb spellings (after the member-repair change)

The spellings of a preverb are written out by hand in nine places. This is
already listed as open in `documentation/lemma-mode-handoff.md:384-393`.

| table | file | what it says |
|---|---|---|
| `SANSKRIT_PREFIXES` | `utils/lexicalResources.py:234` | what `root_any_word` strips, including sandhi spellings (`vy`, `praty`, `ut`, `saṃ`) and `a` |
| `PREFIX_LOOKUP_SPELLINGS` | `utils/lexicalResources.py:256` | forms-table spelling of a stripped prefix (`ud`→`ut`, `vy`→`vi`, `praty`→`prati`) |
| `samMap` | `utils/lexicalResources.py:334` | `sam`/`saṃ`/`saṅ`/`san`/`sañ` fold for dictionary lookup (`dictionaryLookup.py:173`, `rootAnyWord.py:232`) |
| `MEMBER_PARTICLES` | `utils/lexicalResources.py:346` | prefixes and particles glued in hyphenated analyses |
| `CompoundVerbIndex.MW_SPELLINGS` | `utils/compoundVerbs.py:34` | Monier-Williams' spelling → lemma mode's (`nir`→`nis`, `dur`→`dus`, `antaH`→`antar`, …) |
| `CompoundVerbIndex.OUR_SPELLINGS` | `utils/compoundVerbs.py:36` | lemma mode's spelling → the compared one (`ut`→`ud`, `saṃ`→`sam`, …) |
| `Inflector.SPLIT_TOKEN_SPELLINGS` | `functions/inflect.py:51` | splitter token → prefix (`niḥ`→`nis`) |
| `PreverbSandhi` constants | `utils/preverbSandhi.py:29, 35, 38-42, 44` | `RETROFLEX_N_PREVERBS`, `RETROFLEX_S_EXCEPTIONS`, `_FINAL_SPELLINGS`, `_NOT_BASES` |
| `PreverbSandhi` code | `utils/preverbSandhi.py:62, 101-106` | the `ut`→`ud` rewrite in `spellings()`, and the names tested in `join` (`sam`/`saṃ`, `ud`/`ut`, `nis`/`nir`/`dus`/`dur`) |

Seven entries of `MEMBER_PARTICLES` (`a`, `nis`, `nir`, `niḥ`, `niṣ`, `niś`,
`nī`) are also produced by `PreverbSandhi.spellings()`. The `dus` spellings
are not, because `dus` is not in `SANSKRIT_PREFIXES`.

**Target:**
- One table in `lexicalResources.py`, one record per preverb:
  - the name lemma mode gives it;
  - the forms-table lookup spelling;
  - Monier-Williams' spellings;
  - the spellings its end takes before a vowel and before a consonant.
- Every table above becomes a value derived from it.

**Constraints:**
- **Characterization tests come first.** Copy each table's current literal
  into a test and assert that the derived value is equal: same keys, values
  and order where order is used.
- **The tables disagree on purpose.** Lookup maps `ud`→`ut`, while compound
  verbs compare `ut`→`ud`. The record must carry both, not choose one.
- **Do not add `nir` to `SANSKRIT_PREFIXES`.** The stripper runs before the
  dictionary lookup, and `nirvikāratvāt` would come back as `nis` +
  `vikāratva`. It is pinned by `NirPrefixTests`; see the `nir-` section of
  `documentation/prefix-segmentation.md`.
- **Keep the lookup entries.** A spelling in `SANSKRIT_PREFIXES` that is not a
  word needs a `PREFIX_LOOKUP_SPELLINGS` entry, or it is stripped and then
  dropped.
- **`REJOINABLE_PREFIXES` (`functions/cleanResults.py:90`) is not a spelling
  table.** Its values are the filler stems `root_any_word` emits for each
  prefix. Leave it alone; see `documentation/prefix-rejoin.md`.
- **`SANSKRIT_PREFIXES_OLD` (`lexicalResources.py:275`)** is a gloss table
  that nothing reads. Ask the maintainer before deleting it.
- **`AffixFilter.preverb_spellings()` reads `PreverbSandhi.spellings()` and
  `spellings(depth=1)`.** Both must give the same sets after the item (17,554
  spellings at depth 3); `AffixFilterTests` and the evaluation's parity test
  pin what they glue.

## 3. A public letters helper (independent)

`ComponentMarks._letters` (`process_sanskrit/utils/componentMarks.py:36`) is
private, but it is called from outside its class:
- `utils/compoundMembers.py:98, 104, 301, 302`;
- `scripts/compound_members/readings.py:114`;
- `scripts/compound_members/lexicon.py:41` (`Spelling.letters`).

Four of these calls wrap it in `unicodedata.normalize("NFC", …)` first.

**Target:**
- A public `ComponentMarks.letters(text)` that applies NFC itself.
- The outside callers use it.
- `Spelling.letters` delegates to it or goes away.

**Constraint:** before the internal callers (`parts`, `prevailing_members`)
switch to the NFC version, check on all Monier-Williams, Edgerton and
Grassmann `components` values that normalizing first changes no result.
Otherwise keep them on the non-normalizing form.

**Done.** `ComponentMarks.letters(text)` composes and then strips. The four
library calls, `WordHyphenRule._cut` (`readings.py`, which did not compose
before) and `Spelling.key` use it, and `Spelling.letters` is gone. Five of the
six outside calls composed already, not four. The check had to cover more than
the three dictionaries. Monier-Williams, Edgerton, Grassmann, Apte, CAE and
CPED are all composed already, but ddsa has 70 decomposed `components`
(`a-kudhryãk`). `multidict` hands the first non-empty `components` of any
dictionary searched to `parts`, so composing there would change `mode="parts"`
for those words. `parts` and `members` therefore stay on `_letters`.
`WordHyphenRule` reads Monier-Williams only, so composing there changes
nothing.

## 4. Two identical branches (independent)

`PreverbSandhi._before_next` (`utils/preverbSandhi.py:77-80`) has an `iī`
branch and a `uū` branch with the same body. Merge them into one test
(`joined[-1] in SEMIVOWEL`). `tests/test_preverb_sandhi.py` and
`tests/test_compound_members.py` cover it.

**Done.** `PreverbSpellingsTests` pins the semivowel cases, and
`PreverbSandhi.spellings()` was compared before and after: the same 17,554
spellings.

## 5. The evaluation reads members through the library (after the member-repair change)

The compound-member evaluation (`scripts/compound_members/`) re-implements
the library's member reading instead of calling it:
- `MemberVariant._members` (`scripts/compound_members/readings.py`) repeats
  `CompoundMembers.members`, including the elisions it hands to the repair
  (`MwReading.elisions`);
- `DictionaryFallback.members` (`readings.py`) repeats
  `CompoundMembers.fallback` (`utils/compoundMembers.py`).

Only the parity test on 3,000 headwords keeps the two equal
(`test_conservative_is_what_lemma_mode_ships` in `tests/compound_members/test_readings.py`).

**Target:**
- `MemberVariant` builds on `CompoundMembers`, by subclassing or composition,
  fed with the bulk-read `ComponentTables`/`Headwords`
  (`scripts/compound_members/lexicon.py`).
- The evaluation's options become overrides: the Monier-Williams reading, the
  fallback condition, the repair class and the lemma form.
- The `conservative` variant is then a plain `CompoundMembers`. Keep the
  parity test.

Library code that only the evaluation uses moves to `scripts/` with this item:
- `CompoundForms.stubs()` (`compoundMembers.py`), used once in
  `scripts/compound_members/evaluation.py:172`;
- `HyphenMarks`' `strict=False` option (`AffixProfiles.hyphen_marks` in `readings.py`).

The member-repair change left both in the library.

**Check:** `scripts/evaluate_compound_members.py` must give the same
`report.json` as before the item, apart from timing fields. Run it as
`scripts/compound_members/README.md` describes, and delete
`build/compound-members/analyses.jsonl` first.

## Done by the member-repair change (2026-11-25)

Not part of this handoff; listed so that nobody redoes them:
- the vowel-junction table;
- the policy for unrepaired pairs;
- excluding headword spellings from the preverb runs;
- moving `SandhiRules` to `scripts/`;
- renaming `AffixFilter.members()` and `LemmaLexicon._reader`;
- unused imports;
- the documentation errors the review found;
- the version bump.
