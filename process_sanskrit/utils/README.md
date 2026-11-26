# Utilities

Shared helpers used by the analysis functions in `../functions/`. Nothing here
runs the pipeline; these modules hold data, connections and small reusable rules.

## Lexicon and resources

- `databaseSetup.py`: engine and session management for the read-only lexicon
  (`session_scope`, `with_session`, `requires_database`).
- `resourcePaths.py`, `loadResources.py`: where packaged resources and the
  database live, and loading the small ones (`type_map`).
- `dictionary_references.py`: `DICTIONARY_REFERENCES`, the headword → dictionaries
  mapping read from the `word_list` index.
- `wordListBuilder.py`: rebuilds that derived index from the dictionary tables.
- `lexicalResources.py`: sandhi tables and `read_final_sandhi` (a word's last
  letter read as the sound it stands for, `-āv` as `-au`), `pausa_readings`
  (the pausa spellings a compound-initial form stands for: `mano` → `manaḥ`),
  prefix lists (`SANSKRIT_PREFIXES`, `PREFIX_LOOKUP_SPELLINGS`),
  `MEMBER_PARTICLES` / `MEMBER_SUFFIXES` (pieces of a hyphenated analysis that
  are no compound member), `SUPPLETIVE_ROOTS` (the root a reader looks up for
  a verb the table files elsewhere: `paś` → `dṛś`), `samMap`, filtered words.
- `atomicFiles.py`: stage-and-rename file publication for the lexicon updater.

## Text

- `transliterationUtils.py`: `transliterate`, scheme-name resolution, avagraha
  normalisation and restoration (`restore_avagraha`, and
  `restore_avagraha_by_token` for a text already cut into tokens).
- `detectTransliteration.py`: transliteration-scheme detection.
- `stopwordsRemoval.py`: the stopword list.

## Analysis support

- `paradigm.py`: reading a stored inflection table (`tags_for`), telling a
  verbal table from a nominal one (`is_verbal_paradigm`, and
  `is_verbal_entry` for a whole morphology entry), and the one-letter
  verb-root test (`is_one_letter_root`).
- `componentMarks.py`: `ComponentMarks` reads a dictionary `components` string —
  every marked piece for `mode="parts"`, Monier-Williams' compound members, and
  (`votes`, `prevailing_members`) the members of the reading most of a
  headword's entries give, so a rare homograph cannot decide a common word
  (`sukha`); another dictionary's marks can be read through a `reader`.
  `letters` gives the composed (NFC) letters of a string; `parts` and `members`
  do not compose first, because composing 70 ddsa analyses would change `parts`.
  `elisions` finds the members written after an avagraha (`mano—'nuga`).
- `compoundMembers.py`: `CompoundMembers`, the members `mode="lemmas"` reports
  (rule 9 of `lemma-mode.md`): Monier-Williams' vote, Edgerton's and Grassmann's
  `-` for headwords Monier-Williams lacks (`HyphenMarks`; `AffixFilter` glues
  prefixes and suffixes back, but not a run of preverbs that is a word), the
  repair of a pair that sandhi hid (`MemberRepair`, which picks among the
  readings `JunctionSandhi` makes at the junction: merged vowels, semivowels,
  an elided `a`, retroflex `ṣ`/`ṇ`, `cch`), and citation forms from
  Monier-Williams' "in compound for" entries, read per homograph and once per
  database (`CompoundForms`). It reads only the database, never the splitter's
  sandhi rules.
- `compoundVerbs.py`: `CompoundVerbIndex`, the preverb compounds Monier-Williams
  records with their parse (`prati+i` → `pratī`), read once per database; lemma
  mode's key for a preverb and the part after it.
- `preverbSandhi.py`: `PreverbSandhi`, preverbs joined to a stem as the
  dictionaries spell the compound (`upa` + `sam` + `arc` → `upasamarc`); the key
  where Monier-Williams records none. `spellings` lists every spelling a run of
  up to three preverbs takes before what follows (`upā`, `prot`, `abhy`).
  Needs no database.
- `engineCache.py`: `EngineCache`, one value per SQLAlchemy engine, made once
  under a lock and dropped with the engine (weak keys: every in-memory database
  has the URL `sqlite://`). `CompoundVerbIndex` and `CompoundForms` keep a table
  in one (`table`), `PieceLemmas` and `LemmaLexicon` a dict filled word by word
  (`memo`). `EngineCache.clear_all()` empties every instance; it is called by
  `_reset_reference_state` in `dictionary_references.py`.
- `analysisTrace.py`: `PrefixTrace` and `ScanTrace`, optional records of how an
  analysis was reached (which prefix was stripped, which pieces the scanner
  matched and how many characters it skipped). A trace only observes.
- `analysisCache.py`: the persistent split and morphology cache, and the
  algorithm versions that key it.

## Other

- `processAPI.py`: a small Flask wrapper around the public functions.

See the documents in `../../documentation/` for the rules these modules encode:
`dictionary-results.md`, `one-letter-roots.md`, `prefix-segmentation.md`,
`lemma-mode.md`, `local-cache.md`, `compound-members-evaluation.md`.
