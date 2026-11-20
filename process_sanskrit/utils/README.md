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
  letter read as the sound it stands for, `-āv` as `-au`), prefix lists (`SANSKRIT_PREFIXES`,
  `PREFIX_LOOKUP_SPELLINGS`), `SUPPLETIVE_ROOTS` (the root a reader looks up
  for a verb the table files elsewhere: `paś` → `dṛś`), `samMap`, filtered
  words.
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
  every marked piece for `mode="parts"`, compound members for `mode="lemmas"`,
  and (`prevailing_members`) the members of the reading most of a headword's
  entries give, so a rare homograph cannot decide a common word (`sukha`).
- `compoundVerbs.py`: `CompoundVerbIndex`, the preverb compounds Monier-Williams
  records with their parse (`prati+i` → `pratī`), read once per database; lemma
  mode's key for a preverb and the part after it.
- `preverbSandhi.py`: `PreverbSandhi`, preverbs joined to a stem as the
  dictionaries spell the compound (`upa` + `sam` + `arc` → `upasamarc`); the key
  where Monier-Williams records none. Needs no database.
- `analysisTrace.py`: `PrefixTrace` and `ScanTrace`, optional records of how an
  analysis was reached (which prefix was stripped, which pieces the scanner
  matched and how many characters it skipped). A trace only observes.
- `analysisCache.py`: the persistent split and morphology cache, and the
  algorithm versions that key it.

## Other

- `processAPI.py`: a small Flask wrapper around the public functions.

See the documents in `../../documentation/` for the rules these modules encode:
`dictionary-results.md`, `one-letter-roots.md`, `prefix-segmentation.md`,
`lemma-mode.md`, `local-cache.md`.
