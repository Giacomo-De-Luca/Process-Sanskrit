# Packaged resources

Data files the library reads at runtime. `*.json`, `*.csv`, `*.tsv` and `*.xlsx`
files here ship in the wheel and the sdist; the SQLite lexicon does not and is
installed separately.

| file | read by | contents |
|---|---|---|
| `SQliteDB.sqlite` | `utils/databaseSetup.py`, `utils/dictionary_references.py` | the lexicon: dictionaries, inflection tables and the derived `word_list` index. Downloaded by `update-ps-database`; see [`documentation/database-setup.md`](../../documentation/database-setup.md). Not in version control. |
| `type_map.tsv` | `utils/loadResources.py` | grammatical-type abbreviations and their descriptions, loaded at import. |
| `MWKeysOnly.json` | `utils/loadResources.py` | Monier-Williams headwords, loaded only when `mwdictionaryKeys` is requested. |
| `stopwords.csv` | `utils/stopwordsRemoval.py` | the stopword list. |
| `HuetInflectionTables/` | nothing at runtime | local Huet exports; see [`documentation/huet-merge-assessment.md`](../../documentation/huet-merge-assessment.md). |
