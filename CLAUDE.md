# CLAUDE.md

## RULES


- After implementing significant changes, write them into a relevant file in the documentation folder, or update existing documentation files. In the CLAUDE.md keep just pointers to the documentation files. 

- Use **uv run python** when you need to launch python. 

- Avoid code duplication whenever possible. Employ a modular approach using classes rather than standalone functions. If you find dysfunctional pattern or duplication in existing code, allert the user directly, before attempting to fix them. 

- Prefer reusable utility functions inside the utils folder rather than stand alone calculation functions.

- Prefer configuration files to command line interfaces. 

- If some of the istructions are unclear or you encounter unexpected roadblocks, alert the user and ask for clarification, rather than writing code that was not agreed upon. 

- If you make a plan, always define and plan tests first, or use a test generting agnet, then run the code against those tests after. 

- NEVER stash changes without being directly asked. 

- NEVER ever commit. 

- For folders with multiple scripts or data files, add a readme explaining both the structure of the folder, the main classes or data structures present there. 

- After finishing a plan, always use the agent: *code-quality-reviewer* to review the quality of the generated code. 




## Project overview

Process-Sanskrit is a Python library for automatic Sanskrit text annotation and inflected dictionary search. It handles text in any transliteration scheme, undoes sandhi, splits compounds, and returns stems, grammatical tags, inflection tables, and dictionary entries. It implements the cascading analysis system from the NAACL 2025 paper "Accessible Sanskrit" and powers the Sanskrit Voyager backend.

## Setup

```bash
pip install -e .            # or .[byt5] for the experimental BYT5 model
update-ps-database          # downloads/sets up the SQLite database (~583 MB) into process_sanskrit/resources/
                            # or from Python: process_sanskrit.update_database(); see documentation/database-setup.md
```

Almost nothing works without the database (`process_sanskrit/resources/SQliteDB.sqlite`); only `transliterate` is database-free.

## Running tests

Tests use `unittest` and live in `tests/`:

```bash
python -m unittest tests.test_splitter_parity   # vendored splitter behaviour pins
python -m unittest tests.test_optimizations
python -m unittest tests.test_reference_comparison
python -m unittest tests.test_analysis_cache tests.test_database_lifecycle
python -m unittest tests.test_taddhita_derivation
python tests/runBenchmarks.py                   # benchmark suite / Yoga Sutra analysis
```

`tests/test_splitter_parity.py` has an upstream-parity half that is skipped unless `sanskrit-parser==0.2.6` and `gensim` are installed (intentionally not dependencies).

## Releasing

Bump the synchronized versions in `pyproject.toml` and `Cargo.toml`, refresh
`Cargo.lock` and the generated notices, then push to `main`; unchanged versions
publish nothing. See `documentation/publishing.md` for the five-platform native
wheel matrix, sdist and installed-wheel gates, approval flow, and release steps.

## Architecture

Public API (`process_sanskrit/__init__.py`) exports three functions:
- `process` (`functions/process.py`) — the main pipeline: transliteration detection → sandhi splitting → compound analysis → root/inflection lookup → dictionary entries. `mode='roots'` returns stems only; `mode='lemmas'` returns, for each input token, the parts of the word and their alternative lemmas.
- `dict_search` (`functions/dictionaryLookup.py`) — multi-dictionary lookup.
- `transliterate` (`utils/transliterationUtils.py`) — scheme detection + conversion to IAST.

Key layers:
- `functions/` — the cascading pipeline: `wordCascade.py` (the single-word cascade `process()` and lemma mode share), `lemmaAnalysis.py` (`mode="lemmas"`), `rootAnyWord.py` (stem identification), `inflect.py` (inflection tables), `sandhiSplitter.py` / `hybridSplitter.py` / `compoundAnalysis.py` (splitting), `SQLiteFind.py` (DB queries), `taddhitaDerivation.py` (productive `-tā`/`-tva` abstract nouns), `cleanResults.py` (output shaping), `model_inference.py` / `processBYT5.py` (optional BYT5 path).
- `splitter/` — the public split-only facade, the default native Rust backend, and the vendored Python differential reference. See `documentation/rust-splitter.md` for architecture, backend selection, builds, assets, release validation status, and remaining publication work; the original vendoring contract is in `documentation/sandhi-splitter.md`.
- `utils/` — database session management (`databaseSetup.py`, with `session_scope`/`with_session`/`requires_database` decorators), transliteration, lexical resources, dictionary reference tables.
- Persistent split/morphology caching is documented in `documentation/local-cache.md`. A result-changing hybrid/process change must bump `ANALYSIS_ALGORITHM_VERSION`; a direct statistical change must bump `STATISTICAL_ANALYSIS_ALGORITHM_VERSION` (both if both paths change), or stale rows can mask the change.
- Splitter-only baseline and Python/Rust benchmark procedure are documented in `documentation/rust-splitter-benchmark.md`.
- The split ranker (`SplitRanker`) and the six splitter input corrections are developed on the `split-ranker` branch and are not on main; main still ranks candidate splits with `functions/sandhiSplitScorer.py` and lets `root_compounds` replace a statistical split scoring below 0.535 in `hybridSplitter.py`.
- Pre-split compounds (`-`/`+`) and option forwarding through the recursive `process()` calls are documented in `documentation/pre-split-compounds.md`.
- The explicit `input_scheme` argument of `process()` (bypassing scheme auto-detection), scheme-name validation in `transliterate()`, and why the recursion pins it to `"IAST"` are documented in `documentation/input-scheme.md`.
- Why `transliterate()` stays on `indic_transliteration` rather than vidyut-lipi (the frontend's engine), the measured differences, the vidyut-lipi bugs found, and the cost of a shared or proprietary Rust engine are in `documentation/transliteration-assessment.md`.
- Analysis data-flow fixes, regression coverage, and remaining parsing issues are documented in `documentation/analysis-audit.md`.
- Known analysis errors that are still open, with cause, reach and possible fix, are listed in `documentation/known-errors.md`, which also links the per-area lists. The most important is the missing masculine paradigm of some `-an` stems (`mahātmā` → `maha` + `atma`). Add an entry when a defect is found and left unfixed.
- Compound `members` in lemma mode come from the reading most of a headword's Monier-Williams entries give (`ComponentMarks.prevailing_members`, rule 9 of `documentation/lemma-mode.md`), so a one-entry homograph cannot cut `sukha` into `su` + `kha` or `bhāvana` into `bhā` + `vana`. Pinned by `PrevailingMemberTests` and `LexiconTests`.
- Huet inflection data — the local export's coverage and how it lost readings, the official Heritage XML that supersedes it (source URL, version, LGPLLR/CC BY-NC licences), and merge feasibility — is assessed in `documentation/huet-merge-assessment.md`. The Huet data is not used at runtime.
- Sentence vs single-word `process()`, convention-neutral F1 of seven annotation pipelines on WSMP gold, and the recommendation for annotating books (sandhi split as base layer, lemmas with alternatives on top) are in `documentation/sentence-vs-word-evaluation.md`; the reviewer verdicts, data excerpts and frozen prototype scripts are in `documentation/wsmp-evaluation/`, which is local only (gitignored, not in the repository).
- Avagraha glyph normalization (OCR/PDF apostrophe variants) is documented in `documentation/avagraha-normalization.md`.
- Productive `-tā`/`-tva` abstract nouns (`niṣyandatā`) are reconstructed from their base; see `documentation/taddhita-derivation.md`.
- Prefix re-joining in `cleanResults.py` (collapsing a `sam` + `ādhi` split back into `samādhi`) is documented in `documentation/prefix-rejoin.md`. `dict_search` never returns `None` — a miss comes back as a *stub* whose slot `[2]` is a list rather than a dict, so a re-join may only fire on `isinstance(voc_entry[0][2], dict)`. Testing the stub for `is not None` overwrites a correct prefix analysis with a headword that does not exist. The three prefix blocks are now one `rejoin_prefix()` over the `REJOINABLE_PREFIXES` table, whose value per prefix is *every stem `root_any_word` emits for the prefix itself* (`sam`→`sa`, `ava`→`av`) — the fillers the walk must step over to reach the real stem. Omitting one halts the walk on the filler and silently loses the join; do not "simplify" the table. `dict_search` already folds `sam`→`saṃ` (`samMap`), so no spelling retry belongs here, and `DICTIONARY_REFERENCES` is *not* a safe pre-check (`samyoga` is absent from it yet resolves). That fold is lookup-only — `dict_search` echoes the query back at slot `[0]` — so the merged lemma is taken from the payload headword (`_canonical_headword`), which costs no extra lookup and stops the re-join being the only path that reports `samvedana` where the forms DB and MW both say `saṃvedana`. `duḥ` is in the table too (with an empty absorbed set): it is no upasarga, it only ever arrives as a compound cut. Pinned by `tests/test_prefix_merge.py`.
- Prefix *segmentation* — a stripped upasarga is a word, not a rival reading of the stem — is documented in `documentation/prefix-segmentation.md`. `entry[4]` is overloaded: `extract_roots` folds entries sharing it into one tuple of alternatives, while the `-n` lemma rule and the whole-word dictionary check read it expecting the whole word *as matched*. So prefix entries must keep their own surface and only the entries that resolved the remainder get the whole word stamped (`_stamp_whole_word`). Every tempting simplification breaks one of the three consumers; the four that were tried and their casualties are listed in the doc. Pinned by `tests/test_prefix_segmentation.py`.
- A prefix stripped in a sandhi spelling (`vy`, `praty`, `ud`) is looked up under `PREFIX_LOOKUP_SPELLINGS` (`utils/lexicalResources.py`); see the sandhi-spelling section of `documentation/prefix-segmentation.md`. A spelling added to `SANSKRIT_PREFIXES` without an entry there is stripped and then dropped, or read as an unrelated word that shares the spelling.
- `nir-` is read as the prefix `nis` from the splitter's `niḥ` token (`Inflector.SPLIT_TOKEN_SPELLINGS`); see the `nir-` section of `documentation/prefix-segmentation.md`. Do not add `nir` to `SANSKRIT_PREFIXES` instead: the stripper runs before the dictionary lookup and the `-tā`/`-tva` deriver, and `nirvikāratvāt` comes back as `nis` + `vikāratva`. Pinned by `NirPrefixTests`.
- One-letter verb roots (`i`, `ṛ`) on the split path are documented in `documentation/one-letter-roots.md`. The one-letter filter in `process()` and `Inflector._fragments` must change together: relaxing the filter alone readmits scanner fragments (`ayet` out of `arcayet`).
- Which marks of a dictionary `components` field are boundaries in `parts` output is documented in `documentation/dictionary-results.md`: `/` and `^` are accent marks, not boundaries.
- Lemma mode (`mode="lemmas"`: one item per input token, parts AND, lemmas OR) is documented in `documentation/lemma-mode.md`; it is built by `LemmaAnalyzer`/`LemmaBuilder` in `functions/lemmaAnalysis.py`. The traces it reads (`utils/analysisTrace.py`, filled by `root_any_word`, `root_compounds` and `Inflector.resolve`) only observe: those functions must return the same with and without one. `LemmaBuilder` must never modify the entries it reads or call `clean_results`. A preverb is an ordinary part and a verb is never re-joined with it; the scanner's pieces are kept unless it skipped two or more characters (`ScannerPolicy`). Two notions of headword are kept apart: `attested_headword` (`functions/dictionaryLookup.py`; any dictionary, also what `dict_search` uses) and `LemmaLexicon.has_sanskrit_headword` (Monier-Williams or Apte; the test for letting a whole word stand in place of its preverb split, directly or through the noun re-join). Dictionary patterns (`_`, `%`, `*`) are honoured only for one-token input; in running text they are dropped, or a stray `%` returns every headword. Avagrahas are restored over the whole text and cut back by token (`restore_avagraha_by_token`), never token by token. The part after a preverb also carries the compound per lemma (`pratī`, built by `LemmaBuilder.join_preverbs`/`preverb_keys`): the Monier-Williams headword whose recorded parse matches (`utils/compoundVerbs.py`), else the compound `PreverbSandhi` spells (`utils/preverbSandhi.py`); never a `+` key, which search tokenizers cut, a word typed as a Monier-Williams or Apte headword is one of its own lemmas (`with_typed_headword`), and the two passes run in that order in `LemmaBuilder.finished`. Parts are memoised and shared between repeats of a word, so a part is never modified once made. Pinned by `tests/test_lemma_builder.py`, `tests/test_word_cascade.py` and `tests/test_lemma_mode.py`.
- `SQLite_find_verb` returns one entry per root of a verb form the table files under several (`kariṣyati` → `kṝ`, `kṛ`), the old last-match answer first. Only lemma mode reads them all: `LemmaAnalyzer.analyze` holds `every_verb_root()` (`functions/rootAnyWord.py`), and outside it `_direct_roots` reports the first root, so `roots`, `parts` and detailed output are unchanged. Do not let those modes read every root before `clean_results`' prefix re-join and `-n` rule handle alternatives: the re-join leaves the other roots as separate pieces (`saṃkariṣyati` → `saṃkṝ`, `kṛ`). `SUPPLETIVE_ROOTS` (`utils/lexicalResources.py`) adds `dṛś` beside `paś` in lemma mode, for verbal entries only. All in the "Verb forms of several roots" section of `documentation/lemma-mode.md`; pinned by `tests/test_shared_verb_forms.py`.
- The first pass of the lemma-mode plan in `documentation/lemma-mode-handoff.md` is done; that file now holds its results, the older defects found on the way (the last entry `clean_results` never visits; the `jj` rewrite and `-āv` are fixed), and the items still open, among them extensive testing of the `-āv` fix and a de-duplication of the four preverb spelling tables.
- A final `-āv` reads `-au` (`samādhāv` → `samādhi`) via `read_final_sandhi` (`utils/lexicalResources.py`) in `WordCascade` and a retry in `root_any_word` for split tokens; hybrid cache at `hybrid-morphology-v13`. Measured on the Yoga Sūtra only, so it needs extensive testing; see `documentation/final-av.md`, pinned by `tests/test_final_av.py`.
- A `jj` is cut as `j j` (the sandhi of t/d + j, `tajjapas`) only in a word the single-word cascade cannot resolve whole; `lajjā`, `majjati`, `lajj` keep it. Decided by `WordCascade.read_double_j`, called on the cleaned text in `process()` and `LemmaAnalyzer._words`, never in `clean_iast`. No cache bump was needed, because no cache key that existed before changes its computation. See `documentation/double-j.md`; pinned by `tests/test_double_j.py`.
- `process()` never cuts its input; see `documentation/long-input.md`. A text longer than `max_length` (measured before cleaning, as the old cut measured it) is analysed word by word (`_process_each`), with pattern characters removed from its words, and a single word is analysed whole. Do not put a length limit back into `preprocess`. Pinned by `tests/test_long_input.py`.
- The `word_list` dictionary index is *derived* from the dictionary tables and is rebuilt, never patched; see `documentation/word-list-index.md`. It also flags bare Monier-Williams variant-reading pointers such as `tanni`, which rank behind genuine compound cuts while remaining eligible fallbacks. The flag must stay lexical: keying it to "heads no inflection table" instead silently destroys `gacchatā` → `gacchat`, since `gacchat` and `niṣyanda` have no paradigm either and are perfectly real.
- `setup/updateDB.py` — `update_database()` (exported as `process_sanskrit.update_database`, raises `DatabaseUpdateError`) and the `main()` behind the `update-ps-database` console script. Every write is staged as a sibling and atomically renamed via `utils/atomicFiles.py`, so no partial `SQliteDB.sqlite` survives an interrupt; helpers raise with the cause chained and only `main()` prints; the library never downloads on its own. See `documentation/database-setup.md`.

## Conventions and cautions

- Pipeline processing is in IAST, while the private splitter backend boundary uses canonical SLP1 as documented in `documentation/rust-splitter.md`.
- Do not "modernize" the vendored Python reference modules toward upstream style — they are pinned by `tests/test_splitter_parity.py`; behaviour changes there silently alter split results.
- The character class in `preprocess` (`process.py`) must keep `-` escaped and last. Folding it back into a range (`*-+`) silently turns `-` into a range operator and strips hyphens, disabling pre-split compounds. Pinned by `tests/test_presplit_options.py`.
- The recursive `process()` calls (wildcards, pre-split compounds, and the word-by-word fallback for long text) re-enter the public entry point. Any new parameter added to `process` must also be added to `_forwarded_options` (`process.py`), or it is silently dropped on those paths. The exception is `input_scheme`, which is pinned to `"IAST"` there because the text is already IAST (see `documentation/input-scheme.md`).
- `process` is optimized for single words; sentences should be split on whitespace by callers (`mode="lemmas"` does this itself).
- Root logging is silenced to CRITICAL at import time in `__init__.py` (the vendored splitter logs every sandhi rule at DEBUG).
- The `-tā`/`-tva` deriver must stay *after* the whole-word dictionary lookup in the single-word cascade (`WordCascade.resolve`, `functions/wordCascade.py`, which `process()` and lemma mode both use) and out of `root_any_word`. Moving it earlier lets a manufactured analysis outrank an attested word (`vārtā` → `vār` + `tā`). Pinned by `tests/test_taddhita_derivation.py`.
