# CLAUDE.md

## RULES



- After implementing significant changes, write them into a relevant file in the documentation folder, or update existing documentation files. In the AGENTS.md keep just pointers to the documentation files. 

- Use **uv run python** when you need to launch python. 

- Avoid code duplication whenever possible. Employ a modular approach using classes rather than standalone functions. If you find dysfunctional pattern or duplication in existing code, allert the user directly, before attempting to fix them. 

- Prefer reusable utility functions inside the utils folder rather than stand alone calculation functions.

- Prefer configuration files to command line interfaces. 

- If some of the istructions are unclear or you encounter unexpected roadblocks, alert the user and ask for clarification, rather than writing code that was not agreed upon. 

- If you make a plan, always define and plan tests first, or use a test generting agnet, then run the code against those tests after. 

- Never stash changes without being directly asked. 

- For folders with multiple scripts or data files, add a readme explaining both the structure of the folder, the main classes or data structures present there. 

- After finishing a plan, always use the agent: *code-quality-reviewer* to review the quality of the generated code. 




## Project overview

Process-Sanskrit is a Python library for automatic Sanskrit text annotation and inflected dictionary search. It handles text in any transliteration scheme, undoes sandhi, splits compounds, and returns stems, grammatical tags, inflection tables, and dictionary entries. It implements the cascading analysis system from the NAACL 2025 paper "Accessible Sanskrit" and powers the Sanskrit Voyager backend.

## Setup

```bash
pip install -e .            # or .[byt5] for the experimental BYT5 model
update-ps-database          # downloads/sets up the SQLite database (~583 MB) into process_sanskrit/resources/
```

Almost nothing works without the database (`process_sanskrit/resources/SQliteDB.sqlite`); only `transliterate` is database-free.

## Running tests

Tests use `unittest` and live in `tests/`:

```bash
python -m unittest tests.test_splitter_parity   # vendored splitter behaviour pins
python -m unittest tests.test_optimizations
python -m unittest tests.test_reference_comparison
python -m unittest tests.test_analysis_cache tests.test_database_lifecycle
python tests/runBenchmarks.py                   # benchmark suite / Yoga Sutra analysis
```

`tests/test_splitter_parity.py` has an upstream-parity half that is skipped unless `sanskrit-parser==0.2.6` and `gensim` are installed (intentionally not dependencies).

## Architecture

Public API (`process_sanskrit/__init__.py`) exports three functions:
- `process` (`functions/process.py`) — the main pipeline: transliteration detection → sandhi splitting → compound analysis → root/inflection lookup → dictionary entries. `mode='roots'` returns stems only.
- `dict_search` (`functions/dictionaryLookup.py`) — multi-dictionary lookup.
- `transliterate` (`utils/transliterationUtils.py`) — scheme detection + conversion to IAST.

Key layers:
- `functions/` — the cascading pipeline: `rootAnyWord.py` (stem identification), `inflect.py` (inflection tables), `sandhiSplitter.py` / `hybridSplitter.py` / `compoundAnalysis.py` (splitting), `SQLiteFind.py` (DB queries), `cleanResults.py` (output shaping), `model_inference.py` / `processBYT5.py` (optional BYT5 path).
- `splitter/` — **vendored** reduced copy of `kmadathil/sanskrit_parser` v0.2.6 (MIT, see `splitter/NOTICE.md` and `LICENSE.upstream`). Provides `Parser.split()` only. Two deliberate substitutions: a precomputed marisa-trie (`forms.trie`) replaces the sqlite/generative validity oracle, and a numpy scorer replaces gensim — the scorer intentionally reproduces gensim quirks (saturated-term skip, integer-division sigmoid scale); "fixing" them changes which split wins. `tools/build_splitter_data.py` regenerates `splitter/data/` from upstream.
- `utils/` — database session management (`databaseSetup.py`, with `session_scope`/`with_session`/`requires_database` decorators), transliteration, lexical resources, dictionary reference tables.
- Persistent split/morphology caching is documented in `documentation/local-cache.md`.
- External lexicon database paths are documented in `documentation/database-location.md`.
- `setup/updateDB.py` — the `update-ps-database` console script.

## Conventions and cautions

- All internal processing is in IAST; input is auto-transliterated on entry.
- Do not "modernize" or refactor `splitter/` toward upstream style — it is pinned by `tests/test_splitter_parity.py`; behaviour changes there silently alter split results.
- `process` is optimized for single words; sentences should be split on whitespace by callers.
- Root logging is silenced to CRITICAL at import time in `__init__.py` (the vendored splitter logs every sandhi rule at DEBUG).
