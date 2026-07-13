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
publish nothing. See `documentation/publishing.md` for the four-platform native
wheel matrix, sdist and installed-wheel gates, approval flow, and release steps.

## Architecture

Public API (`process_sanskrit/__init__.py`) exports three functions:
- `process` (`functions/process.py`) — the main pipeline: transliteration detection → sandhi splitting → compound analysis → root/inflection lookup → dictionary entries. `mode='roots'` returns stems only.
- `dict_search` (`functions/dictionaryLookup.py`) — multi-dictionary lookup.
- `transliterate` (`utils/transliterationUtils.py`) — scheme detection + conversion to IAST.

Key layers:
- `functions/` — the cascading pipeline: `rootAnyWord.py` (stem identification), `inflect.py` (inflection tables), `sandhiSplitter.py` / `hybridSplitter.py` / `compoundAnalysis.py` (splitting), `SQLiteFind.py` (DB queries), `taddhitaDerivation.py` (productive `-tā`/`-tva` abstract nouns), `cleanResults.py` (output shaping), `model_inference.py` / `processBYT5.py` (optional BYT5 path).
- `splitter/` — the public split-only facade, the default native Rust backend, and the vendored Python differential reference. See `documentation/rust-splitter.md` for architecture, backend selection, builds, assets, release validation status, and remaining publication work; the original vendoring contract is in `documentation/sandhi-splitter.md`.
- `utils/` — database session management (`databaseSetup.py`, with `session_scope`/`with_session`/`requires_database` decorators), transliteration, lexical resources, dictionary reference tables.
- Persistent split/morphology caching is documented in `documentation/local-cache.md`. A result-changing hybrid/process change must bump `ANALYSIS_ALGORITHM_VERSION`; a direct statistical change must bump `STATISTICAL_ANALYSIS_ALGORITHM_VERSION` (both if both paths change), or stale rows can mask the change.
- Splitter-only baseline and Python/Rust benchmark procedure are documented in `documentation/rust-splitter-benchmark.md`.
- Pre-split compounds (`-`/`+`) and option forwarding through the recursive `process()` calls are documented in `documentation/pre-split-compounds.md`.
- Avagraha glyph normalization (OCR/PDF apostrophe variants) is documented in `documentation/avagraha-normalization.md`.
- Productive `-tā`/`-tva` abstract nouns (`niṣyandatā`) are reconstructed from their base; see `documentation/taddhita-derivation.md`.
- The `word_list` dictionary index is *derived* from the dictionary tables and is rebuilt, never patched; see `documentation/word-list-index.md`. It also flags bare Monier-Williams variant-reading pointers such as `tanni`, which rank behind genuine compound cuts while remaining eligible fallbacks. The flag must stay lexical: keying it to "heads no inflection table" instead silently destroys `gacchatā` → `gacchat`, since `gacchat` and `niṣyanda` have no paradigm either and are perfectly real.
- `setup/updateDB.py` — the `update-ps-database` console script.

## Conventions and cautions

- Pipeline processing is in IAST, while the private splitter backend boundary uses canonical SLP1 as documented in `documentation/rust-splitter.md`.
- Do not "modernize" the vendored Python reference modules toward upstream style — they are pinned by `tests/test_splitter_parity.py`; behaviour changes there silently alter split results.
- The character class in `preprocess` (`process.py`) must keep `-` escaped and last. Folding it back into a range (`*-+`) silently turns `-` into a range operator and strips hyphens, disabling pre-split compounds. Pinned by `tests/test_presplit_options.py`.
- The recursive `process()` calls inside `handle_special_characters` re-enter the public entry point. Any new parameter added to `process` must also be added to the `forwarded` dict there, or it is silently dropped for wildcard and pre-split input.
- `process` is optimized for single words; sentences should be split on whitespace by callers.
- Root logging is silenced to CRITICAL at import time in `__init__.py` (the vendored splitter logs every sandhi rule at DEBUG).
- The `-tā`/`-tva` deriver must stay *after* the whole-word dictionary lookup in `process()` and out of `root_any_word`. Moving it earlier lets a manufactured analysis outrank an attested word (`vārtā` → `vār` + `tā`). Pinned by `tests/test_taddhita_derivation.py`.
