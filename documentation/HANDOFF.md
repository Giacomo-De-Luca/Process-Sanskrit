# Status: app-root vs library evaluation → fixes COMPLETE (2026-07-13)

All work described in the earlier version of this file is finished and validated. This file now serves as a changelog; safe to delete once reviewed. Nothing is committed — the working tree also contains unrelated pre-existing modifications.

## What was evaluated

The four parsing differences between the legacy app (`references/app-root/app/processSanskrit.py`, not importable — needs Flask + PostgreSQL) and the library. An A/B harness re-implements the legacy strategies on top of the library and toggles each change in isolation:

- **Harness:** `tests/test_reference_comparison.py` (12 tests)
  - `.venv/bin/python -m unittest tests.test_reference_comparison` — <1s
  - `.venv/bin/python -m tests.test_reference_comparison --report` → `tests/results/reference_comparison_report.txt`
- **Pre-fix baseline report:** `tests/results/reference_comparison_report_baseline.txt`

**Verdicts:** (1) scored compound cuts: net positive vs longest-match, flaws now fixed; (2) all-dictionary lexicon: huge perf win (~1ms O(n) scan → 0.07µs cached), quality-neutral; (3) prefix stripping in `root_any_word`: positive after the precedence fix below; `samādhi` never broke, `samādhipariṇāmaḥ` (broken in app-root) works; (4) IAST switch: parity with app-root tables modulo documented additions, locked by a test.

## Fixes implemented (all validated, uncommitted)

1. **Prefix precedence** — `process_sanskrit/functions/rootAnyWord.py`: new `allow_prefixes` kwarg; the `variableSandhi` recursion passes `False` so a prefix split found deep in one sandhi-variant branch can no longer preempt a whole-word match reachable through a later variant. `utkrāntiś → utkrānti` (was `ut + krānti`), `apratisaṃkramāyās → apratisaṃkrama`; the `asaṃsargaḥ → a + saṃsarga` gain is preserved. Yoga-Sutra-wide: divergences from sensible app-root analyses dropped 2 → 0, gains kept.
2. **Vowel-restoration-aware remainder scoring** — `process_sanskrit/functions/compoundAnalysis.py`: `evaluate_compound_split` retries the remainder with the initial vowels sandhi may have swallowed (same table `root_compounds` uses). `rājapuruṣottamaḥ → rājapuruṣa + uttamaḥ` (was `rāja + puruṣottamaḥ`).
3. **Balance bonus keyed to the first part alone** — same file: the +0.2 bonus is now awarded on `len(first_part) >= 2` and no longer looks at the tail at all. Any tail-dependent term (its length, or its plausibility) lets a shorter cut outscore a longer one whenever the shorter cut happens to leave a nicer tail — that is exactly how `dev + aq` beat `deva + q`, and, in an intermediate attempt that keyed the bonus to tail *continuability*, how `he + tur` beat `hetu` on `heturdvividhe`. Tail evidence is already priced in by the ±0.4 remainder term; with the bonus tail-independent, equally-evidenced cuts tie and the longest wins for free (the cursor walks longest-first and a tie does not displace the incumbent). `devaq → deva`, matching what app-root's longest-match gave. The gate's real job — refusing `ka`/`sa` over-splits — is unaffected (`devaka` + junk scores 0.10). Guarded by `test_no_shorter_cut_can_outscore_a_longer_one_on_a_junk_tail`.
4. **Whole-word dictionary credit** — same file: an exact headword match leaves no remainder to justify, so the remainder credit counts as satisfied. Previously it scored 0.4, fell below the gate, and fragmented dictionary words missing from the forms DB: `vṛtti` became `vṛ + tti`, and `process('yogaścittavṛttinirodhaḥ')` emitted a junk `vṛ` root — it now returns `cittavṛtti` correctly.

## Validation results

- 18 tests green (12 harness + 6 `tests/test_optimizations.py`).
- Report diff vs baseline is minimal and contains no regressions: `rājapuruṣottamaḥ` fixed in both scored configs, prefix divergences 2 → 0, one neutral shift on `āgacchantv` (`cham` → `chand`; both configs, both wrong, neither is the gold). Every other gold, benchmark and Yoga-Sutra case is byte-identical to baseline.
- Pipeline spot-checks: `samādhipariṇāmaḥ → ['samādhi', 'pariṇāma']`, `samādhi → ['samādhi', 'samādhin']`, `yogaścittavṛttinirodhaḥ → [('yoga','yogas'), 'cittavṛtti', 'nirodha']`.
- Perf: 5.5ms avg per benchmark compound (baseline 3.4ms; the extra cost is the vowel-restoration `root_any_word` probes, all memoized per request).

## Design note for whoever touches the scorer next

The scoring bug class here is subtle and recurred twice under different disguises: **any term in the score that depends on the tail can promote a shorter cut over a longer one.** Keep the tail's contribution confined to the ±0.4 remainder-evidence term, and keep the +0.2 bonus a function of `first_part` only. If you add a signal (e.g. attestation weighting, below), check it against `devaq → deva` and `heturdvividhe → hetu + dvividhe`, which are the two guards for this.

## Optional follow-ups (not started)

- **Attestation weighting:** obscure single-dictionary headwords still create occasional junk cuts (`bhāvepyeteṣāṃ` gets `pyā`). `DICTIONARY_REFERENCES[word]` returns the attesting dictionaries, a free scoring signal. Tune with the harness.
- **Dead code:** the explicit nested-prefix loop in `rootAnyWord.py` is likely unreachable (the remainder recursion already nests prefixes implicitly). Verify and remove.
- **Pre-existing bug (both versions):** multichar sandhi variants (`o → aḥ`) over-advance `current_pos` by 1 in `root_compounds`.
- 8/17 gold cases return whole (compound lexicalized in the forms DB) under every config; the gold set can't discriminate strategies there.
