# Should the split ranker ship?

24 November 2026

**Open a draft PR for the integrated ranker, keep `coarse` as the proposed default, and hold the release.** The integrated ranker improves both public lemma outputs on every corpus tested. Five integration tests fail, and a configuration error can silently change the lemmas returned to a caller. Those issues deserve the next round of work. Further weight fitting can wait.

## What the branch changes

`split-ranker` (`7879cec`) and `main` (`0421f45`) diverged at `8e1aca9`. The Rust splitter is already on `main`. The branch changes how the pipeline chooses a split from up to 20 candidates, combining model likelihood, word count, corpus frequency and the number of unseen forms. It retires the fixed 0.535 threshold that let the dictionary scanner replace a whole split. The scanner still helps analyze individual unresolved tokens. The branch also revises Unicode, avagraha, final-`o`, whitespace and duplicate-path handling.

The older `fork-sandhi-splitter` branch is already contained in `main`. A direct merge of `split-ranker` into committed `main` produces 24 file conflicts. An existing detached worktree, `Process-Sanskrit-ranker-merge`, already contains a staged resolution, including the newer lemma-mode caller and cache version. I tested that resolution from an isolated copy. The active checkout’s subsequent, uncommitted compound-member changes were outside this comparison.

## What improves for a caller

Each row uses 1,000 reference sentences. Scores are micro F1 percentages. For `lemmas`, I called the public API on the sentence and scored the returned parts. For `roots`, I called it once per whitespace token. Higher is better.

| Held-out sample | `lemmas` main | `coarse` | `fine` | `roots` main | `coarse` | `fine` |
|---|---:|---:|---:|---:|---:|---:|
| WSMP dev | 78.32 | 80.94 | 82.29 | 74.44 | 77.42 | 78.65 |
| General DCS | 69.04 | 71.17 | 71.66 | 65.22 | 67.16 | 67.65 |
| DCS śāstra | 71.53 | 74.38 | 75.23 | 67.58 | 70.75 | 71.55 |

On śāstra, `coarse` improved the lemma score for 197 sentences and lowered it for 35. For `fine`, the counts were 249 and 51. On WSMP they were 132 and 19 for `coarse`, and 186 and 24 for `fine`. That trade-off favors `coarse` for general annotation and makes `fine` a worthwhile option for callers seeking individual compound members.

The direct splitter comparison used the 500 shipped reference tokens in each set, the actual hybrid chooser on `main`, and 20 candidates on each side. An acceptable split follows the reference boundaries or keeps members of the same compound together.

| Sample | Acceptable on main | Acceptable on `coarse` | Fixed / regressed | `coarse` misses with no acceptable candidate |
|---|---:|---:|---:|---:|
| WSMP dev | 378/500 | 436/500 | 62 / 4 | 48 of 64 |
| General DCS | 335/500 | 395/500 | 63 / 3 | 94 of 105 |
| DCS śāstra | 308/500 | 371/500 | 71 / 8 | 109 of 129 |

For example, `coarse` recovered `tam + apṛcchat + yudhiṣṭhiraḥ` where `main` gave `tama + pṛcchat + yudhiṣṭhiraḥ`. On `cetanācetanasvabhāvayoḥ` the change goes the other way. Both readings are available, but `main` keeps `cetana + acetana` and `coarse` chooses `cetanā + cetana`, losing the negative. This is worth a targeted regression case for philosophical prose.

## What blocks release

The branch alone ran 463 tests with no failures and three optional skips. Committed `main` ran 676 with five expected failures and three skips. The trial merge ran 746 with **five unexpected failures** and three skips. All six profile-and-fixture accuracy checks passed, and the Python reference backend passed all 64 ranker tests.

| Trial-merge failure | Observed change | Release decision needed |
|---|---|---|
| Standalone avagraha in explicit IAST input | In `yoga ऽ dharma`, lemma mode assigns the isolated sign the lemma `a`. | Preserve the unknown part while retaining elision recovery in context. |
| Compound grouping | `tadarthabhāvanam` loses the single searchable `tadartha` part. The new parts are `tat`, `artha`, and `bhāvanam`. | Decide whether the grouped headword must remain searchable. Both versions preserve `bhāvana`. |
| Abstract-noun ending | `rudrakopabhavatvena` ends in `tvena`, which yields the roots `tve` and `ina`. The earlier reading also fragments the word. | Pin the intended analysis of the ending before changing the verb regression test. |
| Hybrid fallback | The former hybrid path returned `ab` for `xyzabc`. The new path reports it as unknown. | Update the test and the split-source documentation to reflect the deliberate removal of the old gate. |
| Repeated word at the length boundary | Whole-input `ko 'pi tatra tatra` now preserves both occurrences of `tatra`. The test used the old collapsed output to infer which path ran. | Test the length boundary directly and pin the desired word count separately. |

**A sixth issue sits outside those five tests.** With a misspelled `PROCESS_SANSKRIT_SPLIT_PROFILE`, `roots` raises `SplitRankerError`. On the same input, `lemmas` catches the exception in `LemmaAnalyzer._split` and continues with the scanner. I reproduced `astyuttarasyāṃdiśi` changing from three parts to five, including spurious `sya` and `am`. Add tests for a bad profile and a damaged frequency table. Configuration and asset errors should reach the caller while genuine unparseable tokens retain their fallback.

Direct users of the hybrid helpers also need a migration note. The branch removes `score_threshold`, `session` and `_memo` from those functions, removes `sandhiSplitScorer`, and changes the meaning of detailed scores. The public `process()` signature is preserved. The trial already updates internal callers and distinguishes the weighted profiles in cache keys.

## Where to spend the next iteration

1. Use the staged trial as the starting point for a draft PR. Reconcile the five tests, expose ranker configuration errors, and rerun database-backed discovery with the pending compound-member changes included. The PR’s database-free CI suite alone would miss the five observed failures.
2. For further accuracy gains, inspect the missing candidates first. They account for 75–90% of remaining `coarse` misses on these fixtures. Separate reference-annotation conventions from genuine omissions such as a missing compound member. For ranking changes, start with a small reviewed set of long-vowel and negative-`a` boundaries where the right candidate already exists.
3. Retain 20 candidates for `process()` for now. The branch’s earlier trials of 50 and 100 cost more time for little downstream gain. In a fresh local timing check on 200 WSMP sentences, median time was 1.35 s on `main` and 1.25 s on `coarse` for lemma mode. Word-by-word roots took 1.42 s and 1.26 s. Three warm runs with overlapping timing ranges support treating speed as secondary to accuracy.

## Verification details

I used Python 3.12, the same installed native extension and copies of the same SQLite lexicon. The Rust source is identical at the two tips. Full discovery used `PROCESS_SANSKRIT_FULL_NATIVE_PARITY=1 uv run --no-sync python -m unittest discover -s tests`. Cross-platform installed-wheel jobs were outside the local run.

The WSMP sample uses the branch evaluator’s alignment filter and seed 13. The DCS and śāstra samples replay the texts and gold lemmas in `split-ranker:documentation/dcs-evaluation/results.json`. Their samples can overlap. All corpus calls used explicit IAST and disabled the analysis cache. Each nonempty lemma list counted as one prediction with alternative readings. Scoring used the branch’s one-to-one sentence-level matcher and preverb conventions. This measures lemma recovery. It gives no extra credit for optional dictionary members or word order, so it cannot settle the search consequences of losing a grouped headword. The direct split fixtures also favor difficult, multi-segment inputs. Their percentages describe that subset.
