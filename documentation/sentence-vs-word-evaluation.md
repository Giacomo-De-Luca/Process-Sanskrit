# Sentence vs word processing, and how to annotate books

Evaluation of 2026-11-12. The data, reviewer verdicts and prototype scripts are in
`documentation/wsmp-evaluation/`, which is **local only**: it is excluded by
`.gitignore` and is not in the repository, so a fresh clone has only this summary.

## Questions

1. Is `process()` more accurate when given a whole sentence or one word at a time?
2. What is the F1 once known annotation-convention differences are no longer
   counted as errors?
3. For annotating books, should the base text be turned into stems or into a sandhi
   split?

## The two paths are different code, not different context

- **Sentence input** (any space in the text) skips the whole single-word block of
  `process()`. That block holds special-character handling, `root_any_word`, the
  `ṅ`/`ṁ` retries, the direct dictionary lookups, the whole-word headword insert and
  the `-tā`/`-tva` deriver. The whole string goes to `analyze_hybrid` once, so one
  score and one statistical-vs-`root_compounds` decision cover the entire sentence.
  The result is one flat list with no mapping back to tokens.
- **Single-word input** looks the word up first and stops as soon as the lexicon
  resolves it. It falls back to the splitter only when that fails.

So the comparison is really *splitter-first* against *lexicon-first*. The splitter
does respect whitespace, so word boundaries usually agree between the paths.
`morphologicalParallel2.py` in the travelogue project calls `process()` once per
whitespace token. Book annotation has therefore always used the single-word path.

## Method

- **Gold.** A seed-13 sample of 1000 WSMP dev sentences (Krishnan, Kulkarni & Huet
  2021, CC BY 4.0), restricted to the 982 that also carry a ByT5 annotation, so all
  pipelines are scored on the same sentences.
- **Metric.** A bag of lemmas per sentence, the only granularity every pipeline
  shares. A roots-mode tuple counts as one predicted item, and it is correct if any
  of its alternatives is correct. Predicted and gold items are paired by maximum
  one-to-one bipartite matching.
- **Pipelines.**

  | pipeline | what it is |
  |---|---|
  | `sentence` | `process(sentence)` |
  | `word` | `process(token)` per whitespace token |
  | `word-nohw` | `word`, minus the inserted whole-word headword |
  | `byt5` | ByT5's own lemmas |
  | `byt5+ps` | ByT5 split, then `process()` on each segment (the current book pipeline) |
  | `byt5+ps-nohw` | `byt5+ps`, minus inserted headwords |
  | `byt5+ps-sfx` | `byt5+ps-nohw` with `-tva`/`-tā` suffix segments re-attached to the segment before them |

- **Convention levels.** These are cumulative. Each level adds credit only for a
  documented convention difference, never for a different word.

  | level | adds |
  |---|---|
  | L0 | exact lemma match only |
  | L1 | a separate preverb is accepted for a gold lemma with the preverb fused (`upa` + `jan` for `upajan`); an unmatched bare preverb is not counted as a false positive; DCS pronoun lemmas are mapped (`tvad` → `yuṣmad`) |
  | L2 | a lexicalised compound kept whole matches the run of gold members it spells, counted as one gold unit; a predicted base followed by `tva`/`tā` matches a gold whole `-tva`/`-tā` noun |
  | L3 | a verb root matches a gold participle, absolutive or infinitive (DCS lemmatises these to the root, WSMP to the participle stem) |

- **Blind review.** Six Sonnet reviewers judged 120 sentences, stratified by how the
  two modes differ. Each saw gold, sentence-mode output and word-mode output, with
  the modes shown as X and Y at random. They rated the gold and listed real system
  errors, separately from convention differences.

## Results

| pipeline | L0 | L1 | L2 | L3 | best estimate |
|---|---|---|---|---|---|
| `sentence` | 0.751 | 0.776 | 0.822 | 0.830 | ≈ 0.81 |
| `word` | 0.719 | 0.743 | 0.804 | 0.811 | ≈ 0.79 |
| `word-nohw` | 0.739 | 0.764 | 0.827 | 0.835 | ≈ 0.81 |
| `byt5` | 0.859 | 0.860 | 0.862 | 0.939 | ≈ 0.93 † |
| `byt5+ps` | 0.807 | 0.829 | 0.831 | 0.837 | ≈ 0.83 |
| `byt5+ps-nohw` | 0.844 | 0.867 | 0.869 | 0.875 | ≈ 0.87 |
| `byt5+ps-sfx` | 0.844 | 0.868 | 0.869 | 0.875 | ≈ 0.87 |

The `-nohw` rows drop only the genuine insert: a leading dictionary entry whose
headword equals the surface carried by the last entry. A first version also dropped
any leading 3-field dictionary entry, which on the split path removed real first
segments (`abhimanyu` from `abhimanyuvadhūgarbhe`, `dṛṣṭvā` from `dṛṣṭvaiva`). The
correction moved these rows by at most 0.003.

**How the best estimate is formed.** Each rule's credits were checked against
evidence:

- *Compound-kept-whole credits (L2).* Cross-checked against the reviewers on the
  reviewed sentences: 64 of 90 credits were accepted and 26 were flagged as real
  errors (e.g. `kharadūṣaṇa`, a dvandva of two names, and `tanmūla`). So only about
  71% of the L1→L2 gain counts.
- *Derivative-to-root credits (L3).* A spot check found about 90% of them correct.
  Wrong ones include `vidhā` = `ukta` and `vā` = `vaktum`.

† ByT5's L3 gain is almost entirely the participle-to-root convention (505
credits). Its figures may also be inflated by training-set overlap; see Caveats.

**What the blind reviewers found.**

- **The gold.** They rated it correct for 97 of 120 sentences, minor issues for 23,
  and wrong for none. There was one high-confidence split error: `trikṛtvaḥ` is cut
  into `tri` + `kṛtvas`, but it is the single adverb `trikṛtvas`, which both modes
  and the frontend keep whole. There were two wrong genders in the tags, which don't
  affect lemmas. The systems were right where the gold was wrong only 4 times.
- **The gold's oddity is a consistent convention.** It splits every transparent
  compound, including lexicalised ones (`bhūtala`, `mahāmuni`, `svargaloka`,
  `ādityavāra`). It keeps absolutives as their own lemma (`chittvā`), and it tags
  `eva` as "prep.".
- **System errors.**

  | | `sentence` | `word` | `word`, headword items excluded |
  |---|---|---|---|
  | total errors | 203 | 232 | 187 |
  | spurious items | 38 | 84 | |
  | wrong splits | 41 | 29 | |

  45 of word mode's errors land on the inserted headword.
- **Weighted to the whole sample.** Weighting each stratum by its size, word mode
  makes about 351 more errors than sentence mode per 1000 sentences as it stands,
  and about 31 fewer once headword items are excluded.

## What the numbers say

- **Sentence vs word.** Sentence mode's lead at L0/L1 has two sources:
  - Word mode inserts the whole-word headword (`vāsukiḥ` beside `vāsuki`; Apte's
    nominatives `paraḥ`, `śatruḥ`).
  - Word mode keeps lexicalised compounds whole (`kṛtopanayana`), which WSMP always
    splits.

  Without headwords and with the granularity convention neutralised, the two paths
  are level (≈ 0.81). The reviewers put word mode slightly ahead, because it makes
  fewer wrong splits. Word mode is also about 3× faster: 5.1 s against 14.2 s for
  1000 sentences.
- **The headword insert is the largest fixable loss.** It costs 0.04 F1 in the book
  pipeline (`byt5+ps` 0.83 → `byt5+ps-nohw` 0.87). This is the bug behind the
  proposed `mode="lemmas"`: a headword should be a fallback alternative, not an
  extra item.
- **Dropping the headword is not the fix either.** Word mode inserted 421 headwords
  across the 982 sentences:
  - **In 304 sentences the headword was only noise:** Apte's nominative beside the
    stem (`paraḥ` + `para`), a citation spelling (`tat` + `tad`), or a finite-verb
    headword from the BHS or Pali dictionaries (`jānāti` + `jñā`, `arhati` + `arh`).
  - **In 21 sentences it was the only correct lemma.** Examples: the feminine
    `śarkarā`, where the inflection table returns `śarkara`; `bhāratī`; and
    absolutives such as `kṛtvā`, which the gold keeps as their own lemma.

  Appending the headword is wrong roughly 15 times as often as dropping it. Offering
  it as an alternative reading is right in both cases.
- **`-tva`/`-tā` re-attachment does nothing measurable on WSMP.** The gold has very
  few such words. It matters for śāstra prose: the old Yoga Sūtra run lost 15 of
  1179 words to detached `tvāt`/`tvāni`. Note that re-gluing alone is not enough:
  `adhimātratvāt` → `adhi` + `mātratva`, because prefix stripping wins before the
  deriver runs.
- **ByT5 split + per-segment `process()` beats `process()` alone** (≈ 0.87 vs
  ≈ 0.81). ByT5's own lemmas score higher still, subject to the contamination
  caveat below.

## Issues found along the way

- **The headword insert lands as a separate item.** `process.py` inserts the
  whole-word headword with no `entry[4]`, so `extract_roots` reports it as a
  separate item rather than an alternative: `gacchati` → `['gacchati', 'gam']`. The
  roots format cannot express "X or (A and B)" at all.
- **`extract_roots` folds identical adjacent words.** On sentence input it treats
  two adjacent identical words as rival readings of one span: `tatra tatra` →
  `['tatra']`.
- **`max_length` truncates silently.** The default `max_length=100` cut sentence
  input without warning; 4 of the 1000 sampled sentences were affected. Fixed
  since: a longer text is analysed word by word, see
  [long-input.md](long-input.md).
- **Errors shared by both modes:**
  - causatives shattered into fragments (`dāpayet`, `yojayiṣyasi` → `id`, `yu`, `ji`)
  - broken `-ayan` participles
  - lost initial `ā` (`āśraya` → `aśraya`)
  - `akrūra` read as `krūra`
- **Dead or overreaching preprocessing.**
  - `text[0:1] == "ch"` (`process.py`) can never be true, so the `ch` → `ś` retry is
    dead code.
  - `preprocess` rewrites every `jj` as `j j`, which sends single words such as
    `sajjana` down the sentence path.
- **Headwords from non-Sanskrit or empty entries.** The headword for finite verbs
  mostly comes from BHS and CPED (Pali). `gacchanti`'s only headword is an empty
  CPED entry (`[None]`), which the insert accepts.

## Caveats

- **ByT5 contamination.** WSMP sentences come from DCS, and ByT5's conventions
  match DCS exactly: participles lemmatised to the root, `tvad` for the second
  person pronoun, `-tva` split off, and preverbs fused (`parivarjay`). It was almost
  certainly trained on DCS-derived data, and whether these sentences were in its
  training split is unknown. Treat the `byt5*` rows as upper bounds until they are
  measured on held-out text.
- **Genre.** WSMP is mostly short epic and purāṇic verse. Śāstra prose with long
  compounds (the Yoga Sūtra and its commentaries) is not represented.
- **Heuristics.** The L2 matcher compares consonant skeletons and ignores vowels, so
  a wrong-vowel split can earn credit. It is confined to the runs of members inside
  one gold token. The L3 rule pairs roots with derivatives by their first sound.
  Both rules over-credit somewhat, as quantified above.
- **Review depth.** One reviewer per sentence, with no measurement of agreement
  between reviewers. The gold "issues" were mostly low-confidence.

## Recommendation for annotating books

**Make the sandhi split the base layer and derive lemmas from it as a second layer
that keeps its alternatives. Do not store stems alone.**

- **Conventions live in the lemma layer.** Every convention difference measured
  here belongs to the lemma layer, not the split: compound granularity is worth
  4–6 F1 points, and participle vs root is worth 7 points for ByT5. A stems-only
  output freezes one convention for good. A split with lemma alternatives can be
  projected later to either the DCS or the Heritage convention, and to whatever the
  search side needs.
- **Context is needed only for the split.** The split is the step that needs
  context, which is what ByT5 supplies. Lemmatising each segment works best
  word-by-word: it is grounded in the dictionary, it is fast, and it is accurate
  once the headword insert is fixed. The best measured `process()`-based pipeline is
  exactly ByT5 split → per-segment `process()`.
- **The split can be checked.** It can be re-joined and compared with the original
  text, so a reader can see and correct an error. A sequence of stems cannot.

The concrete pipeline:

1. Keep ByT5's split, lemma and morphology per segment, aligned to its span in the
   original text.
2. Run `process()` per segment. `mode="lemmas"` now exists
   ([lemma-mode.md](lemma-mode.md)): the headword is an alternative lemma of a
   one-part word, listed last. It has no `suffix` part; a `-tva`/`-tā` derivative
   is one part whose lemma is the derivative. In the other modes, treat the
   inserted headword as an alternative reading of the segment, not as another
   part. It is the leading entry whose headword equals the last entry's `[4]`.
   Dropping it outright loses the only correct lemma in about 5% of cases.
   When ByT5's lemma is `tva`/`tā`, take the preceding segment as the base instead
   of analysing the suffix on its own.
3. Store per segment:
   - the split form
   - ByT5's lemma
   - `process()`'s alternatives, with their source and role
   - an agreement flag: is ByT5's lemma among the alternatives?

   ByT5 then disambiguates in context, the library supplies the dictionary-grounded
   candidates, and disagreements are flagged for review.
4. Index lemmas for search, and keep the split forms for display, highlighting and
   exact-phrase search.

Do not use sentence mode for books. It is slower, it loses repeated words, it maps
nothing back to tokens, and it is no more accurate once conventions are accounted
for.

## Next steps

1. Fix the headword insert (`mode="lemmas"` design), then rerun `evaluate_conventions.py`.
2. Measure ByT5 on held-out text: either a WSMP subset from texts outside its
   training split, or a small hand-checked śāstra sample.
3. Measure the agreement flag: how often is ByT5's lemma among `process()`'s
   alternatives, and which one wins when they disagree?
4. Turn the prototype into a config-driven benchmark with regression floors. The
   data location (a copy in `tests/datasets/` or an external path in config) is
   still to be decided.
