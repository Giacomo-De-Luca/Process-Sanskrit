# Known errors and possible fixes

The central list of analysis errors that are known and not yet fixed. Each entry
gives what goes wrong, why, how far it reaches, and how it could be fixed. Add an
entry when a defect is found and left open; when one is fixed, remove it and
record the fix in the document of the area it touched.

Several areas keep their own lists of open items; they are linked at the end
rather than copied here.

## 1. The masculine forms of some `-an` stems are missing — important

Found 2026-11-20. Affects every mode.

| Input | Now | Should be |
|---|---|---|
| `mahātmā` | `maha` + `atma` | `mahātman` |
| `mahātmānam`, `mahātmānaḥ` | `maha` + `ātman` | `mahātman` |
| `yuvānam` | `yuvā` or `yuṣmad` + `am` | `yuvan` |
| `yūnaḥ` | `yu` + `a` | `yuvan` |

The oblique cases `mahātmanaḥ`, `mahātmanā`, `mahātmanām` and `mahātmabhiḥ`
come back correctly as `mahātman`. The forms that fail are the ones these words
appear in most: `mahātmā` is one of the most frequent words of epic narrative. A
search index never gets its lemma for them, and indexes junk (`atma`, `am`)
instead.

**Cause.** The forms table (`lgtab1` / `lgtab2`) has these stems with the neuter
paradigm `n_an` only (`mahātma`, `mahātmanī`, `mahātmāni`, …). The masculine
forms are not rows, so the forms lookup misses. Monier-Williams has no headword
spelled `mahātmā` either, the splitter leaves the word unsplit, and the fallback
scanner then cuts it into whatever short headwords it finds.

**Reach.** The forms table has 767 `-an` stems with the neuter paradigm only.
For 13 of them Monier-Williams marks a masculine or adjectival use (by a text
match on the entry, so the list is approximate): `agniṣṭomasāman`, `anarvan`,
`arvan`, `atiyuvan`, `devatātman`, `janman`, `kṣatriyayuvan`, `mahātman`,
`parijman`, `punaryuvan`, `satomaghavan`, `vīrahan`, `yuvan`. The other
`-ātman` compounds are not affected: `jitātmā`, `yatātmā`, `durātmā`,
`pāpātmā`, `sarvātmā`, `kṛtātmā`, `viśuddhātmā` and `paramātmā` all reach their
`-ātman` lemma. Whether other stem classes have the same gap (a stem filed under
fewer genders than Monier-Williams gives it) has not been measured.

**Possible fixes.**

- Add the masculine paradigm (`m_an`) for these stems to the forms table in the
  database build. The database is downloaded by `update-ps-database`
  ([database-setup.md](database-setup.md)), so this ships as a rebuilt database,
  not as a runtime patch. Check each stem's gender in Monier-Williams first:
  `janman` is mostly neuter.
- Ship a small supplementary paradigm table with the library, read when the
  forms lookup misses. This needs no database rebuild, but it adds a second
  source of forms.
- In the longer term, take paradigms from the official Heritage data, which has
  a paradigm for each gender ([huet-merge-assessment.md](huet-merge-assessment.md)).

Before choosing, measure the general case: every stem for which Monier-Williams
gives more genders (`mfn.`) than the forms table has paradigms.

**Test once fixed.** `mahātmā`, `mahātmānam`, `mahātmānaḥ` → `mahātman`, and
`yuvā`, `yuvānam`, `yūnaḥ` → `yuvan`, in `roots` and `lemmas` modes.

## 2. Compound members cut after a merged vowel

Found 2026-11-20. Affects `members` in `mode="lemmas"` and the pieces of
`mode="parts"`. Lemmas are not affected.

| Headword | `members` now | Should be |
|---|---|---|
| `mahātman` | `mahā` + `tman` | `mahā` + `ātman` |
| `nānārtha` | `nānā` + `rtha` | `nānā` + `artha` |
| `yathārha` | `yathā` + `rha` | `yathā` + `arha` |
| `manonuga` | `mano` + `nuga` | `manas` + `anuga` |
| `citrāyudha` | `ci` + `trāyudha` | `citra` + `āyudha` |
| `daśāvasthā` | `da` + `śāvasthā` | `daśā` + `avasthā` |

**Cause.** Monier-Williams marks the boundary where the joined spelling is cut,
which after a merged vowel is behind the vowel the second member began with.
In some entries the mark also sits inside a member (`ci—trāyudha`), which looks
like an offset error in how the field was built. `ComponentMarks.members`
(`utils/componentMarks.py`) cuts the string as written.

**Reach.** Of the 105,467 headwords with members, about 312 (0.3%) have a member
that lost its initial vowel, and about 203 (0.2%) have a mark inside a member.
Both counts come from a heuristic (a member that is no headword, where a vowel
plus the member is one, or where neither side is one). The errors are rare, but
they include frequent words such as `mahātman`.

**Possible fix.** When a member is not a headword, put back the vowel the
sandhi absorbed (`a`, `ā`, `i`, `ī`, `u`, `ū`, `ṛ` + the member), and take the
first result that is a Monier-Williams headword (`tman` → `ātman`). The first
member may need its own final restored too (`mano` → `manas`). For a mark inside
a member, re-cut where both sides are headwords. Failing either, report no
members rather than a wrong cut. Measure the result on all 105,467 headwords.

## 3. `parts` mode lets a rare homograph decide a word's pieces

Found 2026-11-20.

| Input | `mode="parts"` now |
|---|---|
| `sukha` | `['su', 'kha']` |
| `bhāvana` | `['bhā', 'vana']` |
| `ayoga` | `['ayo', 'ga']` |

**Cause.** `roots_splitted` (`functions/cleanResults.py`) reads the `components`
of one dictionary entry. Lemma mode was given a vote over all of a headword's
entries on 2026-11-20 (rule 9 of [lemma-mode.md](lemma-mode.md),
`ComponentMarks.prevailing_members`), and `parts` mode was not.

**Possible fix.** Let `roots_splitted` take the reading most of the headword's
entries give. Its vote has to be over `ComponentMarks.parts` rather than over
members, because `parts` also reports prefix boundaries (`nirodha` →
`['ni', 'rodha']`). This changes `parts` output, so check the consumers of that
mode (Sanskrit Voyager) before changing it.

## Open items kept elsewhere

- [analysis-audit.md](analysis-audit.md), "Remaining issues that need separate
  work": `jj` inside compound tokens, the position-dependent `-n` replacement,
  one-character output, the type of an empty `roots` result.
- [lemma-mode.md](lemma-mode.md), "Known limitations".
- [lemma-mode-handoff.md](lemma-mode-handoff.md), "Later, not in this pass":
  participles, absolutives and causatives not linked to their root, Pali
  headwords winning over a failed prefix strip, wider testing of the `-āv` fix,
  the four preverb spelling tables.
- [prefix-rejoin.md](prefix-rejoin.md), "Known defects remaining in the same block".
- [taddhita-derivation.md](taddhita-derivation.md), "Known gaps".
- [double-j.md](double-j.md), "Known limits".
- [pre-split-compounds.md](pre-split-compounds.md), "Known wart".
- [database-location.md](database-location.md), "Known issue".
- [avagraha-normalization.md](avagraha-normalization.md), "Known duplication".
- [sentence-vs-word-evaluation.md](sentence-vs-word-evaluation.md), "Issues found
  along the way".
