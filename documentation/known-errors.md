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

Found 2026-11-20. Affects the pieces of `mode="parts"`, and in `mode="lemmas"`
the pairs listed under *Still open*.

| Headword | `parts` and the old `members` | Should be | `members` since 2026-11-25 |
|---|---|---|---|
| `mahātman` | `mahā` + `tman` | `mahā` + `ātman` | `mahā` + `tman` |
| `nānārtha` | `nānā` + `rtha` | `nānā` + `artha` | fixed |
| `yathārha` | `yathā` + `rha` | `yathā` + `arha` | fixed |
| `manonuga` | `mano` + `nuga` | `manas` + `anuga` | fixed |
| `citrāyudha` | `ci` + `trāyudha` | `citra` + `āyudha` | none (a tie) |
| `daśāvasthā` | `da` + `śāvasthā` | `daśa` + `avasthā` | fixed |
| `triṣṭubh` | `tri` + `ṣṭubh` | `tri` + `stubh` | fixed |

`daśāvasthā` is "the ten conditions (of a lover)" in Monier-Williams, so its
first member is the numeral `daśa`, not `daśā` "state" (corrected 2026-11-22).

**Cause.** Monier-Williams marks the boundary where the joined spelling is cut,
which after a merged vowel is behind the vowel the second member began with.
In some entries the mark also sits inside a member (`ci—trāyudha`), which looks
like an offset error in how the field was built. `ComponentMarks` cuts the
string as written.

**Fixed in lemma mode on 2026-11-25**, except as listed. `MemberRepair` and
`JunctionSandhi` (`utils/compoundMembers.py`, rule 9 of
[lemma-mode.md](lemma-mode.md)) read a pair with a side that is no Sanskrit
headword again where sandhi joined it: a merged vowel at the mark or up to two
letters after it, a semivowel, an `a` elided where an avagraha marks it, a
retroflex `ṣ`/`ṇ`, and `cch`/`ch`. An avagraha is followed even between two
headwords (`paro—'kṣa` → `paras` + `akṣa`, `go—'śva` → `go` + `aśva`). A pair
no reading makes words of keeps the dictionary's cut.

This replaced the repair of 2026-11-22, which re-cut such a pair with the
vendored splitter's sandhi rules and dropped its members when no re-cut gave
two words. Those rules re-cut any pair, not only one at a junction, and any
headword counted as a word, also `na`, `ī` and `ka`: `vitṛṣṇa` (Yoga Sūtra
1.15) became `vitṛṣ` + `na`, `guṇavatī` became `guṇavat` + `ī`, and
`samaprabha` became `sam` + `aprabha`. The code review of 2026-11-24 counted
258 headwords with a one-letter member and 407 with a new two-letter one, and
12,189 headwords that dropping left with no members, among them correct cuts
such as `para` + `cittajñāna`. Loading the rules also took about 200 MB per
process.

**Still open.**

- **`mahātman`** (3,195 DCS occurrences). `tman` is itself a Vedic word, so
  nothing marks the pair as wrong. Restoring a vowel whenever it is possible
  makes `mahākāla` `mahā` + `akāla` and `rājaputra` `rāj` + `aputra`
  (measured, `p5-vowel-always` in
  [compound-members-evaluation.md](compound-members-evaluation.md)). A fix
  needs to know which words: a short list, or a frequency prior.
- **Names now get their etymology in real words.** Before, they got it in
  junk. `yudhiṣṭhira` had `yudhi` + `ṣṭhira` and has `yudhi` + `sthira`
  (2,689 DCS occurrences, all of the name). `suṣeṇa` had `su` + `ṣeṇa` and has
  `su` + `sena`. A search for `sthira` therefore also reaches Yudhiṣṭhira.
  This is the name problem of section 4: over all words DCS never splits, the
  repair changes the members of 397 headwords (9,025 occurrences), and the
  rest are mostly right (`agniṣṭoma` → `agni` + `stoma`).
- **A restored vowel that reads as a negation.** `yathākramam` → `yathā` +
  `akramam` (298 DCS occurrences, all unsplit), `māhātmya` → `māhā` + `ātmya`.
  Restoring a vowel only before a right side that no word could start with
  (`rtha`, `ṅga`, `ttama`) avoids these. Measured on 2026-11-25, it also loses
  61 DCS-verified occurrences (`mahā` + `aṭavī`, `kathā` + `ādi`), so it was
  not adopted.
- **Rare words a misplaced mark can still spell**, above the dictionary floor:
  `keśaromā` → `keśara` + `umā`, `itihoti` → `itiha` + `ūti`. The length of a
  restored vowel is also a guess: `kanyāgāra` → `kanya` + `āgāra`, where DCS
  has `kanyā`.
- **A mark further off than two letters, or a different bracketing**, keeps
  the dictionary's cut: `go` + `stanākāra` (DCS: `gostana` + `ākāra`),
  `bīja` + `pūrarasa` (DCS: `bījapūra` + `rasa`).
- **`parts` mode** reads the components as written, so every row of the table
  is still wrong there; see section 3 for the same split between the modes.

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

## 4. Compound members that are an affix, or the etymology of a simple word

Found 2026-11-22. Affects `members` in `mode="lemmas"`.

| Headword | `members` | DCS occurrences | Should be |
|---|---|---:|---|
| `tattva` | `tat` + `tva` | 2,315 | none |
| `duryodhana` | `dur` + `yodhana` | 1,458 | none (a name) |
| `dhīmat` | `dhī` + `mat` | 1,059 | none, or `dhī` |
| `tadvat` | `tad` + `vat` | 1,015 | none |
| `āyuṣmat` | `āyus` + `mat` | 917 | none |
| `pitāmaha` | `pitā` + `maha` | 1,488 | none |
| `subhaga` | `su` + `bhaga` | 438 | none |

**Cause.** Monier-Williams writes `—` also before a suffix (`tat—tva`, `dhī—mat`)
and after a particle (`dur—yodhana`, `su—bhaga`), and for the etymology of
lexicalised words and names (`pitā—maha`, `dhṛta—rāṣṭra`).

**Reach.** It is the largest precision cost in lemma mode's members. Of the
occurrences of words DCS never splits that get members today (134,821), a
blind judgment of a frequency-weighted sample rejected 85%: 30% as wrong, 55%
as etymology. Every such occurrence is indexed under its members, so `tattva`
lines are found by a search for `tat`.

**Measured** (`p6-mw-affixes`, `p6-mw-affixes-narrow` in
[compound-members-evaluation.md](compound-members-evaluation.md)). Gluing the
affixes back removes these members (70,130 occurrences with the narrow rule)
and changes no error DCS can check. But the search simulation loses matches,
because gluing a suffix back also removes the useful base. `svarūpatva` had
`svarūpa` + `tva` and is then no longer indexed under `svarūpa`, and
`sumahātman` loses `mahātman` the same way.

**Possible fix.** Keep the base as the only member of a productive formation
(`svarūpatva` → `svarūpa`), and give a lexicalised word none (`tattva`).
- It changes the members contract: a single member would act as an extra
  lemma, so the search side has to agree.
- It needs a test for productive formations; the `-tā`/`-tva` deriver
  ([taddhita-derivation.md](taddhita-derivation.md)) already reconstructs them.
- Measure it with the evaluation before shipping.

## 5. Members in a compound-initial spelling no entry covers

Found 2026-11-22. Affects `members` in `mode="lemmas"`.

| Headword | `members` | Should be |
|---|---|---|
| `tryaha` | `try` + `aha` | `tri` + `aha` |
| `dvyaṅgula` | `dvy` + `aṅgula` | `dvi` + `aṅgula` |
| `haryaśva` | `hary` + `aśva` | `hari` + `aśva` |
| `prāṅmukha`, `udaṅmukha` | `prāṅ`, `udaṅ` + `mukha` | `prāñc`/`prāk`, `udañc`/`udak` + `mukha` |
| `ṣaṇmāsa` | `ṣaṇ` + `māsa` | `ṣaṣ` + `māsa` |
| `catuṣpada` | `catuṣ` + `pada` | `catur` + `pada` |
| `tanmātra` | `tan` + `mātra` | `tad` + `mātra` |
| `antarātman` | `antar` + `ātman` | `antara` + `ātman` |

**Cause.** The citation forms come from Monier-Williams' "in compound for X"
entries (rule 9 of [lemma-mode.md](lemma-mode.md)).
- These spellings have no such entry, or they are a word of their own (`tan`
  is also the root "stretch"), so they are left as written.
- `MemberRepair` does not touch them either, when the spelling is itself a
  headword.

**Reach.** The rows above are 1,628 DCS occurrences. All are among the wrong
members the evaluation still finds for the shipped rule (`conservative`).

**Possible fix.** Read a member that is no Sanskrit headword through
`pausa_readings` (`try` → `tri`, `prāṅ` → `prāk`), and take the reading lemma
mode gives as a lemma of the split piece, guarded as the citation forms are.
Measure it before shipping. `tan` needs the homograph question settled
another way.

## 6. The split ranker: what a trial merge with main showed

Found 2026-11-22, in an uncommitted trial merge of main and branch
`split-ranker` (`hybrid-morphology-v14`). These are not errors on main. They
must be decided when the branch is merged.

- **The branch predates lemma mode.** `LemmaAnalyzer._split` calls
  `analyze_hybrid(text, session=…, _memo=…)`, and the ranker's
  `analyze_hybrid` takes neither argument. The `TypeError` is swallowed, so a
  word on the `"hybrid"` split source comes back unsplit. The trial merge
  calls `analyze_hybrid(text)`.
- **Five main-era tests pin splits made without the ranker.** These fail:
  - `tadarthabhāvanam` is now `tat` + `artha` + `bhāvanam`, not `tadartha` +
    `bhāvanam`; its members test still holds;
  - `rudrakopabhavatvena` ends in `tve` + `ina`, so it no longer exercises the
    `-n` rule (`test_the_n_rule_sees_the_list_it_always_saw`), and it may be a
    worse split;
  - `ko 'pi tatra tatra` gives four roots, not three
    (`test_length_is_measured_before_cleaning_as_the_old_cut_measured_it`);
  - a lone `ऽ` comes back as the word `a`, not as an unknown part
    (`test_token_the_splitter_rejects_is_an_unknown_part`). This comes from
    the branch's avagraha correction meeting lemma mode's token restoration;
  - the `"hybrid"` and `"statistical"` split sources now give the same split
    (`test_hybrid_split_is_really_a_different_split`). Separately, the merged
    scheme detection reads the ASCII test word `xyzabc` as SLP1 (`ḷyṣabc`).
- **Everything else passes**: 741 of 746 tests. The compound-member
  evaluation's member errors are unchanged with the ranker, and its search
  gains grow ([compound-members-evaluation.md](compound-members-evaluation.md),
  "With the split ranker").

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
- [compound-members-evaluation.md](compound-members-evaluation.md), "What
  `conservative` still gets wrong" and "Limitations" (DCS lemmatises a
  participle under its root, so it is not perfect gold).
- [double-j.md](double-j.md), "Known limits".
- [pre-split-compounds.md](pre-split-compounds.md), "Known wart".
- [database-location.md](database-location.md), "Known issue".
- [avagraha-normalization.md](avagraha-normalization.md), "Known duplication".
- [sentence-vs-word-evaluation.md](sentence-vs-word-evaluation.md), "Issues found
  along the way".
