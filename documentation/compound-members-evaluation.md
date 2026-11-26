# Compound members for lemma search: which reading errs least

This document evaluates the options in
[compound-members-brief.md](compound-members-brief.md) on the Digital Corpus of
Sanskrit (DCS): 739,200 sentences, 4.1 M tokens. The evaluation lives in these
files:

- code: [`scripts/compound_members/`](../scripts/compound_members/README.md);
- settings: [`benchmarks/compound-members.json`](../benchmarks/compound-members.json);
- hand-checked cases: [`benchmarks/compound-members-cases.json`](../benchmarks/compound-members-cases.json);
- tests: the package [`tests/compound_members/`](../tests/compound_members/README.md).

Measured 2026-11-22 on the main branch with the installed database, and again
on 2026-11-25 (library 1.8.0), after the merged-vowel repair was rebuilt to
work at the sandhi junction only (*The merged-vowel repair*). The results
tables give the second run, which also follows an avagraha between two
headwords (`go—'śva` → `go` + `aśva`, not `śva`). The `conservative` variant is what lemma mode
ships (`utils/compoundMembers.py`, rule 9 of [lemma-mode.md](lemma-mode.md)):
the evaluation builds it from the library's own classes, and lemma mode's
members equal it on all 206,670 headwords with components.

## Answer

| point | option | verdict |
|---|---|---|
| 1 | Edgerton/Grassmann members when Monier-Williams **lacks the headword** | **yes**: 241 headwords gain correct members and no search match is lost. The brief's filters suffice here. |
| 1 | the same also when Monier-Williams has the word but gives no members | **no**: it gives Grassmann's etymologies to frequent words (`pṛthak` → `pṛth` + `ak`) and, with the brief's filters, `sukha` → `su` + `kha` |
| 2 | members in citation form via Monier-Williams' "in compound for X" | **yes, guarded**: follow an entry only where the stub is mostly such an entry and lemma mode gives X for the stub's split spelling |
| 3 | Monier-Williams `-` between two words | **not as proposed**: its gains come with members for the etymology of very frequent simple words (`nṛpa` → `nṛ` + `pa`, `dvija` → `dvi` + `ja`). A blind judgment rejects 87% of the members it adds to words DCS cannot check, and 91% with a three-letter minimum |
| 4 | ties give members | **no**: on words DCS splits, it gives 2,424 occurrences wrong members (`mahāsura` → `mahā` + `sura`) against 1,205 right |
| — | merged-vowel repair (known-errors.md §2), when a member is no headword | **yes, at the junction only**: reading the pair again where sandhi joined it fixes 6,042 occurrences (the `ṛṣi` family, `mahārṇava`, `kāmārtha`, `daśāvasthā`) and breaks 16. Where no reading helps, keeping the dictionary's cut beats dropping the members. It does not fix `mahātman`. |
| — | the same repair also before a second member that is a word | **no**: it adds 12,516 errors (`drākṣārasa` → `drākṣā` + `arasa`, `mahāpadma` → `mahā` + `apadma`) |
| — | prefixes and suffixes glued back in Monier-Williams' own members (not in the brief) | **not yet**: it removes junk such as `tattva` → `tat` + `tva` (70k occurrences), but also the useful base member of productive formations (`svarūpatva` loses `svarūpa`), and the search loses matches |

The combination of the "yes" rows (`conservative`) does best overall, and its
gains are robust:

- errors fall from 35,582 to 20,488 weighted DCS occurrences (−42%; −52%
  weighted by split occurrences only);
- the search simulation reaches more occurrences of a typed stem (+1,769
  [+1,229, +2,288]) and of an inflected form (+1,745 [+918, +2,746]), and
  loses none in the sample;
- it adds members to words DCS cannot check in only 433 occurrences, half of
  them Grassmann's `agnīṣoma` (`agnī` + `ṣoma`);
- it gets 36 of the 45 hand-checked cases right, and adds members to none of
  the words that should have none.

## Results

### Errors against DCS, and the search simulation

The error columns count wrong and missing members, weighted by DCS
occurrences. The "index" yardstick takes the members the index holds for a
split compound; the "lemma" yardstick takes citation lemmas (both are
explained under *What is measured*).

- **Unverified added / removed**: occurrences of words DCS never splits
  that the variant newly gives members, or whose members it removes.
- **Search**: occurrences reached beyond the baseline, net over the strata,
  with a 95% bootstrap interval.
- **Cases**: right (correct or acceptable) / missed / spurious / wrong, out of
  45.

The search figures compare variants within one run only. Its sample is
stratified by which variants change a headword, so it changes with the list
of variants: the second run's figures differ from the first's even for
variants whose members did not change (`p2-xref-guarded`).

| variant | errors (index) | Δ | Δ, split occ. only | Δ (lemma) | unverified added / removed | search: stem query | search: form query | cases |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline` | 35,582 |  |  |  | 0 / 0 |  |  | 24 / 11 / 0 / 10 |
| `p1-bhs` | 34,805 | −777 | −767 | −649 | 154 / 0 | +507 [+317, +748] | +14 [0, +40] | 27 / 8 / 0 / 10 |
| `p1-bhs-gra` | 34,487 | −1,095 | −1,025 | −908 | 433 / 0 | +525 [+334, +762] | +14 [0, +40] | 27 / 8 / 0 / 10 |
| `p1-bhs-gra-brief-filters` | 34,487 | −1,095 | −1,025 | −908 | 454 / 0 | +527 [+334, +764] | +14 [0, +40] | 27 / 8 / 0 / 10 |
| `p1-loose` | 30,953 | −4,629 | −1,388 | −4,252 | 15,033 / 0 | +571 [+363, +818] | +1,414 [+37, +4,133] | 27 / 7 / 0 / 11 |
| `p1-loose-brief-filters` | 30,945 | −4,637 | −1,389 | −4,260 | 66,950 / 0 | +573 [+367, +820] | +1,421 [+39, +4,140] | 24 / 7 / 3 / 11 |
| `p2-xref` | 29,808 | −5,774 | −3,212 | −13,156 | 0 / 0 | −54 [−1,272, +691] | +730 [+71, +1,656] | 29 / 11 / 0 / 5 |
| `p2-xref-guarded` | 27,829 | −7,753 | −3,929 | −9,907 | 0 / 0 | +555 [+251, +967] | +875 [+220, +1,765] | 29 / 11 / 0 / 5 |
| `p3-hyphens` | 32,539 | −3,043 | −1,742 | −1,957 | 23,713 / 0 | +2,144 [+44, +6,200] | +2,107 [−301, +6,376] | 24 / 9 / 0 / 12 |
| `p3-hyphens-min3` | 32,938 | −2,644 | −1,612 | −1,812 | 11,845 / 0 | +2,108 [+15, +6,162] | +2,077 [−352, +6,308] | 24 / 11 / 0 / 10 |
| `p3-hyphens-any-left` | 32,137 | −3,445 | −2,131 | −2,262 | 35,318 / 0 | +2,917 [+135, +7,640] | +2,483 [+40, +6,935] | 25 / 6 / 0 / 14 |
| `p4-ties` | 34,352 | −1,230 | −550 | −1,170 | 7,792 / 0 | +209 [+61, +414] | +481 [+89, +1,094] | 23 / 8 / 2 / 12 |
| `p5-junction` | 29,598 | −5,984 | −4,725 | −3,705 | 0 / 0 | +453 [+207, +719] | +711 [+238, +1,394] | 27 / 11 / 0 / 7 |
| `p5-junction-drop` | 31,215 | −4,367 | −3,166 | −2,294 | 0 / 21,672 | +302 [−18, +641] | +563 [+50, +1,303] | 27 / 11 / 0 / 7 |
| `p5-vowel-always` | 48,098 | +12,516 | +7,006 | +4,508 | 0 / 0 | +259 [−222, +612] | +662 [+184, +1,368] | 26 / 11 / 0 / 8 |
| `p6-mw-affixes` | 45,207 | +9,625 | +3,485 | +8,975 | 0 / 83,365 | −1,933 [−5,038, −205] | −3,888 [−13,546, +2,143] | 24 / 11 / 0 / 10 |
| `p6-mw-affixes-narrow` | 35,572 | −10 | −43 | −7 | 0 / 70,128 | −1,821 [−4,942, −111] | −3,559 [−13,170, +2,660] | 24 / 11 / 0 / 10 |
| `brief-literal` | 20,532 | −15,050 | −6,973 | −21,501 | 150,001 / 0 | +5,123 [+988, +11,056] | +5,324 [+1,693, +10,252] | 29 / 0 / 6 / 10 |
| `conservative` | 20,488 | −15,094 | −9,876 | −14,817 | 433 / 0 | +1,769 [+1,229, +2,288] | +1,745 [+918, +2,746] | 36 / 8 / 0 / 1 |
| `conservative-drop` | 22,199 | −13,383 | −8,227 | −13,325 | 175 / 21,672 | +1,613 [+1,016, +2,128] | +1,597 [+759, +2,611] | 36 / 8 / 0 / 1 |
| `conservative-hyphens` | 17,215 | −18,367 | −11,831 | −17,062 | 24,146 / 0 | +3,976 [+1,472, +8,200] | +3,982 [+1,178, +8,909] | 37 / 6 / 0 / 2 |
| `conservative-hyphens-min3` | 17,659 | −17,923 | −11,661 | −16,862 | 12,278 / 0 | +3,926 [+1,434, +8,159] | +3,929 [+1,131, +8,793] | 36 / 8 / 0 / 1 |
| `conservative-affixes` | 30,654 | −4,928 | −5,862 | −5,275 | 433 / 83,365 | −203 [−3,393, +1,661] | −2,223 [−11,730, +4,184] | 36 / 8 / 0 / 1 |
| `conservative-affixes-narrow` | 20,536 | −15,046 | −9,862 | −14,770 | 433 / 70,128 | −76 [−3,182, +1,877] | −1,850 [−11,340, +4,607] | 36 / 8 / 0 / 1 |

The error columns cannot charge a variant for members it adds to a word DCS
never splits: such members are *unverified*. The search simulation measures
recall, which extra members can only raise. The *unverified added* column is
therefore the precision cost: every one of those occurrences is indexed
under the added members. `brief-literal` has the fewest errors only because
it pays the largest such cost (150,001 occurrences, `sukha` → `su` + `kha`
among them).

### With the split ranker

These figures are from the first run (2026-11-22), with the merged-vowel
repair of that date (`p5-vowel-unattested`, since replaced by
`p5-junction`); they were not measured again.

The `split-ranker` branch was cut before lemma mode reached main, so it cannot
run the evaluation as it stands. This comparison was made before `conservative`
moved into the library, so its figures belong to that run; within it, the two
sides are paired. The evaluation was run a second time on an
uncommitted trial merge of main and `split-ranker` (2026-11-22,
`hybrid-morphology-v14`).

- **Member errors are identical for every variant.** They compare dictionary
  readings with DCS, and lemma mode's analysis of the segments, stems and
  stubs they use does not depend on the ranker.
- **The search sample is the same**, so the two runs are paired. With the
  baseline's members the ranker alone reaches a few more occurrences (stem
  queries 64,924 of 78,278 against 64,516; form queries 129,185 of 182,281
  against 128,765).
- **Every ranking above holds.** The variants that help reach more
  occurrences with the ranker, because it cuts more compounds into pieces and
  members matter more. The variants that hurt lose more.

| variant | stem query, main | stem query, with ranker | form query, main | form query, with ranker |
|---|---:|---:|---:|---:|
| `p1-bhs-gra` | +638 [+399, +885] | +695 [+441, +976] | +32 [0, +69] | +37 [0, +86] |
| `p2-xref-guarded` | +927 [+464, +1,398] | +1,293 [+693, +2,105] | +2,184 [+937, +3,482] | +3,156 [+1,172, +5,027] |
| `p3-hyphens-min3` | +222 [+121, +364] | +219 [+109, +339] | +210 [−532, +767] | +182 [−502, +665] |
| `p4-ties` | +58 [+30, +91] | +113 [+46, +217] | +123 [+47, +216] | +264 [+62, +556] |
| `p5-vowel-unattested` | +593 [+347, +971] | +619 [+375, +985] | +948 [+295, +1,826] | +1,014 [+356, +1,869] |
| `p5-vowel-always` | −5,400 [−11,528, −1,692] | −8,910 [−21,245, −2,065] | −8,018 [−14,962, −1,912] | −17,010 [−38,221, −2,436] |
| `p6-mw-affixes-narrow` | −1,295 [−2,541, −290] | −1,208 [−2,443, −234] | −776 [−1,353, −285] | −630 [−1,106, −226] |
| `conservative` | +2,192 [+1,599, +2,846] | +2,653 [+1,933, +3,505] | +3,164 [+1,738, +4,868] | +4,208 [+2,220, +6,072] |
| `conservative-hyphens-min3` | +2,479 [+1,901, +3,148] | +3,018 [+2,232, +3,806] | +3,469 [+1,871, +5,069] | +4,599 [+2,468, +6,607] |

### Words DCS cannot check

The error counts cannot judge members of words DCS never splits. A fresh
agent judged a sample of them blind, without the variant that produced them or
any of the analysis above. It gave each analysis one of three verdicts:

- `yes`: a transparent compound with the right members;
- `etymology`: historically a compound, but lexicalised, so the members would
  mislead a search (`dvija` → `dvi` + `ja`);
- `no`: a wrong cut, a non-word piece, or an affix taken as a member.

Per option the sample is the 15 most frequent such words plus 15 drawn at
random. The estimate weights the verdicts by DCS occurrences: exact for the 15
most frequent, a ratio estimate for the rest. 182 distinct analyses were
judged.

| members added to words DCS never splits, by | occurrences | rejected (`no` + `etymology`) | of which `no` |
|---|---:|---:|---:|
| `p1-bhs-gra` (strict fallback) | 426 | 62% | 60% |
| `p1-loose` | 15,072 | 98% | 36% |
| `p3-hyphens` | 24,490 | 87% | 20% |
| `p3-hyphens-min3` | 12,306 | 91% | 30% |
| `p4-ties` | 7,838 | 97% | 83% |
| `p5-vowel-unattested` (its re-cuts, the repair of 2026-11-22) | 11,250 | 55% | 37% |
| the baseline's own members of such words | 134,821 | 85% | 30% |

- **Strict fallback.** Its 62% is almost all one word: Grassmann's `agnīṣoma`
  → `agnī` + `ṣoma`, 216 of the 426 occurrences. Without it, 23% are
  rejected. `ālayavijñāna`, `hāhākāra`, `pṛthivīcāla` and `asthiśakalā` are
  judged right.
- **Points 1-loose, 3 and 4.** What they add on such words is overwhelmingly
  etymology or wrong. For point 3: `nṛpa`, `dvija`, `oṣadhi`, `vanaspati`,
  `paraspara` (etymology), and `kaścana`, `itihāsa`, `purātana` (no).
- **The baseline.** The library as it stands gives members to such words in
  134,821 occurrences, and the judge rejects 85% of them: `yudhiṣṭhira` →
  `yudhi` + `ṣṭhira`, `tattva` → `tat` + `tva`, `duryodhana` → `dur` +
  `yodhana`, `dhīmat` → `dhī` + `mat`, `tadvat` → `tad` + `vat`; `pitāmaha`,
  `tathāgata` and `dhṛtarāṣṭra` are etymology. This is the largest precision
  cost in lemma mode's members today, and no option of the brief touches it.
  See *Affixes in Monier-Williams' own members*.
- **The junction repair** (`p5-junction`) adds members to no such word.
  - It changes the members of 397 words that had some (9,025
    occurrences), mostly from junk to the right words (`agniṣṭoma` →
    `agni` + `stoma`, `triṣṭubh` → `tri` + `stubh`).
  - Names now get their etymology in real words instead of junk
    (`yudhiṣṭhira` → `yudhi` + `sthira`, 2,689 occurrences).
  - A restored vowel can read as a negation (`yathākramam` → `yathā` +
    `akramam`); see [known-errors.md](known-errors.md) §2.
  - These changes were not judged blind.

The verdicts are a model's, not a Sanskritist's. Spot checks agree with the
judgments one would expect (`ālayavijñāna`, `oṃkāra` yes; `nṛpa`, `apsaras`
etymology; `pṛthak` → `pṛth` + `ak` no). The sample was drawn from the run
before last, whose members for these words are unchanged.

### Point 1: other dictionaries

- **Strict reading (Monier-Williams lacks the headword).**
  - 241 headwords gain the members DCS gives them (1,079 occurrences):
    `ṛddhiprātihārya`, `śrāvakabodhi`, `antarakalpa`, `agnibala`, `hemajāla`.
  - 44 words DCS never splits gain members (433 occurrences): `hāhākāra`,
    `pṛthivīcāla`, `triṣkṛtvas`, and the brief's `ālayavijñāna`. DCS
    lemmatises all 13 of `ālayavijñāna`'s occurrences whole; with Edgerton's
    members the stem query reaches 11 of them, from none.
  - Grassmann contributes a few bad ones (`agnīṣoma` → `agnī` + `ṣoma`) and a
    third of the gain.
  - Under this reading the brief's filters and the extended ones give the
    same errors.
  - Edgerton is worth little here because DCS has few Buddhist hybrid texts;
    a Buddhist corpus would weigh it more.
- **Loose reading (Monier-Williams has the word, but its vote gives no members).**
  - It adds 4,593 correct occurrences (`svarga`, `viśvakarman`, `hṛdroga`,
    `upāyakauśalya`).
  - It also adds 15,033 unverified occurrences of Grassmann's etymologies of
    frequent words: `dvādaśan` → `dvā` + `daśan`, `pṛthak` → `pṛth` + `ak`,
    `antarikṣa` → `antari` + `kṣa`, `adbhuta` → `ad` + `bhuta`, `ṛtvij` →
    `ṛtv` + `ij`.
  - With the brief's filters it adds 66,950: `sukha` → `su` + `kha`, `salila`
    → `sa` + `lila`.
  - Counted by split occurrences only, its gain over the strict reading
    shrinks from 3,516 to 345. Most of it is `svarga`, which DCS splits 10
    times in 2,708.

### Point 2: members in citation form

Monier-Williams has about 980 "in compound for X" entries (`taj` for `tad`,
`mano` for `manas`, `ṛg` by sandhi for `ṛc`). Two rules decide which to
follow.

- **Homographs.** An entry is followed only when the stub has more such
  entries than entries with a sense of its own. That leaves 754 stubs.
  - The first run followed the first such entry of every stub. It mapped
    `para` → `paras` (339 headwords), `pura` → `puras`, `mahā` → `mahat`,
    `rāja` → `rājan`, `sena` → `senā` (`bhīmasena` → `bhīma` + `senā`) and
    `dharma` → `dharman`, and turned 6,358 occurrences from exact to wrong.
  - `para` has 31 entries, one of them "for °ras"; `mahā` is also "a cow".
  - "In compound for" has to be read per homograph.
- **Unguarded (`p2-xref`).**
  - It fixes 8,278 occurrences: `tapodhana` → `tapas`, `manohara` → `manas`,
    `havirdhāna` → `havis`, `svastyayana` → `svasti`, `ṣaḍguṇa` → `ṣaṣ`,
    `jagannātha` → `jagat`.
  - It turns 2,509 from exact to wrong, nearly all `ātma` → `ātman`
    (`ātmaja`, `ātmavat`). Lemma mode indexes a split `ātma` as `ātma` only,
    so the index never holds `ātman` for it.
  - By the lemma yardstick those mappings are right, and this variant then
    errs least of the two (−13,156 against −9,907).
- **Guarded (`p2-xref-guarded`).** It also needs lemma mode to give X for
  the stub as the splitter hands it over, in pausa:
  - `manaḥ` gives {mana, manas, manaḥ}, so `manas` is followed;
  - `ātma` gives {ātma}, so `ātman` is refused.

  It fixes 7,740 occurrences, turns none from exact to wrong, and loses no
  occurrence in the search sample (the unguarded table loses 29).

### Point 3: Monier-Williams `-` between two words

- **Gains.** It gives correct members to 312 headwords (2,774 occurrences:
  `mitrāvaruṇa`, `viśvakarman`, `sarvaguṇa`, `gadāpāṇi`, `kriyāśakti`). It
  also cuts 1,194 coarse members into DCS's segments.
- **Etymologies.** Monier-Williams writes the etymology of old simple words
  the same way. So the variant also gives 415 headwords DCS never splits
  members in 23,637 occurrences: `nṛpa` → `nṛ` + `pa` (3,326), `dvija` →
  `dvi` + `ja` (2,856), `dvādaśan` → `dvā` + `daśan`, `oṣadhi` → `oṣa` +
  `dhi`, `apsaras` → `ap` + `saras`, `nanu` → `na` + `nu`, `kiṃca` → `kiṃ` +
  `ca`.
- **Finer cuts.** It cuts 1,540 occurrences finer than DCS (`dvijasattama` →
  `dvi` + `ja` + `sattama`). A finer cut also takes the intermediate word out
  of the index: `dvādaśakapāla` → `dvā` + `daśa` + `kapāla` is no longer
  indexed under `dvādaśa`. A search for `dvādaśa` then loses those lines,
  which is the 240-occurrence loss in the form queries.
- **Three-letter minimum.** Requiring both pieces to have at least three
  letters (`p3-hyphens-min3`) halves the etymological members (11,779
  occurrences) and keeps most of the gain. `oṣadhi` and `dvādaśan` still
  pass it.
- **The brief's `sarvābharaṇabhūṣita`.** It is cut only if the piece before
  the `-` may be no headword (`p3-hyphens-any-left`). `sarvābharaṇa` is in
  no dictionary, so the index never holds that member, and it cannot help a
  search.

### Point 4: ties

On the 849 headwords with tied readings, ties do more harm than good:

- on words DCS splits, 2,424 occurrences get wrong members (`mahāsura` →
  `mahā` + `sura`, DCS: `asura`; `mahāhava` → `mahā` + `hava`; `sukhādi` →
  `su` + `khādi`; `alpālpa` → `al` + `pālpa`), against 1,205 right
  (`sarvadā`, `hṛdroga`, `godāna`);
- words DCS never splits get suffixes and particles as members, in 7,793
  occurrences: `samantatas` → `samanta` + `tas`, `bahudhā` → `bahu` + `dhā`,
  `itihāsa` → `iti` + `hāsa`.

### The merged-vowel repair

- **`p5-junction`** (what `conservative` and lemma mode use). A pair with a
  side that is no Sanskrit headword is read again where sandhi joined it
  (`JunctionSandhi`). The readings undo a merged vowel at the mark or up to
  two letters after it, a semivowel, an `a` elided where an avagraha marks it,
  a retroflex `ṣ`/`ṇ`, and `cch`/`ch`. A reading replaces the pair only under
  four rules:
  - both its sides are headwords;
  - a side that already is one keeps its spelling;
  - a reading that moves the cut past the mark needs members that two and
    three of the seven dictionaries list;
  - the nearest reading wins, then the one with fewer long vowels.

  A pair no reading makes words of keeps the dictionary's cut.
  - **Fixes**: 6,042 occurrences (270 headwords). Among them are the
    `ṛṣi`/`ṛṣabha` family (`devarṣi`, `rājarṣi`, `brahmarṣi`), `mahārṇava`,
    `kāmārtha`, `daśāvasthā`, `kathādi` and `śvetacchattra`. `maharṣi`
    becomes `maha` + `ṛṣi`: Monier-Williams' own "in compound for `mahā`
    before `ṛ`", which the index holds for a split `mahā`.
  - **Breaks**: 16 occurrences, all of `mahāmani`, Monier-Williams' spelling
    of `mahāmaṇi`, which becomes `mahā` + `amani`. A further 69 go from finer
    to wrong (`agni` + `stoma` + `sāman`). That is an artefact: the
    evaluation's joiner does not apply the retroflex `ṣ`, so it cannot join
    `agni` + `stoma` back to `agniṣṭoma` inside a longer compound.
  - **Unverified**: it adds members to no word DCS never splits, and changes
    those of 397 that had some (9,025 occurrences; see *Words DCS cannot
    check*).
  - It does not fix `mahātman` (3,195 occurrences, the largest single error
    left). `tman` is itself a Vedic word, and DCS splits `mahātman` only 4
    times.
- **`p5-junction-drop`.** The same, but a pair no reading makes words of
  costs the headword its members, as the repair of 2026-11-22 did.
  - It removes the members of 2,198 words DCS never splits (21,672
    occurrences, mostly junk).
  - But 1,616 occurrences go from exact, coarse or finer to missing.
  - So it makes 1,617 more errors than `p5-junction`, and the search loses
    occurrences in the sample (22 for stems, 41 for forms).
- **The repair of 2026-11-22** (`p5-vowel-unattested`, no longer in the
  configuration) re-cut a pair near the mark with the vendored splitter's
  sandhi rules, and dropped the members when no re-cut gave two words. It
  removed 5,181 errors against `p5-junction`'s 5,984. It also made junk
  members of real words (`vitṛṣṇa` → `vitṛṣ` + `na`) and cost about 200 MB
  per process ([known-errors.md](known-errors.md) §2).
- **`p5-vowel-always`.** It also gives a vowel back before a second member
  that is a word: `drākṣārasa` → `drākṣā` + `arasa`, `kanyādāna` → `kanyā` +
  `adāna`, `mahāpadma` → `mahā` + `apadma`. That turns 21,755 occurrences from
  exact to wrong, for 9,415 fixed.

### Affixes in Monier-Williams' own members (not in the brief)

The brief excludes stubs, suffixes and prefixes from the other dictionaries'
members. Monier-Williams' own `—` makes the same cuts: `tat—tva`, `dhī—mat`,
`tad—vat`, `dur—yodhana`, `su—bhaga`. `p6-mw-affixes` glues those pieces back
with the extended filters; `p6-mw-affixes-narrow` uses only the library's
preverbs, the particles and the suffixes DCS does not treat as words (`-tva`,
`-tā`, `-mat`, `-tas`, `-tara`, `-tama`, `-śas`).

- **Gains.** The narrow rule removes members from 4,576 words DCS never splits
  (70,128 occurrences): `tattva`, `dhīmat`, `āyuṣmat`, `durlabha`,
  `dakṣiṇatas`. It changes no error DCS can check (−10).
- **Losses.** The search still loses matches (stem queries −1,418 [−2,762,
  −433]). Gluing a suffix back also removes the useful base member:
  `svarūpatva` had `svarūpa` + `tva` and has none, so its lines are no longer
  indexed under `svarūpa`. `sumahātman` and `suduścara` lose `mahātman` and
  `duścara` the same way.
- **The broad rule.** It also glues `-vat` and `-maya`, which DCS treats as
  words and lemma mode splits (`balavataḥ`). It turns 9,635 occurrences from
  exact to missing.
- **What would work.** A rule that drops only the affix and keeps the base
  member (`svarūpatva` → `svarūpa`) for productive formations, and gives
  lexicalised words (`tattva`) none. That needs to know which formations are
  productive; the library's `-tā`/`-tva` deriver
  ([taddhita-derivation.md](taddhita-derivation.md)) already reconstructs
  them.

### What `conservative` still gets wrong

- **`mahātman`.**
- **A mark further off than two letters, or another bracketing**, keeps
  the dictionary's cut (`go` + `stanākāra` for DCS's `gostana` + `ākāra`),
  and the length of a restored vowel is a guess (`kanyāgāra` → `kanya` +
  `āgāra`; `jaladāgama` → `jalada` + `agama`).
- **Sandhi spellings of first members no stub covers**: `try` (`tri`), `dvy`
  (`dvi`), `hary` (`hari`), `prāṅ` and `udaṅ` (`prāk`, `udak`), `ṣaṇ` (`ṣaṭ`),
  `catuṣ` (`catur`), `tan` (`tad`). Reading a member that is no headword in
  pausa (`Pausa.readings`) would cover most of them.
- **Members only a tie or a `-` marks** (`mahāsura`, `hṛdroga`,
  `mitrāvaruṇa`, `viśvakarman`). These are reported as missing.
- **DCS's own etymological splits** (`svarga` → `svar` + `ga`, `svadhā` →
  `sva` + `dhā`, `nyāsa` → `ni` + `āsa`). These also count as missing, but
  they would be noise for search. The split-only weighting gives each of
  them its few split occurrences.

## Findings outside the brief's options

- **Lemma mode's analysis of a split piece decides which members can
  match.** A split `ātma` is indexed as `ātma`, a split `rāja` as `rāja` and
  `rāj`, a split `mahā` as `maha` and `mahā`. None of these is the citation
  lemma. Giving a split piece the lemma too, from the same "in compound for"
  table, would let members be citation lemmas everywhere: the unguarded
  table would then be right, and split compounds would also match a member
  written on its own.
- **Members are flat.** Cutting finer than Monier-Williams' `—` takes the
  intermediate words out of the index (`dvādaśa` in `dvādaśakapāla`).
  Indexing the members of members would keep both, if the search wants that.
- **A wrong expected value in known-errors.md.** Section 2 lists `daśāvasthā`
  as `daśā` + `avasthā`. Monier-Williams glosses it "the ten conditions (of a
  lover)", so it is `daśa` + `avasthā`. The case book uses `daśa`. The
  repair of 2026-11-22 broke ties by how many dictionaries list a word, and
  therefore picked `daśā`; the junction repair takes the reading with fewer
  long vowels, `daśa`.
- **DCS also cuts some words etymologically** (`svarga`, `svadhā`,
  `sarvadā`, `nyāsa` as `ṇi` + `āsa`). Gold from DCS is not free of what
  point 3 is faulted for.

## What is measured

Lemma search indexes what `process(text, mode="lemmas")` returns. A line is
indexed under every lemma and every member of its parts. A query part
matches one of its lemmas, or all of its members (`MatchRule`).

### Gold: DCS

DCS annotates every member of a compound as a word of its own (`Case=Cpd`),
with its lemma and its unsandhied form. A headword occurs in DCS in one of
two ways:

- **whole**, as the lemma of a segment (`sukha`, also inside a compound);
- **split**, as a run of segments whose forms join to the headword by the
  vendored splitter's sandhi rules (`tad` + `jātīya` → `tajjātīya`). A member
  is also joined through its pausa readings (`antaḥ` as `antar` + `gata` →
  `antargata`). A participle head is joined through its stem, read off its
  form.

Evidence is collected for every headword that some variant gives members:
115,627 headwords, of which 48,749 occur in DCS.

### Yardsticks

A member *stands for* a DCS segment under one of two yardsticks:

- **index**: it is a term the index holds for the segment when the compound
  is written split. That is a lemma lemma mode gives the segment's form as
  the splitter hands it over, in pausa.
  - A split `manaḥ` gives `mana`, `manas` and `manaḥ`.
  - A split `rāja` gives `rāja` and `rāj`.
  - A split `mahā` gives `maha` and `mahā`.

  Such members let one form of a compound reach another, which is what the
  brief's `ālayavijñāna` case needs.
- **lemma**: it is the segment's citation lemma. This is DCS's lemma, with
  DCS's `-ant` read as the dictionaries' `-at` (`mahant` = `mahat`). Such
  members also reach the member written as a word of its own (`tad …
  jātīya`).

### Judgments and weights

| judgment | meaning |
|---|---|
| `exact` | one member per segment, each standing for it |
| `coarse` / `finer` | members that join runs of segments, or the reverse |
| `wrong` | anything else |
| `missing` | no members for a headword DCS splits |
| `unverified` | members for a headword DCS never splits |

*Errors* are `wrong` + `missing`. Each headword is judged against its three
most frequent DCS cuts, and weighted two ways:

- by all its DCS occurrences, as a proxy for how often it is searched and
  indexed;
- by its split occurrences only. Some frequent words DCS splits only a
  handful of times (`mahātman` 4 in 3,195, `svarga` 10 in 2,708), and this
  weighting keeps those few splits from deciding so much weight.

### Search simulation

The brief's matching rule is run over lemma mode's real analyses of the
tokens where a headword occurs, up to the 30 most frequent tokens per
headword, each counted as often as it occurs. A query is either the
headword typed as a stem, or one of the three forms it most often takes.

The headwords are sampled by stratum:

- a stratum is the set of variants that change a headword;
- headwords that no variant changes form the 23,241-headword stratum
  `unchanged`;
- 60 headwords are drawn per stratum and weighted back to the stratum's size.

Only differences against the baseline are reported, with a 95% bootstrap
interval over the draws. The absolute recall of a sample depends on its
strata, so it means nothing by itself.

### Running text

The running-text sample is one DCS sentence in 150, each token analysed on
its own, as lemma mode analyses running text. It records how often the
members that reach the index change, and how they are judged.

### Hand-checked cases

There are 45 cases: the brief's examples and its "must not change" list,
plus cases found while reviewing the brief. `expected` is the members a
reader would give, in citation form. `acceptable` is another spelling of the
same word that a split piece is indexed under (`rāja` for `rājan`).

### One analysis for every variant

Each variant's members are filled in after a single analysis, not by running
lemma mode once per variant. Members are attached in one place
(`LemmaBuilder._part`) and never read back, so this gives the same result.
`RecordedAnalysisTests` pins it against `process()`.

## The variants

| name | what it does |
|---|---|
| `baseline` | the library rule: Monier-Williams `—`, the reading most entries give (rule 9 of [lemma-mode.md](lemma-mode.md)) |
| `p1-bhs`, `p1-bhs-gra` | point 1: Edgerton's (then Grassmann's) `-` for headwords Monier-Williams lacks, with the extended filters |
| `p1-bhs-gra-brief-filters` | the same with only the filters the brief names (`a-`, `an-`, the library's preverbs, `-tā`, `-tva`, stubs) |
| `p1-loose`, `p1-loose-brief-filters` | point 1 read loosely: also when Monier-Williams has the word but its vote gives no members |
| `p2-xref` | point 2: Monier-Williams' "in compound for X" for stubs that are mostly such entries |
| `p2-xref-guarded` | the same, and only where lemma mode gives X for the stub's pausa spelling |
| `p3-hyphens` | point 3: a Monier-Williams `-` between two Sanskrit headwords is a member boundary. Preverbs are not: this includes runs of them in sandhi spelling, generated with `PreverbSandhi`. Neither are suffixes |
| `p3-hyphens-min3` | the same, with both pieces at least three letters long |
| `p3-hyphens-any-left` | also when the piece before the `-` is no headword |
| `p4-ties` | point 4: a tie between a compound reading and no compound gives the members |
| `p5-junction` | the merged-vowel repair of [known-errors.md](known-errors.md) §2: a pair with a side that is no headword is read again at its sandhi junction, and keeps the dictionary's cut where no reading helps |
| `p5-junction-drop` | the same, but where no reading helps the headword gets no members (the rule of 2026-11-22) |
| `p5-vowel-always` | the junction repair, also restoring a vowel before a second member that is a word |
| `p6-mw-affixes`, `p6-mw-affixes-narrow` | not in the brief: prefixes and suffixes glued back in Monier-Williams' own `—` members, with the extended filters or the `member-affixes` profile |
| `brief-literal` | everything the brief asks, read literally |
| `conservative` | strict fallback, guarded citation forms, junction repair: what lemma mode ships |
| `conservative-drop` | `conservative` with `p5-junction-drop` |
| `conservative-hyphens`, `conservative-hyphens-min3` | `conservative` plus point 3, without and with the three-letter minimum |
| `conservative-affixes`, `conservative-affixes-narrow` | `conservative` plus the affix rule, broad or narrow |

The extended filters add these to what the brief names:

- runs of preverbs in their sandhi spellings (`upā`, `vyā`, `abhy`, `niṣ`,
  `prot`), except a run of two or more that is itself a Sanskrit headword
  (`samādhi`, `vipra`);
- the particles `su`, `ku`, `sa`, `dus` and `nis` in theirs;
- the suffixes `-tara`, `-tama`, `-vat`, `-mat`, `-maya`, `-tas`, `-tra`,
  `-dhā`, `-śas` and `-ka`;
- a check that the pieces spell the headword, with no one-letter piece and
  no bracketed context.

## Limitations

- **Gold covers only what DCS splits.** Members for words DCS keeps whole are
  counted (*unverified*); a blind sample of them was judged separately, by a
  model.
- **DCS is not perfect gold.** It lemmatises a participle or gerundive under
  its root, also as a compound member: `dagdha` → `dah`, `hata` → `han`,
  `saṃskṛta` → `saṃskṛ`, `pūjya` → `pūjay`. In the first 4,000 files that is
  13,847 compound members.
  - The index yardstick reads the segment's form and is unaffected.
  - The lemma yardstick takes DCS's lemma, so it rejects a correct member
    such as `dagdha` or `kṛta`, for every variant alike.
  - DCS also splits some words etymologically (see above).
- **The search figures depend on the sample.** It is stratified by which
  variants change a headword, so adding a variant redraws it. A loss in a
  large stratum weighs about 40 times: in the last run the merged-vowel repair
  alone gains 178 occurrences in the sample and loses 17, but one loss
  (`triṣavaṇa`) puts its stem-query estimate at −16 [−985, +663]. Between the
  last two runs the figures moved by up to a factor of two, inside their
  intervals; no direction changed.
- **The extended filters' generated preverb runs collide with a few words**
  (`vyādhi` is vi + ā + adhi, `samādhi`, `vipra`). This moved no result of
  points 1 or 3 by more than 45 occurrences; it is why the narrow affix rule
  uses the library's own preverb list.
- **DCS is one corpus.** It is weighted towards epic, purāṇic and śāstric
  texts, so Edgerton's Buddhist vocabulary gets less weight than in a
  Buddhist corpus.
- **The simulation scores tokens, not lines.** It does not count a query that
  matches only through another word of the same line. It also does not
  measure false matches; the *unverified added* column stands in for them.
- **Analyses are made with `input_scheme="IAST"`.** The search's own calls
  detect the scheme of a whole line. That can differ for ASCII-only tokens,
  but it is the same for every variant.
- **The DCS-to-headword join can still miss compounds** whose internal
  sandhi the rules do not spell. Such headwords count as *unverified* or
  *no members* for every variant alike.
