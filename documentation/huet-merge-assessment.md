# Huet inflection merge assessment

## Conclusion

A merge is technically feasible and worthwhile, especially for verbs, but the
supplied export should be an **additional morphology source**. It cannot safely
replace the existing paradigms or be appended blindly to their tables.

The local inventories offer **531,673 additional distinct surface spellings
(+15.8%)**. Of those, **463** contain characters outside the assessment's basic
IAST alphabet and need review. Excluding those leaves **531,210 candidate
additions**. These are inventory gains, not measured annotation-accuracy gains.

Both source databases and TSVs remained unchanged. This work added an assessment
script, its checks, and documentation; it did not implement the merge or change
runtime analysis behavior.

**Follow-up, 2026-11-12:** Huet's official morphology export is publicly
downloadable and supersedes the local files as an import source. The local
nominal export kept only the last reading of each source entry, which explains
the missing paradigm cells described below. The official file carries those
readings, nearly complete paradigms, present-tense classes, and participle tags.
Its licence (LGPLLR) and the dictionary's (CC BY-NC 4.0) are now known. See
[Official source data](#official-source-data-follow-up-2026-11-12). Unless
marked otherwise, the measurements in the other sections still describe the
local export.

## Measured coverage

Counts below use canonical TSV spellings, Unicode NFC, and final `m`/`ṃ`
equivalence. Inflected spellings are counted once, regardless of how many
grammatical readings they have. Existing nominal lookup keys include bare stems
and indeclinables as well as inflected forms.

| Inventory | Current distinct forms | Huet distinct forms | Shared | Additional | Growth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 3,269,598 | 817,911 | 360,978 | **456,933** | **14.0%** |
| Verbal | 98,441 | 173,800 | 85,693 | **88,107** | **89.5%** |
| Combined, deduplicated across categories | 3,363,514 | 976,706 | 445,033 | **531,673** | **15.8%** |

Nominal and verbal additions must not be added to obtain the combined number:
some spellings already occur in the other category. Exact spelling comparison,
without final `m`/`ṃ` folding, gives the same additional counts in each category.

The basic-IAST character screen produces 3,362,645 current forms and 3,893,855
merged forms, including 531,210 additions. This screen checks characters only;
it neither validates Sanskrit grammar nor repairs unusual source notation.

### Lemmas and dictionary support

Hyphens are removed from lemma spellings, and current paradigms are resolved
through their first dictionary reference, as the existing lookup does. Source
homonym indexes are retained in the input but collapsed for these lemma counts.

| Inventory | Current lemmas | Huet lemmas | Additional lemmas | Additional lemmas already in the dictionary |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 172,953 | 28,870 | **13,254** | **986** |
| Verbal | 628 | 867 | **280** | **270** |

Dictionary support means exact membership in the installed `word_list` after
lemma normalization, not that every source homonym has a matching definition.

- Of the 456,933 additional nominal forms, **90,011** have at least one lemma
  already represented in current morphology, and **111,394** have at least one
  dictionary headword. Many remaining additions are derived forms, including
  participles, whose connection to a dictionary root needs additional work.
- Of the 88,107 additional verbal forms, **62,941** have a root already in
  current morphology, and **87,360** have a dictionary headword.
- Huet contributes **493,281** new nominal form–lemma pairs and **95,743** new
  verbal pairs. Of these, **35,303** and **5,869**, respectively, attach a new
  lemma reading to a surface already present in that category.
- These pair counts omit additional case/person/tense readings for an existing
  pair and do not distinguish homonyms. They are not counts of fully distinct
  grammatical analyses available through today's runtime API.

The verbal growth includes 45,603 new forms tagged `ca`, 11,712 tagged `des`,
3,047 tagged `int`, and 28,090 tagged `prim`. These feature counts overlap and
must not be summed. Derived verb coverage is a particularly useful benefit.

## Input quality and information loss

### The SQLite export contains two batches

Full row-by-row checks establish that the second half of each Huet table is
identical to its TSV. The first half has identical metadata and differs only
by final `s` becoming `ḥ` in 336,385 nominal rows and 26,667 verbal rows.

| Input | SQLite rows | TSV rows | Distinct complete TSV records | Identical duplicate SQLite rows, ignoring ID |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 2,078,792 | 1,039,396 | 1,038,226 | 704,551 |
| Verbal | 361,426 | 180,713 | 180,713 | 154,046 |

The nonidentical original/visarga spellings are aliases, not extra grammatical
analyses. Even the canonical nominal TSV contains 1,170 duplicate records.
Use the verified TSV batch as the canonical import and retain useful alternate
spellings separately if needed.

### The nominal export does not contain complete paradigms

Grouping by `(stem, stem_index, gender)` yields 91,882 groups. **None contains
all 24 case–number cells**; the maximum is 19. The masculine `rāma` group has
only five supplied cells. Across all genders, `rāmam` occurs only as neuter
nominative singular, although the current database also supplies its masculine
and neuter accusative readings. `rāmau` lacks its accusative-dual reading.

This establishes loss or omission of grammatical alternatives in the supplied
export, not a defect in Huet's original linguistic resources. The follow-up
comparison with Huet's published XML identifies the cause: the export kept only
the last reading of each source entry. See
[How the local export lost readings](#how-the-local-export-lost-readings).

There are also multiple distinct forms for the same grouped cell: 24,086
nominal cells across 13,306 groups, and 15,831 verbal cells across 2,177 groups.
The verbal groups omit conjugation-class information because the export has
no class field; some apparently multi-valued cells may combine classes.
The official XML records the class. Both partial cells and alternatives need
explicit representation.

### Spelling and provenance need attention

There are 463 nominal surface spellings outside the basic IAST character
filter, all absent from the current morphology inventory. Examples include
`a_olupā` and `anu+kramaṇi`; some may be source notation, others conversion
errors. They should be investigated rather than silently deleted or rewritten.
No verbal spelling is rejected by this character screen.

The supplied folder has no source URL, exporter, version, or licence metadata.
The follow-up identified the upstream source and its licences (see
[Official source data](#official-source-data-follow-up-2026-11-12)); the local
files' version and exporter remain unknown. This assessment does not establish
redistribution permissions.

## Official source data (follow-up, 2026-11-12)

This section records a follow-up made after the assessment above. Its figures
come from one-off probes, not from `assess_huet_merge.py`; see
[Reproduction and validation](#reproduction-and-validation).

### Where Huet publishes the data

The Sanskrit Heritage site's
[linguistic resources page](https://sanskrit.uohyd.ac.in/SKT/xml.html) offers
the generated morphology as `SL_morph.xml.gz` (SLP1) and `WX_morph.xml.gz` (WX),
each with a DTD. The Hyderabad mirror serves the files directly, while
`sanskrit.inria.fr` answers scripted requests with a bot-check page.

| Property of the SLP1 file | Value |
| --- | --- |
| URL | `https://sanskrit.uohyd.ac.in/SKT/DATA/XML/SL_morph.xml.gz` |
| Generated from | Sanskrit Heritage Dictionary 3.84, 2026-09-30 |
| Server modification date | 2026-10-13 |
| Size | 8,325,209 bytes compressed; 184,548,856 bytes uncompressed |
| SHA-256 | `a6249412343fa81b15a798d83cef41d2bde76c6e31e5248d78d507bc1a189ea4` |

The file changes between dictionary releases, so an import must pin the
version and checksum of the copy it uses. Each line holds one entry: the form
(`<f form="…">`), all its readings, and the generating stem (`<s stem="…"/>`),
with any homonym index after `#` (`aMSaka#2`). The DTD states that this stem
indexes an entry in the Heritage dictionary.

The licences differ by resource:

- **Morphology XML:** "Copyrighted Gérard Huet 1994-2026"; use is granted under
  the Lesser General Public Licence for Linguistic Resources (LGPLLR), a copy of
  which the resources page links.
- **Dictionary:** the
  [GoldenDict page](https://sanskrit.uohyd.ac.in/SKT/goldendict.html) releases
  the dictionary resources under Creative Commons Attribution-NonCommercial 4.0
  International.

Process-Sanskrit is CC-BY-NC-ND-4.0. A merged database must at least credit
Huet and carry the LGPLLR terms for the imported morphology. This assessment
does not establish whether that combination is permitted.

The dictionary is Sanskrit–French. It is offered as a PDF and as a Babylon-format
archive, `Heritage_du_sanskrit_san-fra.tar.gz`, that can be converted for
StarDict or GoldenDict. The English (`san-eng`) archives on the same page hold
declension tables, grammar links, and Monier-Williams, not an English edition.
These archives are hosted on `sanskrit.inria.fr` and must be downloaded through
a browser. The `Heritage_Resources` and `Heritage_Platform` repositories on
`gitlab.inria.fr` sit behind the same check and were not inspected.

csl-inflect's own `huetdata` extraction uses Huet data dated 2020-06-15
([csl-inflect assessment](csl-inflect-update-assessment.md)); this release is
newer.

### How the local export lost readings

The local nominal TSV was joined to the XML on IAST form, stem, and homonym
index, after converting SLP1 to IAST and final `s`/`r` to visarga. For 849,520
of the 902,607 XML form–stem pairs (94.1%), the export holds **exactly the last
reading of each XML entry**:

| Outcome | Pairs | Share |
| --- | ---: | ---: |
| Last reading of each entry kept, others dropped | 485,806 | 53.8% |
| Last reading kept, nothing dropped (one reading per entry) | 363,714 | 40.3% |
| Pair absent from the export | 27,299 | 3.0% |
| Neither pattern | 25,710 | 2.8% |
| Every reading kept although the rule would drop some | 78 | <0.1% |

Where the rule holds, the export drops **957,582 nominal readings**. Overall,
the XML has 2,088,157 nominal readings, including those of participles, against
1,039,396 TSV rows. In the other direction, 6,349 of the export's 881,657
form–stem pairs (0.7%) are absent from the XML. The export's source version is
unknown, so version differences or conversion details may explain the residual
rows; they were not investigated.

`rāmam` shows the rule. Its XML entry lists masculine accusative, neuter
accusative, and neuter nominative, in that order; the export kept only the
last.

The verbal export likely lost readings the same way, on a smaller scale: the XML
has 193,773 finite verbal readings against 180,713 TSV rows, and 9,920 entries
hold more than one finite reading. This was not checked row by row.

### What the official file adds

| Property | Local export | Official XML |
| --- | --- | --- |
| Nominal readings | 1,039,396 TSV rows | 2,088,157 |
| Finite verbal readings | 180,713 TSV rows | 193,773 |
| `(stem, gender)` groups with all 21 non-vocative cells | none has all 24 cells; maximum 19 | 94,503 of 94,557; the 54 exceptions are grammatically defective (`ap` plural only, `ubha` dual only) |
| Masculine `rāma` | 5 cells | 23 cells; only the vocative dual is absent |
| Present-tense class (gaṇa) | absent | on all 50,728 primary present-system finite readings; causative, desiderative, and intensive readings have none |
| Participles | plain nominal rows under the participle stem | `<pa>` readings with case, number, gender, conjugation, and participle type |
| Compound-initial (`iic`), pre-auxiliary (`iiv`), and indeclinable (`uf`) forms | absent from the nominal TSV, whose case column holds only the eight cases | 22,515, 21,686, and 14,000 entries |

Vocatives are the remaining gap: 63,256 groups have none, 44,058 of them
participle groups. The vocative dual and plural always equal the nominative, so
they can be filled in; the vocative singular must be generated by declension.

For example, `BAvayan` (bhāvayan) is tagged nominative singular masculine of a
causative present active participle, parasmaipada, under the stem `BAvayat#1`.

### Participles still need a link to their root

A participle's generating stem is the participle itself (`BAvayat#1`), not its
root, so linking it to a root still needs rules; the `<pa>` tag says which rule
applies. For active `-at` participles, appending `-i` gives the third-person
singular, which the verbal readings map to a root: bhāvayat → bhāvayati → bhū
(causative). On the local TSVs, this rule links 2,395 of the 2,495 present and
future `-at` lemmas that are both new and absent from `word_list`. Rules for
`-māna`, `-tavat`, `-vas`, and gerundives have not been written or measured.

### New nominal lemmas without a dictionary entry

Of the 13,254 additional nominal lemmas in the local export, 12,268 are absent
from `word_list`. Grouped by ending, about nine in ten are participles and
verbal adjectives, many from causative, desiderative, or intensive stems. The
grouping is approximate: a few names and compounds fall into participle rows.

| Kind | Lemmas | Examples |
| --- | ---: | --- |
| Middle/passive participles (`-māna`, `-āna`, future `-syamāna`) | 3,361 | sarjayamāna, rodayiṣyamāṇa, siṣyandāna |
| Gerundives (`-tavya`, `-anīya`, `-ya`) | 2,692 | medhayitavya, nāthanīya, snapya |
| Future active participles (`-syat`, `-iṣyat`) | 1,386 | kleśayiṣyat, pūjayiṣyat, vakṣyat |
| Past active participles (`-tavat`, `-navat`) | 1,147 | smṛtavat, śocitavat, riktavat |
| Other `-vat`/`-mat` stems, mostly past active participles | 221 | snigdhavat, gāḍhavat, mṛgāṅkavat |
| Present active participles (`-at`, causative `-ayat`, desiderative `-sat`) | 1,109 | kṣāpayat, suṣupsat, carat |
| Past passive participles, many desiderative | 659 | jijīviṣita, cicariṣita, bhandita |
| Perfect active participles (`-vas`) | 495 | cakṛṣvas, pipiśvas, ruruhvas |
| `-in` stems and agent nouns in `-tṛ` | 52 | sarvakarmaphalatyāgin, uttarapakṣin, prasthātṛ |
| Other: names, compounds, abstracts, ordinals | 1,146 | gaṅgaikoṇḍacolapura, svabhāvasiddhatva, antyajātitā, ṣaṭtriṃśattama |

The participles need a root link rather than a definition. The other
roughly 1,200 lemmas need a definition source. By the DTD, each should have a
Heritage dictionary entry, which would be in French; this was not checked.

## Compatibility with the current code

The current `lgtab1`/`vlgtab1` tables store colon-separated paradigms;
`lgtab2`/`vlgtab2` index their surface forms. Nominal paradigms use 24 cells
(8 cases × 3 numbers), and verbal paradigms use 9 cells (3 persons × 3 numbers).
Huet instead stores explicit grammatical fields on individual form rows.

Common case, number, gender, person, and voice fields can be mapped. However:

1. **Preserve source lemma IDs and homonyms.** `stem_index` belongs to Huet;
   it must not be treated as a dictionary reference ID. String equality does
   not establish sense identity. Preserve `dei` and other source feature
   distinctions until their intended interpretation is verified.
2. **Verbal model codes are not interchangeable.** Existing model names can
   include conjugation class, such as `pre-1a`; the local export omits that
   field, but the official XML records it on primary present-system readings.
   Some rows also have blank tense/mood fields, including the `bhavitā`
   series (`system_type=tp`). Their interpretation needs source-aware mapping.
3. **The existing verb reader drops ambiguity.** `_VERB_QUERY` chooses the
   last index row and first matching paradigm. A naive append can replace a
   previously returned reading based on insertion order, while hiding others.
4. **Existing person labels are reversed.** Stored `gam/pre-1a` starts with
   `gacchati` and ends with `gacchāmi`, but `VERBAL_PERSONS` is
   `("First", "Second", "Third")`. A runtime probe confirms `gacchati`
   receives `First/Sg` and `gacchāmi` receives `Third/Sg`.
5. **Existing alternative-cell matching is incomplete.** `tags_for()` compares
   whole cell strings. A slash-separated cell such as `babhūtha/babhūvitha`
   fails to match `babhūtha`; the runtime returns a lemma with no tags.
   Adding Huet alternatives to such strings would inherit that problem.

Items 3–5 were reported during this investigation. Item 4 has since been fixed
(see [the audit notes](analysis-audit.md#verb-person-correction-2026-09-16));
items 3 and 5 remain open.
The inventory audit also found 80 verbal index rows without a corresponding
paradigm and excludes them from the baseline.

## Practical probe

The bundled Yoga Sūtra fixture contains **566 unique whitespace tokens** after
limited punctuation stripping and final `m`/`ṃ` normalization. Direct table
coverage changes from **227 to 234 tokens**: seven additional hits, or **1.24
percentage points**. This deliberately does not split sandhi or compounds and
is not an end-to-end benchmark.

The added hits are `anekeṣāṃ`, `anāśayaṃ`, `ekarūpatvāt`, `panthāḥ`, `upasargā`,
`vitarkā`, and `viśeṣārthatvāt`.

A separate current-runtime probe shows:

- `upasargā` and `vitarkā` already resolve through spelling fallbacks.
- Huet's `anāśayam` match is a verbal reading under `naś`; this is not evidence
  of an improvement for the Yoga Sūtra occurrence.
- The other four tokens miss `root_any_word()`, but later dictionary,
  derivation, and compound stages may still help. No claim is made that all
  four currently fail in `process()`.
- Useful examples absent from current `root_any_word()` include `gacchantī`
  and `gacchatā` → `gacchat`, `jigamiṣati` → `gam`, `bodhayati` → `budh`, and
  `bhāvayati` → `bhū`. Some also have nominal participle readings in Huet.

Thus the likely benefit is broader morphology and fewer unnecessary fallback
analyses, especially for derived verbs and participles. Accuracy and timing
improvements need a separate before/after corpus evaluation.

A follow-up on 2026-11-12 checked reported analysis errors against Huet and the
current runtime (`process(..., mode='roots')`):

| Input | Current result | Huet reading |
| --- | --- | --- |
| dāpayet | da + apa + yā/yad + id | dā, causative optative, 3rd sg. |
| hāpayet | hā/ha + apa + yā/yad + id | hā, causative optative, 3rd sg. |
| sthāpayet | stha + apa + yā/yad + id | sthā, causative optative, 3rd sg. |
| yojayiṣyasi | yu + ji | yuj, causative future, 2nd sg. |
| yojayati | yok | yuj, causative present, 3rd sg. |
| bhāvayan | bhāva/bhā + yad | bhāvayat, nom. sg. masc. |
| kārayan | kāra + yad | kārayat, nom. sg. masc. |
| dhārayan | dhāraya, an `-a` noun reached by a fallback | dhārayat, nom. sg. masc. |

None of these forms is in the current tables, so `process()` falls through to
splitting. Because `process()` looks up the whole word before splitting, a Huet
lookup in `_direct_roots` would stop these splits. Huet gives the participles
only their participle stem; see
[Participles still need a link to their root](#participles-still-need-a-link-to-their-root).

Two other reported errors are not Huet's to fix: an initial `ā` being lost
(`āśraya` read as `aśraya`) and `akrūra` read as `krūra`. Both words already
resolve in the current tables in every case form tested, so these errors are
split choices at ambiguous vowel junctions. Neither was reproduced in isolation
or in phalāśraya, jīvāśraya, tadāśrayam, nirāśrayaḥ, kriyāphalāśrayatvam,
ca/tathā/sadā/mahā + akrūraḥ, or so'krūraḥ. A related failure is `mamāśrayaḥ`
→ asmad + aśri/śri.

## Suggested integration

Before implementing, pin tests for source deduplication, spelling aliases,
homonym preservation, missing versus alternate paradigm cells, person order,
multiple verb readings, dictionary fallback, and unchanged established analyses.
Include the concrete examples above and existing compound/prefix regressions.

Then:

1. Import from the official XML rather than the local TSVs; see
   [Official source data](#official-source-data-follow-up-2026-11-12). It
   restores the dropped readings and carries present-tense classes and
   participle tags. Convert SLP1 to IAST and final `s`/`r` to visarga, pin the
   release version and checksum, and record the attribution and licence. The
   local TSVs can serve as a cross-check for the conversion.
2. Import deduplicated Huet analyses into separate, indexed tables in a staged
   copy of the existing SQLite database. Keep provenance and the full original
   feature tuple. This allows one physical database without forcing two data
   models into the same table prematurely.
3. Add a shared lookup adapter that maps source analyses into the public result
   shape. Preserve current paradigms and readings; do not let row insertion
   order decide precedence. Treat complete paradigms and partial evidence
   differently when rendering inflection tables.
4. Evaluate verb-first integration: its coverage almost doubles, and almost all
   added forms have a dictionary headword. Nominal integration still needs
   participle-to-root links and a definition source for non-derived lemmas;
   the official XML supplies the missing grammatical alternatives.
5. Compare roots, tags, ambiguity, unresolved words, and runtime on held-out
   annotated text. Revalidate affected caches and publish the new database
   through the existing staged/atomic installation workflow.

Adding rows to the main database does not by itself update the native sandhi
splitter's separate lexical assets. Any intended splitter-vocabulary expansion
requires its own asset-build and parity work.

## Reproduction and validation

```bash
uv run --no-sync python -m unittest tests.test_huet_assessment
uv run --no-sync python scripts/assess_huet_merge.py
```

The script uses `AssessmentConfiguration` for paths and writes the detailed
report to `build/huet-merge-assessment.json`. It streams source SQLite data
through read-only immutable connections, verifies both Huet batches against
the complete TSVs, and checks SHA-256 hashes before and after the run. It holds
large inventory sets in memory; run it on a development machine, not in a
latency-sensitive service. SQLite databases must be quiescent, with no pending
WAL changes, for these immutable snapshots to represent the full input.

Validation: **5 assessment tests pass**, full comparison completes, input
hashes remain unchanged, and the required code-quality-reviewer found no
remaining blocking issue. The reviewer caught an initial duplicate-count
edge case, which is covered by a regression test. No runtime code changed,
so the complete processing test suite was not rerun.

The 2026-11-12 follow-up figures come from one-off probes that are not in the
repository. They stream the downloaded `SL_morph.xml.gz` line by line with
regular expressions, since each line is one `<f>` entry. They group nominal
readings by stem (with homonym index) and gender. They join the XML to the
local TSVs after converting SLP1 to IAST with `indic_transliteration` and final
`s`/`r` to visarga. The reported-error results come from
`process(..., mode='roots')` and `root_any_word()` on the 2026-11-12 working
tree. Folding these probes into `assess_huet_merge.py` would make them
reproducible.

Source SHA-256 identifiers:

| File | SHA-256 |
| --- | --- |
| Main SQLite | `7e747993ac6ed106df7f368b756408cf464c7de0f8f2839d23cd20a91715b0a4` |
| Huet SQLite | `8e1b8e1319cff8a8f35ce1370e080fc2332541dfce4b044b43694a1a6d6cc4fb` |
| Nominal TSV | `6f277ace75e9b864eb9bbbe89bdd5c9019a2f919c4d97f25df3977dc6d9d0e0d` |
| Verbal TSV | `35201ffad4a4e2b4768a559feb245486204488fc248580c6db6d2dd62878b9a5` |
| Official SLP1 XML, downloaded 2026-11-12, not stored in the repository | `a6249412343fa81b15a798d83cef41d2bde76c6e31e5248d78d507bc1a189ea4` |
