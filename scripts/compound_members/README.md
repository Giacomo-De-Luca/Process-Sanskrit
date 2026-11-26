# Compound-member evaluation

Measures which way of reading a compound headword's `members` gives lemma
search the fewest errors. The options come from
[`documentation/compound-members-brief.md`](../../documentation/compound-members-brief.md).
The results and their interpretation are in
[`documentation/compound-members-evaluation.md`](../../documentation/compound-members-evaluation.md).

The library ships the winning reading, `CompoundMembers`
(`process_sanskrit/utils/compoundMembers.py`, rule 9 of `lemma-mode.md`). This
folder builds that same reading from the library's classes as the variant
`conservative`, and adds the options the library does not ship, so that they
can be measured against it. Nothing here is imported by the library.

## Running

Settings live in [`benchmarks/compound-members.json`](../../benchmarks/compound-members.json).
The DCS CoNLL-U dump is located through an environment variable, as for the
split-ranker build:

```bash
PROCESS_SANSKRIT_DCS_CONLLU_DIR=/path/to/dcs/conllu/files \
    uv run python scripts/evaluate_compound_members.py
```

The output goes to `build/compound-members/`, which is not committed:

- `report.json` holds everything, per variant;
- `summary.md` holds the headline tables;
- `evidence.json` lists where each judged headword occurs in DCS;
- `analyses.jsonl` stores the library analyses, so a rerun analyses only new words;
- `run.log` is the progress log, if you redirect it there.

The store's first line records the library version, `ANALYSIS_ALGORITHM_VERSION`
and the database file. A store written under another of these is discarded, and
a word whose analysis failed is analysed again. A change to lemma mode that
alters the analyses changes none of these, so it must bump the library version
(as 1.8.0 did), or the old analyses hide it; while you edit under one version,
delete the store. Members are filled in after the analysis and never stored, so
a change to members alone needs neither. A run takes about four minutes on 8
workers, or about two when the store is current.

The tests are the package [`tests/compound_members/`](../../tests/compound_members/README.md):

```bash
uv run python -m unittest discover -s tests/compound_members -t .
```

## The yardsticks

The search indexes what lemma mode returns. A line is indexed under every
lemma and every member of its parts, and a query part matches one of its
lemmas or all of its members (`MatchRule`). Members are judged against DCS,
which annotates every compound member as a word of its own. Two yardsticks say
when a member stands for a DCS segment:

- **index**: the member is a term the index holds for the segment when the
  compound is written split. That is a lemma lemma mode gives the segment's
  form as the splitter hands it over, in pausa (`pausa_readings`): a split
  `manaḥ` gives `mana`, `manas` and `manaḥ`; a split `rāja` gives `rāja` and
  `rāj`, never `rājan`; a split `mahā` gives `maha` and `mahā`, never `mahat`.
  Such members let one form of a compound reach another.
- **lemma**: the member is the segment's citation lemma. This is DCS's lemma,
  with DCS's `-ant` read as the dictionaries' `-at`. Such members also reach
  the member written as a word of its own (`tad … jātīya`). DCS gives a
  participle its root as lemma (`dagdha` → `dah`), so this yardstick rejects
  a correct participle member.

The evaluation uses three measures:

1. **Members against DCS** (`MemberScorer`). A headword occurs in DCS either
   whole, as a lemma, or split, as a run of segments whose forms join to it by
   sandhi. Its members are judged against the split occurrences. Every
   headword weighs as much as it occurs, counted two ways: over all its
   occurrences, and over its split occurrences only.
2. **Search simulation** (`RetrievalSimulation`). The brief's rule is run
   over the library's real analyses of the tokens where a headword occurs, for
   two kinds of query: the headword typed as a stem, and its most frequent
   inflected forms. The sample is stratified by which variants change a
   headword, and only differences against the baseline are reported, with a
   bootstrap interval.
3. **Hand-checked cases** (`CaseBook`,
   [`benchmarks/compound-members-cases.json`](../../benchmarks/compound-members-cases.json)).

Each variant's members are filled in after the analysis, not by running
lemma mode once per variant. `AnalysedPart.member_lemma` records the lemma
whose members lemma mode would attach to a part. This is exact because
members are attached in one place (`LemmaBuilder._part`) and never read back.
`tests/compound_members/test_analyses.py` checks it against
`process(..., mode="lemmas")`.

## Files and main classes

From the library, this folder uses:

- `HyphenMarks`, `AffixFilter`, `MemberRepair`, `JunctionReading` and
  `CompoundForms`, from `utils/compoundMembers.py`;
- `pausa_readings`, `MEMBER_PARTICLES` and `MEMBER_SUFFIXES`, from
  `utils/lexicalResources.py`;
- `PreverbSandhi.spellings`;
- `ComponentMarks.votes`, `ComponentMarks.prevailing_members` and
  `ComponentMarks.elisions`;
- `PieceLemmas`.

It does not call `CompoundMembers` itself: `MemberVariant._members` and
`DictionaryFallback` repeat its steps over the bulk-read tables, and
`tests/compound_members/test_readings.py` checks on 3,000 headwords that
`conservative` gives what lemma mode gives. Building the variants on
`CompoundMembers` is item 5 of
[`documentation/unification-handoff.md`](../../documentation/unification-handoff.md).

| file | contents |
|---|---|
| `lexicon.py` | `Spelling` (letters, the DCS comparison key), `JoinSpellings` (the spellings a DCS segment is joined from), `ComponentTables` and `Headwords` (the library's tables, read once for 200,000 headwords) |
| `readings.py` | what the library does not ship: `MwReading` (the rule before 2026-11-22, with `WordHyphenRule` for point 3, ties for point 4 and affix gluing), `DictionaryFallback` (also its loose reading), `AlwaysRestoreRepair`, `LemmaForm` (also unguarded), `AffixProfiles` (the configuration's filter profiles); `MemberVariant` combines them and `VariantFactory` builds a variant from its spec |
| `dcs.py` | `ConlluReader` → `Sentence` / `Token` / `Segment`; `GoldWord`; `DcsEvidence` and `HeadwordEvidence` (whole and split occurrences of wanted headwords) |
| `joiner.py` | `SandhiJoiner`: the vendored Python splitter's joining rules in IAST, with which `DcsEvidence` and `MemberScorer` join DCS segments (`tad` + `jātīya` → `tajjātīya`); one per process, about 200 MB, and only the evaluation loads it |
| `analyses.py` | `LibraryAnalyses`: lemma-mode analyses with members left open (`RecordingLexicon`), stored as JSON lines and computed in worker processes with the analysis cache off |
| `scoring.py` | `MatchRule`, `MemberScorer`, `CaseBook`, `RetrievalSimulation`, `RunningTextImpact` |
| `evaluation.py` | `EvaluationConfiguration`, `Resources` (the shared library objects and bulk tables), `CompoundMemberEvaluation` (the run), `EvaluationSummary` (Markdown) |

## Overlaps to resolve

- **`ConlluReader` and the DCS join.** These do the work of the split-ranker
  build's `DcsCorpus` and `GoldMatcher._joined` (`scripts/build_split_ranker.py`,
  on the `split-ranker` branch only). The reading conventions are the same:
  - a parsed chapter is read only when its plain twin is missing;
  - U+FFB1 is read as `ḷ`;
  - `-tva`/`-tā` are folded onto their base.

  The joins differ: the split-ranker always adds plain juxtaposition, and this
  one tries pausa readings (`JoinSpellings`). When the branch is merged, one
  reader and one joiner should remain.
- **`PreverbSandhi.spellings`.** It generates preverb spellings from the
  library's own list, rather than adding a fifth hand-written table next to
  the four that `documentation/lemma-mode-handoff.md` proposes to merge.
