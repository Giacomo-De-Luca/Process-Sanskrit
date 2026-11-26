# Brief: compound members for lemma search

Written 2026-11-22 by the agent building lemma search in `flaskapitest`, for the
agent working on Process-Sanskrit. It is self-contained.

## How search uses lemma mode

The search builds a BM25 index from `process(text, mode="lemmas")` on the
releases it pins (1.7.0 now). A line is indexed as every lemma and every
member of each of its parts. A query is analysed the same way, and each of its
parts is required: one of its lemmas, **or all of its members**. A compound
typed whole can reach the same compound written split or inflected only through
its members.

## The problem

A compound headword the library resolves whole, but that Monier-Williams does
not mark with `—`, comes back with no members. Its inflected forms resolve
differently, so the two never meet:

| Input | 1.7.0 output (parts) |
|---|---|
| `ālayavijñāna` | `ālayavijñāna` (no members) |
| `ālayavijñānam` | `ālaya` + `vijñāna` (two parts) |
| `ālayavijñānasya` | `ālaya` + `vijñāna` (two parts) |

In the corpus (5.8M lines), 487 lines contain *ālayavijñāna*. The query
`ālayavijñāna` requires the term `ālayavijñāna`, and matches 127 of them. With
`ālaya` + `vijñāna` as an alternative it would match 470.

## Cause

`LemmaLexicon.members()` (`functions/lemmaAnalysis.py`) reads
`headword_components(lemma)`, which defaults to `mw`, and
`ComponentMarks.members` cuts on `—` only (rule 9 of
[lemma-mode.md](lemma-mode.md)). Other dictionaries mark the split:

| Dictionary | Mark for members | Example | Headwords that would gain members* |
|---|---|---|---|
| BHS (Edgerton) | `-` | `ālaya-vijñāna`, `mithyā-praṇidhāna`, `grantha-dhāraṇī` | +1,139 (no member reading in mw) |
| Grassmann | `-` (also after prefixes: `a/-rīḍha`) | `indra-jūta`, `agni-tapta` | +395 |
| MW itself | `-` between two words, not only after preverbs | `sarvābharaṇa-bhūṣita`, `ṛg-yajus`, `vasoṣ-pati` | ~9,350, noisy |
| Apte, Macdonell | none usable (`--kaḥ` = endings; `tush-ita` = roots and suffixes) | | |

\*Counted on flaskapitest's `resources/merged_formdb.sqlite`, with a rough
rule. BHS also writes stubs (`-ālopaka`, `nikranda-`) and suffixes
(`vijugupsana-tā`), which are not members.

## What we ask for

1. **Members from BHS and Grassmann** when Monier-Williams gives no member
   reading. Their `-` separates members. A stub, a suffix piece (`-tā`, `-tva`)
   or a prefix (`a-`, `an-`, the preverbs) is not a member.
2. **Members in lemma form.** Today `tajjātīya` → `taj` + `jātīya`, and the
   index has `tad`, so `tajjātīyena` does not find `tad … jātīya` lines.
   Likewise `ṛg` → `ṛc`, `ab` → `ap`, `dveṣo` → `dveṣas`. The merged-vowel case
   `mahātman` → `mahā` + `tman` is already in [known-errors.md](known-errors.md).
3. **Optional: MW's `-` between two words.** `sarvābharaṇa-bhūṣita` and
   `ṛg-yajus` are compounds, but most of MW's `-` marks follow a preverb
   (`anu-cara`, `prati-buddhi`), including combined preverbs (`vyā-han`,
   `atipra-yam`). This needs the library's full preverb knowledge; measure it
   before keeping it.
4. **Optional: ties.** `muktāhāra` has one plain entry and one `muktā—hāra`, so
   it gets no members. For search, members are only an extra alternative, so a
   tie could give them. Your call: it changes rule 9 for every consumer.

## Expected results

- **Should gain members:**
  - `ālayavijñāna` → `ālaya`, `vijñāna`
  - `tajjātīya` → `tad`, `jātīya`
  - with point 3: `sarvābharaṇabhūṣita` → `sarvābharaṇa`, `bhūṣita`, and `ṛgyajus` → `ṛc`, `yajus`
- **Must not change:**
  - `sukha`, `bhāvana`, `ayoga` and `vikalpa` stay without members (rule 9's table).
  - `pratiprasava` stays without members (`prati-` is a preverb).
  - `svarūpa` keeps `sva`, `rūpa`.
  - `yogānuśāsana` and `athātas` stay without members: no dictionary splits them.

## Constraints

- **Output shape:** keep it the same (`word`, `parts`, `form`, `lemmas`,
  optional `members`).
- **Docs and tests:** update rule 9 of [lemma-mode.md](lemma-mode.md) and
  [dictionary-results.md](dictionary-results.md), and pin the examples above in
  tests.
- **Release:** put all of this in one release and tell us the version. After it
  ships we redeploy and rebuild the corpus index, which takes about 12
  CPU-hours, so we would rather rebuild once.

## Related, separate issues

The search work also found these. They are not part of this task:

- **A rare headword wins over the common sandhi split.** `ceti` → `ceti`
  (corpus: `ca iti`); `cāpi` → `cāpin` (`ca api`); `tato` → `tata`, `tato` but
  not `tatas`.
- **Some long compounds are mis-split.** `citrālaṃkārabhūṣitām` → `citra`
  `laṅkā` `rabhū` `uṣita`; `nakhakeśadaśākumbhavāriṇā` → `nakhaka` `īśa` …
