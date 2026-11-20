# Lemma mode: next pass (handoff)

Written 2026-11-18 for the agent who takes this over. It is self-contained: you
do not need the conversation that produced it.

## Status: the first pass is done (2026-11-18)

The four changes below are implemented, tested and documented in
[lemma-mode.md](lemma-mode.md) (rules 4, 5, 8 and 9, and "Verb forms of several
roots"). What follows the status notes is the plan as it was written; read it as
history, and the "Later" section as the open list.

Results, measured as the plan asked:

- Root self-check: 624 of the 628 verb-table roots come back among their own
  lemmas (467 before). The four that do not are spelled with `jj`, which text
  cleaning cuts in two before lemma mode sees the word (below).
- Four-root check: 9, 11, 10 and 9 forms of `gam`, `kṛ`, `bhū`, `dṛś` (9, 9, 10
  and 4 before); nothing went down.
- Other modes, over 2,386 inputs (the plan's list plus 13 words of the tests):
  identical `roots`, `parts` and detailed output before and after the pass.
- Lemma mode: 181 of 2,193 distinct tokens changed, every one through a named
  rule: the typed headword changed 18, the joined keys 143, `paś` → `dṛś` 7, the
  other roots of shared verb forms 20 (some tokens through two rules).
- Suite: 607 tests, 4 skipped, 6 expected failures (the new one is the `jj`
  roots; after the `jj` fix, 620 tests, 5 expected failures). The complete output of the other modes (tags and dictionary payloads
  included) was also compared against the committed lookups over 2,404 inputs:
  identical.

Decisions the owner took during the pass:

- **Change 4 is lemma mode only.** With every root in every mode, the 2,386
  inputs changed in 20 places, but an independent review found that the prefix
  re-join of `clean_results` takes the first root only, so a prefixed shared
  form came back with its other roots as separate pieces (`saṃkariṣyati` →
  `saṃkṝ`, `kṛ`), and the `-n` rule rewrote `vena`. As the plan said, that was
  reported rather than widened. The lookup returns every root; `every_verb_root`
  (`functions/rootAnyWord.py`), held by lemma mode, decides that they are read.
  No cache version was bumped, since the cached output did not change.
- **The typed headword replaces several parts in running text too**, not only in
  one-token input: the `-āv` locatives (`samādhāv`, `vidhāv`) become a verb root
  `X-dhāv` rather than junk parts, and in exchange a word analyses the same in a
  one-word query and in an indexed text.

Where the implementation departs from the plan:

- Change 1 is applied by `LemmaBuilder.with_typed_headword`, which
  `LemmaAnalyzer._analyse` calls through `LemmaBuilder.finished`, so the rule is
  testable without the lexicon.
- Change 2: a fourth way a preverb part arises was found and covered (the
  inflector reading two preverb tokens together, `vi` + `ava` of
  `tadvyavaharati`), and `api` is not a preverb (`PARTICLE_PREFIXES`): all 8
  `api+` keys of the held-out tokens tied the particle to the next word.
- Change 3 is applied in `LemmaBuilder._lemmas`, not `_part`, because only there
  can a verbal `paś` be told from the feminine noun `paś` the forms table also
  has.
- Change 4: a later root is read from the first of its paradigms that has the
  form (`juhumahe` → `hu` from its present), and the root naming rule is one
  function, `paradigm_headword`, shared by both lookups.

Found on the way, not fixed (they are older than this pass, and each needs a
decision of its own):

- `clean_iast` rewrote every `jj` as `j j`, in every mode: right for `tajjayāt`,
  wrong for `lajjā` (`laj` + `jā`), `majjati`, and the roots `lajj`, `majj`,
  `bhrajj`, `ujjh`. **Fixed after the pass** (2026-11-19): a word that resolves
  whole keeps its `jj`, so all 628 roots now come back as themselves. See
  [double-j.md](double-j.md).
- The cascade reads a final `-v` as `-u` only, so `samādhāv` (for `samādhau`)
  never reaches `samādhi`. **Fixed after the pass** (2026-11-19), on both the
  single-word and the split path, with the hybrid cache at v13; it still needs
  extensive testing on a larger corpus. See [final-av.md](final-av.md).
- `clean_results` never visits the last entry of the list it cleans, so whether
  its `-n` rule fires depends on position, and its prefix re-join knows one root
  per stem. Both stand between the other modes and the extra roots of change 4.
- `LemmaPart.with_lemma` keeps a compound's members when rule 8 adds a second
  lemma (`ṛtaṃbharā`, whose member cut `ṛta` + `ṃbhara` is also wrong), against
  rule 9.

## Where things stood before the first pass

`process(text, mode="lemmas")` exists, is tested and is documented in
[lemma-mode.md](lemma-mode.md). Read that file first; it has the contract, the
rules and the known limitations. `process()` no longer truncates in any mode
([long-input.md](long-input.md)).

An agent building a BM25 search index on top of lemma mode reported four
problems. All were checked against the code and the data, and the figures below
are from that check. This document plans the fixes that are small, and notes how
the harder ones could be approached later.

**State of the tree**

- Branch `main`, with a large amount of uncommitted work. Do not commit, stash,
  reset or switch branches.
- The split ranker lives on branch `split-ranker` and is not in this tree. This
  tree uses the older hybrid split. Nothing below depends on the split.
- Another session may edit files at the same time. Re-read a file immediately
  before editing it, and make small targeted edits.
- Baseline at hand-over: 535 tests pass, 4 skipped, 5 expected failures
  (`uv run python -m unittest discover -s tests -p "test_*.py" -t .`).

**Rules** are in `CLAUDE.md`. The ones that bite: write the tests before the
code; run Python with `uv run python`; update the docs and keep only pointers in
`CLAUDE.md`; finish with an independent quality review (no `code-quality-reviewer`
agent is defined in this environment, so use a general-purpose one and give it
the code, not your conclusions — the last two reviews both found real defects).

## Decisions already taken (change them before starting if they are wrong)

These are defaults chosen when this plan was written. The owner has seen the
options but has not confirmed every one.

| Decision | Default |
|---|---|
| Scope of change 1 | Lemma mode only. The other modes keep their output. |
| Change 1 when the analysis has several parts | The typed headword replaces the split. |
| Joined preverb key (change 2) | Emitted by the library as an extra lemma of the stem part, written `prati+i`. No marker field on preverb parts. **Changed by the owner on 2026-11-19:** `+` is not searchable, so the key is now the compound as one word, the Monier-Williams headword (`pratī`) or else a sandhi join (`upasamarc`); see lemma-mode.md, rule 5. |
| `paśyati` (change 3) | `dṛś` is added beside `paś`; `paś` is kept. |
| Change 4 | Applies to every mode, because the lookup is shared. |

Earlier decisions that still hold: no `known` flag; a preverb is an ordinary
part; a verb is never re-joined with its preverb; nothing is ever truncated or
dropped silently.

## First pass: four changes, in this order

### 1. A word typed as a headword is one of its own lemmas

**Problem.** A root typed on its own comes back as something else:

```
as   -> at | ad        dṛś -> dṛ        vad -> vat        yuj -> yut | yudh
```

161 of the 628 roots in the verb table do not come back as themselves. Roots
ending in a consonant come back 69% of the time, roots ending in a vowel 98%
(nominal stems: 98% and 99.7%). All 628 are Monier-Williams headwords.

**Cause.** The word is rewritten before the dictionary is ever asked:

- `sanskritFixedSandhiMap`, applied at the top of `WordCascade.resolve`
  (`functions/wordCascade.py`), rewrites a final `y r v d b g ś` (71 of the 161).
- `variableSandhi`, the last-letter retries inside `root_any_word`
  (`functions/rootAnyWord.py`), rewrites a final `s c j n k t p …` (90 of the 161).

The word *is* looked up as typed first, but in the forms table, where a bare root
is not a row. The retry then finds some other word, and the dictionary, which
has the root, is never consulted. The existing headword rule (rule 8 of the
contract) looks at the form that was *matched* (`vat`), not the word that was
*typed* (`vad`).

**Change.** In `LemmaAnalyzer._analyse` (`functions/lemmaAnalysis.py`), for an
ordinary word (not the pattern or pre-split branches), after the parts are built:
if `self.lexicon.has_sanskrit_headword(word)` and `word` is not already among
the lemmas,

- one part: add it with `LemmaPart.with_lemma(word)`;
- several parts: return one part for the whole word (`form` = `word`, lemma =
  `word`, members as `LemmaBuilder._part` gives them).

`word` here is the cleaned word as typed, not `resolution.text`.

**Evidence that it is safe.** Simulated, it brings all four groups to 100%. On
running text (4,633 tokens: the Yoga Sūtra and the gold segments) 0.63% of tokens
gain a lemma, and most gains are corrections: `punar`, `antar` and `bahir` do not
get their own lemma today (`punar` → `puna | pu | punaḥ`).

**Tests first.**

- `as`, `dṛś`, `vad`, `yuj`, `punar`, `antar` include themselves.
- `sev` (analysed as several parts today) is one part.
- Unchanged: `gacchati`, `te`, `upagacchati`, `virati`, the three-word example.
- A regression pin over the data: every root in the verb table comes back among
  its own lemmas.

**Check before finishing.** List every token of the Yoga Sūtra whose parts
change. If a several-parts replacement makes running text worse (candidate:
`samādhāv`, which is also a Monier-Williams prefixed root), restrict the
replacement to one-token input and say so in the doc.

### 2. A joined key for preverb and stem

**Problem.** `pratyeti` is returned as `prati` + `i`, by design. A bag-of-words
index loses the fact that the two belong to one word, and both are very common
terms.

**Change.** The part that follows one or more preverb parts gains one extra
lemma per lemma it has: the preverbs and the lemma joined with `+`.

```
pratyeti     ->  prati(prati) + eti(i | prati+i)
vyavaharati  ->  vi(vi) + ava(ava) + harati(hṛ | vi+ava+hṛ)
```

The key is mechanical. It uses no sandhi, so it never fails, and `+` cannot occur
in a lemma, so it cannot collide with a real word. A search for `i` still finds
the word; the key is rare and ties the pair.

**Where.**

- `LemmaPart` gets an internal flag `preverb` (default false) that `as_dict`
  does not export.
- Set it where preverb parts are made: the prefix parts in
  `LemmaBuilder.from_lookup`, and `LemmaBuilder._lone_preverb`.
- Add one pass, `LemmaBuilder.join_preverbs(parts)`, and call it once per word
  from `LemmaAnalyzer`. It walks the parts, collects consecutive preverb parts,
  and adds the keys to the next part that is not a preverb. A preverb with
  nothing after it adds nothing. One pass covers the three ways a preverb part
  arises (prefix stripper, splitter token joined by `Inflector`, splitter token
  left on its own).

Apply it after change 1, so a word replaced by its typed headword has nothing to
join.

**Tests first.** `pratyeti`, `vyavaharati`, `samarcayet` (preverb cut off by the
splitter), a stem with several lemmas (each gets a key), a standalone `prati`
(unchanged), `virati` (unchanged). Several existing tests pin exact lemma lists
for preverb verbs and must be updated deliberately, not loosened:
`PreverbTests` and the three-word example in `tests/test_lemma_mode.py`,
`LookupPartTests` and `TokenPartTests` in `tests/test_lemma_builder.py`.

**Docs.** The contract in `lemma-mode.md` (rules 4 and 5, the example at the
top), and the example in `README.md`. A lemma is no longer always a real word;
say so plainly.

### 3. `paśyati` also reaches `dṛś`

**Problem.** The verb table files the present of `dṛś` under the root `paś`, so
`paśyati` returns `paś` only. Of 16 common forms of `dṛś`, 4 reach it.

**Change.** A table in `utils/lexicalResources.py`, `SUPPLETIVE_ROOTS =
{"paś": "dṛś"}`, applied in lemma mode where a part's lemmas are finalised
(`LemmaBuilder._part`): the mapped root is added right after the root that has
one. `paś` stays. Because this runs before `join_preverbs`, both get a key.

Keep the table to this one entry unless you verify another. The other
suppletive verbs checked already come back under a root a reader would look for:
`uvāca` → `vac`, `bravīti` → `brū`, `avadhīt` and `jaghāna` → `han`.

**Tests first.** `paśyati` → `paś`, `dṛś`; `anupaśyati` (today `anu` + `paś`);
an unrelated verb unchanged.

### 4. A verb form shared by several roots returns all of them

**Problem.** 2,127 of the 98,441 forms in the verb table (2.2%) belong to more
than one root, and the lookup returns one:

```
kariṣyati  roots in the table: kṛ, kṝ     returned: kṝ
cakāra     roots in the table: kṛ, kṝ     returned: kṝ
dhīyate    roots in the table: dhā, dhī   returned: dhī
```

**Cause.** `SQLite_find_verb` (`functions/SQLiteFind.py`) keeps the last matching
row ("the historical last-match rule") and `LIMIT 1`. The ambiguity sits at two
levels of the data, and both must be covered: two `vlgtab2` rows with different
stems (`kariṣyati`: `kṛ`, `kṝ`), and one stem and model with two `vlgtab1` rows
(`dhīyate`: stem `dhīy`, model `pre-p`, refs `dhā` and `dhī`).

**Change.** Return one entry per distinct root. Put first the entry the old rule
returned, so a form with one root gives byte-identical output and a consumer
reading the first entry sees what it always saw. Each entry's tags must be
computed from its own paradigm row.

**This is the only change of the four that touches every mode.** `roots` mode
will return a tuple of alternatives for those forms. Therefore:

- bump `ANALYSIS_ALGORITHM_VERSION` (`utils/analysisCache.py`, `v12` → `v13`),
  with a comment line, and update the two documents that quote the current
  value (`local-cache.md`, `prefix-segmentation.md`);
- compare the other modes before and after (procedure below): every difference
  must be a form that gained a root, and nothing else.

Do it last. If it grows beyond the lookup and its tests, stop and report rather
than widen it.

**Tests first.** `kariṣyati` and `dhīyate` return both roots in lemma mode and
in `roots` mode; `gacchati` returns exactly what it returns today;
`tests/test_verb_persons.py` and `tests/test_optimizations.py` still pass.

## Verifying the pass

1. The full suite, with the same skips and expected failures.
2. **The other modes.** Before touching anything, save `roots`, `parts` and the
   lemma list of detailed output (`cached=False`) for a fixed input list: the
   lines and the distinct words of `tests/datasets/yogaSutra.py`, the verbs
   listed at the end of this section, and the 1,500 held-out tokens
   (`git show split-ranker:tests/datasets/split_gold/wsmp_dev.json` and its two
   siblings; read them, do not check the branch out). Compare after each change.
   Changes 1 to 3 must change nothing here. Change 4 must change only forms that
   gained a root. The earlier pass compared 2,378 inputs this way; the scripts
   were scratch files and are not in the repository.
3. **Lemma mode.** Save its output for the same list before and after, and
   account for every difference by the rule that caused it.
4. The four-root check: 16 forms each of `gam`, `kṛ`, `bhū`, `dṛś`, counting the
   forms whose lemmas include the root. Today 9, 9, 10 and 4. Changes 3 and 4
   should move `dṛś` and `kṛ`; nothing should go down.
5. The independent review, then fix what it finds and re-run 1 to 3.

Verbs for steps 2 and 3 (prefixed finite verbs, then the four-root forms):

```
upagacchati anugacchati saṃgacchati udgacchati nirgacchati āgacchati samāgacchati
pratyāgacchati pratipadyate vyavaharati adhīte upaiti praṇamati niṣīdati abhyasyati
pratyeti anveti paryeti vyākaroti saṃskaroti uttiṣṭhati avatiṣṭhate pratiṣṭhate
upadiśyate anubhavati abhibhavati sambhavati vijānāti anujānāti parityajati nivartate
pravartate vinivartate upalabhate viharati saṃharati pratyāharati udāharati
upasaṃharati adhigacchati avagacchati atikrāmati nirdiśati upāsate apekṣate upekṣate
samīkṣate abhidhīyate vidhīyate ādatte pradadāti utpadyate niṣpadyate sampadyate

gacchati gacchanti agacchat gacchet gacchatu gamiṣyati jagāma agamat gamyate gamayati
gata gatavat gacchan gantavya gatvā gantum
karoti kurvanti akarot kuryāt karotu kariṣyati cakāra akārṣīt kriyate kārayati
kṛta kṛtavat kurvan kartavya kṛtvā kartum
bhavati bhavanti abhavat bhavet bhavatu bhaviṣyati babhūva abhūt bhūyate bhāvayati
bhūta bhūtavat bhavan bhavitavya bhūtvā bhavitum
paśyati paśyanti apaśyat paśyet paśyatu drakṣyati dadarśa adrākṣīt dṛśyate darśayati
dṛṣṭa dṛṣṭavat paśyan draṣṭavya dṛṣṭvā draṣṭum
```

## Not library work: for the agent building the index

- **Pass whole sentences, not single tokens.** A leading apostrophe is an
  avagraha only after a word in `-e` or `-o`, and lemma mode decides that from
  the word before (`tatra 'yasya` → `yasya`; alone, `'yasya` → `ayasya`). Do not
  strip apostrophes: `'pi` would become `pi` instead of `api`.
- **Re-join `-tva`/`-tā` suffix tokens** to the word before them if the corpus
  splits them off. `śūnya tvāt` reads `tvāt` as the pronoun `yuṣmad`;
  `śūnyatvāt` gives `śūnyatva`.
- **Index the preverb compound like any other lemma** (change 2). The part after
  a preverb carries the compound as one word (`pratī` for `pratyeti`, `upagam`
  for `upagacchati`; see rule 5 of [lemma-mode.md](lemma-mode.md)). It contains
  no `+` or other character a tokenizer cuts at, so nothing else is needed to
  tie a preverb to its verb. (The plan below proposed a `prati+i` key; the owner
  replaced it on 2026-11-19.)
- **Leave the extra surface lemma alone.** About 11% of parts carry their own
  form as one more lemma. These are frequent terms and weigh little.

## Later, not in this pass

Brief notes so the next attempt does not start from zero. None of these depends
on the split: none of 64 verb forms tested changes between the statistical and
the hybrid split.

**Participles in `-an`, `-ān` (`gacchan` → `gaccha`, `vidvān` → `vidu`).** The
forms table has no paradigm for most present and perfect active participles
(`gacchantam` is not a row). Looking the word up "as typed" cannot help. Two
routes:

- a rule after the dictionary miss, in the manner of the `-tā`/`-tva` deriver
  (`functions/taddhitaDerivation.py`): undo the `-at` endings, and accept the
  stem only if stem + `i` is a present form in the verb table (`gacchat` →
  `gacchati` → `gam`);
- import the participle paradigms of the Heritage data, assessed in
  [huet-merge-assessment.md](huet-merge-assessment.md).

**Participles, absolutives, infinitives, causatives (`gata`, `gatvā`, `gantum`,
`gamayati`).** They come back as themselves because the lexicon has no link from
a derived stem to its root; the Heritage data does not have it ready either.
What is needed is a derived table, stem → root, built offline, and then the
root is added as one more lemma, exactly as change 3 does for `paś`. No contract
change. Candidate sources, none of them tried here:

- the Digital Corpus of Sanskrit, which lemmatises these forms to their root;
  the build script on branch `split-ranker` already reads its CoNLL-U dump, and
  attested pairs come with frequencies;
- a rule-based generator of derived stems for the 628 roots;
- the Heritage participle readings.

Measure any of them with the four-root check above.

**Smaller items.**

- The fix of change 1 for the other modes: it belongs in `WordCascade`, and
  changes `roots` and detailed output for typed roots.
- A headword found only in the Pali or Buddhist Hybrid dictionaries still wins
  when the prefix stripper fails (`anveti`, `upekṣate`); restricting that early
  return to Monier-Williams and Apte was not tested.
- **Test the `-āv` fix extensively** ([final-av.md](final-av.md)): it was
  measured on the Yoga Sūtra, the 235 headwords ending in `v` and a few
  sentences only. Every `-āv` token of an annotated corpus (DCS, WSMP gold), in
  all four modes, against the gold lemma.
- **De-duplicate the preverb spellings.** The spellings of a preverb now live in
  four places that already disagree: `PREFIX_LOOKUP_SPELLINGS`
  (`utils/lexicalResources.py`, `ud` → `ut`, `vy` → `vi`, `praty` → `prati`, for
  lookup), `CompoundVerbIndex.OUR_SPELLINGS` and `MW_SPELLINGS`
  (`utils/compoundVerbs.py`, `ut` → `ud`, `saṃ` → `sam`, `nir` → `nis`, …),
  `Inflector.SPLIT_TOKEN_SPELLINGS` (`functions/inflect.py`, `niḥ` → `nis`), and
  the names written into `PreverbSandhi.join` (`utils/preverbSandhi.py`). One
  table of preverbs in `lexicalResources.py`, each with its canonical name and
  its spellings (pausa, before a vowel, before a voiced sound, Monier-Williams'),
  should feed all four. Only `CompoundVerbIndex` knows `vy` and `praty` today.
- The dictionary spelling of a prefixed verb (`pratī`, `upagam`) is now the
  lemma that ties a preverb to its verb (rule 5 of
  [lemma-mode.md](lemma-mode.md)); it could also serve as a display label.
- The split source: re-examine the default when `split-ranker` is merged
  ([lemma-mode.md](lemma-mode.md), "The split source").
