# Lemma mode: the analysis as a search-index query

`process(text, mode="lemmas")` returns the same analysis as the other modes in a
shape that can be turned directly into a query.

```python
process("yogaścittavṛttinirodhaḥ upagacchati xyzabc", mode="lemmas")

[ {"word": "yogaścittavṛttinirodhaḥ", "parts": [
      {"form": "yogaḥ",      "lemmas": ["yoga", "yogas"]},
      {"form": "cittavṛtti", "lemmas": ["cittavṛtti"], "members": ["citta", "vṛtti"]},
      {"form": "nirodhaḥ",   "lemmas": ["nirodha"]}]},
  {"word": "upagacchati", "parts": [
      {"form": "upa",      "lemmas": ["upa"]},
      {"form": "gacchati", "lemmas": ["gam", "upagam"]}]},
  {"word": "xyzabc", "parts": [{"form": "xyzabc", "lemmas": []}]} ]
```

The meaning is fixed by position: the **parts** of a word are all required, the
**lemmas** of a part are alternatives. `upagam` is the compound of the preverb
and the verb, the Monier-Williams headword, and ties the two parts together
(rule 5).

## Why `roots` mode is not enough

| | `mode="roots"` | `mode="lemmas"` |
|---|---|---|
| Two readings of one word | `['gacchati', 'gam']` | one part, lemmas `gam`, `gacchati` |
| Two pieces of one word | `['bhāvaya', 'id']` — same shape | two parts |
| A sentence | one flat list | one item per input token |
| `tatra tatra` | `['tatra']` | two items |
| An unknown word | a guess (`xyzabc` → `['ab']`) | a part with no lemmas |

## The contract

1. **One item per whitespace token of the input, in order.** The result lines up
   with `text.split()`. `word` is the token in IAST, as typed. A token with no
   letters (`|`, `50`, `-`, `%`) is an item with no parts; a token that cleans to
   several words (`te'pi`) keeps the parts of all of them.
2. **Nothing is truncated or dropped.** Every token comes back whatever the
   length of the text or of a single word, and a word that yields no analysis is
   one part with `"lemmas": []`. `max_length` has no effect in this mode.
3. **`form` is the un-sandhied form of the part** (`yogaḥ` for `yogaś`, `kaḥ`
   for the `ko` of `ko 'pi`), the token the splitter and the forms table work
   with.
4. **Lemmas are plain lowercase IAST**, in the order the analysis found them,
   without repeats. No lemma contains `+` or any other character a search
   tokenizer cuts at: the compounds of rule 5 are written as one word.
5. **A preverb is an ordinary part** before the verb: `pratyeti` → `prati` + `i`.
   The part carries the preverb only, never the homographs the forms table
   returns beside it (`sam` → `sa`), whether the prefix stripper found it, the
   splitter cut it off as a token of its own, or the inflector read it together
   with the token after it, that token being the stem or a second preverb
   (`tadvyavaharati`: the split `vi`, `ava`, `harati` is read `vi` + `ava`,
   then `harati`, and `ava` is a preverb, not a stem with the homograph `av`). A
   search for `i` finds every preverb compound of it. Position tells the preverb
   from the indeclinable: a standalone `prati` is a word with one part.

   **The part after one or more preverbs also carries their compound** (the
   *key*) for each of its lemmas, listed after them.

   ```
   pratyeti     ->  prati (prati) + eti (i, pratī)
   vyavaharati  ->  vi (vi) + ava (ava) + harati (hṛ, vyavahṛ)
   anubhavati   ->  anu (anu) + bhavati (bhavatī, bhavat, bhū,
                                         anubhavatī, anubhavat, anubhū)
   ```

   A bag-of-words index loses the fact that `prati` and `i` belong to one word,
   and both are very common terms; the key is rare and ties them. It is written
   as the dictionaries write the compound, so a search for the headword `pratī`
   finds `pratyeti`, and the reader can be sent to its entry:

   - **the Monier-Williams headword** whose parse it records
     (`<info verb="pre" parse="prati+i"/>` on `pratī`), read once per database
     by `CompoundVerbIndex` (`utils/compoundVerbs.py`, 6,148 parses). Our
     preverb names and Monier-Williams' are brought to one spelling first:
     `ut` reads `ud` (`ut` + `gam` → `udgam`), and Monier-Williams' `nir`, `dur`,
     `praty`, `vy` read `nis`, `dus`, `prati`, `vi`. Of several headwords, the
     one the joiner below spells the same way is taken (`vi` + `dhī` → `vidhī`,
     not `vidīdhī`), else all of them;
   - **otherwise, the compound `PreverbSandhi` builds** (`utils/preverbSandhi.py`):
     `upa` + `sam` + `arc` → `upasamarc`, `a` + `saṃsarga` → `asaṃsarga`,
     `vi` + `avahita` → `vyavahita`. Its rules follow Monier-Williams' spelling
     (`saṃgam` but `sampad`, `nirgam`, `niḥsṛ`, `praṇam`, `upaidh`, `abhiṣañj`)
     and spell 5,979 of the 6,148 parses as Monier-Williams does; the rest are
     its own choices word by word (`abhisap` but `abhiṣañj`). The `s` → `ṣ`
     rule also reaches nouns, where it is sometimes wrong (`vi` + `sarga` →
     `viṣarga`); it only applies to a word no dictionary lists whole, since an
     attested word is not split.

   The index is cached per database engine and cleared with the other lexicon
   caches (`_reset_reference_state`, `utils/dictionary_references.py`).

   The key used to be the names joined with `+` (`prati+i`), which a search
   tokenizer cuts in two. A preverb with nothing after it adds nothing, and a
   part without lemmas takes no key. Every stripped prefix counts, not only the
   preverbs of verbs: the privative `a` (`asaṃsargaḥ` → `a` + `saṃsarga`,
   `asaṃsarga`) and the outer prefix of a re-joined noun (`prati` +
   `saṃvedana`, `pratisaṃvedana`). The one exception is `api`
   (`PARTICLE_PREFIXES`, `utils/lexicalResources.py`): it is stripped and cut
   off like a preverb, but running text uses it as the particle "also, even"
   (`nāpyasti`), so it is a part with its own name that ties nothing and takes
   no key. Of the 97 parts of the 1,500 held-out tokens that carried a key, 8
   carried an `api` key, and all 8 were the particle in the gold segmentation.
   A key never crosses a word: not a whitespace token (in `prati eti`, `prati`
   is a word of its own), and not a segment of a pre-split compound
   (`upa-gacchati` is `upa` and `gacchati`).
6. **A verb is never re-joined with its preverb.** `roots` mode joins `sam`,
   `anu` and `ava` to the verb when the result is attested (`saṃgam`); lemma mode
   returns `sam` + `gam`, or a search for `gam` would miss it. A prefixed *noun*
   is re-joined when Monier-Williams or Apte has the joined word
   (`prati` + `saṃvedana`).
7. **A whole word wins over its preverb split when Monier-Williams or Apte has
   it.** `virati` is one part, not `vi` + `rati`. A headword found only in the
   Pali or Buddhist Hybrid dictionaries does not win, directly or through the
   re-join of rule 6: `upagacchati` is `upa` + `gam`, `anubuddha` is `anu` +
   `buddha`.
8. **The whole-word headword is one more lemma of a one-part word**, listed last:
   `gacchati` → `gam`, `gacchati`; `te` → `ta`, `tā`, `tad`, `yuṣmad`, `te`.
   That is the word as the analysis *matched* it, filed in any dictionary.
   **The word as typed is one of its own lemmas too**, when Monier-Williams or
   Apte has it: `vad` → `vat`, `vad`; `punar` → `puna`, `pu`, `punaḥ`, `punar`.
   The cascade reads a word's last letter as the sound it stands for before any
   dictionary is asked (`vad` → `vat`, `as` → `at`, `dṛś` → `dṛ`), so a root
   typed on its own used to come back as something else; 161 of the 628 roots of
   the verb table did. A typed headword that the analysis cut into several parts
   is one part, the whole word, as rule 7 has it for a preverb split: `sev` was
   `se` + `u`. "Typed" means the word after text cleaning, so it is the word
   lemma mode analyses, not the raw token.
9. **`members`** lists the members of a lexicalised compound, read from the
   Monier-Williams components by cutting on `—` only (see
   [dictionary-results.md](dictionary-results.md)). It is given only when the
   part has exactly one lemma, and the members are those of **the reading most
   of the headword's entries give**; a tie gives none
   (`ComponentMarks.prevailing_members`). Each Monier-Williams entry votes for
   the members its `components` marks, and an entry that marks none — the plain
   word, or a prefix reading such as `a-yoga` — votes for "no compound".
   Homographs share a headword, and a rare compound homograph must not lend its
   cut to a common word:

   | Headword | Entries | Members |
   |---|---|---|
   | `sukha` | 10 plain, 1 `su—kha` | none |
   | `bhāvana` | 15 plain, 1 `bhā—vana` ("a forest of rays") | none |
   | `ayoga` | 9 `a-yoga`, 1 plain, 1 `ayo—ga` | none |
   | `vikalpa` | 17 `vi-kalpa`, 1 `vi—kalpa` | none |
   | `svarūpa` | 14 `sva—rūpa`, 1 `sva-rūpa` | `sva`, `rūpa` |
   | `sukṛta` | 11 `su—kṛta` (accented or not), 1 plain | `su`, `kṛta` |

   Over the 194,068 Monier-Williams headwords this vote gives 326 fewer
   headwords members, and 348 more, than the rule it replaced (members only
   when every compound analysis agreed, which let a single `su—kha` decide
   `sukha` and a single `sva-rūpa` deny `svarūpa`); no compound's members
   changed. The key is absent when there are none. A key of rule 5 does not
   count as a second lemma: it is the lemma with its preverbs, not a rival
   reading, so a compound after a preverb keeps its members beside its key.
10. **The dictionary names passed to `process()` do not apply.** Lemma mode
    reports no definitions.

### Special input

| Input | Result |
|---|---|
| `dharma-kṣetra`, `hetu+pada` | one item whose parts are the segments' parts |
| `gatvā*`, alone | the headword if there is one, otherwise the word analysed without the star |
| `dharm_`, `yog%`, alone | one part whose lemmas are the matching headwords |
| `*`, `_`, `%` inside a text of several tokens | removed from the word; they are patterns only in a one-token query |
| empty text | `[]` |

A pattern is a dictionary query, and the other modes honour it only for
single-word input. In running text the same characters are noise, and a stray
`%` would bring back every headword there is (194,068 lemmas for one token).

### Avagrahas and quotation marks

An avagraha is decided by the word before it, and undoing one rewrites that word
too: `ko 'pi` is `kaḥ api`. The text is therefore restored as a whole and cut
back at the token boundaries (`restore_avagraha_by_token`, see
[avagraha-normalization.md](avagraha-normalization.md)), never token by token.
An apostrophe that ends a word is a closing quotation mark and adds nothing.

## Unknown tokens: when to believe the scanner

When the lexicon does not know a split token, the fallback scanner
(`root_compounds`) looks for dictionary words inside it and silently steps over
characters it cannot match. `ScannerPolicy` decides what to keep:

- every piece it found is kept, however short;
- the token gets **no lemmas only when the scanner skipped two or more
  characters**, or found nothing.

```
deśabandhaś  ->  deśa + bandha
śrutvā       ->  śrut + vā
bhāvayet     ->  bhāvaya + id
śāmyat       ->  śam               (one character skipped: an ending it could not read)
xyzabc       ->  lemmas: []        (four characters skipped)
```

The scanner knows no verb morphology, so a kept analysis often ends in a junk
piece (`id`, `vā`) and the stem is a related headword rather than the root
(`śrut`, not `śru`). That is the accepted price: a junk tail is a frequent word
and weighs little in a TF-IDF index, while a lost stem cannot be found at all.

Measured through lemma mode on this tree: the scanner analysed 218 tokens of the
1,500 held-out corpus tokens (202 with nothing skipped, 16 with one character
skipped) and 45 tokens of the 572 distinct Yoga Sūtra words (none skipped). None
of these 263 skipped two or more characters. Of six made-up words, three skipped
two or more and two found nothing.

A stricter rule was measured on the `split-ranker` tree, which has the corpus
frequency table it needs, and rejected. Requiring every piece to have four
letters and the token to be absent from the corpus removed 77 of 88 junk or
doubtful multi-piece analyses, but it also discarded stems such as `śrut` and
`bhāvaya`.

**Not caught:** a made-up word the *splitter* can cut into real short words never
reaches the scanner. `foobar` comes back as `phu` + `a` + `ba`.

**A word the splitter cannot handle** is analysed unsplit instead of failing the
call. The splitter raises on a single token of about a thousand characters, and
on a token that normalises to nothing; such a word goes to the scanner whole, and
comes back as an unknown part if that finds nothing either.

## The split source

A word the cascade cannot resolve is split, and `LemmaAnalyzer` takes the split
from one seam, `split_source`:

- `"statistical"` (default): the statistical splitter's split, with the scanner
  run per unknown token, where the characters it skips can be seen.
- `"hybrid"`: the split the other modes use on this tree. The scanner may
  replace the whole split unseen, so the rule above cannot fire: `xyzabc` → `ab`.

To compare the two:

```python
from process_sanskrit.functions.lemmaAnalysis import LemmaAnalyzer
from process_sanskrit.utils.databaseSetup import session_scope

with session_scope() as session:
    for source in LemmaAnalyzer.SPLIT_SOURCES:
        analyzer = LemmaAnalyzer(session=session, cached=False, split_source=source)
        print(source, analyzer.analyze("ekātmatevāsmitā"))
```

On the Yoga Sūtra the two give different parts for 13 of 572 distinct words. By
reading those 13, the hybrid split is better in 7 (`svabuddhisaṃvedanam` →
`sva` + `buddhi`, where the statistical split gives `su` + `abuddhi`), the
statistical split in 3 (`ekātmatevāsmitā` → `eka` + `ātmatā` + `iva` +
`asmitā`), and 3 are equivalent. The statistical split on this tree is the
unranked one; the ranked split on branch `split-ranker` is expected to change
this comparison, and the default should be re-examined when it is merged.

## Verb forms of several roots

**In lemma mode, a form the verb table files under several roots has all of
them as lemmas.** 2,127 of the 98,441 forms in the table (2.2%) belong to more
than one root, and `SQLite_find_verb` (`functions/SQLiteFind.py`) used to return
one: the last matching `vlgtab2` row and its first paradigm ("the historical
last-match rule"). It now returns every root, and the ambiguity sits at two
levels of the data, both read:

| Form | Where | Roots, the old answer first |
|---|---|---|
| `kariṣyati`, `cakāra` | two `vlgtab2` rows with different stems | `kṝ`, `kṛ` |
| `ucyate` | two `vlgtab2` rows with different models (`pre-4m`, `pre-p`) | `uc`, `vac` |
| `abhajyata` | one `vlgtab2` row (stem `abhajy`, model `ipf-p`) whose stem and model have two `vlgtab1` paradigms | `bhaj`, `bhañj` |

767 of the 2,127 forms reach one of their roots only through a second paradigm
of one stem and model, as `abhajyata` does. Each root is reported once, with its
tags read from the paradigm it comes from (`abhakṣata` is a third singular of
`bhakṣ` and a third plural of `bhaj`). The first entry is the one the old rule
returned, so a form of one root gives exactly what it always gave, and a caller
that reads only the first entry sees what it always saw. A later root is read
from the first of its rows whose paradigm has the form, or from its first row
when none does: `juhumahe` takes `hu` from its present (first plural), not from
a perfect whose table lacks the form.

**Only lemma mode reads the other roots.** `every_verb_root`
(`functions/rootAnyWord.py`) is a context manager that `LemmaAnalyzer.analyze`
holds for its whole analysis; inside it, `_direct_roots`, where the nominal and
the verbal readings of a word are put together, reports every root of a verb
form, and outside it the first only. The memo keeps every root and serves either
setting. The one part of lemma mode that reads the first root only is the
`"hybrid"` split source, which is the other modes' split and is computed as they
compute it. The setting belongs to the current thread: work handed to another
thread must hold the block itself. Lemma mode lists the roots as lemmas of one part, and the keys of
rule 5 are made for each: `kariṣyati` → `kṝ`, `kṛ`; `abhidhīyate` → `abhi` +
`dhīyate` (`dhī`, `dhā`, `abhidhī`, `abhidhā`); `saṃkariṣyati` → `sam` +
`kariṣyati` (`kṝ`, `kṛ`, `saṃkṝ`, `saṃkṛ`). A one-letter root keeps the place
the one-letter rule gives it ([one-letter-roots.md](one-letter-roots.md)): the
word `agāt` has `aga`, `i` and `gā`, while a piece the scanner cuts out of an
unknown token keeps `gā` and still withholds `i`.

`roots`, `parts` and detailed output give exactly what they gave before: over
the 2,386 comparison inputs (the Yoga Sūtra lines and distinct words, the verbs
of [lemma-mode-handoff.md](lemma-mode-handoff.md), the 1,500 held-out tokens and
13 words of the tests) not one differs. They were tried with every root and
dropped it, on the owner's decision, because two rules of `clean_results` expect
one root per form:

- *The prefix re-join* joins `sam`, `anu` or `ava` to the stem after it when the
  joined word is attested, and takes the first root only. The other roots were
  left behind as pieces of their own, without the prefix: `saṃkariṣyati` gave
  `saṃkṝ`, `kṛ` where it gives `saṃkṝ`, and `samucyate` gave `samuc`, `vac`. For
  1,505, 1,021 and 618 of the 2,127 forms, `sam`, `anu` or `ava` and the first
  root make an attested word.
- *The `-n` rule* replaces an entry whose lemma ends in `-n` by the dictionary
  entry of its surface form, and never visits the last entry of the list. `vena`
  is filed under `van` and `ven`; with a root after it, `van` was no longer last
  and was replaced (`('va', 'vena', 'van')` became `('va', 'vena')` followed by
  `ven`). It is the one form of the 98,441 exposed to that rule.

Giving those modes every root needs both rules taught about alternatives first,
and `roots` mode a shape for the alternatives of a re-joined word. Because their
output is unchanged, no cache version was bumped: the cached hybrid morphology is
theirs, and lemma mode caches only the statistical split, which never reads the
verb table.

**The verb table files the present of `dṛś` under the root `paś`**, so
`paśyati` reached `paś` only, and 4 of 16 common forms of `dṛś` reached `dṛś`.
Lemma mode adds the root a reader looks up right after the one the table gives,
which stays: `paśyati` → `paś`, `dṛś`; `anupaśyati` → `anu` + `paśyati` (`paś`,
`dṛś`, `anupaś`, `anudṛś`). The table is `SUPPLETIVE_ROOTS`
(`utils/lexicalResources.py`), and it holds that one entry: the other suppletive
verbs checked already come back under the root a reader looks for (`uvāca` →
`vac`, `bravīti` → `brū`, `avadhīt` and `jaghāna` → `han`). The mapping applies
to verbal entries only, because the forms table also has a feminine noun `paś`.
It too is lemma mode only.

Sixteen common forms each of `gam`, `kṛ`, `bhū` and `dṛś`, counted by whether
their lemmas include the root: 9, 9, 10 and 4 before these two changes; 9, 11,
10 and 9 after (`kariṣyati` and `cakāra` reach `kṛ`; the five present-system
forms reach `dṛś`). The forms still missing are participles, absolutives,
infinitives and causatives, which the lexicon does not link to a root
([lemma-mode-handoff.md](lemma-mode-handoff.md), "Later").
`CommonFormsReachTheirRootTests` keeps every form that reaches its root
reaching it.

## How it is built

Lemma mode needs three facts the flat entry list has already lost by the time
`clean_results` runs. Each is recorded in an optional trace
(`utils/analysisTrace.py`) that changes nothing when absent:

| Fact | Trace | Filled in by |
|---|---|---|
| which prefix was stripped, and what was left | `PrefixTrace` | `root_any_word`, before `_stamp_whole_word` overwrites it |
| which pieces the scanner matched, and how many characters it skipped | `ScanTrace` | `root_compounds` |
| which split token an entry came from | `TokenAnalysis` | `Inflector.resolve` (`analyze` is its flattening) |

The single-word cascade (forms table → dictionary → `-tā`/`-tva` deriver) was
moved out of `process()` into `WordCascade` (`functions/wordCascade.py`), so both
the older modes and lemma mode follow one ordering. `LemmaAnalyzer`
(`functions/lemmaAnalysis.py`) cuts the text into tokens, cleans each
(`to_iast` and `clean_iast`, the two halves of `preprocess`), runs the cascade
per word and hands the result to `LemmaBuilder`; `LemmaLexicon` is the builder's
only window on the dictionaries.

`LemmaBuilder.finished` makes two passes over a word's parts, in this order:
`with_typed_headword` (rule 8, the word as typed) and `join_preverbs` (rule 5,
the keys). The order matters: a word replaced by its typed headword is one part
and has no preverb left to join. A preverb part is marked by an internal flag,
`LemmaPart.preverb`, set where such parts are made (`LemmaBuilder._preverb`,
called for the stripped prefixes of `from_lookup`, a remainder that is itself a
prefix, and the lone preverb tokens of `from_tokens`) and never exported.
`_lemmas`, where entries become lemmas, adds the roots of `SUPPLETIVE_ROOTS`.
Both passes run once per word, on the segments of a pre-split compound one
segment at a time.

Built when lemma mode was introduced, the older modes were unchanged: 2,378
inputs gave identical `roots`, `parts` and detailed output before and after,
apart from the six inputs over 100 characters that are no longer truncated (see
[long-input.md](long-input.md)). The typed headword, the joined keys, the
suppletive roots and the other roots of a shared verb form are lemma mode only as
well: none of 2,386 inputs changes in the other modes. With caching on, lemma
mode reuses the statistical split cache (rows keyed with `attempts=20`) and
stores nothing else; the statistical split never reads the verb table, so that
cache is valid whichever roots are read.

Timing on the whole Yoga Sūtra (666 tokens, 1,179 parts), on a loaded machine:
1.6 s in lemma mode (0.7 s of CPU), against 2.4 s for `roots` mode called word by
word. The two passes of `finished` cost nothing measurable, and the verb query
that returns every root takes the time the single-row query took.

## Invariants to preserve

- **A trace only observes.** `root_any_word`, `root_compounds` and
  `Inflector.analyze` must return the same thing with and without one.
- **`LemmaBuilder` never modifies the entries it reads**, and never calls
  `clean_results`, which does.
- **A part is never modified once made.** Parts are memoised per word and
  shared between its repeats; `with_lemma`, `with_lemmas` and both passes of
  `finished` return new parts.
- **A withheld one-letter root is not a skipped character.** `arcayet` keeps
  `arc` because the withheld `ayet` does not count against it.
- **Every word with a letter in it yields at least one part.**
  `LemmaAnalyzer._parts` falls back to a part with no lemmas; removing that
  drops words silently.
- **Two notions of headword, kept apart.** `attested_headword`
  (`functions/dictionaryLookup.py`) asks whether any dictionary has a word; it
  is what `dict_search` uses, and what makes a dictionary-only piece a lemma.
  `LemmaLexicon.has_sanskrit_headword` asks whether Monier-Williams or Apte has
  it, and is the only test for rules 6 and 7 and for the typed headword of rule
  8. Using the first where the second belongs lets a Pali headword replace a
  preverb split.
- **The typed headword runs before the joined keys**, and both run after the
  parts are built, never inside `from_lookup` or `from_tokens`.
- **Patterns are for one-token input only.**
- **Avagrahas are restored over the whole text**, then cut back by token.

## Known limitations

- Rule 8 also adds Apte's nominative headwords (`samādhiḥ` beside `samādhi`).
- A headword found only in the Pali or Buddhist Hybrid dictionaries still wins
  when the prefix stripper fails: `upekṣate` and `anveti` come back as
  themselves, with no root.
- Rule 7 lets a Monier-Williams noun take the place of a finite verb: `āyāti`
  is one part `āyāti`, and its verb reading (`ā` + `yā`) is not offered.
- "Sanskrit dictionary" means Monier-Williams and Apte. A word only Cappeller,
  Macdonell or Grassmann has is not treated as a whole-word lemma (`abhaṇita` →
  `a` + `bhaṇita`).
- A derived `-tā`/`-tva` word is one part whose only lemma is the derivative
  (`niṣyandatā`); its base is not offered as a lemma.
- `ud` is reported as `ut`, as in the other modes.
- A re-joined part's form is built from the prefix's name and the stem's form:
  `samvedanā` for the `saṃvedanā` of `pratisaṃvedanā`.
- A scanner piece's form is the piece, so an ending the scanner could not read
  is not shown: `arc` for `arcayet`.
- When a preverb token is joined to the next token by `Inflector`, the whole
  word checked against the dictionaries is their plain concatenation
  (`vi` + `avahitānām`), so rule 7 cannot recognise `vyavahita`.
- An unknown `mode` name is not refused: `mode="lemma"` returns detailed output,
  as any unrecognised mode always has.
- **A `jj` is cut only in a word that does not resolve whole**
  ([double-j.md](double-j.md)). `lajjā`, `majjati` and the roots `lajj`, `majj`,
  `bhrajj`, `ujjh` come back as themselves. `tajjapas` is still `taj` (`tad`,
  `tat`, `taj`) + `japas`, and `tajjayāt` is now the dictionary's `tajjaya`
  (members `taj`, `jaya`). A compound token holding a `jj` word
  (`lajjāvismayamanmathaiḥ`) is still cut.
- **In running text the typed headword mostly adds a spelling of a word the
  part already has.** The roots it was made for are typed on their own in
  queries and grammatical texts; a running text spells most words in sandhi,
  and Monier-Williams or Apte also keys many of those spellings: `kiṃ` → `kim`,
  `kiṃ`; `so` → `sa`, `tad`, `so`; `tato` → `tata`, `tato`; `kaivalyaṃ` →
  `kaivalya`, `kaivalyaṃ` (`kiṃ`, `so`, `tato` are Monier-Williams keys,
  `kaivalyaṃ` an Apte one). Of the 572 distinct Yoga Sūtra words it changed 10:
  `punar` and `bahir` gain their Monier-Williams spelling, `tato`, `kiṃ`,
  `hānaṃ` and `kaivalyaṃ` gain their sandhi spelling, `tajjayāt`, `tajjapas`
  and `tajjaḥ` gain `taj` (above), and `samādhāv` becomes the wrong word
  (below). These are frequent, low-weight index terms.
  Folding `-ṃ`/`-m`, `-o`/`-aḥ` and `-r`/`-ḥ` before the test would drop the
  spellings, and `punar` and `antar` with them.
- **A typed headword can be a second, rarer word.** `-āv` before a vowel is
  the sandhi spelling of a locative in `-au`, and Monier-Williams has a root
  `X-dhāv` for most of the prefixes. Since a final `-āv` is read as `-au`
  ([final-av.md](final-av.md)), `samādhāv` → `samādhi`, with the root
  `samādhāv` beside it as the typed headword; before, the root was its only
  lemma. The typed headword is kept for running text rather than restricted to
  one-token input, so that a word analyses the same in a one-word query and in
  an indexed text.
- A key inherits the split it is built from: a preverb the splitter cut
  wrongly ties a wrong pair, and the joiner then spells a word nobody looks for
  (`puruṣasyāpariṇāmitvāt` → `āpariṇāmitva`, `upaiti` → `upeti`). The split
  parts stay beside it, so retrieval loses nothing.
- The members vote of rule 9 counts dictionary entries, not usage in texts. It
  is right when the common word has more entries than its rare homograph, as it
  usually does; an obscure homograph with more entries would win.
- Members are cut where Monier-Williams puts the mark, which after a merged
  vowel loses the second member's initial vowel: `mahātman` → `mahā` + `tman`,
  not `ātman`. See [known-errors.md](known-errors.md).
- A part can carry members beside two real lemmas when rule 8 adds the second
  one to a compound: `ṛtaṃbharā` → `ṛtaṃbhara`, `ṛtaṃbharā`, with the members
  `ṛta`, `ṃbhara`, whose cut is also wrong. It is the one such part in the 2,386
  comparison inputs.

Pinned by `tests/test_lemma_builder.py` (the rules, without the lexicon),
`tests/test_word_cascade.py` (the cascade and the traces),
`tests/test_lemma_mode.py` (the contract through `process()`, and `LemmaLexicon`
against the installed dictionaries) and `tests/test_shared_verb_forms.py` (verb
forms of several roots: the lookup, `every_verb_root`, lemma mode, and the other
modes unchanged).
