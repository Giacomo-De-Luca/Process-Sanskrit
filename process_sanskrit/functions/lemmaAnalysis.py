"""`process(text, mode="lemmas")`: the analysis as a search-index query.

The result has one item per whitespace token of the input, in order:

    {"word": "upagacchati", "parts": [
        {"form": "upa",      "lemmas": ["upa"]},
        {"form": "gacchati", "lemmas": ["gam", "upagam"]}]}

The parts of a word are all required; the lemmas of a part are alternatives.
`upagam`, the compound of the preverb and the verb, ties the two.  Nothing is
truncated or dropped: every token comes back, and a token that cannot be
analysed comes back as a part with no lemmas.

The classes, from the data outwards:

* `LemmaPart`      one part of a word;
* `ScannerPolicy`  when to believe the fallback scanner;
* `LemmaLexicon`   the builder's only window on the dictionaries;
* `PieceLemmas`    the lemmas of one word on its own, for the members' citation forms;
* `LemmaBuilder`   turns an analysis into parts;
* `LemmaAnalyzer`  runs the analysis for a text.

See documentation/lemma-mode.md for the rules and the measurements behind them.
"""

import re
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Optional

import regex

from process_sanskrit.functions.cleanResults import REJOINABLE_PREFIXES, rejoin_prefix
from process_sanskrit.functions.dictionaryLookup import (
    attested_headword,
    dict_search,
    headword_components,
)
from process_sanskrit.functions.inflect import Inflector, TokenAnalysis
from process_sanskrit.functions.process import WILDCARD_CHARACTERS, clean_iast, to_iast
from process_sanskrit.functions.rootAnyWord import every_verb_root
from process_sanskrit.functions.wordCascade import WordCascade, WordResolution
from process_sanskrit.utils.analysisTrace import PrefixTrace, ScanTrace
from process_sanskrit.utils.compoundMembers import CompoundForms, CompoundMembers
from process_sanskrit.utils.compoundVerbs import CompoundVerbIndex
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils.engineCache import EngineCache
from process_sanskrit.utils.lexicalResources import (
    PARTICLE_PREFIXES,
    SANSKRIT_PREFIXES,
    SUPPLETIVE_ROOTS,
)
from process_sanskrit.utils.paradigm import is_verbal_entry
from process_sanskrit.utils.preverbSandhi import PreverbSandhi
from process_sanskrit.utils.transliterationUtils import restore_avagraha_by_token


@dataclass
class LemmaPart:
    """One part of a word: its un-sandhied form and the lemmas it may be.

    `lemmas` are alternatives and may be empty, which is how an unknown part is
    reported.  `members` are the members of a lexicalised compound, when the
    dictionary marks them.  `preverb` marks a part that is a preverb of the
    part after it; `LemmaBuilder.join_preverbs` reads it, and it is never
    exported.

    A part is never modified once made: parts are memoised per word and shared
    between its repeats, so every change returns a new one.
    """

    form: str
    lemmas: List[str]
    members: List[str] = field(default_factory=list)
    preverb: bool = False

    def with_lemma(self, lemma: str) -> "LemmaPart":
        """This part with one more alternative, listed last; unchanged if present."""
        return self.with_lemmas([lemma])

    def with_lemmas(self, lemmas: List[str]) -> "LemmaPart":
        """This part with the lemmas it lacks appended in order; unchanged if none."""
        added = [lemma for lemma in dict.fromkeys(lemmas) if lemma not in self.lemmas]
        if not added:
            return self
        return replace(self, lemmas=self.lemmas + added)

    def as_dict(self) -> dict:
        shape = {"form": self.form, "lemmas": list(self.lemmas)}
        if self.members:
            shape["members"] = list(self.members)
        return shape


class ScannerPolicy:
    """When to believe what the fallback scanner found in an unknown token.

    Its pieces are kept, however short, unless it had to step over more than
    `max_skipped` characters.  One unmatched character is an ending it could not
    read (`śāmyat` still gives `śam`); two or more mean it did not understand the
    word (`xyzabc`).  The scanner knows no verb morphology, so a kept analysis
    often ends in a junk piece (`bhāvayet` -> `bhāvaya` + `id`).  That is the
    accepted price: a junk tail costs a search index little, a lost stem a lot.
    """

    def __init__(self, max_skipped: int = 1):
        self.max_skipped = max_skipped

    def accepts(self, scan: ScanTrace) -> bool:
        return bool(scan.pieces) and scan.skipped <= self.max_skipped


class LemmaLexicon:
    """What the builder asks the dictionaries, and nothing else.

    Lemma mode reports no definitions, so the dictionaries a caller names do not
    matter to it.

    Two notions of "headword" are kept apart on purpose.  `headword` asks whether
    *any* dictionary has the word, which is what decides a dictionary hit in the
    other modes too.  `has_sanskrit_headword` asks whether Monier-Williams or
    Apte has it, and is the stricter test for letting a whole word stand as a
    lemma in place of its preverb split: the Pali and Buddhist Hybrid
    dictionaries file inflected forms (`upagacchati`), which must not win.
    """

    SANSKRIT_DICTIONARIES = frozenset({"mw", "ap90"})

    ## members depend on the database only, so they outlive one `process()` call.
    ## Every lexicon on an engine shares them: a subclass that answers
    ## `has_sanskrit_headword` or `_compound_members` its own way must also
    ## override `members` (as `_MemberlessLexicon` and the evaluation's
    ## `RecordingLexicon` do), or it would mix its answers into the shared ones.
    _members = EngineCache()

    def __init__(self, session):
        self.session = session
        self._compound_members_reader: Optional[CompoundMembers] = None

    def has_sanskrit_headword(self, word: str) -> bool:
        return bool(self.SANSKRIT_DICTIONARIES.intersection(DICTIONARY_REFERENCES.get(word, ())))

    def headword(self, word: str) -> Optional[str]:
        """The headword some dictionary files `word` under, or None."""
        return attested_headword(word)

    def members(self, lemma: str) -> List[str]:
        """The members of a compound headword (`CompoundMembers`; rule 9 of lemma-mode.md).

        Monier-Williams' `—` by the reading most of its entries give: `sukha` is
        ten plain entries and one `su—kha`, `bhāvana` fifteen and one `bhā—vana`
        ("a forest of rays"), and the rare compound must not lend its cut to the
        common word; a tie gives none.  For a headword Monier-Williams lacks,
        Edgerton's and Grassmann's `-` (`ālaya` + `vijñāna`).  A pair with a
        side that is no word is read again where sandhi joined it (`nānā` +
        `rtha` -> `nānā` + `artha`), or else kept as the dictionary cuts it,
        and a compound-initial spelling is put in citation form where lemma mode
        gives that form for the split piece (`taj` -> `tad`).
        """
        found = self._members.memo(self.session)
        if lemma not in found:
            found[lemma] = self._compound_members().members(lemma)
        return list(found[lemma])

    def _compound_members(self) -> CompoundMembers:
        if self._compound_members_reader is None:
            self._compound_members_reader = CompoundMembers(
                CompoundForms(self.session),
                components=lambda dictionary, word: headword_components(word, dictionary, session=self.session),
                is_word=self.has_sanskrit_headword,
                breadth=lambda word: len(DICTIONARY_REFERENCES.get(word, ())),
                in_monier_williams=lambda word: "mw" in DICTIONARY_REFERENCES.get(word, ()),
                piece_lemmas=PieceLemmas(self.session),
            )
        return self._compound_members_reader

    def joined_headword(self, prefix: str, stem: str) -> Optional[str]:
        """The headword for `prefix` + `stem` as one word, or None.

        Asks `rejoin_prefix`, the join the other modes perform, on a scratch
        pair, so the spelling of the joined word is found in one place.  The
        join then has to pass the same test as any other whole word that
        stands in place of its preverb split.
        """
        scratch = [[prefix], [stem]]
        rejoin_prefix(scratch, 0, REJOINABLE_PREFIXES[prefix], session=self.session)
        if len(scratch) != 1:
            return None
        headword = scratch[0][0]
        return headword if self.has_sanskrit_headword(headword) else None

    def compound_verbs(self, preverbs: List[str], stem: str) -> List[str]:
        """The Monier-Williams headwords of `preverbs` + `stem`; [] if it records none."""
        return CompoundVerbIndex(self.session).headwords(preverbs, stem)

    def headwords_matching(self, pattern: str) -> List[str]:
        """Every headword a `_` / `%` pattern matches, in dictionary order."""
        payload = dict_search([pattern], session=self.session)[0][2]
        if not isinstance(payload, dict):
            return []
        return list(dict.fromkeys(headword for found in payload.values() for headword in found))


class LemmaBuilder:
    """Turn an analysis into the parts of a word.  Never modifies what it reads."""

    def __init__(self, lexicon, policy: Optional[ScannerPolicy] = None):
        self.lexicon = lexicon
        self.policy = policy or ScannerPolicy()

    @staticmethod
    def _lemmas(entries) -> List[str]:
        """The lemmas of morphology `entries`, in order and without repeats.

        A verb the table files under another root than the one a reader looks
        up brings that root along, right after its own (`paś`, `dṛś`).  Only a
        verb: the forms table also has a feminine noun `paś`.
        """
        lemmas = []
        for entry in entries:
            lemmas.append(entry[0])
            if entry[0] in SUPPLETIVE_ROOTS and is_verbal_entry(entry):
                lemmas.append(SUPPLETIVE_ROOTS[entry[0]])
        return list(dict.fromkeys(lemmas))

    def _part(self, form: str, lemmas: List[str]) -> LemmaPart:
        """A part, with its compound members when it has exactly one reading.

        With several alternative lemmas the members would belong to one of them
        and nothing in the part would say which.
        """
        members = self.lexicon.members(lemmas[0]) if len(lemmas) == 1 else []
        return LemmaPart(form, lemmas, members)

    def _headword_part(self, word: str) -> LemmaPart:
        """A piece only the dictionaries can know: a headword, or nothing."""
        headword = self.lexicon.headword(word)
        return self._part(word, [headword] if headword else [])

    def _rejoined(self, prefixes: List[str], stems: list, remainder: str) -> Optional[LemmaPart]:
        """The innermost prefix and a nominal stem as one attested word, if they are one.

        A verb is never re-joined: its preverb stays a part, so that a search for
        the root finds every preverb compound of it.
        """
        innermost = prefixes[-1]
        if innermost not in REJOINABLE_PREFIXES:
            return None
        if any(is_verbal_entry(stem) for stem in stems):
            return None
        joined = [
            headword
            for headword in (
                self.lexicon.joined_headword(innermost, lemma) for lemma in self._lemmas(stems)
            )
            if headword
        ]
        if not joined:
            return None
        return self._part(innermost + remainder, list(dict.fromkeys(joined)))

    def from_lookup(self, entries: list, prefixes: Optional[PrefixTrace] = None) -> List[LemmaPart]:
        """The parts of one lookup of one span.

        `entries` is what `root_any_word` returned for it, or the bare pieces of
        a scan that only the dictionaries know.  `prefixes` is what was stripped
        to resolve it.
        """
        analysed = [entry for entry in entries if isinstance(entry, list)]
        if not analysed:
            return [self._headword_part(entry) for entry in entries]

        ## the stem's entries carry the whole word as matched; a stripped prefix's
        ## own entries -- homographs included -- carry the prefix, and are not read
        whole = analysed[-1][4]
        stems = [entry for entry in analysed if entry[4] == whole]
        if not prefixes:
            return [self._part(whole, self._lemmas(stems))]

        ## a whole word Monier-Williams or Apte files is a lemma, not a split
        if self.lexicon.has_sanskrit_headword(whole):
            return [self._part(whole, [whole])]

        outer = list(prefixes.prefixes)
        ## the inflector reads two preverbs the splitter cut off together (`upa`
        ## + `sam` of `upasaṃharati`), and the inner one comes back as the stem,
        ## homograph `sa` included: it is a preverb too, and the run goes on
        if prefixes.remainder in SANSKRIT_PREFIXES:
            return [self._preverb(prefix) for prefix in outer + [prefixes.remainder]]
        stem_part = self._rejoined(outer, stems, prefixes.remainder)
        if stem_part is None:
            stem_part = self._part(prefixes.remainder, self._lemmas(stems))
        else:
            outer = outer[:-1]
        return [self._preverb(prefix) for prefix in outer] + [stem_part]

    def from_resolution(self, resolution: WordResolution) -> List[LemmaPart]:
        """The parts of a word the single-word cascade resolved."""
        if resolution.stage == WordResolution.MORPHOLOGY:
            entries = [entry for entry in resolution.entries if entry is not resolution.headword]
            parts = self.from_lookup(entries, resolution.prefixes)
            ## the surface form is itself a headword somewhere: one more reading
            ## of a one-part word, and nothing a split word can carry
            if resolution.headword is not None and len(parts) == 1:
                parts = [parts[0].with_lemma(resolution.headword[0])]
            return parts
        if resolution.stage == WordResolution.DICTIONARY:
            word = resolution.entries[0][0]
            return [self._part(word, [self.lexicon.headword(word) or word])]
        if resolution.stage == WordResolution.DERIVED:
            ## the components belong to the base the word was derived from
            entry = resolution.entries[0]
            return [LemmaPart(entry[4], [entry[0]])]
        return []

    @staticmethod
    def _preverb(name: str) -> LemmaPart:
        """A preverb part: its own name only, marked for `join_preverbs`.

        A prefix that running text uses as a particle (`api`, see
        `PARTICLE_PREFIXES`) carries its name too, but is no preverb.
        """
        return LemmaPart(name, [name], preverb=name not in PARTICLE_PREFIXES)

    @classmethod
    def _lone_preverb(cls, analysis: TokenAnalysis) -> Optional[LemmaPart]:
        """A preverb the splitter cut off and nothing joined to the next token.

        Looked up as a word it brings its homographs along (`sam` -> `sa`), just
        as it does when the stripper finds it; here too it is the preverb only.
        """
        if len(analysis.tokens) != 1 or analysis.prefixes:
            return None
        if analysis.tokens[0] not in SANSKRIT_PREFIXES:
            return None
        return cls._preverb(analysis.entries[0][4])

    def from_tokens(self, analyses: List[TokenAnalysis]) -> List[LemmaPart]:
        """The parts of a word that had to be split, one analysis per split token."""
        parts: List[LemmaPart] = []
        for analysis in analyses:
            if analysis.source == TokenAnalysis.LOOKUP:
                preverb = self._lone_preverb(analysis)
                if preverb is not None:
                    parts.append(preverb)
                else:
                    parts.extend(self.from_lookup(analysis.entries, analysis.prefixes))
            elif analysis.source == TokenAnalysis.UNRESOLVED:
                parts.extend(self.from_lookup(analysis.entries))
            elif self.policy.accepts(analysis.scan):
                for piece in analysis.scan.pieces:
                    parts.extend(self.from_lookup(piece.entries, piece.prefixes))
            else:
                parts.append(LemmaPart("".join(analysis.tokens), []))
        return parts

    def with_typed_headword(self, word: str, parts: List[LemmaPart]) -> List[LemmaPart]:
        """`word` among its own lemmas, as typed, when Monier-Williams or Apte has it.

        The cascade reads a word's last letter as the sound it stands for before
        any dictionary is asked, and its retries then find some other word: a
        root typed on its own came back as something else (`vad` -> `vat`,
        `as` -> `at`, `dṛś` -> `dṛ`).  The headword rule of `from_resolution`
        looks at the form that was *matched*, so it cannot see the word that
        was typed.

        One part gains the word as one more alternative, listed last.  Several
        parts give way to one part for the whole word, as a whole word wins over
        its preverb split (`sev` was `se` + `u`).
        """
        if not self.lexicon.has_sanskrit_headword(word):
            return parts
        if any(word in part.lemmas for part in parts):
            return parts
        if len(parts) == 1:
            return [parts[0].with_lemma(word)]
        return [self._part(word, [word])]

    def preverb_keys(self, preverbs: List[str], lemma: str) -> List[str]:
        """The compound `preverbs` and `lemma` make, as the lemmas that tie them.

        The Monier-Williams headword whose parse it records (`prati` + `i` ->
        `pratī`); of several, the one `PreverbSandhi` spells the same way, else
        all of them.  Without one, the compound `PreverbSandhi` builds
        (`upa` + `sam` + `arc` -> `upasamarc`).  Never anything with `+` in it:
        a search tokenizer cuts there, and the key would be lost.
        """
        joined = PreverbSandhi.join_all(preverbs, lemma)
        headwords = self.lexicon.compound_verbs(preverbs, lemma)
        if not headwords or joined in headwords:
            return [joined]
        return headwords

    def join_preverbs(self, parts: List[LemmaPart]) -> List[LemmaPart]:
        """Tie each run of preverbs to the part after it, with its compound per lemma.

        `pratyeti` is `prati` + `i` by design, so that a search for `i` finds
        every preverb compound of it -- but a bag of words then loses the fact
        that the two belong to one word.  The part after a run of preverbs
        gains, for each of its lemmas, the compound of the preverbs and the
        lemma (`preverb_keys`): `eti` (`i`, `pratī`), `harati` (`hṛ`,
        `vyavahṛ`).  A preverb carries its own name only, so that name is its
        lemma.  A preverb with nothing after it adds nothing; a part without
        lemmas, or a particle such as `api`, takes no key, and still ends the
        run.
        """
        joined: List[LemmaPart] = []
        run: List[str] = []
        for part in parts:
            if part.preverb:
                run.append(part.lemmas[0])
                joined.append(part)
                continue
            if run and not PARTICLE_PREFIXES.intersection(part.lemmas):
                part = part.with_lemmas(
                    [key for lemma in part.lemmas for key in self.preverb_keys(run, lemma)]
                )
            run = []
            joined.append(part)
        return joined

    def finished(self, word: str, parts: List[LemmaPart]) -> List[LemmaPart]:
        """The parts of `word` as they are reported.

        The typed headword goes first: a word it replaces is one part, and has
        no preverb left to join.
        """
        return self.join_preverbs(self.with_typed_headword(word, parts))


class _MemberlessLexicon(LemmaLexicon):
    """A lexicon that gives no members, so that `PieceLemmas` cannot recurse into them."""

    def members(self, lemma: str) -> List[str]:
        return []


class PieceLemmas:
    """The lemmas lemma mode gives one word on its own, members aside.

    `CompoundMembers` asks it whether the citation form of a compound-initial
    spelling is a lemma of the piece the splitter would hand over (`manaḥ`
    for `mano`).  It runs the single-word cascade and the builder lemma mode
    runs, reading every root of a shared verb form as lemma mode does.  The
    answers depend on the database only, so they are kept per database
    engine, as `CompoundVerbIndex` keeps its table: indexing a corpus asks the
    same few hundred spellings again and again.
    """

    _answers = EngineCache()

    @classmethod
    def reset(cls) -> None:
        """Forget every answer; the next question works it out again."""
        cls._answers.clear()

    def __init__(self, session):
        self.session = session
        self._cascade: Optional[WordCascade] = None
        self._builder: Optional[LemmaBuilder] = None

    def __call__(self, word: str) -> FrozenSet[str]:
        answers = self._answers.memo(self.session)
        if word not in answers:
            answers[word] = self._lemmas(word)
        return answers[word]

    def _lemmas(self, word: str) -> FrozenSet[str]:
        if self._cascade is None:
            self._cascade = WordCascade(session=self.session, memo={})
            self._builder = LemmaBuilder(_MemberlessLexicon(self.session))
        with every_verb_root():
            resolution = self._cascade.resolve(word)
            if not resolution.resolved:
                return frozenset()
            parts = self._builder.from_resolution(resolution)
        return frozenset(lemma for part in parts for lemma in part.lemmas)


class LemmaAnalyzer:
    """Analyse a text word by word and report it in lemma-mode shape.

    `split_source` chooses the split for a word the cascade cannot resolve:

    * "statistical" (the default): the statistical splitter's split, with the
      dictionary scanner run per unknown token, where the characters it skips
      can be seen and `ScannerPolicy` can judge the result;
    * "hybrid": the split the other modes use, in which the scanner may replace
      the whole split unseen (`xyzabc` -> `ab`).  Kept selectable so the two can
      be compared.
    """

    SPLIT_SOURCES = ("statistical", "hybrid")
    ATTEMPTS = 20
    _HAS_LETTER = regex.compile(r"\p{L}")

    def __init__(self, *, session, memo=None, cached=None, debug=False,
                 split_source: str = "statistical"):
        if split_source not in self.SPLIT_SOURCES:
            raise ValueError(
                f"unknown split source {split_source!r}; expected one of {self.SPLIT_SOURCES}"
            )
        from process_sanskrit.utils.analysisCache import resolve_cache_enabled

        self.session = session
        self.memo = {} if memo is None else memo
        self.cached = resolve_cache_enabled(cached)
        self.split_source = split_source
        self.lexicon = LemmaLexicon(session)
        self.builder = LemmaBuilder(self.lexicon)
        self.cascade = WordCascade(session=session, memo=self.memo, debug=debug)
        self.inflector = Inflector(session=session, memo=self.memo, debug=debug)
        self._parts_of: Dict[str, List[LemmaPart]] = {}

    def analyze(self, text, *, input_scheme: Optional[str] = None) -> List[dict]:
        """One item per whitespace token of `text`, in order.

        A verb form the verb table files under several roots has all of them as
        lemmas (`kariṣyati`: `kṝ`, `kṛ`); the other modes report the first only.
        """
        tokens = to_iast(text, input_scheme).split()
        ## A pattern is a one-word dictionary query.  In running text the same
        ## characters are noise, and a stray `%` would match every headword.
        patterns = len(tokens) == 1
        items = []
        with every_verb_root():
            ## an avagraha is decided by the word before it and rewrites that
            ## word too (`ko 'pi` is `kaḥ api`), so it is undone over the whole text
            for token, restored in zip(tokens, restore_avagraha_by_token(tokens)):
                parts = [
                    part for word in self._words(restored, patterns) for part in self._parts(word)
                ]
                items.append({"word": token, "parts": [part.as_dict() for part in parts]})
        return items

    def _words(self, restored: str, patterns: bool) -> List[str]:
        """The words one token cleans to: none (punctuation), one, or several."""
        words = clean_iast(restored).split()
        if not patterns:
            words = [WILDCARD_CHARACTERS.sub("", word) for word in words]
        ## `tajjapaḥ` is `taj japaḥ`, `lajjā` is itself
        words = self.cascade.read_double_j(" ".join(words)).split()
        ## `-`, `+` and the pattern characters survive cleaning on their own;
        ## without a letter there is no word to report
        return [word for word in words if self._HAS_LETTER.search(word)]

    def _parts(self, word: str) -> List[LemmaPart]:
        if word not in self._parts_of:
            ## never an empty list: a word that yields nothing is still reported
            self._parts_of[word] = self._analyse(word) or [LemmaPart(word, [])]
        return self._parts_of[word]

    def _analyse(self, word: str) -> List[LemmaPart]:
        if word.endswith("*"):
            stem = word[:-1]
            headword = self.lexicon.headword(stem) if stem else None
            if headword:
                return [LemmaPart(stem, [headword])]
            ## no headword: the stem is read as a word, `jj` decision included
            return [
                part for word in self.cascade.read_double_j(stem).split() for part in self._parts(word)
            ]
        if "_" in word or "%" in word:
            return [LemmaPart(word, self.lexicon.headwords_matching(word))]
        if "-" in word or "+" in word:
            return [
                part
                for segment in re.split(r"[-+]", word)
                if segment
                for part in self._parts(segment)
            ]

        resolution = self.cascade.resolve(word)
        if resolution.resolved:
            parts = self.builder.from_resolution(resolution)
        else:
            parts = self.builder.from_tokens(self.inflector.resolve(self._split(resolution.text)))
        ## the word as typed, not `resolution.text`: the cascade's rewrite of the
        ## last letter is what hid a typed headword in the first place
        return self.builder.finished(word, parts)

    def _split(self, text: str) -> List[str]:
        """The split tokens of a word the cascade could not resolve.

        A word the splitter cannot handle -- it gives up on a token of a
        thousand characters or so -- is handed on unsplit.  It is still a word
        of the text and must come back, as an unknown part if need be, rather
        than take every other word of the call down with it.
        """
        try:
            if self.split_source == "hybrid":
                from process_sanskrit.functions.hybridSplitter import analyze_hybrid

                ## the split the other modes use, so read as they read it
                with every_verb_root(False):
                    return analyze_hybrid(text, session=self.session, _memo=self.memo).split
            from process_sanskrit.functions.sandhiSplitter import sandhi_splitter

            return sandhi_splitter(text, cached=self.cached, attempts=self.ATTEMPTS)
        except Exception:
            return [text]


__all__ = ["LemmaAnalyzer", "LemmaBuilder", "LemmaLexicon", "LemmaPart", "PieceLemmas", "ScannerPolicy"]
