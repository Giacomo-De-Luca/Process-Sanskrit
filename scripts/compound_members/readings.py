"""The ways of reading a compound headword's members that the evaluation compares.

The library reads members with `CompoundMembers` (`process_sanskrit/utils/
compoundMembers.py`): the variant `conservative` of benchmarks/compound-members.json,
built from the same classes.  This module adds what the library does not ship,
so that every option of documentation/compound-members-brief.md can be measured:

* `MwReading`        Monier-Williams' `—` by majority vote (the rule before
                     2026-11-22, `baseline`), optionally reading a `-` between two
                     words (`WordHyphenRule`, point 3), breaking a tie in favour
                     of the compound (point 4), or gluing affixes back (p6);
* `DictionaryFallback` Edgerton's and Grassmann's `-` also for headwords
                     Monier-Williams has (the loose reading of point 1);
* `AlwaysRestoreRepair` the merged-vowel repair that also restores a vowel
                     before a second member that is a word (`mahā` + `tman`);
* `LemmaForm`        the "in compound for" table without the library's guard;
* `AffixProfiles`    the filter profiles a configuration names;
* `MemberVariant`, `VariantFactory` a variant, and its construction from a spec.
"""

from collections import Counter
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from process_sanskrit.utils.componentMarks import ComponentMarks
from process_sanskrit.utils.compoundMembers import (
    AffixFilter,
    CompoundForms,
    HyphenMarks,
    JunctionReading,
    MemberRepair,
)
from process_sanskrit.utils.lexicalResources import (
    MEMBER_PARTICLES,
    MEMBER_SUFFIXES,
    PREFIX_LOOKUP_SPELLINGS,
    SANSKRIT_PREFIXES,
)
from process_sanskrit.utils.preverbSandhi import VOWELS, PreverbSandhi

from .lexicon import ComponentTables, Headwords

Analyses = Sequence[Tuple[str, int]]


class AffixProfiles:
    """The filter profiles of a configuration, as library `AffixFilter`s and `HyphenMarks`.

    A profile names its preverbs (`library`: `SANSKRIT_PREFIXES` and its
    spellings; `combined`: `PreverbSandhi.spellings`), its particles and its
    suffixes (a list, or `library` for `MEMBER_PARTICLES` / `MEMBER_SUFFIXES`),
    whether `HyphenMarks` is `strict`, and whether a run of preverbs that is a
    word is a member (`word_runs`: `samādhi`, `vipra`; `AffixFilter.runs`).
    """

    def __init__(self, profiles: Mapping[str, Mapping], is_word: Optional[Callable[[str], bool]] = None):
        self.profiles = profiles
        self.is_word = is_word
        self._filters: Dict[str, AffixFilter] = {}

    @staticmethod
    def library_preverbs() -> frozenset:
        return frozenset(SANSKRIT_PREFIXES) | frozenset(PREFIX_LOOKUP_SPELLINGS)

    def affixes(self, name: str) -> AffixFilter:
        if name not in self._filters:
            profile = self.profiles[name]
            preverbs = {"library": self.library_preverbs, "combined": PreverbSandhi.spellings}[profile["preverbs"]]()
            particles = MEMBER_PARTICLES if profile.get("particles") == "library" else frozenset(profile.get("particles", ()))
            suffixes = MEMBER_SUFFIXES if profile.get("suffixes") == "library" else frozenset(profile.get("suffixes", ()))
            runs, is_word = (), None
            if profile.get("word_runs"):
                if self.is_word is None:
                    raise ValueError(f"filter profile {name!r} has word_runs, but no is_word was given")
                runs, is_word = AffixFilter.preverb_spellings().runs & frozenset(preverbs), self.is_word
            self._filters[name] = AffixFilter(frozenset(preverbs) | particles, suffixes, runs, is_word)
        return self._filters[name]

    def hyphen_marks(self, name: str) -> HyphenMarks:
        return HyphenMarks(self.affixes(name), strict=bool(self.profiles[name].get("strict", False)))


def prevailing_with_ties(votes: Counter) -> List[str]:
    """The reading with most entries; on a tie, the compound if exactly one tied reading is one."""
    if not votes:
        return []
    top = max(votes.values())
    leaders = [reading for reading, entries in votes.items() if entries == top]
    if len(leaders) == 1:
        return list(leaders[0])
    compounds = [reading for reading in leaders if reading]
    return list(compounds[0]) if len(compounds) == 1 else []


class WordHyphenRule:
    """Which of Monier-Williams' `-` separate two words (point 3 of the brief).

    Most of them follow a preverb (`anu-cara`, `upā-dāna`) or precede a suffix
    (`bila-vat`); those are glued back by the `AffixFilter`.  A remaining `-`
    is a member boundary when the piece after it is a word, and, if
    `left_must_be_word`, the piece before it too, and both have at least
    `min_letters` letters (Monier-Williams writes the etymology of simple
    words the same way: `nṛ-pa`, `dvi-ja`).  `rewrite` turns those `-` into
    `—`, so that the library's vote reads them.
    """

    def __init__(self, affixes: AffixFilter, is_word: Callable[[str], bool], left_must_be_word: bool,
                 min_letters: int = 2):
        self.affixes = affixes
        self.is_word = is_word
        self.left_must_be_word = left_must_be_word
        self.min_letters = min_letters

    def rewrite(self, components: str) -> str:
        first, comma, rest = components.partition(ComponentMarks.ALTERNATIVES)
        members = [self._cut(member) for member in first.split(ComponentMarks.MEMBER)]
        return ComponentMarks.MEMBER.join(members) + comma + rest

    def _cut(self, member: str) -> str:
        if ComponentMarks.AFFIX not in member:
            return member
        pieces = [piece for piece in (ComponentMarks.letters(raw) for raw in member.split(ComponentMarks.AFFIX)) if piece]
        if not pieces:
            return member
        pieces = self.affixes.merge(pieces)
        words = [pieces[0]]
        for piece in pieces[1:]:
            left_ok = self.is_word(words[-1]) or not self.left_must_be_word
            long_enough = min(len(words[-1]), len(piece)) >= self.min_letters
            if left_ok and long_enough and self.is_word(piece):
                words.append(piece)
            else:
                words[-1] += piece
        return ComponentMarks.MEMBER.join(words)


class MwReading:
    """Monier-Williams' members by majority vote: the rule before 2026-11-22 when no option is set.

    With `affixes`, the prevailing members are passed through the filter: a
    prefix is glued to the member after it, a suffix to the one before, and
    fewer than two members left means none.
    """

    def __init__(self, word_hyphens: Optional[WordHyphenRule] = None, ties: bool = False,
                 affixes: Optional[AffixFilter] = None):
        self.word_hyphens = word_hyphens
        self.ties = ties
        self.affixes = affixes

    def members(self, analyses: Analyses) -> List[str]:
        analyses = self._rewritten(analyses)
        if self.ties:
            members = prevailing_with_ties(ComponentMarks.votes(analyses))
        else:
            members = ComponentMarks.prevailing_members(analyses)
        if members and self.affixes is not None:
            members = self.affixes.merge(members)
            members = members if len(members) > 1 else []
        return members

    def elisions(self, analyses: Analyses, members: Sequence[str]) -> frozenset:
        """`ComponentMarks.elisions` for `members` as this reading gives them (none where gluing affixes changed them)."""
        return ComponentMarks.elisions(self._rewritten(analyses), list(members))

    def _rewritten(self, analyses: Analyses) -> Analyses:
        if self.word_hyphens is None:
            return analyses
        return [(self.word_hyphens.rewrite(components), entries) for components, entries in analyses]


class DictionaryFallback:
    """Edgerton's, then Grassmann's members for a headword Monier-Williams cannot answer for.

    `when` is `mw_absent` (Monier-Williams has no entry for the headword: what
    the library does) or `mw_no_members` (its vote gives none: the loose
    reading, under which a Grassmann `su-kha/` decides `sukha`).
    """

    WHEN = ("mw_absent", "mw_no_members")

    def __init__(self, tables: ComponentTables, headwords: Headwords, dictionaries: Sequence[str], when: str,
                 hyphens: HyphenMarks):
        if when not in self.WHEN:
            raise ValueError(f"unknown fallback condition {when!r}; expected one of {self.WHEN}")
        self.tables = tables
        self.headwords = headwords
        self.dictionaries = tuple(dictionaries)
        self.when = when
        self.hyphens = hyphens

    def applies(self, headword: str) -> bool:
        return self.when == "mw_no_members" or not self.headwords.in_monier_williams(headword)

    def members(self, headword: str) -> List[str]:
        for dictionary in self.dictionaries:
            members = ComponentMarks.prevailing_members(
                self.tables.analyses(dictionary, headword),
                reader=lambda components: self.hyphens.members(components, headword),
            )
            if members:
                return members
        return []


class AlwaysRestoreRepair(MemberRepair):
    """The library's repair, but also for a vowel-final member before a consonant.

    A vowel is put back before the second member even when it is a word, and a
    reading that does so is preferred: `mahā` + `tman` -> `mahā` + `ātman`, but
    also `mahā` + `kāla` -> `mahā` + `akāla`.
    """

    def suspect(self, left: str, right: str) -> bool:
        return super().suspect(left, right) or (left[-1] in VOWELS and right[0] not in VOWELS)

    def admissible(self, left: str, right: str, reading: JunctionReading) -> bool:
        if super().admissible(left, right, reading):
            return True
        return (self._restores_vowel(right, reading) and reading.left == left
                and self.is_word(reading.left) and self.is_word(reading.right))

    def rank(self, left: str, right: str, reading: JunctionReading) -> tuple:
        return (not self._restores_vowel(right, reading),) + super().rank(left, right, reading)

    @staticmethod
    def _restores_vowel(right: str, reading: JunctionReading) -> bool:
        return reading.distance == 0 and reading.right[0] in VOWELS and right[0] not in VOWELS


class LemmaForm:
    """Each member in citation form (point 2 of the brief).

    `xref_guarded` is the library's rule (`CompoundForms.citation`); `xref`
    follows the table without asking lemma mode (`ātma` -> `ātman`).
    """

    MODES = ("xref", "xref_guarded")

    def __init__(self, mode: str, forms: CompoundForms, piece_lemmas: Callable[[str], Iterable[str]]):
        if mode not in self.MODES:
            raise ValueError(f"unknown lemma-form mode {mode!r}; expected one of {self.MODES}")
        self.mode = mode
        self.forms = forms
        self.piece_lemmas = piece_lemmas

    def normalize(self, members: Sequence[str]) -> List[str]:
        if self.mode == "xref":
            return [self.forms.target(member) or member for member in members]
        return [self.forms.citation(member, self.piece_lemmas) for member in members]


class MemberVariant:
    """One way of answering "what are the members of this headword?"."""

    def __init__(self, name: str, tables: ComponentTables, mw_reading: MwReading,
                 fallback: Optional[DictionaryFallback] = None,
                 repair: Optional[MemberRepair] = None,
                 lemma_form: Optional[LemmaForm] = None, description: str = ""):
        self.name = name
        self.description = description
        self.tables = tables
        self.mw_reading = mw_reading
        self.fallback = fallback
        self.repair = repair
        self.lemma_form = lemma_form
        self._memo: Dict[str, Tuple[str, ...]] = {}

    def members(self, headword: str) -> List[str]:
        if headword not in self._memo:
            self._memo[headword] = tuple(self._members(headword))
        return list(self._memo[headword])

    def _members(self, headword: str) -> List[str]:
        analyses = self.tables.analyses("mw", headword)
        members = self.mw_reading.members(analyses)
        elided = self.mw_reading.elisions(analyses, members)
        if not members and self.fallback is not None and self.fallback.applies(headword):
            members, elided = self.fallback.members(headword), frozenset()
        if members and self.repair is not None:
            members = self.repair.repair(members, elided)
        if members and self.lemma_form is not None:
            members = self.lemma_form.normalize(members)
        return members


class VariantFactory:
    """Builds the variants a configuration names, from resources shared between them.

    A variant spec has a `name` and any of: `fallback` (`dictionaries`, `when`,
    `filters`), `mw_hyphens` (`left`: `word` or `any`, `filters`, `min_letters`),
    `ties`, `mw_affixes` (a filter profile), `merged_vowel` (`junction`, the
    library's, or `always`) with `unrepaired` (`keep`, the library's, or
    `drop`), and `lemma_form` (`xref_guarded`, the library's, or `xref`).
    """

    KEYS = frozenset({"name", "description", "fallback", "mw_hyphens", "ties", "mw_affixes",
                      "merged_vowel", "unrepaired", "lemma_form"})
    REPAIRS = {"junction": MemberRepair, "always": AlwaysRestoreRepair}

    def __init__(self, tables: ComponentTables, headwords: Headwords, forms: CompoundForms,
                 filter_profiles: Mapping[str, Mapping], piece_lemmas: Callable[[str], Iterable[str]]):
        self.tables = tables
        self.headwords = headwords
        self.forms = forms
        self.profiles = AffixProfiles(filter_profiles, is_word=headwords.sanskrit)
        self.piece_lemmas = piece_lemmas

    def is_word(self, piece: str) -> bool:
        """A Sanskrit headword, or a compound-initial spelling Monier-Williams names."""
        return self.headwords.sanskrit(piece) or self.forms.target(piece) is not None

    def build(self, spec: Mapping) -> MemberVariant:
        unknown = set(spec) - self.KEYS
        if unknown:
            raise ValueError(f"variant {spec.get('name')!r} has unknown keys {sorted(unknown)}")
        word_hyphens = None
        if spec.get("mw_hyphens"):
            hyphens = {} if spec["mw_hyphens"] is True else spec["mw_hyphens"]
            word_hyphens = WordHyphenRule(
                self.profiles.affixes(hyphens.get("filters", "extended")), self.is_word,
                left_must_be_word=hyphens.get("left", "word") == "word",
                min_letters=hyphens.get("min_letters", 2),
            )
        fallback = None
        if spec.get("fallback"):
            options = spec["fallback"]
            fallback = DictionaryFallback(self.tables, self.headwords, options["dictionaries"], options["when"],
                                          self.profiles.hyphen_marks(options["filters"]))
        repair = None
        if spec.get("merged_vowel"):
            if spec["merged_vowel"] not in self.REPAIRS:
                raise ValueError(f"unknown merged-vowel mode {spec['merged_vowel']!r}")
            repair = self.REPAIRS[spec["merged_vowel"]](self.headwords.sanskrit, self.headwords.breadth,
                                                        unrepaired=spec.get("unrepaired", "keep"))
        elif "unrepaired" in spec:
            raise ValueError(f"variant {spec.get('name')!r} sets unrepaired without merged_vowel")
        lemma_form = LemmaForm(spec["lemma_form"], self.forms, self.piece_lemmas) if spec.get("lemma_form") else None
        affixes = self.profiles.affixes(spec["mw_affixes"]) if spec.get("mw_affixes") else None
        return MemberVariant(
            spec["name"], self.tables, MwReading(word_hyphens, ties=bool(spec.get("ties")), affixes=affixes),
            fallback=fallback, repair=repair, lemma_form=lemma_form, description=spec.get("description", ""),
        )
