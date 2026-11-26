"""The members of a compound headword, for lemma mode (rule 9 of documentation/lemma-mode.md).

`CompoundMembers.members(headword)` reads them in four steps:

1. Monier-Williams' `—`, the reading most of the headword's entries give
   (`ComponentMarks.prevailing_members`);
2. for a headword Monier-Williams lacks, Edgerton's and then Grassmann's `-`,
   read by `HyphenMarks`: a stub, a prefix or particle, or a suffix is no member;
3. a pair with a side that is no Sanskrit headword is read again where sandhi
   joined it (`MemberRepair`, `JunctionSandhi`: `nānā` + `rtha` -> `nānā` +
   `artha`); a pair no such reading makes words of keeps the dictionary's cut;
4. each member in its citation form where Monier-Williams says its spelling is
   "in compound for" another word and lemma mode gives that word for the piece
   the splitter would hand over (`CompoundForms.citation`: `taj` -> `tad`).

These four, and the alternatives that were measured and rejected (the other
dictionaries also for words Monier-Williams has, Monier-Williams' `-`, ties,
restoring a vowel whenever possible, dropping a pair that cannot be read
again), are in documentation/compound-members-evaluation.md.

The classes here know no lemma mode: what they need from it -- the components
of a headword, whether a word is a headword, the lemmas of a piece -- is handed
in, so that the evaluation can run the same code on bulk-read tables.  They read
only the database, never the splitter's sandhi rules.
"""

import re
from functools import lru_cache
from typing import Callable, Collection, Dict, FrozenSet, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

from sqlalchemy import text

from process_sanskrit.utils.componentMarks import ComponentMarks
from process_sanskrit.utils.engineCache import EngineCache
from process_sanskrit.utils.lexicalResources import MEMBER_PARTICLES, MEMBER_SUFFIXES, pausa_readings
from process_sanskrit.utils.preverbSandhi import VOWELS, PreverbSandhi

Analyses = List[Tuple[str, int]]


class PreverbSpellings(NamedTuple):
    """The prefixes `AffixFilter.default` glues, and which of them are runs of preverbs."""

    ## preverbs and their runs in sandhi spelling, and the particles
    prefixes: FrozenSet[str]
    ## the runs of two or more preverbs (`samā`, `samādhi`, `vipra`)
    runs: FrozenSet[str]


class AffixFilter:
    """Which pieces of a hyphenated analysis are a prefix or a suffix, not a member.

    A run of preverbs can spell a word of its own: `sam` + `ā` + `adhi` is
    `samādhi`, `vi` + `pra` is `vipra`.  Given `is_word`, a piece of `runs` that
    is a word is a member, not a prefix (Edgerton's `samādhi-saṃyuktaka`).  A
    single preverb is a prefix even when it is a word (`vi`, `upa`).
    """

    def __init__(self, prefixes: Iterable[str], suffixes: Iterable[str], runs: Iterable[str] = (),
                 is_word: Optional[Callable[[str], bool]] = None):
        self.prefixes = frozenset(prefixes)
        self.suffixes = frozenset(suffixes)
        self.runs = frozenset(runs)
        self.is_word = is_word

    @classmethod
    def default(cls, is_word: Optional[Callable[[str], bool]] = None) -> "AffixFilter":
        """The filter lemma mode uses: preverbs and their runs, the particles, the suffixes."""
        spellings = cls.preverb_spellings()
        return cls(spellings.prefixes, MEMBER_SUFFIXES, spellings.runs, is_word)

    @staticmethod
    @lru_cache(maxsize=1)
    def preverb_spellings() -> PreverbSpellings:
        spellings = PreverbSandhi.spellings()
        return PreverbSpellings(spellings | MEMBER_PARTICLES, spellings - PreverbSandhi.spellings(depth=1))

    def is_prefix(self, piece: str) -> bool:
        if piece not in self.prefixes:
            return False
        return self.is_word is None or piece not in self.runs or not self.is_word(piece)

    def merge(self, pieces: Sequence[str]) -> List[str]:
        """`pieces` with each prefix glued to the piece after it and each suffix to the one before."""
        glued: List[str] = []
        carried = ""
        for index, piece in enumerate(pieces):
            if index < len(pieces) - 1 and self.is_prefix(piece):
                carried += piece
                continue
            glued.append(carried + piece)
            carried = ""
        merged: List[str] = []
        for piece in glued:
            if merged and piece in self.suffixes:
                merged[-1] += piece
            else:
                merged.append(piece)
        return merged


class HyphenMarks:
    """The members of an analysis that marks them with `-` (Edgerton, Grassmann).

    A stub (`-ālāpaka`, `nikranda-`, `°ṇā`) gives none; a bracketed piece is
    context, not part of the word (`akṣuṇṇa(-vyākaraṇa)`); prefixes and
    suffixes are glued back (`a-kalpya`, `daurmanasya-tā`).  With `strict`, an
    analysis with a one-letter piece (`tatha-r-iva`) or whose pieces do not
    spell the headword gives none either.
    """

    _BRACKETS = re.compile(r"\([^)]*\)")

    def __init__(self, affixes: Optional[AffixFilter] = None, strict: bool = True):
        self.affixes = affixes or AffixFilter.default()
        self.strict = strict

    def members(self, components: str, headword: str) -> List[str]:
        first = components.split(ComponentMarks.ALTERNATIVES)[0].strip()
        if not first or "°" in first:
            return []
        first = self._BRACKETS.sub("", first).strip()
        if first.startswith(ComponentMarks.AFFIX) or first.endswith(ComponentMarks.AFFIX):
            return []
        pieces = [ComponentMarks.letters(piece) for piece in first.split(ComponentMarks.AFFIX)]
        if not all(pieces):
            return []
        pieces = self.affixes.merge(pieces)
        if self.strict:
            spelled = ComponentMarks.letters(headword)
            if "".join(pieces) != spelled or any(len(piece) < 2 for piece in pieces):
                return []
        return pieces if len(pieces) > 1 else []


class JunctionReading(NamedTuple):
    """Two members read again at their junction (`JunctionSandhi.readings`)."""

    left: str
    right: str
    ## letters between the mark and the junction when the mark stands early, else 0
    distance: int
    ## how many long vowels the reading gives the members at the junction
    long_vowels: int
    ## it puts back the `a` an avagraha marks as elided
    elision: bool = False


class JunctionSandhi:
    """The readings of two members that undo the sandhi where they meet.

    Monier-Williams marks the cut in the joined spelling, so where sandhi
    merged or changed sounds a piece is no word: `maha—rṣi` for `ṛṣi`,
    `nānā—rtha` for `artha`.  `readings` undoes what sandhi does at the
    junction, and nothing else:

    * a merged vowel (`MERGES`) at the mark (`nānā—rtha`, `maha—rṣi`), or up
      to `WINDOW` letters after it, for a mark that stands a syllable early
      (`kā—mārtha` for `kāma` + `artha`); never before it, so the `ya` of
      `sūrya—prabha` is not read as `ī` + `a`;
    * a semivowel before the mark, for its vowel (`madhv—āsava`,
      `abhy—upapādana`);
    * an elided `a`, only where an avagraha marks it (`mano—'nuga` for
      `manas` + `anuga`, `uru—'bja` for `uru` + `abja`; `namo—vṛdha` has none);
    * an initial `ṣ` of the right member read as `s`, and a `ṇ` in it as `n`,
      as a member before may have made them retroflex (`tri—ṣṭubh`,
      `tri—ṇāman`); and `ch` written `cch` after a vowel (`gāyatra—cchandas`),
      or for `ś` after a stop (`bṛhac—chravas`, `dik—chabda`).

    It knows no words; `MemberRepair` picks the reading whose sides are words.
    """

    ## a merged sound -> (the final of the left member, the initial of the right)
    MERGES = {
        "ā": (("a", "a"), ("a", "ā"), ("ā", "a"), ("ā", "ā")),
        "e": (("a", "i"), ("a", "ī"), ("ā", "i"), ("ā", "ī")),
        "o": (("a", "u"), ("a", "ū"), ("ā", "u"), ("ā", "ū")),
        "ai": (("a", "e"), ("a", "ai"), ("ā", "e"), ("ā", "ai")),
        "au": (("a", "o"), ("a", "au"), ("ā", "o"), ("ā", "au")),
        "ī": (("i", "i"), ("i", "ī"), ("ī", "i"), ("ī", "ī")),
        "ū": (("u", "u"), ("u", "ū"), ("ū", "u"), ("ū", "ū")),
        "ṝ": (("ṛ", "ṛ"), ("ṛ", "ṝ")),
        "ar": (("a", "ṛ"), ("ā", "ṛ")),
        "ār": (("a", "ṛ"), ("ā", "ṛ")),
        "al": (("a", "ḷ"), ("ā", "ḷ")),
    }
    ## a semivowel before a vowel -> the vowel it stands for
    SEMIVOWELS = {"y": ("i", "ī"), "v": ("u", "ū"), "r": ("ṛ",)}
    LONG = frozenset({"ā", "ī", "ū", "ṝ", "e", "o", "ai", "au"})
    ## letters after the mark that a merged vowel may stand at, when the mark stands early
    WINDOW = 2
    ## an s the member before made retroflex, and what it was
    RETROFLEX_S = (("ṣṭh", "sth"), ("ṣṭ", "st"), ("ṣ", "s"))
    ## the stops before which `ś` is written `ch`
    STOPS_BEFORE_CH = frozenset("kcṭp")

    def readings(self, left: str, right: str, elided: bool = False) -> Iterator[JunctionReading]:
        """The readings of `left` + `right`, in no order; `elided`: an avagraha precedes `right`."""
        if not left or not right:
            return
        for reading in self._vowels(left, right, elided):
            yield reading
            for spelled in self._unretroflex(reading.right):
                yield reading._replace(right=spelled)
        for spelled in self._unretroflex(right):
            yield JunctionReading(left, spelled, 0, 0)
        yield from self._consonants(left, right)

    def _vowels(self, left: str, right: str, elided: bool) -> Iterator[JunctionReading]:
        whole, mark = left + right, len(left)
        for merged, pairs in self.MERGES.items():
            start = whole.find(merged, max(1, mark - len(merged)))
            while start != -1 and start <= mark + self.WINDOW:
                end = start + len(merged)
                if mark <= end < len(whole):
                    for final, initial in pairs:
                        yield JunctionReading(whole[:start] + final, initial + whole[end:], max(0, start - mark),
                                              (final in self.LONG) + (initial in self.LONG))
                start = whole.find(merged, start + 1)
        if len(left) > 1 and left[-1] in self.SEMIVOWELS and left[-2] not in VOWELS and right[0] in VOWELS:
            for final in self.SEMIVOWELS[left[-1]]:
                yield JunctionReading(left[:-1] + final, right, 0, int(final in self.LONG))
        if elided:
            yield JunctionReading(left, "a" + right, 0, 0, elision=True)
            if left[-1] == "o":
                yield JunctionReading(left[:-1] + "as", "a" + right, 0, 0, elision=True)

    def _unretroflex(self, right: str) -> Set[str]:
        """`ṣṭhira` -> `sthira`, `ṇāman` -> `nāman`, `ṣavaṇa` -> `savana`."""
        spelled = set()
        for retroflex, dental in self.RETROFLEX_S:
            if right.startswith(retroflex):
                spelled.add(dental + right[len(retroflex):])
                break
        for word in spelled | {right}:
            if "ṇ" in word:
                spelled |= {word.replace("ṇ", "n", 1), word.replace("ṇ", "n")}
        spelled.discard(right)
        return spelled

    def _consonants(self, left: str, right: str) -> Iterator[JunctionReading]:
        if right.startswith("cch") and left[-1] in VOWELS:
            yield JunctionReading(left, right[1:], 0, 0)
        elif right.startswith("ch") and len(right) > 2 and left[-1] in self.STOPS_BEFORE_CH:
            yield JunctionReading(left, "ś" + right[2:], 0, 0)
            if left[-1] == "c":
                yield JunctionReading(left[:-1] + "t", "ś" + right[2:], 0, 0)


class MemberRepair:
    """Re-cuts a pair of members where sandhi hides their boundary (known-errors.md, section 2).

    A pair with a side that is no Sanskrit headword is read again at its
    junction (`JunctionSandhi`), and a reading whose two sides are words
    replaces it, under three rules measured against DCS:

    * a side that is a word keeps its spelling: `sura` + `rṣi` becomes `sura`
      + `ṛṣi`, not `surā` + `ṛṣi`, and no vowel is put before a right side that
      is a word (`pattrā` + `kṛta` is not `pattra` + `akṛta`).  Only a mark
      that stands early changes a left side that is a word: the cut moves
      right, and the left side takes letters of the right (`kā` + `mārtha` ->
      `kāma` + `artha`);
    * a reading that moves the cut past the mark needs members that at least
      `EARLY_MARK_BREADTH` dictionaries list, the left and the right one: the
      rare words a misplaced cut can spell are mostly wrong (`gandha` +
      `kelikā` is not `gandhaka` + `ilikā`);
    * of the rest, the nearest to the mark wins, then the one with fewer long
      vowels (`daśa` + `avasthā`, not `daśā`).

    An avagraha is the dictionary's own mark of an elided `a`, so a pair
    written after one is read again even when both its sides are words, and
    its `a` goes back before a right side that is a word: `bhūyo—'kṣara` is
    `bhūyo` + `akṣara`, not `kṣara` "perishable".

    A pair with a side that is no word, and that no reading makes words of,
    keeps the dictionary's cut (`vi` + `tṛṣṇa`); with `unrepaired="drop"` the
    headword gets no members instead.  Any other pair of words is left alone,
    so `mahā` + `tman` stays.
    """

    ## dictionaries that must list the left and the right member of a reading
    ## that moves the cut past the mark
    EARLY_MARK_BREADTH = (2, 3)
    ## what becomes of a pair no reading makes words of
    UNREPAIRED = ("keep", "drop")

    def __init__(self, is_word: Callable[[str], bool], breadth: Callable[[str], int],
                 junctions: Optional[JunctionSandhi] = None, unrepaired: str = "keep"):
        if unrepaired not in self.UNREPAIRED:
            raise ValueError(f"unknown policy {unrepaired!r} for an unrepaired pair; expected one of {self.UNREPAIRED}")
        self.is_word = is_word
        self.breadth = breadth
        self.junctions = junctions or JunctionSandhi()
        self.unrepaired = unrepaired

    def repair(self, members: Sequence[str], elided: Collection[int] = ()) -> List[str]:
        """`members` with each suspect pair read again; `elided`: the indexes an avagraha marks."""
        repaired = list(members)
        for index in range(len(repaired) - 1):
            left, right = repaired[index], repaired[index + 1]
            suspect, marked = self.suspect(left, right), index + 1 in elided
            if not (suspect or marked):
                continue
            better = self.recut(left, right, elided=marked)
            if better is not None:
                repaired[index], repaired[index + 1] = better
            elif suspect and self.unrepaired == "drop":
                return []
        return repaired

    def suspect(self, left: str, right: str) -> bool:
        return not (self.is_word(left) and self.is_word(right))

    def recut(self, left: str, right: str, elided: bool = False) -> Optional[Tuple[str, str]]:
        """The best reading of `left` + `right` whose sides are words, or None."""
        candidates = [reading for reading in self.junctions.readings(left, right, elided)
                      if self.admissible(left, right, reading)]
        if not candidates:
            return None
        best = min(candidates, key=lambda reading: self.rank(left, right, reading))
        return best.left, best.right

    def admissible(self, left: str, right: str, reading: JunctionReading) -> bool:
        if reading.right != right and self.is_word(right) and not reading.elision:
            return False
        if reading.distance == 0:
            if reading.left != left and self.is_word(left):
                return False
        else:
            least_left, least_right = self.EARLY_MARK_BREADTH
            if self.breadth(reading.left) < least_left or self.breadth(reading.right) < least_right:
                return False
        return self.is_word(reading.left) and self.is_word(reading.right)

    def rank(self, left: str, right: str, reading: JunctionReading) -> tuple:
        """The sort key of a reading: one an avagraha marks, then nearest to the mark, then fewer long vowels."""
        return (not reading.elision, reading.distance, reading.long_vowels, reading.left, reading.right)


class CompoundForms:
    """Monier-Williams' entries for a spelling used inside compounds.

    `taj` is "in compound for tad", `mano` "in compound for manas", `ṛg` "by
    Sandhi for ṛc"; a `°` in the target stands for the start of the stub
    (`nabhaḥ`, "for °bhas" -> `nabhas`).  Homographs share a headword: `para`
    has 31 entries, one "for °ras", and `mahā` is also "a cow".  A stub is
    followed only when it has more such entries than entries with a sense of
    its own (pointers such as "See p. 225" count as neither), so `mano` and
    `ṛg` are followed and `para`, `pura`, `mahā`, `dharma` are not.

    The table is read once per database engine, like `CompoundVerbIndex`'s.
    """

    _PATTERN = re.compile(
        r"compound or Comparative degree</ab> for (?:<hom>[^<]*</hom> )?<s>(.*?)</s>"
        r"|by <s1 slp1=\"saMDi\">Sandhi</s1> for (?:<hom>[^<]*</hom> )?<s>(.*?)</s>"
    )
    _POINTER = re.compile(r"See <pcol>|for words beginning thus|<ab>sub voce")
    _TAG = re.compile(r"<[^>]*>")
    _QUERY = text(
        "SELECT keys_iast, cleaned_body FROM mw WHERE keys_iast IN ("
        "SELECT keys_iast FROM mw WHERE cleaned_body LIKE :compound OR cleaned_body LIKE :sandhi"
        ") ORDER BY keys_iast, lnum"
    )

    _tables = EngineCache()

    @classmethod
    def reset(cls) -> None:
        """Forget every table read; the next lookup reads its database again."""
        cls._tables.clear()

    def __init__(self, session):
        self.session = session

    def target(self, form: str) -> Optional[str]:
        """The word `form` stands for, if `form` is mostly a compound-form entry."""
        return self._table().get(form)

    def stubs(self) -> List[str]:
        return sorted(self._table())

    def citation(self, member: str, piece_lemmas: Callable[[str], Iterable[str]]) -> str:
        """`member` in citation form, if lemma mode gives that form for the split piece.

        The splitter hands the piece over in pausa (`mano` as `manaḥ`), so the
        target must be a lemma of one of `pausa_readings(member)`: `manaḥ` gives
        `manas`, but a split `ātma` is indexed as `ātma`, never `ātman`, and so
        keeps its spelling.
        """
        target = self.target(member)
        if target is None:
            return member
        if any(target in piece_lemmas(reading) for reading in pausa_readings(member)):
            return target
        return member

    @staticmethod
    def expand(stub: str, target: str) -> Optional[str]:
        """The target written out: `°bhas` for `nabhaḥ` is `nabhas`; None if it cannot be."""
        target = target.strip()
        if not target.startswith("°"):
            return ComponentMarks.letters(target) or None
        tail = ComponentMarks.letters(target[1:])
        start = stub.rfind(tail[0]) if tail else -1
        return stub[:start] + tail if start > 0 else None

    def _table(self) -> Dict[str, str]:
        return self._tables.table(self.session, self._read)

    def _read(self) -> Dict[str, str]:
        targets: Dict[str, str] = {}
        counts: Dict[str, List[int]] = {}
        rows = self.session.execute(self._QUERY, {
            "compound": "%compound or Comparative degree</ab> for%", "sandhi": "%Sandhi</s1> for%",
        })
        for stub, body in rows:
            counts.setdefault(stub, [0, 0])
            found = self._PATTERN.search(body or "")
            if found:
                counts[stub][0] += 1
                target = self.expand(stub, self._TAG.sub("", found.group(1) or found.group(2)))
                if target and target != stub and stub not in targets:
                    targets[stub] = target
            elif not self._POINTER.search((body or "")[:200]):
                counts[stub][1] += 1
        return {stub: target for stub, target in targets.items() if counts[stub][0] > counts[stub][1]}


class CompoundMembers:
    """The members lemma mode reports for a compound headword.

    Everything it reads is handed in:

    * `components(dictionary, headword)` -> `[(components, entries), ...]`,
      what `headword_components` gives;
    * `is_word(word)`: a Sanskrit headword (`LemmaLexicon.has_sanskrit_headword`);
    * `breadth(word)`: how many dictionaries list `word`;
    * `in_monier_williams(word)`: whether Monier-Williams has the headword at all;
    * `piece_lemmas(word)`: the lemmas lemma mode gives `word` on its own.
    """

    FALLBACK_DICTIONARIES = ("bhs", "gra")

    def __init__(self, forms: CompoundForms, *, components: Callable[[str, str], Analyses],
                 is_word: Callable[[str], bool], breadth: Callable[[str], int],
                 in_monier_williams: Callable[[str], bool], piece_lemmas: Callable[[str], Iterable[str]],
                 hyphens: Optional[HyphenMarks] = None, repair: Optional[MemberRepair] = None):
        self.forms = forms
        self.components = components
        self.in_monier_williams = in_monier_williams
        self.piece_lemmas = piece_lemmas
        self.hyphens = hyphens or HyphenMarks(AffixFilter.default(is_word))
        self.repair = repair or MemberRepair(is_word, breadth)

    def members(self, headword: str) -> List[str]:
        analyses = self.components("mw", headword)
        members = ComponentMarks.prevailing_members(analyses)
        elided = ComponentMarks.elisions(analyses, members)
        if not members and not self.in_monier_williams(headword):
            members, elided = self.fallback(headword), frozenset()
        if members:
            members = self.repair.repair(members, elided)
        return [self.forms.citation(member, self.piece_lemmas) for member in members]

    def fallback(self, headword: str) -> List[str]:
        """Edgerton's, then Grassmann's members, by the reading most of their entries give."""
        for dictionary in self.FALLBACK_DICTIONARIES:
            members = ComponentMarks.prevailing_members(
                self.components(dictionary, headword),
                reader=lambda components: self.hyphens.members(components, headword),
            )
            if members:
                return members
        return []


__all__ = ["AffixFilter", "CompoundForms", "CompoundMembers", "HyphenMarks", "JunctionReading", "JunctionSandhi",
           "MemberRepair", "PreverbSpellings"]
