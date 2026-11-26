"""Reading the `components` field of a dictionary headword.

Monier-Williams writes these marks into it:

    citta—vṛtti     `—`  boundary between the members of a compound
    ni-rodha        `-`  prefix or suffix boundary inside one word
    u/pa, agni/     `/`  accent, written after the accented vowel
    akṣitavya^      `^`  accent (svarita)
    mano—'nuga      `'`  avagraha: the member after it lost its initial `a`

and `,` separates alternative accentuations of one headword.  Which marks count
as a cut depends on the question, so both questions are answered here and the
marks are named once.  See documentation/dictionary-results.md.
"""

import re
import unicodedata
from collections import Counter
from typing import Callable, FrozenSet, Iterable, List, Optional, Tuple

import regex


class ComponentMarks:
    """The cuts a `components` string marks, with the accent marks removed."""

    MEMBER = "—"
    AFFIX = "-"
    ALTERNATIVES = ","
    ## an avagraha: the member after it lost its initial `a` (`mano—'nuga`)
    ELISION = "'"

    ## every boundary the dictionary marks: compound members, prefix and suffix
    ## boundaries, and the comma between alternative accentuations.  The accent
    ## marks are deliberately absent: cutting on them returned `upa` as u + pa.
    _ANY_BOUNDARY = re.compile("[" + re.escape(AFFIX + MEMBER + ALTERNATIVES) + "]")
    _NOT_A_LETTER = regex.compile(r"[^\p{L}]")

    @classmethod
    def letters(cls, text: str) -> str:
        """The letters of `text`, composed (NFC): no accents, marks, spaces or other non-letters."""
        return cls._letters(unicodedata.normalize("NFC", text))

    ## `parts` and `members` do not compose first: 70 `components` of ddsa are
    ## decomposed, and composing them would change what `mode="parts"` returns
    ## (`a-kudhryãk`).  Every other dictionary's `components` are composed already.
    @classmethod
    def _letters(cls, piece: str) -> str:
        return cls._NOT_A_LETTER.sub("", piece)

    @classmethod
    def parts(cls, components: str) -> List[str]:
        """Every marked piece, in order, without repeats (`mode="parts"`)."""
        pieces = [cls._letters(piece) for piece in cls._ANY_BOUNDARY.split(components) if piece]
        return list(dict.fromkeys(pieces))

    @classmethod
    def members(cls, components: Optional[str]) -> List[str]:
        """The members of a compound headword; empty when it is not marked as one.

        Only `—` separates members.  A prefix boundary inside a member is not a
        cut (`a-dharma—kṣetra` is `adharma` + `kṣetra`), and of alternative
        accentuations the first is read.
        """
        if not components or cls.MEMBER not in components:
            return []
        first = components.split(cls.ALTERNATIVES)[0]
        members = [cls._letters(member) for member in first.split(cls.MEMBER)]
        members = [member for member in members if member]
        return members if len(members) > 1 else []

    @classmethod
    def votes(cls, analyses: Iterable[Tuple[str, int]],
              reader: Optional[Callable[[str], List[str]]] = None) -> Counter:
        """How many entries vote for each reading: members tuple -> entries.

        `reader` turns one `components` string into members; Monier-Williams'
        `members` unless another dictionary's marks are read (`HyphenMarks`).
        """
        reader = reader or cls.members
        votes: Counter = Counter()
        for components, entries in analyses:
            votes[tuple(reader(components))] += entries
        return votes

    @classmethod
    def prevailing_members(cls, analyses: Iterable[Tuple[str, int]],
                           reader: Optional[Callable[[str], List[str]]] = None) -> List[str]:
        """The members of the reading most of a headword's entries give; [] on a tie.

        `analyses` pairs each `components` string recorded for one headword with
        the number of dictionary entries that record it.  Each entry votes for
        the members its string marks, and a string that marks none -- the plain
        word, or a prefix reading (`a-yoga`) -- votes for "no compound".  A rare
        homograph therefore cannot lend its cut to a common word: `sukha` is
        one `su—kha` entry against ten plain ones.  Accent variants vote
        together (`su—kṛta`, `su—kṛta/`).
        """
        ranked = cls.votes(analyses, reader).most_common(2)
        if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
            return []
        return list(ranked[0][0])

    @classmethod
    def elisions(cls, analyses: Iterable[Tuple[str, int]], members: List[str]) -> FrozenSet[int]:
        """The indexes of `members` written after an avagraha (`mano—'nuga`: 1).

        Monier-Williams writes `—'` where an `a` was elided after `o` or `e`,
        which `members` drops with the other non-letters.  Read from the
        analyses that give exactly `members`; one that marks it is enough.
        """
        found = set()
        for components, _ in analyses:
            if not members or cls.members(components) != members:
                continue
            first = components.split(cls.ALTERNATIVES)[0]
            pieces = [piece for piece in first.split(cls.MEMBER) if cls._letters(piece)]
            found.update(index for index, piece in enumerate(pieces) if piece.lstrip().startswith(cls.ELISION))
        return frozenset(found)


__all__ = ["ComponentMarks"]
