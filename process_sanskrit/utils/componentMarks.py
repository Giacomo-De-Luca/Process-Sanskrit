"""Reading the `components` field of a dictionary headword.

Monier-Williams writes three marks into it, and they mean three things:

    citta—vṛtti     `—`  boundary between the members of a compound
    ni-rodha        `-`  prefix or suffix boundary inside one word
    u/pa, agni/     `/`  accent, written after the accented vowel
    akṣitavya^      `^`  accent (svarita)

and `,` separates alternative accentuations of one headword.  Which marks count
as a cut depends on the question, so both questions are answered here and the
marks are named once.  See documentation/dictionary-results.md.
"""

import re
from collections import Counter
from typing import Iterable, List, Optional, Tuple

import regex


class ComponentMarks:
    """The cuts a `components` string marks, with the accent marks removed."""

    MEMBER = "—"
    AFFIX = "-"
    ALTERNATIVES = ","

    ## every boundary the dictionary marks: compound members, prefix and suffix
    ## boundaries, and the comma between alternative accentuations.  The accent
    ## marks are deliberately absent: cutting on them returned `upa` as u + pa.
    _ANY_BOUNDARY = re.compile("[" + re.escape(AFFIX + MEMBER + ALTERNATIVES) + "]")
    _NOT_A_LETTER = regex.compile(r"[^\p{L}]")

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
    def prevailing_members(cls, analyses: Iterable[Tuple[str, int]]) -> List[str]:
        """The members of the reading most of a headword's entries give; [] on a tie.

        `analyses` pairs each `components` string recorded for one headword with
        the number of dictionary entries that record it.  Each entry votes for
        the members its string marks, and a string that marks none -- the plain
        word, or a prefix reading (`a-yoga`) -- votes for "no compound".  A rare
        homograph therefore cannot lend its cut to a common word: `sukha` is
        one `su—kha` entry against ten plain ones.  Accent variants vote
        together (`su—kṛta`, `su—kṛta/`).
        """
        votes: Counter = Counter()
        for components, entries in analyses:
            votes[tuple(cls.members(components))] += entries
        ranked = votes.most_common(2)
        if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
            return []
        return list(ranked[0][0])


__all__ = ["ComponentMarks"]
