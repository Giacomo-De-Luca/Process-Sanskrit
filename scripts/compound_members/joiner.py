"""Joining DCS segments by sandhi, with the vendored splitter's rules.

The evaluation needs to know which spellings a run of DCS segments has when
joined (`tad` + `jātīya` -> `tajjātīya`), to find a headword's split
occurrences and to judge members against them.  The vendored Python splitter
(`process_sanskrit/splitter/sandhi.py`) has a full table of joining rules;
`SandhiJoiner` puts it behind IAST.

Only the evaluation uses this: the library's members read the database alone
(`process_sanskrit/utils/compoundMembers.py`).  Loading the rules costs about
200 MB, so there is one joiner per process, and its caches are bounded
(`CACHE_SIZE` entries each).
"""

import threading
import unicodedata
from functools import lru_cache
from typing import FrozenSet, Optional, Sequence

from indic_transliteration import sanscript


## entries each cache of the joiner keeps: enough for the forms of a DCS run,
## small enough not to grow by hundreds of megabytes
CACHE_SIZE = 1 << 17


class SandhiJoiner:
    """The vendored splitter's joining rules, in IAST."""

    MAX_JOINS = 8
    _instance: Optional["SandhiJoiner"] = None
    _lock = threading.Lock()

    @classmethod
    def shared(cls) -> "SandhiJoiner":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        from process_sanskrit.splitter.sandhi import Sandhi

        self._sandhi = Sandhi()

    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def to_slp1(word: str) -> str:
        return sanscript.transliterate(unicodedata.normalize("NFC", word), sanscript.IAST, sanscript.SLP1)

    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def to_iast(word: str) -> str:
        return sanscript.transliterate(word, sanscript.SLP1, sanscript.IAST)

    @staticmethod
    def _string(slp1: str):
        from process_sanskrit.splitter.sanskrit_base import SanskritImmutableString

        return SanskritImmutableString(slp1, encoding=sanscript.SLP1)

    @lru_cache(maxsize=CACHE_SIZE)
    def join_pair(self, left: str, right: str) -> FrozenSet[str]:
        """Every spelling of `left` + `right` the rules allow; the bare concatenation if none."""
        joins = self._sandhi.join(self._string(self.to_slp1(left)), self._string(self.to_slp1(right)))
        if isinstance(joins, str):
            joins = {joins}
        if not joins:
            return frozenset({left + right})
        return frozenset(self.to_iast(joined) for joined in joins)

    def join(self, pieces: Sequence[str]) -> FrozenSet[str]:
        """Every spelling of `pieces` joined left to right, at most `MAX_JOINS` per step."""
        joined = frozenset({pieces[0]})
        for piece in pieces[1:]:
            spelled = {whole for left in joined for whole in self.join_pair(left, piece)}
            joined = frozenset(sorted(spelled)[: self.MAX_JOINS])
        return joined


__all__ = ["SandhiJoiner"]
