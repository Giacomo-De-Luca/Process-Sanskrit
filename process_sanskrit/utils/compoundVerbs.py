"""The preverb compounds of Monier-Williams, keyed by the parse it records.

Monier-Williams marks the entry of a preverb compound with its parse, in SLP1:
`<info verb="pre" parse="prati+A+gam"/>` on `pratyāgam`.  `CompoundVerbIndex`
reads those marks once per database (6,338 entries, 6,148 distinct parses, a
tenth of a second) and answers which headwords a run of preverbs and a root
make: `prati` + `i` -> `pratī`, `vi` + `ava` + `hṛ` -> `vyavahṛ`.

Lemma mode uses it for the lemma that ties a preverb to the part after it; see
documentation/lemma-mode.md, rule 5.
"""

import re
from typing import Dict, List, Sequence, Tuple

from indic_transliteration import sanscript
from sqlalchemy import text

from process_sanskrit.utils.engineCache import EngineCache

_PARSE = re.compile(r'<info verb="pre"[^>]*parse="([^"]+)"')
_QUERY = text(
    "SELECT keys_iast, cleaned_body FROM mw WHERE cleaned_body LIKE '%verb=\"pre\"%' ORDER BY lnum"
)


class CompoundVerbIndex:
    """`(preverbs, root)` -> the Monier-Williams headwords of that compound."""

    ## Monier-Williams spells some preverbs as they sound before a voiced
    ## consonant (`nir`, `dur`), some as before a vowel (`praty`, `vy`), and
    ## both `nir` and `nis`.  Both sides are brought to one spelling, the one
    ## lemma mode names the preverb by, before they are compared.
    MW_SPELLINGS = {"nir": "nis", "dur": "dus", "antaH": "antar", "praty": "prati", "vy": "vi"}
    ## lemma mode names `ud` by its pausa form `ut` and keeps `saṃ` as found
    OUR_SPELLINGS = {"ut": "ud", "saṃ": "sam", "vy": "vi", "praty": "prati"}

    ## one table per database engine, dropped with the engine
    _tables = EngineCache()

    @classmethod
    def reset(cls) -> None:
        """Forget every table read; the next lookup reads its database again."""
        cls._tables.clear()

    def __init__(self, session):
        self.session = session

    def headwords(self, preverbs: Sequence[str], root: str) -> List[str]:
        """The headwords of `preverbs` + `root`, in dictionary order; [] if none."""
        if not preverbs or not root:
            return []
        key = tuple(
            self._slp1(self.OUR_SPELLINGS.get(preverb, preverb)) for preverb in preverbs
        ) + (self._slp1(root),)
        return list(self._table().get(key, ()))

    @staticmethod
    def _slp1(word: str) -> str:
        return sanscript.transliterate(word, sanscript.IAST, sanscript.SLP1)

    def _table(self) -> Dict[Tuple[str, ...], Tuple[str, ...]]:
        return self._tables.table(self.session, self._read)

    def _read(self) -> Dict[Tuple[str, ...], Tuple[str, ...]]:
        found: Dict[Tuple[str, ...], Dict[str, None]] = {}
        for headword, body in self.session.execute(_QUERY):
            for parse in _PARSE.findall(body or ""):
                *preverbs, root = parse.split("+")
                key = tuple(self.MW_SPELLINGS.get(preverb, preverb) for preverb in preverbs) + (root,)
                found.setdefault(key, {})[headword] = None
        return {key: tuple(headwords) for key, headwords in found.items()}


__all__ = ["CompoundVerbIndex"]
