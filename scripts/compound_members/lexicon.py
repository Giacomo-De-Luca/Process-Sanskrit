"""What the evaluation reads from the dictionaries in bulk, and how it compares spellings.

The member readings themselves live in the library
(`process_sanskrit/utils/compoundMembers.py`); this module has only what the
evaluation needs on top:

* `Spelling`         a key that ignores the spelling differences between DCS
                     and the dictionaries (letters: `ComponentMarks.letters`);
* `JoinSpellings`    the spellings a DCS segment is joined from (its pausa
                     readings, a final visarga as `r` or `s`);
* `ComponentTables`  the `components` column of Monier-Williams, Edgerton and
                     Grassmann, read once, for 200,000 headwords at a time;
* `Headwords`        `DICTIONARY_REFERENCES`, read once.
"""

import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple

from process_sanskrit.functions.lemmaAnalysis import LemmaLexicon
from process_sanskrit.utils.componentMarks import ComponentMarks
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils.lexicalResources import pausa_readings


class Spelling:
    """Comparison keys of IAST strings."""

    ## A nasal written before a consonant is spelled ṅ/ñ/ṇ/n/m by Monier-Williams
    ## (`nistaraṅga`, `saṃyoga`) and ṃ by DCS (`nistaraṃga`).
    _NASAL_BEFORE_CONSONANT = re.compile("[ṅñṇnm](?=[kgcjṭḍtdpbyrlvśṣsh])")
    ## A sibilant or visarga before k/p is written ṣ, ś, s or ḥ (`niṣphala`, `niḥphala`).
    _SIBILANT_BEFORE_STOP = re.compile("[ṣśḥ](?=[kp])")

    @classmethod
    def key(cls, text: str) -> str:
        """A key equal for spellings of one word: nasals, ṇ / n, sibilants before k/p, final ḥ / s, final ṃ / m."""
        word = ComponentMarks.letters(text).lower().replace("ṁ", "ṃ")
        word = cls._NASAL_BEFORE_CONSONANT.sub("ṃ", word)
        word = cls._SIBILANT_BEFORE_STOP.sub("s", word).replace("ṇ", "n")
        if word.endswith("ḥ"):
            word = word[:-1] + "s"
        if word.endswith("ṃ"):
            word = word[:-1] + "m"
        return word

    @classmethod
    def keys_of(cls, words: Iterable[str]) -> Dict[str, Tuple[str, ...]]:
        """For each key of `words`, the words of `words` that have it."""
        found: Dict[str, set] = defaultdict(set)
        for word in words:
            found[cls.key(word)].add(word)
        return {key: tuple(sorted(spellings)) for key, spellings in found.items()}


class JoinSpellings:
    """The spellings of a DCS compound segment to try when joining it to the next.

    DCS writes a segment unsandhied, in a stem form (`tad`, `vāc`) or in pausa
    (`antaḥ`).  The sandhi rules join pausa forms, so a segment is tried as
    written, in its pausa readings (`pausa_readings`), and with a final
    visarga read as the `r` or `s` of the stem (`antaḥ` + `gata` -> `antargata`).
    """

    @staticmethod
    def of(form: str) -> Tuple[str, ...]:
        found = list(pausa_readings(form))
        if form.endswith("ḥ"):
            found += [form[:-1] + "r", form[:-1] + "s"]
        return tuple(dict.fromkeys(found))


class ComponentTables:
    """The `components` column of the dictionaries that mark members, read once.

    `analyses(dictionary, headword)` gives what `headword_components` gives for
    one headword (the tests pin that): each distinct analysis, in sorted
    order, with the number of entries that record it.  The library reads one
    headword at a time; the evaluation reads all of them.
    """

    DICTIONARIES = ("mw", "bhs", "gra")

    def __init__(self, connection: sqlite3.Connection, dictionaries: Sequence[str] = DICTIONARIES):
        self._analyses: Dict[str, Dict[str, List[Tuple[str, int]]]] = {}
        self._entries: Dict[str, FrozenSet[str]] = {}
        for dictionary in dictionaries:
            if dictionary not in self.DICTIONARIES:
                raise ValueError(f"no components reader for {dictionary!r}")
            analyses: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
            for headword, components, entries in connection.execute(
                f"SELECT keys_iast, components, COUNT(*) FROM {dictionary} "
                "WHERE components IS NOT NULL AND components != '' "
                "GROUP BY keys_iast, components ORDER BY keys_iast, components"
            ):
                analyses[headword].append((components, entries))
            self._analyses[dictionary] = dict(analyses)
            self._entries[dictionary] = frozenset(
                headword for (headword,) in connection.execute(f"SELECT DISTINCT keys_iast FROM {dictionary}")
            )

    def analyses(self, dictionary: str, headword: str) -> List[Tuple[str, int]]:
        return list(self._analyses[dictionary].get(headword, ()))

    def has_entry(self, dictionary: str, headword: str) -> bool:
        """Whether `dictionary` has `headword` at all, with or without components."""
        return headword in self._entries[dictionary]

    def headwords(self) -> FrozenSet[str]:
        """Every headword some dictionary here gives components for."""
        return frozenset(headword for analyses in self._analyses.values() for headword in analyses)


class Headwords:
    """`DICTIONARY_REFERENCES`, read once: which dictionaries list a word.

    `sanskrit` is the library's stricter notion (Monier-Williams or Apte,
    `LemmaLexicon.SANSKRIT_DICTIONARIES`).
    """

    def __init__(self, references=DICTIONARY_REFERENCES):
        self._dictionaries: Dict[str, FrozenSet[str]] = {
            word: frozenset(names) for word, names in references.items()
        }

    def sanskrit(self, word: str) -> bool:
        return bool(LemmaLexicon.SANSKRIT_DICTIONARIES & self._dictionaries.get(word, frozenset()))

    def breadth(self, word: str) -> int:
        """How many dictionaries have `word`: a rough stand-in for how common it is."""
        return len(self._dictionaries.get(word, ()))

    def in_monier_williams(self, word: str) -> bool:
        return "mw" in self._dictionaries.get(word, frozenset())


def read_only_connection(path: Path) -> sqlite3.Connection:
    """The lexicon database, opened so that nothing here can write to it."""
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection
