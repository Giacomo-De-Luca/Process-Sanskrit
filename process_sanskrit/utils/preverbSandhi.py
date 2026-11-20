"""Join preverbs to a stem the way the dictionaries spell a preverb compound.

Lemma mode ties a preverb to the part after it with one more lemma, the
compound they make (`prati` + `i` -> `pratī`).  It takes that word from
Monier-Williams when Monier-Williams records the compound; `PreverbSandhi`
builds it when Monier-Williams does not (`upa` + `sam` + `arc` -> `upasamarc`).
The two must spell alike, so the rules follow Monier-Williams' headwords:
`saṃgam` but `sampad`, `udgam`, `nirgam`, `niḥsṛ`, `praṇam`.

The rules cover the sounds the preverbs of `SANSKRIT_PREFIXES` end in.  They are
a spelling convention for a search key, not a grammar: an `s` after `i`/`u` is
always `ṣ`, where Monier-Williams decides word by word (`abhiṣañj` but
`abhisap`), and a stem is taken as given.  See documentation/lemma-mode.md,
rule 5.
"""

from typing import Sequence

VOWELS = frozenset("aāiīuūṛṝḷeo")
LONG = {"a": "ā", "ā": "ā", "i": "ī", "ī": "ī", "u": "ū", "ū": "ū", "ṛ": "ṝ", "ṝ": "ṝ"}
SEMIVOWEL = {"i": "y", "ī": "y", "u": "v", "ū": "v"}

## voiced consonants: before them `ud` and `nis`/`dus` keep (or take) a voiced end
VOICED = ("gh", "g", "jh", "j", "ḍh", "ḍ", "dh", "d", "bh", "b", "y", "v", "r", "l", "n", "m", "ṅ", "ñ", "ṇ", "h")
LABIALS = ("p", "b", "m")
## the preverbs after which Monier-Williams spells a stem's n as ṇ
RETROFLEX_N_PREVERBS = frozenset({"pra", "parā", "pari"})


class PreverbSandhi:
    """Sandhi between a preverb and what follows it.  Stateless."""

    RETROFLEX_S_EXCEPTIONS = ("sam", "saṃ", "sṛ")

    @classmethod
    def join_all(cls, preverbs: Sequence[str], stem: str) -> str:
        """`preverbs` and `stem` joined: `vi`, `ava`, `hṛ` -> `vyavahṛ`.

        From the right, because a rule is chosen by the preverb on its left:
        `sam` + `ud` + `car` is `sam` + `uccar`, and `samud` is no preverb.
        """
        joined = stem
        for preverb in reversed(list(preverbs)):
            joined = cls.join(preverb, joined)
        return joined

    @classmethod
    def join(cls, left: str, right: str) -> str:
        if not left or not right:
            return left + right
        if left in ("sam", "saṃ"):
            return cls._sam(right)
        if left in ("ud", "ut"):
            return cls._ud(right)
        if left in ("nis", "nir", "dus", "dur"):
            return cls._s_final(left[:-1], right)
        if left[-1] in VOWELS:
            return cls._vowel(left, right)
        return left + right

    @staticmethod
    def _starts_with_vowel(word: str) -> bool:
        return word[0] in VOWELS

    @classmethod
    def _vowel(cls, left: str, right: str) -> str:
        end, start = left[-1], right[0]
        if not cls._starts_with_vowel(right):
            ## `pra` + `nam` -> `praṇam` (Monier-Williams keeps `pratinad`)
            if start == "n" and left in RETROFLEX_N_PREVERBS:
                return left + "ṇ" + right[1:]
            ## `abhi` + `sañj` -> `abhiṣañj`, `adhi` + `sthā` -> `adhiṣṭhā`; never
            ## before a second preverb (`abhisaṃgam`)
            if start == "s" and end in "iīuū" and not right.startswith(cls.RETROFLEX_S_EXCEPTIONS):
                return left + cls._retroflex_s(right)
            ## `ā` + `chad` -> `ācchad`
            if right.startswith("ch"):
                return left + "c" + right
            return left + right
        if left == "a":
            return "an" + right
        diphthong = right[:2] in ("ai", "au")
        if end in "aā":
            ## vṛddhi before `edh` (P. 6.1.89): `upaidh`; Monier-Williams spells
            ## `ūh` with o (`proh`, `upoh`)
            if right.startswith("edh"):
                return left[:-1] + "ai" + right[1:]
            if diphthong:
                return left[:-1] + right
            if start in "aā":
                return left[:-1] + "ā" + right[1:]
            if start in "iī":
                return left[:-1] + "e" + right[1:]
            if start in "uū":
                return left[:-1] + "o" + right[1:]
            if start in "ṛṝ":
                return left[:-1] + "ār" + right[1:]
            ## `pra` + `ej` -> `prej`: before e or o a preverb's a is dropped
            return left[:-1] + right
        if end in "iīuū":
            if LONG.get(start) == LONG[end] and not diphthong:
                return left[:-1] + LONG[end] + right[1:]
            return left[:-1] + SEMIVOWEL[end] + right
        return left + right

    @staticmethod
    def _retroflex_s(right: str) -> str:
        if right.startswith("sth"):
            return "ṣṭh" + right[3:]
        if right.startswith("st"):
            return "ṣṭ" + right[2:]
        return "ṣ" + right[1:]

    @classmethod
    def _sam(cls, right: str) -> str:
        if cls._starts_with_vowel(right) or right.startswith(LABIALS):
            return "sam" + right
        return "saṃ" + right

    @classmethod
    def _ud(cls, right: str) -> str:
        if cls._starts_with_vowel(right):
            return "ud" + right
        if right.startswith("st"):
            return "ut" + right[1:]
        if right.startswith(("ṭ", "ḍ")):
            return "u" + right[0] + right
        if right.startswith("h"):
            return "uddh" + right[1:]
        if right.startswith("l"):
            return "ul" + right
        if right.startswith(("n", "m")):
            return "un" + right
        if right.startswith("c"):
            return "uc" + right
        if right.startswith("j"):
            return "uj" + right
        if right.startswith("ś"):
            return "ucch" + right[1:]
        if right.startswith(VOICED):
            return "ud" + right
        return "ut" + right

    @classmethod
    def _s_final(cls, base: str, right: str) -> str:
        """`nis` and `dus`: `nirgam`, `niṣkṛ`, `niścar`, `nistṝ`, `niḥsṛ`."""
        if cls._starts_with_vowel(right):
            return base + "r" + right
        if right.startswith("r"):
            return base[:-1] + LONG[base[-1]] + right
        if right.startswith("n"):
            return base + "rṇ" + right[1:]
        if right.startswith(VOICED):
            return base + "r" + right
        if right.startswith("kṣ"):
            return base + "ḥ" + right
        if right.startswith(("k", "p")):
            return base + "ṣ" + right
        if right.startswith("c"):
            return base + "ś" + right
        if right.startswith("ṭ"):
            return base + "ṣ" + right
        if right.startswith("t"):
            return base + "s" + right
        return base + "ḥ" + right


__all__ = ["PreverbSandhi"]
