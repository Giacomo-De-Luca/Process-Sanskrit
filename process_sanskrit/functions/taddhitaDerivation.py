"""Productive -tā / -tva abstract nouns.

The taddhita suffixes *-tā* (feminine, ā-stem) and *-tva* (neuter, a-stem) build
an abstract noun out of any nominal stem: śūnya "empty" -> śūnyatā "emptiness",
niṣyanda "flowing" -> niṣyandatā "the state of flowing".  Because they are
productive, a text may coin one from any stem at all, and no lexicon can list
them exhaustively.

The database lexicalises the common ones -- roughly 2050 in -tā and 2340 in -tva,
each stored with a hyphenated stem and a full paradigm ("śūnya-tā", model f_A --
so the storage convention already treats them as base + suffix.  Anything
outside that list resolves nowhere, and the compound splitter then cuts the word
in two and analyses the orphaned suffix as a verb (-tā as a form of tṛ or tan),
destroying the lemma and inventing a root that is not there.

This module rebuilds those derivatives instead.  Given an unresolved word it
strips a -tā/-tva paradigm ending, checks that what remains is a real nominal
stem, and regenerates the paradigm from the database's own exemplar row -- so a
coined derivative comes back with the same lemma, model and case tags a
lexicalised one would have.

Ambiguity.  *-atā* has a second, equally valid parse.  A consonant stem in *-at*
(typically a present participle) makes its Inst. Sg. in *-atā*, its Gen. Pl. in
*-atām* and its Dat. Sg. in *-ate*, all of which collide with base(-a) + tā:

    gacchatā  =  gacchat + ā   Inst. Sg., "by the one going"    <- gacchat is attested
    gacchatā  =  gaccha  + tā  abstract noun, "going-ness"      <- spurious

Nothing in the surface form separates them, so the tie is broken on evidence: if
the competing *-at* stem is itself attested, it owns the word and no derivation
is offered.  This is deliberately conservative -- it also declines the handful of
genuinely ambiguous cases (jayatā: "by the conquering one", or "victoriousness")
rather than overriding an attested stem with a manufactured one.

Order matters as much as the rule.  The deriver is consulted only once every
other layer has missed: a direct hit in the inflection tables, and then a
whole-word dictionary match, both win first.  That is what keeps the look-alikes
safe -- feminine past participles (kṛtā, gatā), instrumentals of -vat stems
(bhagavatā), agent nouns (pitā, kartā) and plain words that merely end in -tā
(sītā, latā) all resolve earlier and never reach this code.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from process_sanskrit.functions.SQLiteFind import SQLite_paradigm, SQLite_stem_exists
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils.loadResources import type_map


## the inflection tables lay a paradigm out as 8 cases x 3 numbers, read across
CASES = ("Nom", "Acc", "Inst", "Dat", "Abl", "Gen", "Loc", "Voc")
NUMBERS = ("Sg", "Du", "Pl")

## A base shorter than this is never accepted.  It is the last line of defence
## against re-analysing a monosyllable as a suffix carrier -- kṛ-tā, ga-tā, jā-tā
## are past participles, not abstract nouns, and their would-be bases (kṛ, ga,
## jā) are all real dictionary headwords, so nothing else would reject them.
MIN_BASE_LENGTH = 3


@dataclass(frozen=True)
class TaddhitaSuffix:
    """One productive suffix, and the stored row whose paradigm it copies.

    The endings are lifted from a lexicalised derivative rather than hardcoded,
    so the forms this module generates are by construction the forms the database
    would have stored for the same word.
    """

    suffix: str
    model: str
    exemplar: str


SUFFIXES: Tuple[TaddhitaSuffix, ...] = (
    TaddhitaSuffix(suffix="tā", model="f_A", exemplar="śūnya-tā"),
    TaddhitaSuffix(suffix="tva", model="n_a", exemplar="a-karuṇa-tva"),
)


@dataclass(frozen=True)
class DerivedForm:
    """A derivative reconstructed from its base."""

    lemma: str
    base: str
    suffix: str
    model: str
    tags: Optional[List[Tuple[str, str]]]
    forms: List[str]
    surface: str

    def as_entry(self) -> list:
        """Render as the 5-slot row SQLite_find_name returns.

        Downstream code must not be able to tell a reconstructed analysis from a
        stored one, so the shape has to match exactly: stem, human-readable
        model, case/number tags, full paradigm, the surface form looked up.
        """
        return [
            self.lemma,
            type_map.get(self.model, self.model),
            self.tags,
            list(self.forms),
            self.surface,
        ]


class TaddhitaDeriver:
    """Reconstructs -tā / -tva derivatives that the lexicon does not list."""

    def __init__(self, suffixes: Sequence[TaddhitaSuffix] = SUFFIXES):
        self._suffixes = tuple(suffixes)
        self._endings: Dict[str, List[str]] = {}

    ## -- paradigm construction ------------------------------------------------

    def clear_cache(self) -> None:
        """Drop the memoised exemplar paradigms (the database may have changed)."""
        self._endings.clear()

    def stored_paradigm(self, stem: str, suffix: str, session=None) -> Optional[List[str]]:
        """The paradigm the database stores for a lexicalised derivative."""
        spec = self._spec(suffix)
        return SQLite_paradigm(stem, spec.model, session=session)

    def endings(self, suffix: str, session=None) -> List[str]:
        """The 24 endings of this suffix, read off its exemplar row.

        The exemplar's own base is stripped from each of its forms, which leaves
        the suffix plus the case ending ("tā", "tayā", "tānām", ...).
        """
        if suffix in self._endings:
            return self._endings[suffix]

        spec = self._spec(suffix)
        forms = SQLite_paradigm(spec.exemplar, spec.model, session=session)
        if not forms:
            ## Silently skipping the derivation would look like "this word has no
            ## analysis", which is indistinguishable from the bug this module
            ## fixes.  A missing exemplar means the database is not the one this
            ## code was built against, and that should be said out loud.
            raise RuntimeError(
                f"cannot build the -{suffix} paradigm: the database has no row for "
                f"stem {spec.exemplar!r} / model {spec.model!r}. "
                "The installed database is older than this code; "
                "re-run `update-ps-database`."
            )

        ## the stem is stored hyphenated ("a-karuṇa-tva"); its base is that stem
        ## with the hyphens and the suffix taken off ("akaruṇa")
        exemplar_base = spec.exemplar.replace("-", "")[: -len(suffix)]
        endings = [form[len(exemplar_base):] for form in forms]

        ## Only the suffix's consonant survives declension: -tā inflects to -te
        ## and -tayā, -tva to -tve and -tvāni, so the final vowel is not an
        ## invariant and must not be asserted on.  What must hold is that every
        ## form is the base plus something that still begins with that consonant.
        stem_consonant = suffix[:-1]
        if not all(
            form.startswith(exemplar_base) and ending.startswith(stem_consonant)
            for form, ending in zip(forms, endings)
        ):
            raise RuntimeError(
                f"the exemplar row {spec.exemplar!r} / {spec.model!r} does not "
                f"inflect as -{suffix} does on the base {exemplar_base!r}; "
                "the database layout has changed."
            )

        self._endings[suffix] = endings
        return endings

    def paradigm(self, base: str, suffix: str, session=None) -> List[str]:
        """The full 24-form table of `base` + `suffix`."""
        return [base + ending for ending in self.endings(suffix, session=session)]

    ## -- derivation -----------------------------------------------------------

    def derive(self, word: str, session=None) -> Optional[DerivedForm]:
        """Reconstruct `word` as base + -tā / -tva, or return None.

        Call only after the ordinary lookups have missed; see the module
        docstring on why order is load-bearing.
        """
        if not word:
            return None

        for spec in self._suffixes:
            base = self._base_of(word, spec, session=session)
            if base is None:
                continue

            forms = self.paradigm(base, spec.suffix, session=session)
            ## one surface form can fill several cells (X-te is Voc. Sg. and both
            ## duals), so report every cell it fills, as the stored lookup does
            positions = [index for index, form in enumerate(forms) if form == word]
            tags = [
                (CASES[index // len(NUMBERS)], NUMBERS[index % len(NUMBERS)])
                for index in positions
            ] or None

            return DerivedForm(
                lemma=base + spec.suffix,
                base=base,
                suffix=spec.suffix,
                model=spec.model,
                tags=tags,
                forms=forms,
                surface=word,
            )

        return None

    ## -- internals ------------------------------------------------------------

    def _spec(self, suffix: str) -> TaddhitaSuffix:
        for spec in self._suffixes:
            if spec.suffix == suffix:
                return spec
        raise KeyError(f"no such taddhita suffix: {suffix!r}")

    def _base_of(self, word: str, spec: TaddhitaSuffix, session=None) -> Optional[str]:
        """The stem `word` is built on, if it is built on one at all."""
        ## longest ending first: -tāyāḥ must not be mistaken for a shorter match
        for ending in sorted(set(self.endings(spec.suffix, session=session)), key=len, reverse=True):
            if not word.endswith(ending):
                continue

            base = word[: -len(ending)]
            if len(base) < MIN_BASE_LENGTH:
                continue
            if not self._is_attested_stem(base, session=session):
                continue
            if self._at_stem_wins(base, spec, session=session):
                return None

            return base

        return None

    def _at_stem_wins(self, base: str, spec: TaddhitaSuffix, session=None) -> bool:
        """Does an attested -at stem already account for this form?

        Only -tā is at risk: the consonant-stem endings that collide (-atā, -atām,
        -ate) all attach to a stem in -at, which exists only when the base ends
        in -a.  -tva has no such competitor.
        """
        if spec.suffix != "tā" or not base.endswith("a"):
            return False
        return self._is_attested_stem(base + "t", session=session)

    def _is_attested_stem(self, candidate: str, session=None) -> bool:
        """Is this a real nominal stem: a dictionary headword, or an inflecting stem?

        Both halves are needed.  niṣyanda is a headword with no inflection table
        of its own, while plenty of stems inflect without heading a dictionary
        entry; a base that fails both is not a word, and the derivation would be
        an invention.
        """
        if candidate in DICTIONARY_REFERENCES:
            return True
        return SQLite_stem_exists(candidate, session=session)


## The exemplar paradigms are structural facts about the lexicon, so one shared
## instance memoises them for the whole process; call clear_cache() if the
## database underneath is ever swapped.
taddhita_deriver = TaddhitaDeriver()


__all__ = [
    "DerivedForm",
    "SUFFIXES",
    "TaddhitaDeriver",
    "TaddhitaSuffix",
    "taddhita_deriver",
]
