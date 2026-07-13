"""Productive -tā / -tva abstract nouns.

The taddhita suffixes *-tā* (feminine, ā-stem) and *-tva* (neuter, a-stem) build
an abstract noun out of any nominal stem: śūnya "empty" -> śūnyatā "emptiness",
niṣyanda "flowing" -> niṣyandatā "the state of flowing".  Because they are
productive, a text may coin one from any stem at all, and no lexicon can list
them exhaustively.

The database lexicalises the common ones -- roughly 2050 in -tā and 2340 in -tva,
each stored with a hyphenated stem and a full paradigm ("śūnya-tā", model f_A), so
the storage convention already treats them as base + suffix.  Anything outside
that list resolves nowhere, and the compound splitter then cuts the word in two
and analyses the orphaned suffix as a verb (-tā as a form of tṛ or tan),
destroying the lemma and inventing a root that is not there.

This module rebuilds those derivatives instead.  Given an unresolved word it
strips a -tā/-tva paradigm ending, checks that what remains is a real nominal
stem, and regenerates the paradigm from the database's own exemplar row -- so a
coined derivative comes back with the same lemma, model and case tags a
lexicalised one would have.

Order matters as much as the rule.  The deriver is consulted only once every
other layer has missed: a direct hit in the inflection tables, and then a
whole-word dictionary match, both win first.  That is what keeps the look-alikes
safe -- feminine past participles (kṛtā, gatā), instrumentals of -vat stems
(bhagavatā), agent nouns (pitā, kartā) and plain words that merely end in -tā
(sītā, latā) all resolve earlier and never reach this code.

Two collisions survive that ordering and are handled explicitly.

1. *-atā* is ambiguous.  A consonant stem in *-at* (typically a present
   participle) makes its Inst. Sg. in *-atā*, its Gen. Pl. in *-atām* and its
   Dat. Sg. in *-ate*, all of which collide with base(-a) + tā:

       gacchatā  =  gacchat + ā   Inst. Sg., "by the one going"    <- gacchat is attested
       gacchatā  =  gaccha  + tā  abstract noun, "going-ness"      <- spurious

   Nothing in the surface form separates them, so the tie is broken on evidence:
   if the competing *-at* stem is itself attested, it owns the word and no
   derivation is offered.  This is deliberately conservative -- it also declines
   the genuinely ambiguous cases (jayatā: "by the conquering one", or
   "victoriousness") rather than overriding an attested stem with a manufactured
   one.

2. *-te* is not usable as evidence at all; see NON_LICENSING_ENDINGS.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from process_sanskrit.functions.SQLiteFind import SQLite_paradigm, SQLite_stem_exists
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils.loadResources import type_map
from process_sanskrit.utils.paradigm import NOMINAL_CASES, tags_for


## A base shorter than this is never accepted.  It is the last line of defence
## against re-analysing a monosyllable as a suffix carrier -- kṛ-tā, ga-tā, jā-tā
## are past participles, not abstract nouns, and their would-be bases (kṛ, ga,
## jā) are all real dictionary headwords, so nothing else would reject them.
MIN_BASE_LENGTH = 3

## Cells that are *generated* into the paradigm but must never *license* a
## derivation.
##
## -te is the ā-stem's Voc. Sg. and all three of its dual cells.  It is also the
## ending of every ātmanepada and every passive 3rd sg. in the language: śocate
## "he grieves", dṛśyate "it is seen", kriyate "it is done".  When such a verb
## misses the verb tables -- which it routinely does -- the deriver would strip
## the -te, find the (perfectly real) nominal stem underneath, and replace a
## correct verbal root with a fabricated abstract noun: śocate -> "śocatā".  That
## is precisely the damage this module exists to undo, running backwards.
##
## The trade is lopsided.  A vocative or dual of an abstract noun -- "O
## emptiness!", "two emptinesses" -- is vanishingly rare; ātmanepada and passive
## 3rd sg. are among the commonest forms in the language.  So -te is generated
## (the table stays complete and matches the stored one form for form) but is
## never accepted as the ending that identifies a derivative.
##
## -tām is deliberately NOT listed here even though it is the 3rd person
## imperative (kurutām "let him do"): it is also the ā-stem's Acc. Sg. (śūnyatām),
## which is common in the philosophical corpus, and the imperatives all resolve in
## the verb tables long before the deriver is consulted -- measured at zero
## changed analyses over an imperative probe set.
NON_LICENSING_ENDINGS = frozenset({"te"})


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

    @property
    def stem_consonant(self) -> str:
        """The part of the suffix that survives declension.

        Only the final vowel inflects -- -tā becomes -te and -tayā, -tva becomes
        -tve and -tvāni -- so this ("t", "tv") is the invariant worth asserting on.
        """
        return self.suffix[:-1]


SUFFIXES: Tuple[TaddhitaSuffix, ...] = (
    TaddhitaSuffix(suffix="tā", model="f_A", exemplar="śūnya-tā"),
    TaddhitaSuffix(suffix="tva", model="n_a", exemplar="a-karuṇa-tva"),
)


@dataclass(frozen=True)
class EndingSet:
    """The two ending lists read off one exemplar row.

    `endings` is the full paradigm in stored order, used to *generate* a table.
    `licensing` is the subset allowed to *identify* a derivative -- deduplicated,
    longest first, and with NON_LICENSING_ENDINGS removed.  They are kept together
    because they must never disagree about which lexicon they came from.
    """

    endings: List[str]
    licensing: Tuple[str, ...]


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
        """Render as the 5-slot row root_any_word yields for an attested word.

        Downstream code must not be able to tell a reconstructed analysis from a
        stored one, so the shape has to match exactly: stem, model, case/number
        tags, full paradigm, the surface form looked up.  The model is mapped
        through type_map here because that is what root_any_word does to
        SQLite_find_name's raw model code before anyone downstream sees it.
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
        ## Keyed on (lexicon identity, suffix), not on suffix alone: an
        ## externally provisioned database (see commit 65151d6) can be pointed at
        ## a different path mid-process, and a stale ending list would then be
        ## reused instead of raising the deliberately fatal error below.
        ##
        ## The two ending sets live in ONE record rather than two parallel dicts:
        ## they must never disagree, and reading them separately would mean
        ## resolving the lexicon identity twice, which can change underneath a
        ## caller mid-lookup and miss on the second read.
        self._paradigms: Dict[Tuple[str, str], EndingSet] = {}

    ## -- paradigm construction ------------------------------------------------

    def clear_cache(self) -> None:
        """Drop the memoised exemplar paradigms."""
        self._paradigms.clear()

    def stored_paradigm(self, stem: str, suffix: str, session=None) -> Optional[List[str]]:
        """The paradigm the database stores for a lexicalised derivative."""
        return SQLite_paradigm(stem, self._spec(suffix).model, session=session)

    def endings(self, suffix: str, session=None) -> List[str]:
        """The 24 endings of this suffix, read off its exemplar row.

        The exemplar's own base is stripped from each of its forms, which leaves
        the suffix plus the case ending ("tā", "tayā", "tānām", ...).
        """
        return self._ending_set(suffix, session=session).endings

    def _ending_set(self, suffix: str, session=None) -> "EndingSet":
        """Everything read off one exemplar row, resolved in a single lookup."""
        spec = self._spec(suffix)
        key = (self._lexicon_identity(), suffix)

        cached = self._paradigms.get(key)
        if cached is not None:
            return cached

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

        if not all(
            form.startswith(exemplar_base) and ending.startswith(spec.stem_consonant)
            for form, ending in zip(forms, endings)
        ):
            raise RuntimeError(
                f"the exemplar row {spec.exemplar!r} / {spec.model!r} does not "
                f"inflect as -{suffix} does on the base {exemplar_base!r}; "
                "the database layout has changed."
            )

        ## longest first, so a longer cell is never shadowed by a shorter one that
        ## happens to be its tail; -te is dropped, see NON_LICENSING_ENDINGS
        licensing = tuple(
            sorted(
                {e for e in endings if e not in NON_LICENSING_ENDINGS},
                key=len,
                reverse=True,
            )
        )
        ending_set = EndingSet(endings=endings, licensing=licensing)
        self._paradigms[key] = ending_set
        return ending_set

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
            ## resolved once and threaded through, so the endings used to find the
            ## base and the endings used to build its table cannot come from two
            ## different lexicons
            ending_set = self._ending_set(spec.suffix, session=session)

            base = self._base_of(word, spec, ending_set, session=session)
            if base is None:
                continue

            forms = [base + ending for ending in ending_set.endings]
            return DerivedForm(
                lemma=base + spec.suffix,
                base=base,
                suffix=spec.suffix,
                model=spec.model,
                ## one surface form can fill several cells, so report them all,
                ## exactly as the stored lookup does
                tags=tags_for(forms, word, rows=NOMINAL_CASES),
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

    def _lexicon_identity(self) -> str:
        """Which lexicon the cached endings were read from.

        Deliberately the configured path, and NOT analysisCache.lexicon_fingerprint():
        that helper is @lru_cache'd on its `db_path` argument, so called the way we
        would call it (with no argument) it freezes at its first answer and never
        notices a swap -- which makes it useless as a cache key here.  get_db_path()
        reflects the currently configured database on every call.
        """
        from process_sanskrit.utils.databaseSetup import get_db_path

        try:
            return get_db_path()
        except OSError:
            ## the identity only keys the cache; a path that cannot be resolved
            ## must not take down a derivation the session can perfectly well answer
            return "unknown"

    def _base_of(
        self,
        word: str,
        spec: TaddhitaSuffix,
        ending_set: EndingSet,
        session=None,
    ) -> Optional[str]:
        """The stem `word` is built on, if it is built on one at all.

        Takes the already-resolved ending set rather than looking it up again:
        resolving it twice means resolving the lexicon identity twice, and a swap
        between the two reads would miss the cache and raise.
        """
        ## No licensing ending is a tail of another (pinned by
        ## test_no_licensing_ending_is_a_tail_of_another), so at most one can
        ## match: the first hit is the only hit, and a rejected hit has nothing to
        ## fall back to.  Hence every rejection below returns rather than continues.
        for ending in ending_set.licensing:
            if not word.endswith(ending):
                continue

            base = word[: -len(ending)]
            if len(base) < MIN_BASE_LENGTH:
                return None
            if not self._is_attested_stem(base, session=session):
                return None
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
## instance memoises them for the whole process; the cache is keyed on the
## lexicon's identity, so swapping the database underneath is safe.
taddhita_deriver = TaddhitaDeriver()


__all__ = [
    "DerivedForm",
    "EndingSet",
    "MIN_BASE_LENGTH",
    "NON_LICENSING_ENDINGS",
    "SUFFIXES",
    "TaddhitaDeriver",
    "TaddhitaSuffix",
    "taddhita_deriver",
]
