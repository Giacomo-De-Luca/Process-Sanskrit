r"""
Productive -tā / -tva abstract nouns.

The secondary (taddhita) suffixes *-tā* (f., ā-stem) and *-tva* (n., a-stem)
turn any nominal stem into an abstract noun: śūnya "empty" -> śūnyatā
"emptiness", niṣyanda "flowing" -> niṣyandatā "the state of flowing".  They are
*productive*: a text may coin one from any stem it likes, so no wordlist can
ever be complete.

The database lexicalises about 2050 of the first and 2340 of the second, stored
with a hyphenated stem and a full paradigm ("śūnya-tā", model f_A).  Anything
outside that list used to fail every lookup and fall through to the compound
splitter, which cut the word in two and then analysed the orphaned suffix as a
verb:

    process("niṣyandatā", mode="roots")
    -> ['niṣyanda', ('tā', 'tṛ', 'tan')]      # lemma destroyed, tṛ/tan invented

The deriver rebuilds such words: it strips a -tā/-tva paradigm ending, checks the
remaining base is a real nominal stem, and regenerates the paradigm from the
database's own exemplar row -- so a coined derivative gets the same lemma, model
and case tags a lexicalised one would.

The hazard is that *-atā* is ambiguous.  A consonant stem in *-at* (a present
participle: gacchat "going") makes its Inst. Sg. in *-atā* and its Gen. Pl. in
*-atām*, which collides head-on with base(-a) + tā:

    gacchatā  =  gacchat + ā   (Inst. Sg., "by the one going")   <- attested
    gacchatā  =  gaccha  + tā  (abstract noun, "going-ness")     <- spurious

Both parses are structurally valid, so the tie is broken on evidence: if the
competing *-at* stem is itself attested (gacchat, paśyat, jayat and vidvat all
are), it owns the form and the derivation is declined.  That check is what keeps
the participles below intact.
"""

import unittest

from process_sanskrit.functions.process import process
from process_sanskrit.functions.taddhitaDerivation import (
    SUFFIXES,
    TaddhitaDeriver,
    TaddhitaSuffix,
    taddhita_deriver,
)
from process_sanskrit.utils import databaseSetup
from process_sanskrit.utils.databaseSetup import session_scope


## niṣyanda ("flowing, oozing") is the running example: it is a real word -- a
## BHS dictionary headword -- but it has no inflection table of its own, and no
## -tā derivative of it is lexicalised.  So every layer of the cascade misses it,
## which is exactly the case the deriver exists to catch.
TARGETS = {
    "niṣyandatā": ("niṣyandatā", [("Nom", "Sg")]),
    "niṣyandatām": ("niṣyandatā", [("Acc", "Sg")]),
    "niṣyandatayā": ("niṣyandatā", [("Inst", "Sg")]),
    "niṣyandatāyām": ("niṣyandatā", [("Loc", "Sg")]),
    "niṣyandatāsu": ("niṣyandatā", [("Loc", "Pl")]),
    "niṣyandatvam": ("niṣyandatva", [("Nom", "Sg"), ("Acc", "Sg")]),
    "niṣyandatvena": ("niṣyandatva", [("Inst", "Sg")]),
}

## Words that merely *end* in a -tā/-tva-looking string.  Each is already
## resolved by an earlier layer, so the deriver must never be consulted for it;
## the values are the outputs recorded before the deriver existed.
UNCHANGED = {
    ## feminine past participles: kṛ-ta + ā, not kṛ + tā
    "kṛtā": ["kṛtā", ("kṛta", "kṛt")],
    "gatā": ["gata"],
    "gītā": ["gītā", "gīta"],
    ## instrumentals of consonant -at / -vat stems
    "mahatā": [("mahatā", "mahat")],
    "bhavatā": ["bhavat"],
    "bhagavatā": ["bhagavat"],
    ## agent nouns in -ṛ
    "pitā": ["pitā", "pitṛ"],
    "kartā": [("kartṛ", "kṛ")],
    ## plain words that happen to end in -tā
    "sītā": ["sītā"],
    "latā": ["latā"],
    ## lexicalised derivatives: these must keep coming from the database, not
    ## from the deriver, so their (possibly irregular) stored analysis wins
    "devatā": ["devatā"],
    "śūnyatā": ["śūnyatā"],
    "laghutā": ["laghutā"],
    "madhuratā": ["madhuratā"],
    "śūnyatvam": ["śūnyatva"],
    "mahattvam": ["mahattva"],
}

## The -atā collision.  These keep their participle reading because the -at stem
## is attested; the deriver must decline them.
PARTICIPLES = {
    "gacchatā": ["gacchat"],
    "jayatā": ["jayat"],
}

## The -te collision, and the reason NON_LICENSING_ENDINGS exists.  -te is the
## ā-stem's Voc. Sg. and all three of its duals, but it is also the ending of
## every ātmanepada and every passive 3rd sg.  These verbs miss the verb tables
## and reach the deriver; if -te were allowed to license a derivation it would
## strip it, find the real nominal stem underneath, and bury a correct verbal
## root under a fabricated noun (śocate "he grieves" -> "śocatā").  The values are
## the outputs recorded before the deriver existed -- noisy, but with the right
## root in them.
ATMANEPADA = {
    "śocate": ["śuc", ("ta", "tā", "tad", "yuṣmad")],
    "nandate": [("nanda", "nand"), ("ta", "tā", "tad", "yuṣmad")],
    "pīḍate": ["pīḍa", ("ta", "tā", "tad", "yuṣmad")],
    "stanate": ["stana", ("ta", "tā", "tad", "yuṣmad")],
}


class TaddhitaParadigmTest(unittest.TestCase):
    """The generated paradigm must be byte-identical to the database's own."""

    def test_regenerates_lexicalised_tables_exactly(self):
        ## The strongest available correctness proof: for every derivative the
        ## database already stores, rebuilding the table from its base must
        ## reproduce the stored table form for form.  If the generator drifts
        ## from the database's conventions this fails loudly.
        with session_scope() as session:
            for stem, suffix in [
                ("śūnya-tā", "tā"),
                ("a-bhinna-tā", "tā"),
                ("a-karuṇa-tva", "tva"),
                ## a base that does not end in -a, to prove nothing assumes one
                ("a-kartṛ-tva", "tva"),
            ]:
                with self.subTest(stem=stem):
                    stored = taddhita_deriver.stored_paradigm(stem, suffix, session=session)
                    self.assertIsNotNone(stored, f"missing exemplar row for {stem}")
                    base = stem.replace("-", "")[: -len(suffix)]
                    generated = taddhita_deriver.paradigm(base, suffix, session=session)
                    self.assertEqual(generated, stored)

    def test_no_licensing_ending_is_a_tail_of_another(self):
        ## _base_of leans on this: because at most one ending can match a given
        ## word, the first hit is the only hit, so every rejection returns instead
        ## of trying the next ending.  A new suffix or a changed exemplar model
        ## could break the invariant silently, and the control flow would then
        ## quietly stop considering a legitimate longer match.
        with session_scope() as session:
            for spec in SUFFIXES:
                endings = taddhita_deriver._ending_set(spec.suffix, session=session)
                overlaps = [
                    (a, b)
                    for a in endings.licensing
                    for b in endings.licensing
                    if a != b and a.endswith(b)
                ]
                self.assertEqual(overlaps, [], f"-{spec.suffix} endings overlap")

    def test_paradigm_is_the_expected_shape(self):
        with session_scope() as session:
            forms = taddhita_deriver.paradigm("niṣyanda", "tā", session=session)
        self.assertEqual(len(forms), 24)
        self.assertEqual(forms[0], "niṣyandatā")  # Nom. Sg.
        self.assertIn("niṣyandatayā", forms)  # Inst. Sg.
        self.assertIn("niṣyandatānām", forms)  # Gen. Pl.


class TaddhitaFailureModeTest(unittest.TestCase):
    """The two ways this can go wrong quietly."""

    def test_a_missing_exemplar_is_fatal(self):
        ## Degrading silently here would look exactly like the bug the module
        ## fixes -- "this word has no analysis" -- so it must say so out loud.
        deriver = TaddhitaDeriver(
            suffixes=(TaddhitaSuffix(suffix="tā", model="f_A", exemplar="no-such-stem"),)
        )
        with session_scope() as session:
            with self.assertRaises(RuntimeError) as caught:
                deriver.derive("niṣyandatā", session=session)
        self.assertIn("update-ps-database", str(caught.exception))

    def test_the_ending_cache_is_keyed_on_the_database(self):
        ## The exemplar endings are memoised, so the key has to change when the
        ## lexicon does -- an externally provisioned database can be pointed
        ## somewhere else mid-process.  This must NOT be keyed on
        ## analysisCache.lexicon_fingerprint(): that helper is @lru_cache'd on its
        ## db_path argument, so called with no argument it freezes at its first
        ## answer and would never notice the swap.
        deriver = TaddhitaDeriver()
        with session_scope() as session:
            deriver.endings("tā", session=session)
        keys_before = set(deriver._paradigms)
        self.assertTrue(keys_before)

        real_get_db_path = databaseSetup.get_db_path
        databaseSetup.get_db_path = lambda: "/some/other/lexicon.sqlite"
        try:
            self.assertNotIn(deriver._lexicon_identity(), {k[0] for k in keys_before})
        finally:
            databaseSetup.get_db_path = real_get_db_path

        self.assertIn(deriver._lexicon_identity(), {k[0] for k in keys_before})

    def test_a_missing_session_is_not_reported_as_a_stale_database(self):
        ## A dropped argument used to be laundered into "the database has no such
        ## row", telling the user to re-download 583 MB to fix a programming
        ## error.  Whatever it raises now, it must not claim that.
        deriver = TaddhitaDeriver()
        with self.assertRaises(Exception) as caught:
            deriver.derive("niṣyandatā", session=None)
        self.assertNotIn("update-ps-database", str(caught.exception))


class TaddhitaDeriverTest(unittest.TestCase):
    """Unit level: does the rule fire, and on what."""

    def test_fires_on_productive_derivatives(self):
        with session_scope() as session:
            for surface, (lemma, tags) in TARGETS.items():
                with self.subTest(word=surface):
                    derived = taddhita_deriver.derive(surface, session=session)
                    self.assertIsNotNone(derived, f"{surface} should derive")
                    self.assertEqual(derived.lemma, lemma)
                    self.assertEqual(derived.base, "niṣyanda")
                    self.assertEqual(derived.tags, tags)
                    self.assertEqual(derived.surface, surface)

    def test_declines_when_the_at_stem_is_attested(self):
        ## gacchat / jayat are real stems, so -atā is their Inst. Sg. and the
        ## abstract-noun reading must not be manufactured on top of it.
        with session_scope() as session:
            for surface in PARTICIPLES:
                with self.subTest(word=surface):
                    self.assertIsNone(taddhita_deriver.derive(surface, session=session))

    def test_declines_a_bare_te_ending(self):
        ## -te may be generated into the paradigm but must never identify a
        ## derivative, or every ātmanepada verb in the language becomes an
        ## abstract noun.
        with session_scope() as session:
            for surface in ATMANEPADA:
                with self.subTest(word=surface):
                    self.assertIsNone(taddhita_deriver.derive(surface, session=session))

    def test_te_is_still_generated_into_the_paradigm(self):
        ## de-licensing it must not punch a hole in the table: the stored tables
        ## have -te in them, and test_regenerates_lexicalised_tables_exactly
        ## compares against those.
        with session_scope() as session:
            forms = taddhita_deriver.paradigm("niṣyanda", "tā", session=session)
        self.assertIn("niṣyandate", forms)

    def test_declines_when_the_base_is_not_a_word(self):
        ## The base carries the meaning; inventing one from an arbitrary string
        ## would let the deriver "explain" any word ending in -tā.
        with session_scope() as session:
            for surface in ("qqqxatā", "blorbatvam", "tā", "tvam"):
                with self.subTest(word=surface):
                    self.assertIsNone(taddhita_deriver.derive(surface, session=session))


class TaddhitaProcessTest(unittest.TestCase):
    """Integration level: what process() actually returns."""

    def test_roots_mode_returns_the_derived_lemma(self):
        for surface, (lemma, _tags) in TARGETS.items():
            with self.subTest(word=surface):
                self.assertEqual(process(surface, mode="roots"), [lemma])

    def test_detailed_mode_carries_the_bases_dictionary_entry(self):
        ## niṣyandatā is in no dictionary, but niṣyanda is -- and that is where
        ## its meaning lives.  The derived entry must therefore be glossed from
        ## the base, not come back empty.
        entries = process("niṣyandatā")
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry[0], "niṣyandatā")
        self.assertEqual(entry[2], [("Nom", "Sg")])
        self.assertEqual(len(entry), 7, "should match the canonical dict_search shape")
        glosses = entry[6]
        self.assertTrue(
            any(glosses.get(name) for name in glosses),
            f"expected a dictionary gloss inherited from the base: {glosses}",
        )

    def test_an_unglossed_base_still_yields_a_well_formed_entry(self):
        ## A base can be attested as an inflecting stem without heading a
        ## dictionary entry.  The gloss then comes back empty -- which is honest,
        ## the lexicon really has nothing to say -- but the entry SHAPE must still
        ## hold, or downstream code that trusts the 7-slot layout breaks.  Pinned
        ## so nobody later "fixes" the empty gloss into a fuzzy match that invents
        ## a meaning.
        entries = process("analānandatā")
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(len(entry), 7)
        self.assertEqual(entry[0], "analānandatā")
        self.assertEqual(process("analānandatā", mode="roots"), ["analānandatā"])

    def test_the_old_shattered_split_is_gone(self):
        ## The regression this whole module exists to prevent: the suffix must
        ## never again come back as a separate word analysed as a verb.
        roots = process("niṣyandatā", mode="roots")
        self.assertNotIn("niṣyanda", roots)
        flattened = [r for item in roots for r in (item if isinstance(item, tuple) else (item,))]
        for spurious in ("tā", "tṛ", "tan"):
            self.assertNotIn(spurious, flattened)

    def test_existing_analyses_are_untouched(self):
        for surface, expected in {**UNCHANGED, **PARTICIPLES, **ATMANEPADA}.items():
            with self.subTest(word=surface):
                self.assertEqual(process(surface, mode="roots"), expected)

    def test_atmanepada_verbs_keep_their_root(self):
        ## the sharper statement of the above: whatever noise surrounds it, the
        ## verbal root must still be in the answer and the fabricated abstract
        ## noun must not be.
        for surface, root in (("śocate", "śuc"), ("nandate", "nand"), ("pīḍate", "pīḍa")):
            with self.subTest(word=surface):
                roots = process(surface, mode="roots")
                flattened = [
                    r for item in roots for r in (item if isinstance(item, tuple) else (item,))
                ]
                self.assertIn(root, flattened)
                self.assertNotIn(surface[:-2] + "tā", flattened)


if __name__ == "__main__":
    unittest.main()
