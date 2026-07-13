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
from process_sanskrit.functions.taddhitaDerivation import taddhita_deriver
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

    def test_paradigm_is_the_expected_shape(self):
        with session_scope() as session:
            forms = taddhita_deriver.paradigm("niṣyanda", "tā", session=session)
        self.assertEqual(len(forms), 24)
        self.assertEqual(forms[0], "niṣyandatā")  # Nom. Sg.
        self.assertIn("niṣyandatayā", forms)  # Inst. Sg.
        self.assertIn("niṣyandatānām", forms)  # Gen. Pl.


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

    def test_the_old_shattered_split_is_gone(self):
        ## The regression this whole module exists to prevent: the suffix must
        ## never again come back as a separate word analysed as a verb.
        roots = process("niṣyandatā", mode="roots")
        self.assertNotIn("niṣyanda", roots)
        flattened = [r for item in roots for r in (item if isinstance(item, tuple) else (item,))]
        for spurious in ("tā", "tṛ", "tan"):
            self.assertNotIn(spurious, flattened)

    def test_existing_analyses_are_untouched(self):
        for surface, expected in {**UNCHANGED, **PARTICIPLES}.items():
            with self.subTest(word=surface):
                self.assertEqual(process(surface, mode="roots"), expected)


if __name__ == "__main__":
    unittest.main()
