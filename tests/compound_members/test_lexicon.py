"""Spelling keys and DCS join spellings (scripts/compound_members/lexicon.py)."""

import random
import unittest

from scripts.compound_members.lexicon import JoinSpellings, Spelling

from .support import has_database, shared_resources


class SpellingTests(unittest.TestCase):
    ## the letters of a string are `ComponentMarks.letters`, tested with it in
    ## test_lemma_builder.py (`ComponentMarkLetterTests`)

    def test_key_folds_a_nasal_before_a_consonant_into_anusvara(self):
        self.assertEqual(Spelling.key("nistaraṃga"), Spelling.key("nistaraṅga"))
        self.assertEqual(Spelling.key("saṃyoga"), Spelling.key("samyoga"))

    def test_key_reads_a_final_visarga_as_s(self):
        self.assertEqual(Spelling.key("āyuḥ"), Spelling.key("āyus"))

    def test_key_folds_retroflexion_that_sandhi_decides(self):
        self.assertEqual(Spelling.key("niṣphala"), Spelling.key("niḥphala"))
        self.assertEqual(Spelling.key("rāmāyaṇa"), Spelling.key("rāmāyana"))

    def test_key_keeps_different_words_apart(self):
        self.assertNotEqual(Spelling.key("tad"), Spelling.key("taj"))
        self.assertNotEqual(Spelling.key("mahā"), Spelling.key("maha"))

    def test_keys_of_groups_spellings(self):
        self.assertEqual(Spelling.keys_of(["saṃyoga", "samyoga", "tad"]),
                         {"saṃyoga": ("samyoga", "saṃyoga"), "tad": ("tad",)})


class JoinSpellingsTests(unittest.TestCase):
    def test_a_segment_is_joined_as_written_and_in_pausa(self):
        self.assertEqual(JoinSpellings.of("tad")[0], "tad")
        self.assertIn("tat", JoinSpellings.of("tad"))

    def test_a_final_visarga_is_also_joined_as_r_or_s(self):
        self.assertIn("antar", JoinSpellings.of("antaḥ"))
        self.assertIn("manas", JoinSpellings.of("manaḥ"))


@unittest.skipUnless(has_database(), "lexicon database is not installed")
class BulkReadsTests(unittest.TestCase):
    """The tables the evaluation reads in bulk agree with the library's lookups."""

    @classmethod
    def setUpClass(cls):
        cls.resources = shared_resources()

    def test_the_tables_give_what_headword_components_gives(self):
        from process_sanskrit.functions.dictionaryLookup import headword_components

        tables = self.resources.tables
        for headword in random.Random(13).sample(sorted(tables.headwords()), 500):
            for dictionary in ("mw", "bhs", "gra"):
                self.assertEqual(tables.analyses(dictionary, headword),
                                 headword_components(headword, dictionary, session=self.resources.session),
                                 (dictionary, headword))

    def test_the_headword_table_is_dictionary_references(self):
        from process_sanskrit.functions.lemmaAnalysis import LemmaLexicon
        from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES

        lexicon = LemmaLexicon(self.resources.session)
        headwords = self.resources.headwords
        for word in ("ālayavijñāna", "tman", "ṛṣi", "rṣi", "taj", "sarvābharaṇa"):
            self.assertEqual(headwords.sanskrit(word), lexicon.has_sanskrit_headword(word), word)
            self.assertEqual(headwords.breadth(word), len(DICTIONARY_REFERENCES.get(word, ())), word)
            self.assertEqual(headwords.in_monier_williams(word), "mw" in DICTIONARY_REFERENCES.get(word, ()), word)


if __name__ == "__main__":
    unittest.main()
