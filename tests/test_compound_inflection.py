"""Compound fallback pieces must retain morphology before dictionary lookup."""

import unittest

from process_sanskrit.functions.inflect import inflect
from process_sanskrit.functions.process import process
from process_sanskrit.utils.databaseSetup import session_scope
from process_sanskrit.utils.resourcePaths import get_database_path


COMPOUND = "sarvatragāminīpratipajjñānabalam"


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class CompoundInflectionTests(unittest.TestCase):
    def test_fallback_resolves_inflected_components(self):
        # Exercise the morphology fallback independently of statistical ranking.
        for word, lemma, surface in (
            ("jñānabalam", "bala", "balam"),
            ("sarvatragāminīpratipaj", "pratipad", "pratipat"),
        ):
            with self.subTest(word=word), session_scope() as session:
                entries = inflect([word], session=session)
                matches = [entry for entry in entries if isinstance(entry, list)
                           and entry[0] == lemma]
                self.assertTrue(matches, entries)
                self.assertTrue(all(len(entry) == 5 for entry in matches))
                self.assertTrue(all(entry[4] == surface for entry in matches))

    def test_dictionary_only_component_survives_fallback(self):
        with session_scope() as session:
            entries = inflect(["niṣyandaguṇa"], session=session)
        self.assertIn("niṣyanda", entries)
        self.assertTrue(any(isinstance(entry, list) and entry[0] == "guṇa"
                            for entry in entries), entries)

    def test_reported_compound_has_populated_lemmas(self):
        entries = process(COMPOUND, cached=False)
        self.assertEqual(
            list(dict.fromkeys(entry[0] for entry in entries)),
            ["sarvatragāmin", "pratipad", "jñāna", "bala"],
        )
        for entry in entries:
            with self.subTest(lemma=entry[0]):
                self.assertEqual(len(entry), 7)
                self.assertIsInstance(entry[-1], dict)
                self.assertTrue(any(entry[-1].values()))
        bala = [entry for entry in entries if entry[0] == "bala"]
        self.assertTrue(any(entry[1].startswith("neuter") and
                            ["Acc", "Sg"] in [list(tag) for tag in entry[2]]
                            for entry in bala))
        self.assertTrue(all(entry[4] == "balam" for entry in bala))

    def test_reported_compound_roots(self):
        self.assertEqual(process(COMPOUND, mode="roots", cached=False),
                         ["sarvatragāmin", "pratipad", "jñāna", "bala"])

    def test_balam_still_resolves_on_its_own(self):
        self.assertEqual(process("balam", mode="roots", cached=False), ["bala"])

    def test_existing_jj_sandhi_analysis_survives(self):
        self.assertEqual(process("tajjñānam", mode="roots", cached=False),
                         ["tad", "jñāna"])


if __name__ == "__main__":
    unittest.main()
