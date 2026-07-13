"""Bare dictionary pointers are fallback compound cuts, never first choice."""

import unittest
from unittest.mock import patch

from process_sanskrit.functions import compoundAnalysis
from process_sanskrit.functions.compoundAnalysis import (
    dict_word_iterative,
    evaluate_compound_split,
    root_compounds,
)
from process_sanskrit.functions.process import process
from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.utils.databaseSetup import session_scope
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES
from process_sanskrit.utils.resourcePaths import get_database_path


requires_lexicon = unittest.skipUnless(
    get_database_path().exists(), "packaged lexicon database is not installed"
)


@requires_lexicon
class LexiconAssumptions(unittest.TestCase):
    def test_tanni_is_a_flagged_dictionary_key_with_no_paradigm(self):
        self.assertIn("tanni", DICTIONARY_REFERENCES)
        self.assertTrue(DICTIONARY_REFERENCES.is_stub("tanni"))
        with session_scope() as session:
            self.assertIsNone(root_any_word("tanni", session=session))

    def test_true_first_members_are_analysable(self):
        with session_scope() as session:
            self.assertTrue(root_any_word("tan", session=session))
            self.assertTrue(root_any_word("tad", session=session))
            self.assertTrue(root_any_word("nirodha", session=session))

    def test_real_unanalysable_headwords_are_not_flagged(self):
        for word in ("niṣyanda", "gacchat", "cakṣūroga", "dhenikā"):
            with self.subTest(word=word):
                self.assertIn(word, DICTIONARY_REFERENCES)
                self.assertFalse(DICTIONARY_REFERENCES.is_stub(word))
        with session_scope() as session:
            self.assertIsNone(root_any_word("niṣyanda", session=session))


@requires_lexicon
class StubCutsAreDemoted(unittest.TestCase):
    def test_walk_does_not_stop_on_the_stub(self):
        with session_scope() as session:
            match = dict_word_iterative("tannirodha", session=session)
        self.assertIsNotNone(match)
        self.assertNotEqual(match[0], "tanni")

    def test_root_compounds_keeps_nirodha_whole(self):
        with session_scope() as session:
            self.assertNotIn("tanni", root_compounds("tannirodha", session=session))

    def test_stub_score_still_clears_the_acceptance_gate(self):
        with session_scope() as session:
            score = evaluate_compound_split("tanni", "rodha", session=session)
        self.assertGreaterEqual(score, 0.6)


@requires_lexicon
class PipelineOutput(unittest.TestCase):
    def test_tannirodha(self):
        roots = process("tannirodha", mode="roots")
        flat = [root[0] if isinstance(root, tuple) else root for root in roots]
        self.assertNotIn("tanni", flat)
        self.assertIn("nirodha", flat)

    def test_reported_long_compound(self):
        roots = process("ananyathāviparyāsatannirodhāryagocaraiḥ", mode="roots")
        flat = [root[0] if isinstance(root, tuple) else root for root in roots]
        self.assertNotIn("tanni", flat)
        self.assertIn("nirodha", flat)
        self.assertIn("viparyāsa", flat)

    def test_unanalysable_headword_survives(self):
        self.assertEqual(process("niṣyandaguṇa", mode="roots"), ["niṣyanda", "guṇa"])


class SyntheticPreferenceTests(unittest.TestCase):
    """Preference and numeric acceptance remain independent."""

    class References:
        def __init__(self, words, stubs):
            self.words = set(words)
            self.stubs = set(stubs)

        def __contains__(self, word):
            return word in self.words

        def is_stub(self, word):
            return word in self.stubs

    @staticmethod
    def roots_except_whole(whole):
        return lambda word, **_kwargs: word != whole

    def match(self, word, references):
        with patch.object(
            compoundAnalysis, "DICTIONARY_REFERENCES", references
        ), patch.object(
            compoundAnalysis,
            "root_any_word",
            side_effect=self.roots_except_whole(word),
        ):
            return dict_word_iterative(word)

    def test_genuine_candidate_outranks_higher_numeric_stub(self):
        references = self.References(
            words={"asax", "asa", "a"},
            stubs={"asax"},
        )
        self.assertEqual(self.match("asaxrest", references)[0], "asa")

    def test_sole_stub_remains_eligible_at_the_numeric_gate(self):
        references = self.References(words={"asa", "a"}, stubs={"asa"})
        self.assertEqual(self.match("asarest", references)[0], "asa")

    def test_ineligible_genuine_candidate_does_not_hide_eligible_stub(self):
        references = self.References(
            words={"asax", "asa", "a"},
            stubs={"asax"},
        )
        with patch.dict(compoundAnalysis.SANSKRIT_ENDINGS, {"sa": compoundAnalysis.EndingProperties(0.5, "test")}, clear=True):
            self.assertEqual(self.match("asaxrest", references)[0], "asax")


if __name__ == "__main__":
    unittest.main()
