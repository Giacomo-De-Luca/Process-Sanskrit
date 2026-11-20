"""The verb roots `i` and `ṛ` are words, not one-letter noise.

`process()` drops every one-letter lemma on its split path.  The filter is a guard
against the compound scanner (`root_compounds`), which walks an unresolved token
one character at a time -- and every single vowel is a dictionary headword.  But it
tested the *lemma's* length, and two real verb roots are one letter long, so a
sentence lost its verb:

    process("eti")        -> ['eti', 'i']     single-word path, no filter
    process("tatra eti")  -> ['tatra']        split path, `i` filtered away

Two clauses decide what is kept, and each has a test class below:

* **verbal analyses only.**  The forms table also files `i` and `ṛ` as nouns (the
  letter names); those are still noise.
* **whole tokens only.**  A root is kept when it resolves a token the splitter
  produced, not when the scanner cut it out of a token the lexicon does not know.
  `arcayet` is such a token: the scanner leaves the tail `ayet`, which happens to
  be an optative of `i`.

`cached=False` unless the cache is what is under test: the assertions are about
what the pipeline computes.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from process_sanskrit import process
from process_sanskrit.functions.compoundAnalysis import root_compounds
from process_sanskrit.functions.inflect import inflect
from process_sanskrit.utils.analysisCache import reset_analysis_cache
from process_sanskrit.utils.databaseSetup import session_scope
from process_sanskrit.utils.paradigm import is_one_letter_root, is_verbal_paradigm
from process_sanskrit.utils.resourcePaths import get_database_path


VERBAL_FORMS = ["eti", "itaḥ", "yanti", "eṣi", "ithaḥ", "itha", "emi", "ivaḥ", "imaḥ"]
NOMINAL_FORMS = ["x"] * 24


def lemmas(roots):
    """Every lemma in a `mode="roots"` result, alternatives included."""
    flat = []
    for item in roots:
        flat.extend(item if isinstance(item, tuple) else (item,))
    return flat


class VerbalParadigmTests(unittest.TestCase):
    """A verbal table is three persons by three numbers; needs no lexicon."""

    def test_nine_cells_is_verbal(self):
        self.assertTrue(is_verbal_paradigm(VERBAL_FORMS))

    def test_nominal_and_indeclinable_tables_are_not(self):
        self.assertFalse(is_verbal_paradigm(NOMINAL_FORMS))
        self.assertFalse(is_verbal_paradigm(["upa"]))

    def test_a_missing_table_is_not(self):
        self.assertFalse(is_verbal_paradigm(None))


class OneLetterRootPredicateTests(unittest.TestCase):
    """`is_one_letter_root` on hand-built entries; needs no lexicon."""

    verb = ["i", "verb, present tense", [("Third", "Sg")], VERBAL_FORMS, "eti"]
    noun = ["i", "masculine noun/adjective ending in -i", [("Voc", "Sg")], NOMINAL_FORMS, "e"]

    def test_one_letter_verb_root(self):
        self.assertTrue(is_one_letter_root(self.verb))

    def test_entry_extended_by_the_dictionary_lookup(self):
        self.assertTrue(is_one_letter_root(self.verb + ["i", {"mw": {}}]))

    def test_one_letter_nominal_lemma_is_not(self):
        self.assertFalse(is_one_letter_root(self.noun))

    def test_longer_verb_root_is_not(self):
        self.assertFalse(is_one_letter_root(["gam"] + self.verb[1:]))

    def test_unresolved_tokens_are_not(self):
        self.assertFalse(is_one_letter_root("i"))
        self.assertFalse(is_one_letter_root(["i", "i", ["i"]]))


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class RootSurvivesTheSplitPathTests(unittest.TestCase):
    """A one-letter verb root that resolves a whole token is kept."""

    def test_root_i_beside_another_word(self):
        self.assertEqual(process("tatra eti", mode="roots", cached=False), ["tatra", "i"])

    def test_root_r_beside_another_word(self):
        self.assertEqual(process("tatra ṛcchati", mode="roots", cached=False), ["tatra", "ṛ"])

    # Needs the split ranker (split-ranker branch): without it the statistical
    # split is replaced by root_compounds, which reads `eti` as `iti`. When the
    # ranker merges this passes and must lose the marker.
    @unittest.expectedFailure
    def test_root_i_after_a_preverb(self):
        self.assertEqual(process("paryeti", mode="roots", cached=False), ["pari", "i"])

    # Needs the split ranker (split-ranker branch); see test_root_i_after_a_preverb.
    @unittest.expectedFailure
    def test_root_i_inside_a_sandhied_token(self):
        self.assertEqual(
            process("prasādamanveti", mode="roots", cached=False), ["prasāda", "anu", "i"]
        )
        self.assertIn("i", lemmas(process("māmavehi", mode="roots", cached=False)))


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class ConsequencesOfKeepingTheRootTests(unittest.TestCase):
    """What else changes once the root is no longer filtered away.

    Neither is a separate rule: both are what the single-word path already did,
    now reached on the split path too.
    """

    def test_a_word_in_a_sentence_reads_as_it_does_alone(self):
        """`itaḥ` is `ita` and a dual of `i`; the sentence used to lose the second."""
        alone = process("itaḥ", mode="roots", cached=False)
        self.assertEqual(alone, [("ita", "i")])
        self.assertEqual(process("tatra itaḥ", mode="roots", cached=False), ["tatra"] + alone)

    def test_kept_root_takes_part_in_the_prefix_rejoin(self):
        """`sam` + `i` re-joins to the attested headword, as `sam` + `gam` does."""
        self.assertEqual(process("tatra sameti", mode="roots", cached=False), ["tatra", "sami"])


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class NoiseIsStillFilteredTests(unittest.TestCase):
    """What the filter was written to remove is still removed."""

    def setUp(self):
        context = session_scope()
        self.session = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def scanner_finds_a_one_letter_root(self, token):
        pieces = root_compounds(token, inflection=True, session=self.session, _memo={})
        return any(is_one_letter_root(piece) for piece in pieces)

    def test_scanner_fragment_does_not_introduce_a_root(self):
        # Without this precondition the test passes for the wrong reason the day
        # the lexicon learns `arcayet` and the scanner is never reached.
        self.assertTrue(self.scanner_finds_a_one_letter_root("arcayet"))
        for text in ("śārṅgamarcayet", "svatantratvādvibhāgamavabhāsayet"):
            with self.subTest(text=text):
                self.assertNotIn("i", lemmas(process(text, mode="roots", cached=False)))

    def test_one_letter_nominal_lemmas_are_still_dropped(self):
        roots = lemmas(process("ājagmurdevagandharvā", mode="roots", cached=False))
        self.assertEqual([lemma for lemma in roots if len(lemma) == 1], [])

    def test_inflector_tells_a_whole_token_from_a_fragment(self):
        self.assertTrue(self.scanner_finds_a_one_letter_root("arcayet"))
        whole = inflect(["eti"], session=self.session)
        fragments = inflect(["arcayet"], session=self.session)
        self.assertTrue(any(is_one_letter_root(entry) for entry in whole))
        self.assertFalse(any(is_one_letter_root(entry) for entry in fragments))

    def test_token_left_with_no_pieces_is_reported_unresolved(self):
        """When the scanner finds nothing but a withheld root, the token survives.

        It used to vanish from the result altogether: the root was emitted, then
        filtered.  An unknown word now comes back as itself, the same as a token
        the scanner finds nothing in at all.
        """
        self.assertTrue(self.scanner_finds_a_one_letter_root("ñiyāt"))
        self.assertEqual(inflect(["ñiyāt"], session=self.session), ["ñiyāt"])
        self.assertEqual(
            process("tatra ñiyāt", mode="roots", cached=False), ["tatra", "ñiyāt"]
        )


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class CacheRoundTripTests(unittest.TestCase):
    """The filter runs after the cache is read, so a hit must keep the root too."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.cache_path = Path(self.temporary_directory.name) / "roots.sqlite3"
        reset_analysis_cache()
        self.addCleanup(reset_analysis_cache)

    def test_root_survives_a_cache_hit(self):
        from process_sanskrit.functions import hybridSplitter

        with patch.dict(os.environ, {"PROCESS_SANSKRIT_CACHE_PATH": str(self.cache_path)}), patch.object(
            hybridSplitter, "analyze_hybrid", wraps=hybridSplitter.analyze_hybrid
        ) as split:
            cold = process("tatra eti", mode="roots", cached=True)
            split.side_effect = AssertionError("cache hit recomputed the split")
            warm = process("tatra eti", mode="roots", cached=True)
        self.assertEqual(cold, ["tatra", "i"])
        self.assertEqual(warm, cold)


if __name__ == "__main__":
    unittest.main()
