"""`process()` never cuts its input.

`preprocess` used to truncate the text at `max_length` (100 characters by
default) without a word to the caller, so a passage lost most of its words: the
first fifteen sūtras came back as 12 lemmas.  Nothing is truncated any more:

* a text longer than `max_length` is analysed word by word and the results are
  combined;
* a single word is analysed whole, whatever its length.

`max_length` therefore no longer limits what is analysed, only how: it is the
longest text handed to the splitter as one unit.

See documentation/long-input.md.  `cached=False` throughout.
"""

import unittest

from process_sanskrit import process
from process_sanskrit.functions.process import preprocess
from process_sanskrit.utils.resourcePaths import get_database_path
from tests.datasets.yogaSutra import ys


PASSAGE = " ".join(line.strip() for line in ys[:15])


class PreprocessDoesNotCutTests(unittest.TestCase):
    """Needs no lexicon."""

    def test_whole_text_is_returned(self):
        cleaned = preprocess(PASSAGE)
        self.assertGreater(len(cleaned), 400)
        self.assertTrue(cleaned.endswith(preprocess(PASSAGE.split()[-1])))

    def test_long_single_word_is_returned_whole(self):
        word = "dharma" * 40
        self.assertEqual(preprocess(word), word)


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class LongTextFallsBackToWordByWordTests(unittest.TestCase):
    def setUp(self):
        self.assertGreater(len(preprocess(PASSAGE)), 100)
        self.words = preprocess(PASSAGE).split()

    def test_roots_are_the_words_roots_in_order(self):
        expected = []
        for word in self.words:
            expected.extend(process(word, mode="roots", cached=False))
        self.assertEqual(process(PASSAGE, mode="roots", cached=False), expected)

    def test_parts_are_the_words_parts_merged(self):
        expected = {}
        for word in self.words:
            expected.update(process(word, mode="parts", cached=False))
        self.assertEqual(process(PASSAGE, mode="parts", cached=False), expected)

    def test_detailed_entries_are_the_words_entries_in_order(self):
        expected = []
        for word in self.words:
            expected.extend(entry[0] for entry in process(word, cached=False))
        self.assertEqual([entry[0] for entry in process(PASSAGE, cached=False)], expected)

    def test_last_word_is_analysed(self):
        last = process(self.words[-1], mode="roots", cached=False)
        self.assertTrue(last)
        self.assertEqual(process(PASSAGE, mode="roots", cached=False)[-len(last):], last)


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class MaxLengthMeaningTests(unittest.TestCase):
    """`max_length` chooses between one unit and word by word; it never cuts."""

    TEXT = "tatra tatra"

    def test_text_within_the_limit_is_analysed_as_one_unit(self):
        """As one unit the two identical words are folded, as they always were."""
        self.assertEqual(
            process(self.TEXT, mode="roots", cached=False, max_length=len(self.TEXT)),
            ["tatra"],
        )

    def test_text_over_the_limit_is_analysed_word_by_word(self):
        self.assertEqual(
            process(self.TEXT, mode="roots", cached=False, max_length=len(self.TEXT) - 1),
            ["tatra", "tatra"],
        )

    def test_small_limit_loses_nothing(self):
        roots = process("yogaḥ cittavṛtti nirodhaḥ", mode="roots", cached=False, max_length=5)
        self.assertEqual(roots[-1], "nirodha")

    def test_single_word_over_the_limit_is_not_cut(self):
        word = "yogaścittavṛttinirodhaḥ"
        self.assertEqual(
            process(word, mode="roots", cached=False, max_length=5),
            process(word, mode="roots", cached=False),
        )

    def test_empty_result_shapes_are_unchanged(self):
        self.assertEqual(process("", mode="roots", cached=False), "")
        self.assertEqual(process("", mode="parts", cached=False), [])
        self.assertEqual(process("", cached=False), [])

    def test_length_is_measured_before_cleaning_as_the_old_cut_measured_it(self):
        """`ko 'pi tatra tatra` is 18 characters and cleans to 19 (`kaḥ api ...`).

        A text that was short enough to be analysed as one unit must still be:
        measuring the cleaned text instead sent it word by word, and changed the
        result of an input the old cut never touched.
        """
        text = "ko 'pi tatra tatra"
        self.assertEqual(
            process(text, mode="roots", cached=False, max_length=len(text)),
            ["ka", "api", "tatra"],
        )
        self.assertEqual(
            process(text, mode="roots", cached=False, max_length=len(text) - 1).count("tatra"), 2
        )


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class WildcardsStayOutOfRunningTextTests(unittest.TestCase):
    """A pattern is a one-word query; word by word must not turn noise into one."""

    LONG = " ".join(["tatra"] * 30) + " 50 % eti dharm_ kṣetre"

    def test_stray_percent_does_not_match_every_headword(self):
        self.assertGreater(len(self.LONG), 100)
        self.assertLess(len(process(self.LONG, cached=False)), 100)

    def test_pattern_characters_are_not_roots(self):
        roots = process(self.LONG, mode="roots", cached=False)
        self.assertNotIn("%", roots)
        self.assertFalse([root for root in roots if isinstance(root, str) and "_" in root])

    def test_one_word_pattern_still_works(self):
        self.assertIn("dharma", process("dharm_", cached=False)[0][2]["mw"])


class PreprocessSignatureTests(unittest.TestCase):
    """Needs no lexicon."""

    def test_a_positional_second_argument_is_refused(self):
        """It used to be `max_length`; read as `debug` it would be silently wrong."""
        with self.assertRaises(TypeError):
            preprocess("dharma", 150)


if __name__ == "__main__":
    unittest.main()
