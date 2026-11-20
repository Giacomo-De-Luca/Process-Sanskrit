"""`jj`: a double consonant of the word, or the sandhi of a final t/d before j.

`lajjā`, `rajju`, `ujjvala` and the roots `lajj`, `majj`, `ujjh` are spelled
with `jj`; `tajjapaḥ` is `tat` + `japaḥ`, a sandhi the splitter does not undo.
Text cleaning used to cut every `jj` as `j j`, so `lajjā` reached the lookups
as `laj` + `jā`.  Now a word keeps its `jj` when it resolves whole, and only a
word that does not is cut, as before (`WordCascade.read_double_j`).

See documentation/double-j.md.  `cached=False` throughout.
"""

import unittest
from unittest.mock import patch

from process_sanskrit import process
from process_sanskrit.functions.process import clean_iast
from process_sanskrit.functions.wordCascade import WordCascade
from process_sanskrit.utils.databaseSetup import session_scope
from process_sanskrit.utils.resourcePaths import get_database_path


def roots(text):
    return process(text, mode="roots", cached=False, input_scheme="IAST")


def parts(word):
    """The parts of a single word in lemma mode, as `(form, lemmas)` pairs."""
    result = process(word, mode="lemmas", cached=False, input_scheme="IAST")
    assert len(result) == 1, result
    return [(part["form"], part["lemmas"]) for part in result[0]["parts"]]


class CleaningTests(unittest.TestCase):
    def test_cleaning_leaves_a_double_j_alone(self):
        self.assertEqual(clean_iast("lajjā"), "lajjā")
        self.assertEqual(clean_iast("tajjapas tadarthabhāvanam"), "tajjapas tadarthabhāvanam")


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class ReadingTests(unittest.TestCase):
    def read(self, text):
        with session_scope() as session:
            return WordCascade(session=session, memo={}).read_double_j(text)

    def test_a_word_that_resolves_whole_keeps_its_double_j(self):
        for word in ("lajjā", "rajjuḥ", "ujjvalam", "sajjanaḥ", "majjati", "lajj", "ujjh"):
            with self.subTest(word=word):
                self.assertEqual(self.read(word), word)

    def test_a_word_that_does_not_is_cut_as_before(self):
        self.assertEqual(self.read("tajjapas"), "taj japas")
        self.assertEqual(self.read("tajjñānam"), "taj jñānam")

    def test_each_word_of_a_text_is_read_on_its_own(self):
        self.assertEqual(self.read("lajjā tajjapas"), "lajjā taj japas")

    def test_each_segment_of_a_pre_split_word_is_read_on_its_own(self):
        self.assertEqual(self.read("lajjā-tajjapas"), "lajjā-taj japas")

    def test_a_text_without_double_j_is_never_looked_up(self):
        with patch.object(WordCascade, "resolve", side_effect=AssertionError("looked up")):
            self.assertEqual(self.read("yogaś cittavṛttinirodhaḥ"), "yogaś cittavṛttinirodhaḥ")

    def test_a_pattern_is_a_headword_query_and_keeps_its_double_j(self):
        with patch.object(WordCascade, "resolve", side_effect=AssertionError("looked up")):
            for pattern in ("lajj%", "laj_jā", "lajj*"):
                with self.subTest(pattern=pattern):
                    self.assertEqual(self.read(pattern), pattern)

    def test_in_running_text_a_pattern_character_makes_no_pattern(self):
        """Only a one-word query is a pattern; in running text the word is read as any other."""
        self.assertEqual(self.read("tajjapas% tadartha"), "taj japas% tadartha")

    def test_the_decision_leaves_the_request_memo_alone(self):
        """A filled memo hands back copies whose tags are lists, not tuples."""
        with session_scope() as session:
            memo = {}
            WordCascade(session=session, memo=memo).read_double_j("lajjā")
        self.assertEqual(memo, {})


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class DetailedOutputTests(unittest.TestCase):
    def test_a_double_j_word_has_tags_of_the_same_type_as_any_other(self):
        lajja = process("lajjā", cached=False, input_scheme="IAST")
        rama = process("rāmaḥ", cached=False, input_scheme="IAST")
        self.assertEqual(type(lajja[0][2][0]), type(rama[0][2][0]))

    def test_running_text_with_a_pattern_character_is_cut_as_before(self):
        result = roots("tajjapas% tadarthabhāvanam")
        self.assertEqual(result[0], "tad")


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class RootsModeTests(unittest.TestCase):
    def test_words_spelled_with_double_j(self):
        self.assertEqual(roots("lajjā"), ["lajjā"])
        self.assertEqual(roots("lajjate"), ["lajj"])
        self.assertEqual(roots("ujjvalam"), ["ujjvala"])
        self.assertEqual(roots("sajjanaḥ"), ["sajjana"])
        self.assertIn("rajju", roots("rajjuḥ"))
        self.assertIn("majj", roots("majjati"))
        self.assertIn("ujjh", roots("ujjhati"))

    def test_the_sandhi_of_t_before_j_is_still_undone(self):
        self.assertEqual(roots("tajjapas"), ["tad", "japa"])
        self.assertEqual(roots("tajjñānam"), ["tad", "jñāna"])

    def test_in_a_sentence(self):
        self.assertIn("lajj", roots("sa lajjate"))
        result = roots("tajjapas tadarthabhāvanam")
        self.assertEqual(result[:2], ["tad", "japa"])

    def test_in_a_text_analysed_word_by_word(self):
        """Longer than `max_length`, so `_process_each` analyses each word."""
        text = "sa lajjate tajjapas " + " ".join(["yogaścittavṛttinirodhaḥ"] * 6)
        result = process(text, mode="roots", cached=False, input_scheme="IAST", max_length=40)
        self.assertIn("lajj", result)
        self.assertIn("japa", result)


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class LemmaModeTests(unittest.TestCase):
    def test_words_spelled_with_double_j(self):
        self.assertEqual(parts("lajjā"), [("lajjā", ["lajjā"])])
        self.assertEqual(parts("ujjvalam"), [("ujjvalam", ["ujjvala"])])

    def test_the_sandhi_of_t_before_j_is_still_undone(self):
        (first, first_lemmas), (second, second_lemmas) = parts("tajjapas")
        self.assertIn("tad", first_lemmas)
        self.assertEqual(second_lemmas, ["japa"])

    def test_a_starred_query_that_finds_no_headword_is_read_as_a_word(self):
        """`tajjapas*` names no headword, so it falls back to the word `tajjapas`."""
        (first, first_lemmas), (second, second_lemmas) = parts("tajjapas*")
        self.assertIn("tad", first_lemmas)
        self.assertEqual(second_lemmas, ["japa"])

    def test_a_compound_keeps_the_double_j_of_its_member_when_the_splitter_does(self):
        # The token does not resolve whole, so it is cut as before; the
        # splitter is then handed `laj jāvismaya...` and cannot do better.
        # Pinned so that a change here is noticed, not because it is right.
        forms = [form for form, _ in parts("lajjāvismayamanmathaiḥ")]
        self.assertEqual(forms[0], "laj")


if __name__ == "__main__":
    unittest.main()
