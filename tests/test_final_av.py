"""A final `-āv` is the sandhi spelling of `-au` before a vowel.

`samādhau eva` is written `samādhāv eva`.  No inflected form ends in `v`, and
the cascade used to read every final `-v` as `-u`, which turned `samādhāv` into
`samādhāu`, a form that does not exist either.  Now a final `-āv` reads `-au`
(`samādhau` -> `samādhi`), on the single-word path and on the split path, and a
final `-v` after any other sound still reads `-u` (`gurv` -> `guru`).

See documentation/final-av.md.  `cached=False` throughout.
"""

import unittest

from process_sanskrit import process
from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.utils.databaseSetup import session_scope
from process_sanskrit.utils.lexicalResources import read_final_sandhi
from process_sanskrit.utils.resourcePaths import get_database_path


def roots(text):
    return process(text, mode="roots", cached=False, input_scheme="IAST")


def lemmas(text):
    return [
        [(part["form"], part["lemmas"]) for part in item["parts"]]
        for item in process(text, mode="lemmas", cached=False, input_scheme="IAST")
    ]


class ReadingTests(unittest.TestCase):
    """`read_final_sandhi`: the last letter as the sound it stands for (no database)."""

    def test_final_av_reads_au(self):
        self.assertEqual(read_final_sandhi("samādhāv"), "samādhau")
        self.assertEqual(read_final_sandhi("nāv"), "nau")

    def test_any_other_final_v_reads_u(self):
        self.assertEqual(read_final_sandhi("gurv"), "guru")
        self.assertEqual(read_final_sandhi("tv"), "tu")

    def test_the_other_final_letters_read_as_before(self):
        self.assertEqual(read_final_sandhi("ity"), "iti")
        self.assertEqual(read_final_sandhi("punar"), "punaḥ")
        self.assertEqual(read_final_sandhi("yoga"), "yoga")
        self.assertEqual(read_final_sandhi(""), "")


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class SingleWordTests(unittest.TestCase):
    def test_a_locative_in_av_reaches_its_stem(self):
        self.assertEqual(roots("samādhāv"), ["samādhi"])
        self.assertEqual(roots("nidhāv"), ["nidhi"])

    def test_a_final_v_after_a_consonant_is_unchanged(self):
        self.assertEqual(roots("gurv"), ["guru"])
        self.assertEqual(roots("madhv"), ["madhu"])
        self.assertEqual(roots("tv"), ["tu"])

    def test_in_lemma_mode_the_typed_root_stays_a_lemma(self):
        """Monier-Williams has a root `samādhāv`; rule 8 keeps it beside the locative."""
        (part,), = lemmas("samādhāv")
        self.assertEqual(part[1], ["samādhi", "samādhāv"])
        (part,), = lemmas("dhāv")
        self.assertIn("dhāv", part[1])


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class SplitPathTests(unittest.TestCase):
    def test_a_split_token_in_av_reaches_its_stem(self):
        with session_scope() as session:
            found = root_any_word("samādhāv", session=session)
        self.assertEqual([entry[0] for entry in found], ["samādhi"])

    def test_in_a_sentence(self):
        self.assertEqual(roots("samādhāv eva"), ["samādhi", "eva"])

    def test_in_lemma_mode_running_text(self):
        result = lemmas("samādhāv eva")
        self.assertIn("samādhi", result[0][0][1])


if __name__ == "__main__":
    unittest.main()
