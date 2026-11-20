"""CSL verb tables store third, second, then first person, each Sg/Du/Pl."""

import unittest

from process_sanskrit import process
from process_sanskrit.utils.paradigm import VERBAL_PERSONS, tags_for
from process_sanskrit.utils.resourcePaths import get_database_path


# The bhū present active paradigm in CSL's stored order. Expected tags are
# explicit grammatical readings, independent of the parser's row arithmetic.
PRESENT_BHU = (
    ("bhavati", ("Third", "Sg")),
    ("bhavataḥ", ("Third", "Du")),
    ("bhavanti", ("Third", "Pl")),
    ("bhavasi", ("Second", "Sg")),
    ("bhavathaḥ", ("Second", "Du")),
    ("bhavatha", ("Second", "Pl")),
    ("bhavāmi", ("First", "Sg")),
    ("bhavāvaḥ", ("First", "Du")),
    ("bhavāmaḥ", ("First", "Pl")),
)


class VerbPersonTests(unittest.TestCase):
    def test_present_paradigm_covers_every_person_and_number(self):
        forms = [word for word, _ in PRESENT_BHU]
        for word, expected in PRESENT_BHU:
            with self.subTest(word=word):
                self.assertEqual(
                    tags_for(forms, word, rows=VERBAL_PERSONS), [expected]
                )


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class PublicVerbPersonTests(unittest.TestCase):
    def test_public_morphology_preserves_person_across_tenses_and_scripts(self):
        cases = PRESENT_BHU + (
            ("abhavat", ("Third", "Sg")),
            ("abhavam", ("First", "Sg")),
            ("bhaviṣyati", ("Third", "Sg")),
            ("bhaviṣyāmi", ("First", "Sg")),
            ("भवति", ("Third", "Sg")),
        )
        for word, expected in cases:
            with self.subTest(word=word):
                # Some surfaces also have nominal/dictionary readings.
                verbs = [
                    entry for entry in process(word, cached=False)
                    if entry[0] == "bhū" and entry[1].startswith("verb,")
                ]
                self.assertTrue(verbs)
                for entry in verbs:
                    self.assertEqual(entry[2], [expected])


if __name__ == "__main__":
    unittest.main()
