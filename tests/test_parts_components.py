"""`mode="parts"` cuts a headword only where the dictionary marks a boundary.

Monier-Williams' `components` field uses three marks that mean three different
things:

    citta—vṛtti     `—`  boundary between the members of a compound
    ni-rodha        `-`  prefix or suffix boundary inside one word
    u/pa, agni/     `/`  accent, written after the accented vowel (úpa, agní)

`roots_splitted` used to cut on all three with one pattern, so every headword
accented anywhere but on its last vowel came back in pieces: `upa` as `u` + `pa`,
`tad` as `ta` + `d`.  A final accent hid the bug, because the piece after it is
empty and is discarded -- which is why `agni` always looked right.

The accent (and the svarita mark `^`) is not a boundary.  It is removed by the
letters-only clean-up that follows the cut.
"""

import unittest

from process_sanskrit import process
from process_sanskrit.functions.cleanResults import roots_splitted
from process_sanskrit.utils.resourcePaths import get_database_path


class ComponentMarkTests(unittest.TestCase):
    """`roots_splitted` on hand-built entries; needs no lexicon."""

    @staticmethod
    def parts(headword, components):
        return roots_splitted([[headword, components, {}]])[headword]

    def test_accent_inside_a_word_is_not_a_boundary(self):
        self.assertEqual(self.parts("upa", "u/pa"), ["upa"])
        self.assertEqual(self.parts("tad", "ta/d"), ["tad"])

    def test_accent_on_the_final_vowel_is_dropped(self):
        self.assertEqual(self.parts("agni", "agni/"), ["agni"])

    def test_svarita_mark_is_dropped(self):
        self.assertEqual(self.parts("akṣitavya", "akṣitavya^"), ["akṣitavya"])

    def test_compound_boundary_still_splits(self):
        self.assertEqual(self.parts("cittavṛtti", "citta—vṛtti"), ["citta", "vṛtti"])

    def test_prefix_boundary_still_splits(self):
        self.assertEqual(self.parts("nirodha", "ni-rodha"), ["ni", "rodha"])

    def test_accent_beside_a_boundary_leaves_the_boundary_alone(self):
        self.assertEqual(self.parts("niṣad", "ni-ṣa/d"), ["ni", "ṣad"])

    def test_alternative_accentuations_collapse_to_one_reading(self):
        """`,` separates two accentuations of one headword; the parts are the same."""
        self.assertEqual(self.parts("aghnya", "a/-ghnya,a-ghnya/"), ["a", "ghnya"])

    def test_seven_field_entry_reads_the_same_marks(self):
        entry = ["upa", "indeclinable (avyaya)", [("Nom", "Sg")], ["upa"], "upa", "u/pa", {}]
        self.assertEqual(roots_splitted([entry]), {"upa": ["upa"]})


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class PublicPartsTests(unittest.TestCase):
    """The same marks, read from the installed dictionaries through `process()`."""

    def test_accented_headwords_come_back_whole(self):
        self.assertEqual(process("upa", mode="parts", cached=False), {"upa": ["upa"]})
        self.assertEqual(process("tad", mode="parts", cached=False)["tad"], ["tad"])

    def test_compound_members_are_unchanged(self):
        self.assertEqual(
            process("cittavṛtti", mode="parts", cached=False),
            {"cittavṛtti": ["citta", "vṛtti"]},
        )


if __name__ == "__main__":
    unittest.main()
