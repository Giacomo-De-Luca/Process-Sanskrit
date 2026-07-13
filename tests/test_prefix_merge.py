"""Regression tests for the prefix re-joining block in `clean_results`.

When `root_any_word` resolves a word by stripping a prefix (sam + upekṣa),
`clean_results` tries to re-join the pieces and look the whole thing up, so that
a genuinely lexicalised form (sam + ādhi -> samādhi) is reported as one word
rather than as its parts.

The trap this pins: `dict_search` never returns None.  Fed a bare word it cannot
find, it comes back as a *stub* entry whose slot [2] is a list holding the word
itself, while a real hit carries a dict keyed by dictionary name.  A re-join that
only checks `is not None` therefore always "succeeds", and overwrites a correct
prefix analysis with a lookup failure for a headword that does not exist -- which
is how `samupekṣa` came back as nothing but itself.

See documentation/prefix-rejoin.md, which also records why the dict-at-[2] test
is only valid on the bare-word path.
"""

import unittest

from process_sanskrit import process
from process_sanskrit.functions.cleanResults import clean_results
from process_sanskrit.functions.dictionaryLookup import dict_search
from process_sanskrit.utils.databaseSetup import get_database_path

requires_lexicon = unittest.skipUnless(
    get_database_path().exists(), "packaged lexicon database is not installed"
)


def _is_dictionary_hit(entry):
    """A real bare-word dict_search hit carries a dict at [2]; a miss carries a list."""
    return isinstance(entry[2], dict)


def _flatten_roots(roots):
    """roots mode groups the stems of one original word into a tuple; flatten them."""
    return [
        root
        for group in roots
        for root in (group if isinstance(group, tuple) else (group,))
    ]


def _prefix_entries(prefix_stem, stem, original):
    """The entry list root_any_word hands clean_results for a stripped prefix."""
    return [
        [prefix_stem, "indeclinable (avyaya)", [("Nom", "Sg")], [prefix_stem], original],
        ["sa", "masculine noun/adjective ending in -a", [("Acc", "Sg")], ["saḥ"], original],
        [stem, "masculine noun/adjective ending in -a", [("Voc", "Sg")], [stem], original],
    ]


@requires_lexicon
class DictSearchStubShapeTests(unittest.TestCase):
    """The contract the re-join guard depends on."""

    def test_miss_returns_a_stub_rather_than_none(self):
        result = dict_search(["samupekṣa"])
        self.assertIsNotNone(result, "dict_search must not be probed with `is not None`")
        self.assertTrue(result)
        self.assertFalse(_is_dictionary_hit(result[0]))
        self.assertEqual(result[0][2], ["samupekṣa"])

    def test_hit_returns_a_dictionary_payload(self):
        self.assertTrue(_is_dictionary_hit(dict_search(["samādhi"])[0]))


@requires_lexicon
class SamPrefixRejoinTests(unittest.TestCase):
    def test_unattested_join_is_not_merged(self):
        """Unit-level: `samupekṣa` heads no entry, so the split must survive intact."""
        entries = _prefix_entries("sam", "upekṣa", "samupekṣa")
        roots = clean_results(entries, mode="roots")
        self.assertEqual(roots, [("sam", "sa", "upekṣa")])
        self.assertNotIn("samupekṣa", _flatten_roots(roots))

    def test_attested_join_is_still_merged(self):
        """Unit-level: the block must keep doing its job -- sam + ādhi is a real word."""
        entries = _prefix_entries("sam", "ādhi", "samādhi")
        entries.append(
            [
                "pariṇāma",
                "masculine noun/adjective ending in -a",
                [("Nom", "Sg")],
                ["pariṇāmaḥ"],
                "pariṇāmaḥ",
            ]
        )
        self.assertEqual(clean_results(entries, mode="roots"), ["samādhi", "pariṇāma"])

    def test_unattested_join_keeps_the_prefix_analysis(self):
        """End to end: sam + upekṣa reaches the caller instead of an invented headword."""
        flat = _flatten_roots(process("samupekṣa", mode="roots"))
        self.assertIn("upekṣa", flat)
        self.assertIn("sam", flat)
        self.assertNotIn(
            "samupekṣa", flat, "the non-existent headword must not replace the analysis"
        )

    def test_unattested_join_inside_a_compound(self):
        """The stub also used to strand the sibling stem it failed to consume."""
        flat = _flatten_roots(process("saṃyojanānyāvaraṇamudvegasamupekṣayoḥ", mode="roots"))
        for expected in ("saṃyojana", "āvaraṇa", "udvega", "upekṣa"):
            self.assertIn(expected, flat)
        self.assertNotIn("samupekṣa", flat)

    def test_lexicalised_prefixed_words_are_unaffected(self):
        """Words the forms DB already knows whole never reach the re-join at all."""
        self.assertEqual(process("samādhipariṇāmaḥ", mode="roots"), ["samādhi", "pariṇāma"])


if __name__ == "__main__":
    unittest.main()
