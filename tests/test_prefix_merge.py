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
from process_sanskrit.functions.cleanResults import REJOINABLE_PREFIXES, clean_results
from process_sanskrit.functions.dictionaryLookup import dict_search
from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.utils.databaseSetup import get_database_path, session_scope

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


def _entry(stem, original):
    """One entry, for the prefixes whose stripping produces no homograph filler."""
    return [stem, "masculine noun/adjective ending in -a", [("Voc", "Sg")], [stem], original]


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

    def test_join_relies_on_dict_search_to_fold_sam_into_saṃ(self):
        """`samyoga` heads no entry of its own: dict_search maps sam -> saṃ (samMap).

        The block must not re-implement that fold.  An earlier attempt to do so
        keyed on `'MW'` while the payload is keyed `'mw'`, so it never ran; had it
        run it would have passed a bare string to a dict_search that wants a list.

        The fold reaches the emitted lemma too -- see CanonicalLemmaTests.
        """
        entries = _prefix_entries("sam", "yoga", "samyoga")
        self.assertEqual(clean_results(entries, mode="roots"), ["saṃyoga"])


@requires_lexicon
class CanonicalLemmaTests(unittest.TestCase):
    """The merged lemma is the dictionary's headword, not the query spelling.

    `dict_search` folds sam -> saṃ (samMap) for the *lookup* but echoes the query
    back at slot [0].  Left alone, the re-join therefore emitted `samvedana` while
    both other authorities in the pipeline say `saṃvedana`: the forms DB agrees
    (`root_any_word("samvedana") -> saṃvedana`) and so does Monier-Williams, which
    files the entry under `saṃvedana`.  Adopting the headword out of the payload we
    already hold costs no extra lookup and stops the re-join being the only path
    that disagrees with the other two.
    """

    def test_merged_lemma_adopts_the_dictionary_headword(self):
        entries = _prefix_entries("sam", "vedana", "samvedana")
        self.assertEqual(clean_results(entries, mode="roots"), ["saṃvedana"])

    def test_the_headword_is_what_root_any_word_independently_says(self):
        """Not a new convention -- the forms DB already canonicalises the same way."""
        with session_scope() as session:
            for folded, canonical in (
                ("samvedana", "saṃvedana"),
                ("samkramaṇa", "saṃkramaṇa"),
                ("samtāra", "saṃtāra"),
            ):
                with self.subTest(word=folded):
                    stems = [m[0] for m in root_any_word(folded, session=session)]
                    self.assertEqual(stems, [canonical])

    def test_a_headword_that_needs_no_fold_is_untouched(self):
        """`sam-` really is canonical before a vowel: samādhi must not become saṃādhi."""
        entries = _prefix_entries("sam", "ādhi", "samādhi")
        self.assertEqual(clean_results(entries, mode="roots"), ["samādhi"])

    def test_non_sam_prefixes_keep_their_headword(self):
        for prefix, stem, expected in (
            ("ava", "graha", "avagraha"),
            ("anu", "graha", "anugraha"),
            ("duḥ", "kha", "duḥkha"),
        ):
            with self.subTest(prefix=prefix):
                entries = [_entry(prefix, expected), _entry(stem, expected)]
                self.assertEqual(clean_results(entries, mode="roots"), [expected])


@requires_lexicon
class AnuPrefixRejoinTests(unittest.TestCase):
    """`anu` is the block that was already correct; it is the reference behaviour."""

    def test_attested_join_is_merged(self):
        entries = [_entry("anu", "anugraha"), _entry("graha", "anugraha")]
        self.assertEqual(clean_results(entries, mode="roots"), ["anugraha"])

    def test_unattested_join_is_not_merged(self):
        entries = [_entry("anu", "anuupekṣa"), _entry("upekṣa", "anuupekṣa"), _entry("x", "x")]
        roots = clean_results(entries, mode="roots")
        self.assertNotIn("anuupekṣa", _flatten_roots(roots))


@requires_lexicon
class PrefixFillerContractTests(unittest.TestCase):
    """REJOINABLE_PREFIXES must absorb exactly the stems root_any_word emits.

    The re-join walks from the prefix to the real stem by stepping over the
    homographs root_any_word also finds for the prefix itself.  If that set is
    wrong the walk stops on a filler and the join is silently lost: dropping `av`
    from `ava` turns `avaruhya` from `avaruh` into an unmerged `ava` + `av`.
    """

    def test_absorbed_sets_match_what_root_any_word_emits(self):
        with session_scope() as session:
            for prefix, absorbed in REJOINABLE_PREFIXES.items():
                with self.subTest(prefix=prefix):
                    ## `or []` is not defensive padding: root_any_word("duḥ") really
                    ## is None -- duḥ is no upasarga, it only ever reaches the entry
                    ## list as a compound cut -- so it contributes no fillers and its
                    ## absorbed set is legitimately empty.
                    emitted = {m[0] for m in (root_any_word(prefix, session=session) or [])}
                    self.assertEqual(
                        emitted,
                        set(absorbed),
                        f"root_any_word({prefix!r}) emits {sorted(emitted)}; the re-join "
                        f"absorbs {sorted(absorbed)}. A filler missing from the absorbed "
                        f"set stops the walk short and the join is lost.",
                    )


@requires_lexicon
class DuhkhaRejoinTests(unittest.TestCase):
    """`duḥ` + `kha` is a re-join with no fillers, not a special case.

    `duḥ` is not an upasarga -- it is absent from SANSKRIT_PREFIXES and
    `root_any_word("duḥ")` is None, so it is never *stripped*.  A `duḥ` entry can
    only arrive from the compound splitter cutting `duḥkha` in two, which is why
    the hand-written block never fired on the corpus.  When it did fire it raised
    IndexError: it read `list_of_entries[1 + 2]` (a typo for `i + 2`) after the
    list had already shrunk, and compared an entry *list* against a string.
    """

    def test_attested_join_is_merged(self):
        entries = [_entry("duḥ", "duḥkha"), _entry("kha", "duḥkha")]
        self.assertEqual(clean_results(entries, mode="roots"), ["duḥkha"])

    def test_attested_join_is_merged_when_a_word_follows(self):
        entries = [
            _entry("duḥ", "duḥkha"),
            _entry("kha", "duḥkha"),
            _entry("nirodha", "nirodha"),
        ]
        self.assertEqual(clean_results(entries, mode="roots"), ["duḥkha", "nirodha"])

    def test_unattested_join_is_not_merged(self):
        """The same stub trap: `duḥupekṣa` heads no entry, so the split must survive."""
        entries = [_entry("duḥ", "duḥupekṣa"), _entry("upekṣa", "duḥupekṣa"), _entry("x", "x")]
        roots = clean_results(entries, mode="roots")
        self.assertNotIn("duḥupekṣa", _flatten_roots(roots))

    def test_lexicalised_duhkha_is_unaffected(self):
        """The forms DB knows `duḥkha` whole, so the re-join is never reached for it."""
        self.assertEqual(process("duḥkha", mode="roots"), ["duḥkha"])


@requires_lexicon
class AvaPrefixRejoinTests(unittest.TestCase):
    """`ava` indexed the entry *after* the stem, so it never re-joined by design.

    It read `list_of_entries[j + 1]` where its two siblings read `[j]`.  That +1
    happened to hop the single `av` filler root_any_word emits beside `ava`, so
    `avaruh` re-joined by accident -- but with a word after the stem it looked up
    `ava` + *the following word* and lost the merge, and with the stem last it ran
    off the end and raised IndexError.
    """

    def test_attested_join_is_merged(self):
        entries = [_entry("ava", "avagraha"), _entry("graha", "avagraha")]
        self.assertEqual(clean_results(entries, mode="roots"), ["avagraha"])

    def test_join_is_merged_across_the_av_filler(self):
        """The real shape: root_any_word puts `av` between the prefix and the stem."""
        entries = [
            _entry("ava", "avaruhya"),
            _entry("av", "avaruhya"),
            _entry("ruh", "avaruhya"),
        ]
        self.assertEqual(clean_results(entries, mode="roots"), ["avaruh"])

    def test_attested_join_is_merged_when_a_word_follows(self):
        entries = [
            _entry("ava", "avagraha"),
            _entry("graha", "avagraha"),
            _entry("nirodha", "nirodha"),
        ]
        self.assertEqual(clean_results(entries, mode="roots"), ["avagraha", "nirodha"])

    def test_stem_in_final_position_does_not_raise(self):
        """The [j + 1] read raised IndexError whenever the stem was the last entry."""
        entries = [_entry("ava", "avatāra"), _entry("tāra", "avatāra")]
        self.assertEqual(clean_results(entries, mode="roots"), ["avatāra"])

    def test_unattested_join_is_not_merged(self):
        entries = [_entry("ava", "avaupekṣa"), _entry("upekṣa", "avaupekṣa"), _entry("x", "x")]
        roots = clean_results(entries, mode="roots")
        self.assertNotIn("avaupekṣa", _flatten_roots(roots))


if __name__ == "__main__":
    unittest.main()
