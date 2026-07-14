"""Regression tests for prefix *segmentation* in `mode='roots'` output.

When `root_any_word` resolves a word by stripping an upasarga, the prefix and the
stem are two **sequential words** (`upadiśyate` -> `upa` + `diś`), not two rival
readings of one word.

`extract_roots` tells the two apart by surface form: it walks the entry list and
folds every run of entries that share `entry[4]` into a single tuple of
"alternative" stems.  That is only sound when the entries came from one lookup of
one span, so the prefix branch must not stamp the whole word onto the prefix
entries -- otherwise `upa` and `diś` share a surface and collapse into
`('upa', 'diś')`, which reads as "upa OR diś" downstream.

The stamp is still load-bearing for the *root* entries: two other consumers read
`entry[4]` expecting the whole word -- the `-n` lemma rule (`cleanResults`) and
the whole-word dictionary check (`process`).  Each class of collateral damage
that showed up while narrowing this is pinned below, because the obvious fixes
(drop the stamp; stamp only the remainder; key the dictionary check on the raw
input) each reintroduce one of them.

`cached=False` throughout: these assertions are about what the pipeline computes,
and a stale cache row would otherwise answer for it.
"""

import unittest

from process_sanskrit import process


class PrefixIsItsOwnWordTests(unittest.TestCase):
    """A stripped prefix is a word, not an alternative reading of the stem."""

    def test_prefixed_verb_yields_two_words(self):
        self.assertEqual(process("upadiśyate", mode="roots", cached=False), ["upa", "diś"])

    def test_prefixed_stem_yields_two_words(self):
        self.assertEqual(
            process("anumitārthe", mode="roots", cached=False), ["anu", "mitārtha"]
        )

    def test_prefix_inside_a_compound(self):
        self.assertEqual(
            process("śabdenopadiśyate", mode="roots", cached=False),
            ["śabda", "upa", "diś"],
        )
        self.assertEqual(
            process("dṛṣṭānumitārthe", mode="roots", cached=False),
            ["dṛṣṭa", "anu", "mitārtha"],
        )

    def test_privative_a_is_its_own_word(self):
        self.assertEqual(
            process("asaṃsargaḥ", mode="roots", cached=False), ["a", "saṃsarga"]
        )

    def test_nested_prefixes_stay_separate(self):
        """a + vi + rati: the outer strip must not re-stamp the inner one.

        The remainder of `a` is `virati`, which is itself resolved by stripping
        `vi`.  Re-stamping everything the recursive call returned would fold that
        inner split back into one surface, so `vi` and `ratī` would collapse again.
        """
        self.assertEqual(
            process("avirati", mode="roots", cached=False), ["avirati", "a", "vi", "ratī"]
        )


    def test_remainder_resolved_through_a_sandhi_variant_is_still_stamped(self):
        """`prati` + a remainder that only matches as a variant.

        `pratisaṃvedanā` reaches its stem through `samMap`/`variableSandhi`, so no
        entry comes back carrying the remainder *as requested*.  Keying the stamp on
        the requested remainder therefore stamps nothing, and the root entry keeps a
        sub-span surface -- which then feeds the `-n` rule and the whole-word
        dictionary check a fragment of the word.  The stamp must read the surface off
        the entries instead.

        The trailing lemma is `saṃvedana`, not the `samvedana` that was queried: the
        re-join takes it from the dictionary payload rather than echoing the query
        back.  Incidental to the stamp under test here -- see CanonicalLemmaTests in
        tests/test_prefix_merge.py.
        """
        roots = process("pratisaṃvedanā", mode="roots", cached=False)
        self.assertEqual(roots, ["pratisaṃvedanā", "prati", "saṃvedana"])


class ApiEarlyReturnTests(unittest.TestCase):
    """`root_any_word('api')` returns a flat 3-list, not a list of 5-slot entries.

    Both the stamp and the whole-word check must tolerate that shape rather than
    index into it -- which is the only reason their `isinstance(..., list)` guards
    exist.  Without a test, someone simplifies them away and this raises.
    """

    def test_api_does_not_raise(self):
        self.assertTrue(process("api", mode="roots", cached=False))

    def test_word_beginning_with_api_does_not_raise(self):
        self.assertTrue(process("apidhāya", mode="roots", cached=False))


class WholeWordEntrySurvivesTests(unittest.TestCase):
    """An attested prefixed word keeps its own dictionary entry alongside the split."""

    def test_lexicalised_prefixed_word_is_still_offered(self):
        """`virati` heads a dictionary entry; the split must not displace it.

        The whole-word check keys off the *matched* whole word carried by the root
        entry.  Reading it from `result[0]` instead breaks here, because `result[0]`
        is now the prefix entry (`vi`), whose surface is just `vi`.
        """
        roots = process("virati", mode="roots", cached=False)
        self.assertEqual(roots, ["virati", "vi", "ratī"])

    def test_normalised_surface_still_finds_the_headword(self):
        """The check must use the matched surface, not the raw input.

        `saṃskāro` is matched as `saṃskāraḥ` and `saṅgrahaḥ` as `saṃgrahaḥ`; keying
        the lookup on the raw input silently loses both entries.
        """
        self.assertEqual(
            process("saṃskāro", mode="roots", cached=False), ["saṃskāraḥ", "saṃskāra"]
        )
        self.assertEqual(
            process("saṅgrahaḥ", mode="roots", cached=False), ["saṃgrahaḥ", "saṃgraha"]
        )

    def test_headword_is_not_offered_twice(self):
        """`pratyakṣam` is already one of the analyses, so it is not re-inserted."""
        self.assertEqual(
            process("pratyakṣam", mode="roots", cached=False),
            [("pratyakṣam", "pratyakṣa")],
        )

    def test_detailed_mode_inserts_one_real_dictionary_entry(self):
        """The default mode is what dictionary consumers see, so pin it too.

        `mode='roots'` collapses each entry to its stem and would hide both a stub
        insertion and a duplicated one.
        """
        entries = process("virati", cached=False)
        whole = [e for e in entries if e[0] == "virati"]
        self.assertEqual(len(whole), 1, "the whole word is offered exactly once")
        self.assertIsInstance(
            whole[0][2], dict, "a stub must never be inserted -- a miss is not None"
        )
        ## `vi` heads two rows; they are rival readings of the *prefix*, and collapse
        ## into one root only in mode='roots'.  What matters here is the order.
        stems = list(dict.fromkeys(e[0] for e in entries))
        self.assertEqual(stems, ["virati", "vi", "ratī"])


class NoLemmaDegradationTests(unittest.TestCase):
    """The root entry keeps the whole word, so the `-n` rule still behaves."""

    def test_n_stem_lemma_is_preserved(self):
        """`viṣayin` must not decay into the surface-ish headword `viṣayī`.

        The `-n` rule in `clean_results` swaps an `-n` stem for `entry[4]` when that
        surface is itself a headword.  Handing it the *segment* surface (`viṣayī`,
        which is a headword) instead of the whole word (`aviṣayī`, which is not)
        destroys the correct lemma.
        """
        self.assertEqual(
            process("aviṣayī", mode="roots", cached=False), ["a", "viṣayin"]
        )
        self.assertEqual(
            process("tasyāviṣayībhūtatvāt", mode="roots", cached=False),
            [("ta", "tad"), "viṣayin", "bhūtatva"],
        )


class GenuineAlternativesStillGroupTests(unittest.TestCase):
    """The tuple must keep meaning "these are rival readings of one word"."""

    def test_ambiguous_word_still_returns_a_tuple(self):
        roots = process("tasya", mode="roots", cached=False)
        self.assertEqual(roots, [("ta", "tad")])

    def test_same_stem_under_two_genders_is_deduplicated(self):
        """śabdena is masculine *and* neuter śabda: one stem, not a tuple."""
        self.assertEqual(process("śabdena", mode="roots", cached=False), ["śabda"])


if __name__ == "__main__":
    unittest.main()
