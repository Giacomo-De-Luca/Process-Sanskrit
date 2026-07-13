"""Regression tests for pre-split compounds and option forwarding.

Two entangled behaviours are pinned here.

1. ``preprocess`` must preserve ``-`` and ``+``.  The historical character
   class was ``[^\\p{L}'_%*-+]``, in which ``*-+`` is a *range* (``*`` through
   ``+``), not three literal characters.  ``+`` therefore survived while ``-``
   was rewritten to a space, which pushed hyphenated input into the whitespace
   branch of ``process`` and left the pre-split branch of
   ``handle_special_characters`` unreachable for ``-``.  A future tidy-up that
   folds the class back into a range would silently resurrect that bug, so the
   literal characters are asserted directly.

2. The pre-split branch recurses into ``process`` once per segment.  That
   recursion must forward the caller's options; otherwise a pre-split compound
   silently falls back to ``mode="detailed"`` and the default ``mw`` dictionary
   no matter what the caller asked for.
"""

import unittest

from process_sanskrit.functions.process import preprocess, process


def is_roots_shaped(result):
    """``mode='roots'`` yields bare stems (str) or stem/alternative pairs.

    An empty list satisfies ``all()`` vacuously, so require a non-empty result:
    otherwise a regression to ``[]`` would slip past every shape assertion.
    """
    return (
        isinstance(result, list)
        and bool(result)
        and all(isinstance(entry, (str, tuple)) for entry in result)
    )


def is_detailed_shaped(result):
    """``mode='detailed'`` yields ``[headword, inflected, {dict: entries}]``."""
    return (
        isinstance(result, list)
        and bool(result)
        and all(isinstance(entry, list) for entry in result)
    )


def dictionaries_in(result):
    """Collect the dictionary names present in a detailed result."""
    names = set()
    for entry in result:
        if isinstance(entry, list):
            for field in entry:
                if isinstance(field, dict):
                    names |= set(field.keys())
    return names


class PreprocessSeparatorTests(unittest.TestCase):
    def test_hyphen_survives_preprocess(self):
        self.assertEqual(preprocess("hetu-pada"), "hetu-pada")

    def test_plus_survives_preprocess(self):
        self.assertEqual(preprocess("hetu+pada"), "hetu+pada")

    def test_punctuation_is_still_stripped(self):
        ## the separators are preserved, but genuine punctuation is not
        self.assertEqual(preprocess("hetu, pada."), "hetu pada")


class PreSplitOptionForwardingTests(unittest.TestCase):
    """The pre-split recursion must not discard the caller's options."""

    def test_roots_mode_forwarded_through_hyphen(self):
        result = process("hetu-pada", mode="roots")
        self.assertTrue(
            is_roots_shaped(result),
            f"mode='roots' was dropped on the hyphen path: {result!r:.120}",
        )

    def test_roots_mode_forwarded_through_plus(self):
        result = process("hetu+pada", mode="roots")
        self.assertTrue(
            is_roots_shaped(result),
            f"mode='roots' was dropped on the plus path: {result!r:.120}",
        )

    def test_detailed_mode_still_detailed_through_hyphen(self):
        self.assertTrue(is_detailed_shaped(process("hetu-pada")))

    def test_dict_names_forwarded_through_hyphen(self):
        result = process("hetu-pada", "ap90")
        self.assertIn(
            "ap90",
            dictionaries_in(result),
            "dict_names was dropped on the hyphen path; only the default "
            "dictionary came back",
        )

    def test_dict_names_forwarded_through_plus(self):
        self.assertIn("ap90", dictionaries_in(process("hetu+pada", "ap90")))

    def test_default_dictionary_unchanged(self):
        ## guards the assertion above: without dict_names the result really is
        ## mw-only, so finding ap90 above is evidence of forwarding
        self.assertEqual(dictionaries_in(process("hetu-pada")), {"mw"})


class PreSplitEquivalenceTests(unittest.TestCase):
    """A pre-split compound is the concatenation of its processed segments."""

    def test_hyphen_matches_segmentwise_roots(self):
        self.assertEqual(
            process("hetu-pada", mode="roots"),
            process("hetu", mode="roots") + process("pada", mode="roots"),
        )

    def test_hyphen_matches_segmentwise_detailed(self):
        self.assertEqual(
            process("hetu-pada"),
            process("hetu") + process("pada"),
        )

    def test_hyphen_and_plus_are_equivalent(self):
        ## the two separators are handled by the same branch and must not drift
        self.assertEqual(
            process("hetu-pada", mode="roots"),
            process("hetu+pada", mode="roots"),
        )

    def test_separators_may_be_mixed(self):
        self.assertEqual(
            process("hetu-pada+adhikam", mode="roots"),
            process("hetu+pada-adhikam", mode="roots"),
        )

    def test_pre_split_boundaries_are_honoured(self):
        ## the splitter is not asked to re-derive the boundaries the caller gave
        self.assertEqual(process("hetu-pada", mode="roots")[0], "hetu")


class WildcardTests(unittest.TestCase):
    """Wildcard lookups honour ``mode`` and ``dict_names`` like anything else.

    Both wildcard branches have two exits: a recursion into ``process`` when the
    pattern finds nothing, and an early ``return voc_entry`` when it hits.  The
    early exit -- which is the common one -- used to hand back raw ``dict_search``
    output without passing through ``clean_results``, so ``mode='roots'`` came
    back as detailed entries.

    For a pattern query (``_``/``%``) the "root" is the literal pattern, since a
    pattern has no stem.  That is intended, not an accident.
    """

    def test_asterisk_honours_roots_mode(self):
        self.assertEqual(process("deva*", mode="roots"), ["deva"])

    def test_underscore_pattern_honours_roots_mode(self):
        self.assertEqual(process("dev_", mode="roots"), ["dev_"])

    def test_percent_pattern_honours_roots_mode(self):
        self.assertEqual(process("deva%", mode="roots"), ["deva%"])

    def test_wildcard_still_detailed_by_default(self):
        self.assertTrue(is_detailed_shaped(process("deva*")))

    def test_wildcard_honours_dict_names(self):
        self.assertIn("ap90", dictionaries_in(process("deva*", "ap90")))


class MultiWordSeparatorTests(unittest.TestCase):
    """A separator inside multi-word input must still act as a boundary.

    ``handle_special_characters`` is gated on ``' ' not in text``, so the
    pre-split branch never sees a sentence.  Once ``preprocess`` stopped
    rewriting ``-`` to a space, a hyphen in a sentence survived all the way into
    the splitter, which does not treat it as a boundary -- and duly *merged* the
    two segments the caller had explicitly separated
    (``yoga-citta-vṛtti nirodhaḥ`` -> ``['yoga', 'cittavṛtti', 'nirodha']``).
    Whitespace *is* honoured by the splitter, so the separator is normalised to
    whitespace on this path.
    """

    SENTENCE_ROOTS = ["yoga", "citta", "vṛtti", "nirodha"]

    def test_hyphen_in_sentence_does_not_merge_segments(self):
        self.assertEqual(
            process("yoga-citta-vṛtti nirodhaḥ", mode="roots"),
            self.SENTENCE_ROOTS,
        )

    def test_plus_in_sentence_does_not_merge_segments(self):
        self.assertEqual(
            process("yoga+citta+vṛtti nirodhaḥ", mode="roots"),
            self.SENTENCE_ROOTS,
        )

    def test_separator_in_sentence_matches_whitespace_equivalent(self):
        ## the caller's boundary must be worth exactly as much as a space
        self.assertEqual(
            process("yoga-citta-vṛtti nirodhaḥ", mode="roots"),
            process("yoga citta vṛtti nirodhaḥ", mode="roots"),
        )

    def test_options_survive_the_sentence_path(self):
        self.assertTrue(
            is_roots_shaped(process("yoga-citta nirodhaḥ", mode="roots"))
        )


class EmptySegmentTests(unittest.TestCase):
    """A separator with nothing beside it yields an empty segment.

    ``re.split`` happily produces ``''`` for a leading, trailing or doubled
    separator, and a trailing hyphen is ordinary in running Sanskrit text.  The
    empty segment used to reach ``preprocess``, which indexed ``text[0]``
    unguarded and raised IndexError, so ``process('hetu-')`` crashed rather than
    analysing ``hetu``.
    """

    def test_trailing_separator(self):
        self.assertEqual(process("hetu-", mode="roots"), ["hetu"])

    def test_leading_separator(self):
        self.assertEqual(process("-hetu", mode="roots"), ["hetu"])

    def test_doubled_separator(self):
        self.assertEqual(
            process("hetu--pada", mode="roots"),
            process("hetu-pada", mode="roots"),
        )

    def test_separator_only(self):
        ## every segment is empty, so nothing is contributed
        self.assertEqual(process("-", mode="roots"), [])
        self.assertEqual(process("-"), [])

    def test_empty_input_does_not_crash(self):
        ## preprocess indexed text[0] before checking for the empty string
        self.assertEqual(process(""), [])

    def test_known_wart_roots_mode_empty_return_type(self):
        """mode='roots' has two different empty returns, and they disagree.

        Empty input short-circuits in process() and returns ``""`` -- a *string*
        -- while a compound whose segments are all empty returns ``[]``.  A
        caller iterating the result gets characters in one case and stems in the
        other.  Pinned as-is rather than fixed: ``""`` is the published 1.0.x
        behaviour for empty input and changing the return type is a breaking
        change that belongs to a deliberate API decision, not to this fix.
        """
        self.assertEqual(process("", mode="roots"), "")
        self.assertEqual(process("-", mode="roots"), [])

    def test_unanalysable_segment_is_dropped_not_fatal(self):
        ## documents current behaviour: a junk segment yields nothing at all
        self.assertEqual(process("hetu-qqqq", mode="roots"), ["hetu"])


if __name__ == "__main__":
    unittest.main()
