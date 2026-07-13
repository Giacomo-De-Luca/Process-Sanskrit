"""
Avagraha handling.

Editions, OCR output and copy-pasted PDFs write the elided initial *a-*
(avagraha) with whatever apostrophe-like glyph was at hand: the ASCII "'", the
typographic "’", the modifier letter "ʼ", a stray backtick or acute.  Only the
ASCII form used to be understood; every other glyph either vanished (the
punctuation ones are dropped by the \p{L} filter in preprocess) or -- worse --
survived as a letter (U+02BC is category Lm) and reached the splitter as a
bogus consonant.  Either way the initial *a-* was never restored and the word
shattered into fragments: "’nupalambhena" -> nu + pa + pa + lambha.

The word used throughout is *anupalambhena* (Inst. Sg. of anupalambha,
"non-apprehension"), which in a text appears elided as "so 'nupalambhena".
"""

import unittest

from process_sanskrit.functions.process import preprocess, process
from process_sanskrit.utils.transliterationUtils import (
    AVAGRAHA_VARIANTS,
    normalize_avagraha,
)

# every glyph we accept as an avagraha, ASCII included
MARKS = ["'", "’", "ʼ", "`", "‘", "´"]


class TestNormalizeAvagraha(unittest.TestCase):
    """The pure normalisation utility, no database needed."""

    def test_every_variant_folds_to_ascii_apostrophe(self):
        for mark in MARKS:
            with self.subTest(mark=mark, codepoint=f"U+{ord(mark):04X}"):
                self.assertEqual(
                    normalize_avagraha(f"{mark}nupalambhena"), "'nupalambhena"
                )

    def test_declared_variants_are_all_covered(self):
        for mark in MARKS:
            self.assertIn(mark, AVAGRAHA_VARIANTS)

    def test_leaves_ordinary_text_alone(self):
        self.assertEqual(normalize_avagraha("yogaścittavṛttinirodhaḥ"),
                         "yogaścittavṛttinirodhaḥ")

    def test_empty_string(self):
        self.assertEqual(normalize_avagraha(""), "")


class TestPreprocessRestoresElidedA(unittest.TestCase):
    """preprocess() must put the elided *a-* back for every glyph."""

    def test_word_initial_avagraha(self):
        for mark in MARKS:
            with self.subTest(mark=mark):
                self.assertEqual(
                    preprocess(f"{mark}nupalambhena"), "anupalambhena"
                )

    def test_avagraha_is_restored_mid_sentence_not_only_at_index_0(self):
        ## the restoration used to be `text[0] == "'"`, so an avagraha on any
        ## word but the first was never restored
        for mark in MARKS:
            with self.subTest(mark=mark):
                self.assertEqual(
                    preprocess(f"tasmāt so {mark}nupalambhena"),
                    "tasmāt saḥ anupalambhena",
                )

    def test_o_plus_avagraha_contraction(self):
        ## "so'nupalambhena" is saḥ + anupalambhena written as one token
        for mark in MARKS:
            with self.subTest(mark=mark):
                self.assertEqual(
                    preprocess(f"so{mark}nupalambhena"), "saḥ anupalambhena"
                )

    def test_devanagari_avagraha(self):
        self.assertEqual(preprocess("सोऽनुपलम्भेन"), "saḥ anupalambhena")


class TestProcessResolvesElidedWord(unittest.TestCase):
    """End to end: the elided word must resolve to a real, populated entry."""

    def _stems(self, results):
        return [r[0] if isinstance(r, list) else r for r in results]

    def test_every_variant_resolves_to_anupalambha(self):
        for mark in MARKS:
            with self.subTest(mark=mark):
                results = process(f"{mark}nupalambhena")
                self.assertIn("anupalambha", self._stems(results))

    def test_resolved_entry_carries_a_full_payload(self):
        ## the symptom of the old bug was a bare fragment head with nothing
        ## attached -- no grammar, no inflection table, no dictionary entry
        for mark in MARKS:
            with self.subTest(mark=mark):
                entry = next(
                    r for r in process(f"{mark}nupalambhena")
                    if isinstance(r, list) and r[0] == "anupalambha"
                )
                self.assertGreaterEqual(len(entry), 5)
                self.assertIn(("Inst", "Sg"), entry[2])
                self.assertIsInstance(entry[-1], dict)  # dictionary entries

    def test_no_spurious_fragments(self):
        ## nu / pa / pa / lambha were what the shattered word produced
        for mark in MARKS:
            with self.subTest(mark=mark):
                stems = self._stems(process(f"{mark}nupalambhena"))
                for junk in ("nu", "pa", "lambha", "pala"):
                    self.assertNotIn(junk, stems)

    def test_elided_word_in_a_sentence(self):
        stems = self._stems(process("so ’nupalambhena"))
        self.assertIn("anupalambha", stems)

    def test_ordinary_words_are_unaffected(self):
        stems = self._stems(process("yogaścittavṛttinirodhaḥ"))
        self.assertIn("yoga", stems)


if __name__ == "__main__":
    unittest.main()
