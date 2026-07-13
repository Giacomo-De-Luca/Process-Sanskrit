r"""
Avagraha handling.

Editions, OCR output and copy-pasted PDFs write the elided initial *a-*
(avagraha) with whatever apostrophe-like glyph was at hand: the ASCII "'", the
typographic "’", the modifier letter "ʼ", a stray backtick or acute.  Only the
ASCII form used to be understood; every other glyph either vanished (the
punctuation ones are dropped by the \p{L} filter in preprocess) or -- worse --
survived as a letter (U+02BC is category Lm) and reached the splitter as a
bogus consonant.  Either way the initial *a-* was never restored and the word
shattered into fragments: "’nupalambhena" -> nu + pa + pa + lambha.

What identifies an avagraha is its *position*, not its glyph: the a- is elided
only after a preceding **e** or **o**, or at the head of a bare token.  An
apostrophe anywhere else is a quotation mark or OCR noise, and must not become a
vowel -- "iti ‘yoga’ ucyate" would otherwise yield the real-but-wrong lemma
*ayoga* ("non-union"), a silent corruption with no exception to warn anyone.

The word used throughout is *anupalambhena* (Inst. Sg. of anupalambha,
"non-apprehension"), which in a text appears elided as "so 'nupalambhena".
"""

import unittest

from process_sanskrit.functions.process import preprocess, process
from process_sanskrit.utils.transliterationUtils import (
    AVAGRAHA_VARIANTS,
    normalize_avagraha,
    restore_avagraha,
)

## drive every loop off the table itself, so a newly declared variant is
## exercised automatically instead of silently going untested
MARKS = sorted(AVAGRAHA_VARIANTS)


class TestNormalizeAvagraha(unittest.TestCase):
    """The pure glyph fold, no database needed."""

    def test_every_variant_folds_to_ascii_apostrophe(self):
        for mark in MARKS:
            with self.subTest(mark=mark, codepoint=f"U+{ord(mark):04X}"):
                self.assertEqual(
                    normalize_avagraha(f"{mark}nupalambhena"), "'nupalambhena"
                )

    def test_leaves_ordinary_text_alone(self):
        self.assertEqual(normalize_avagraha("yogaścittavṛttinirodhaḥ"),
                         "yogaścittavṛttinirodhaḥ")

    def test_empty_string(self):
        self.assertEqual(normalize_avagraha(""), "")


class TestRestoreAvagraha(unittest.TestCase):
    """The positional restoration rules, no database needed."""

    def test_o_from_visarga_is_undone_with_the_elision(self):
        ## saḥ + anupalambhena -> so 'nupalambhena, so both halves come back
        for mark in MARKS:
            with self.subTest(mark=mark):
                self.assertEqual(
                    restore_avagraha(f"so{mark}nupalambhena"), "saḥ anupalambhena"
                )
                self.assertEqual(
                    restore_avagraha(f"so {mark}nupalambhena"), "saḥ anupalambhena"
                )

    def test_original_o_of_indeclinables_is_kept(self):
        ## aho/bho/ho/o end in an original -o, not one from -aḥ: only the
        ## elided a- is restored, the o stays put
        for text, expected in [
            ("aho 'yam", "aho ayam"),
            ("bho 'ham", "bho aham"),
            ("aho’yam", "aho ayam"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(restore_avagraha(text), expected)

    def test_e_avagraha_spaced_and_unspaced(self):
        ## the -e half of the rule: te 'pi, vane 'smin.  The e is original and
        ## stays; only the a- comes back
        for text, expected in [
            ("te 'pi", "te api"),
            ("te'pi", "te api"),
            ("te’pi", "te api"),
            ("vane 'smin", "vane asmin"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(restore_avagraha(text), expected)

    def test_bare_token_initial_avagraha(self):
        for mark in MARKS:
            with self.subTest(mark=mark):
                self.assertEqual(
                    restore_avagraha(f"{mark}nupalambhena"), "anupalambhena"
                )

    def test_quotation_marks_are_not_avagrahas(self):
        ## the regression this guards: folding ‘/’ onto ' and then treating any
        ## post-whitespace apostrophe as an avagraha turned *yoga* into *ayoga*
        self.assertEqual(restore_avagraha("iti ‘yoga’ ucyate"), "iti yoga ucyate")
        self.assertEqual(restore_avagraha("‘tapas’"), "tapas")
        self.assertEqual(restore_avagraha("kṛṣṇa’s"), "kṛṣṇas")

    def test_leaves_ordinary_text_alone(self):
        for text in ["yogaścittavṛttinirodhaḥ", "loka", "yo yogaḥ", "tepi", ""]:
            with self.subTest(text=text):
                self.assertEqual(restore_avagraha(text), text)


class TestPreprocess(unittest.TestCase):
    """preprocess() must put the elided a- back for every glyph."""

    def test_word_initial_avagraha(self):
        for mark in MARKS:
            with self.subTest(mark=mark):
                self.assertEqual(preprocess(f"{mark}nupalambhena"), "anupalambhena")

    def test_avagraha_is_restored_mid_sentence_not_only_at_index_0(self):
        ## the restoration used to be `text[0] == "'"`, so an avagraha on any
        ## word but the first was never restored
        for mark in MARKS:
            with self.subTest(mark=mark):
                self.assertEqual(
                    preprocess(f"tasmāt so {mark}nupalambhena"),
                    "tasmāt saḥ anupalambhena",
                )

    def test_devanagari_avagraha(self):
        self.assertEqual(preprocess("सोऽनुपलम्भेन"), "saḥ anupalambhena")

    def test_quoted_word_survives_unmangled(self):
        self.assertEqual(preprocess("iti ‘yoga’ ucyate"), "iti yoga ucyate")

    def test_is_idempotent(self):
        ## preprocess re-runs on the wildcard / pre-split recursion path in
        ## handle_special_characters, so it must be a fixed point
        for text in ["so ’nupalambhena", "te’pi", "iti ‘yoga’ ucyate", "aho 'yam"]:
            with self.subTest(text=text):
                once = preprocess(text)
                self.assertEqual(preprocess(once), once)


class TestProcessResolvesElidedWord(unittest.TestCase):
    """End to end: the elided word must resolve to a real, populated entry."""

    def _stems(self, results):
        return [r[0] if isinstance(r, list) else r for r in results]

    def test_every_variant_resolves_to_anupalambha(self):
        for mark in MARKS:
            with self.subTest(mark=mark):
                self.assertIn(
                    "anupalambha", self._stems(process(f"{mark}nupalambhena"))
                )

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
        self.assertIn("anupalambha", self._stems(process("so ’nupalambhena")))

    def test_quoted_word_is_not_read_as_an_avagraha(self):
        stems = self._stems(process("iti ‘yoga’ ucyate"))
        self.assertIn("yoga", stems)
        self.assertNotIn("ayoga", stems)

    def test_ordinary_words_are_unaffected(self):
        self.assertIn("yoga", self._stems(process("yogaścittavṛttinirodhaḥ")))


if __name__ == "__main__":
    unittest.main()
