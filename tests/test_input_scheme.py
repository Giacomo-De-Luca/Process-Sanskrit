"""Tests for the explicit ``input_scheme`` argument of ``process``.

Auto-detection cannot tell the ASCII schemes apart on short input, so a
caller that already knows the scheme passes it instead.  The inputs here are
chosen because detection gets each one wrong:

* SLP1 ``Darma`` (dharma) is read as Harvard-Kyoto, giving ``ḍarma``;
* WX ``Xarma`` (dharma) is read as SLP1, giving ``ḹarma``;
* ITRANS ``chakra`` (cakra) is read as Harvard-Kyoto, giving ``chakra``.

``preprocess`` converts the text to IAST before anything else, and the
wildcard and pre-split branches then re-enter ``process`` on that IAST text.
Those calls must not read it in the caller's scheme a second time.  The WX
cases pin this, because IAST ``dharma`` read as WX is ``ḍharma``.
"""

import unittest

from process_sanskrit.functions.dictionaryLookup import dict_search
from process_sanskrit.functions.process import preprocess, process
from process_sanskrit.utils.detectTransliteration import TransliterationScheme
from process_sanskrit.utils.transliterationUtils import resolve_scheme, transliterate


class ResolveSchemeTests(unittest.TestCase):
    def test_names_are_case_insensitive(self):
        ## the frontend sends "Tamil", "Kolkata", "DEVANAGARI" ...
        self.assertEqual(resolve_scheme("slp1"), resolve_scheme("SLP1"))
        self.assertEqual(resolve_scheme("Tamil"), resolve_scheme("TAMIL"))

    def test_every_detected_scheme_resolves(self):
        ## transliterate() feeds detect()'s answer straight to resolve_scheme()
        for scheme in TransliterationScheme:
            with self.subTest(scheme=scheme.value):
                resolve_scheme(scheme.value)

    def test_unknown_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_scheme("NOPE")

    def test_non_scheme_attribute_raises_value_error(self):
        ## sanscript.SCHEMES is a module attribute, but not a scheme
        with self.assertRaises(ValueError):
            resolve_scheme("schemes")


class TransliterateSchemeValidationTests(unittest.TestCase):
    def test_unknown_output_scheme_raises_value_error(self):
        with self.assertRaises(ValueError):
            transliterate("dharma", "NOPE")

    def test_unknown_input_scheme_raises_value_error(self):
        with self.assertRaises(ValueError):
            transliterate("dharma", "IAST", "NOPE")

    def test_explicit_input_scheme_is_used(self):
        self.assertEqual(transliterate("Darma", "IAST", "SLP1"), "dharma")


class PreprocessInputSchemeTests(unittest.TestCase):
    def test_slp1(self):
        self.assertEqual(preprocess("Darma", input_scheme="SLP1"), "dharma")

    def test_wx(self):
        self.assertEqual(preprocess("Xarma", input_scheme="WX"), "dharma")

    def test_itrans(self):
        self.assertEqual(preprocess("chakra", input_scheme="ITRANS"), "cakra")

    def test_scheme_name_is_case_insensitive(self):
        self.assertEqual(preprocess("Darma", input_scheme="slp1"), "dharma")

    def test_default_still_autodetects(self):
        self.assertEqual(preprocess("धर्म"), "dharma")
        self.assertEqual(preprocess("dharma", input_scheme=None), preprocess("dharma"))

    def test_unknown_scheme_raises_value_error(self):
        with self.assertRaises(ValueError):
            preprocess("dharma", input_scheme="NOPE")


class ProcessInputSchemeTests(unittest.TestCase):
    """Each explicit-scheme call must match the same word given in IAST."""

    def assertSameAnalysis(self, text, scheme, iast_text, *dict_names, **options):
        self.assertEqual(
            process(text, *dict_names, input_scheme=scheme, **options),
            process(iast_text, *dict_names, **options),
        )

    def test_slp1_word(self):
        self.assertSameAnalysis("Darma", "SLP1", "dharma", mode="roots")

    def test_wx_word(self):
        self.assertSameAnalysis("Xarma", "WX", "dharma", mode="roots")

    def test_itrans_word(self):
        self.assertSameAnalysis("chakra", "ITRANS", "cakra", mode="roots")

    def test_detailed_mode_with_dictionaries(self):
        self.assertSameAnalysis("Darma", "SLP1", "dharma", "mw", "ap90")

    ## The two recursion tests build their expectation from plain words, which
    ## never reach the recursion.  Comparing against "dharma-kāya" instead would
    ## send both sides through the same recursion, and a wrongly forwarded
    ## scheme would corrupt them identically and still compare equal.

    def test_presplit_recursion_does_not_reapply_scheme(self):
        expected = process("dharma", mode="roots") + process("kāya", mode="roots")
        self.assertEqual(
            process("Xarma-kAya", input_scheme="WX", mode="roots"), expected
        )

    def test_wildcard_fallback_does_not_reapply_scheme(self):
        ## an inflected form is no headword, so the star lookup misses and
        ## falls back to process() on the IAST text (dharmeṇa would not do:
        ## it is an MW headword, and the hit returns before the fallback)
        self.assertTrue(
            isinstance(dict_search(["dharmasya"])[0][2], list),
            "dharmasya became a headword; pick a form whose star lookup misses",
        )
        self.assertEqual(
            process("Xarmasya*", input_scheme="WX", mode="roots"),
            process("dharmasya", mode="roots"),
        )

    def test_unknown_scheme_raises_value_error(self):
        with self.assertRaises(ValueError):
            process("dharma", input_scheme="NOPE")


if __name__ == "__main__":
    unittest.main()
