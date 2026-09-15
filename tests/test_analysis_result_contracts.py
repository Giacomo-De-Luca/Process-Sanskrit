"""Regression checks for data lost between analysis and output stages."""

import copy
import unittest
from unittest.mock import patch

from process_sanskrit import dict_search, process
from process_sanskrit.functions.cleanResults import clean_results
from process_sanskrit.functions.dictionaryLookup import consult_references
from process_sanskrit.functions.inflect import inflect
from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.utils.databaseSetup import session_scope
from process_sanskrit.utils.resourcePaths import get_database_path


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class AnalysisResultContractTests(unittest.TestCase):
    missing = "ḍḍḍḍ"

    def setUp(self):
        context = session_scope()
        self.session = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def test_morphology_survives_a_missing_dictionary_headword(self):
        entry = [self.missing, "n_a", [("Acc", "Sg")], [self.missing + "am"],
                 self.missing + "am"]
        original = copy.deepcopy(entry)
        result = dict_search([entry], session=self.session)
        self.assertEqual(result, [entry + [self.missing, [self.missing]]])
        self.assertEqual(entry, original)
        self.assertEqual(clean_results(result, mode="roots"), [self.missing])
        self.assertEqual(clean_results(result, mode="parts"),
                         {self.missing: [self.missing]})

    def test_empty_dictionary_payload_is_a_miss(self):
        result = consult_references(self.missing, "mw", session=self.session)
        self.assertEqual(result, [self.missing, [self.missing]])

    def test_wildcard_miss_is_a_stub(self):
        query = self.missing + "%"
        self.assertEqual(dict_search([query], session=self.session),
                         [[query, query, [query]]])

    def test_wildcard_miss_terminates_and_preserves_pattern(self):
        for suffix in ("%", "_"):
            query = self.missing + suffix
            with self.subTest(query=query):
                self.assertEqual(process(query, cached=False),
                                 [[query, query, [query]]])
                self.assertEqual(process(query, mode="parts", cached=False),
                                 {query: [self.missing]})

    def test_star_miss_still_falls_back_to_morphology(self):
        self.assertEqual(process("balam*", mode="roots", cached=False), ["bala"])

    def test_dictionary_selection_is_case_insensitive(self):
        for name in ("ap90", "AP90"):
            with self.subTest(name=name):
                entry = dict_search(["deva"], name, session=self.session)[0]
                self.assertEqual(set(entry[-1]), {name})
                self.assertTrue(entry[-1][name])

    def test_unknown_token_survives_inflection(self):
        self.assertEqual(inflect([self.missing], session=self.session),
                         [self.missing])

    def test_unknown_token_survives_public_processing(self):
        self.assertEqual(process(self.missing, cached=False),
                         [[self.missing, self.missing, [self.missing]]])
        self.assertEqual(process(self.missing, mode="roots", cached=False),
                         [self.missing])

    def test_api_produces_one_well_formed_analysis(self):
        for word in ("api", "āpi"):
            with self.subTest(word=word):
                entries = root_any_word(word, session=self.session)
                self.assertEqual(len(entries), 1)
                self.assertEqual(len(entries[0]), 5)
                self.assertEqual(entries[0][0], "api")
                self.assertEqual(entries[0][4], word)
                result = process(word, cached=False)
                particles = [entry for entry in result if entry[0] == "api"]
                self.assertEqual(len(particles), 1)
                self.assertEqual(len(particles[0]), 7)
                self.assertTrue(particles[0][-1])

    def test_presplit_parts_preserves_component_mappings(self):
        expected = {}
        for word in ("hetu", "pada"):
            expected.update(process(word, mode="parts", cached=False))
        for separator in ("-", "+"):
            with self.subTest(separator=separator):
                self.assertEqual(process("hetu" + separator + "pada",
                                         mode="parts", cached=False), expected)

    def test_empty_presplit_parts_has_mapping_shape(self):
        self.assertEqual(process("-+-", mode="parts", cached=False), {})

    def test_prefix_rejoin_uses_selected_dictionary_and_session(self):
        entries = dict_search(["anu", "bhū"], "ap90", session=self.session)
        with patch("process_sanskrit.functions.cleanResults.dict_search",
                   wraps=dict_search) as lookup:
            result = clean_results(entries, dict_names=("ap90",),
                                   session=self.session)
        self.assertEqual([entry[0] for entry in result], ["anubhū"])
        self.assertEqual(set(result[0][-1]), {"ap90"})
        self.assertIs(lookup.call_args.kwargs["session"], self.session)

    def test_public_prefix_rejoin_preserves_dictionary_selection(self):
        # Force a known prefix cut to isolate cleanup from statistical ranking.
        from process_sanskrit.functions.hybridSplitter import HybridAnalysis

        analysis = HybridAnalysis(["anu", "bhū"], 0.7, {}, "statistical", "success")
        with patch("process_sanskrit.functions.hybridSplitter.analyze_hybrid",
                   return_value=analysis), patch(
                       "process_sanskrit.functions.inflect.inflect",
                       return_value=["anu", "bhū"]):
            result = process("anu bhū", "ap90", cached=False, session=self.session)
        self.assertEqual([entry[0] for entry in result], ["anubhū"])
        self.assertEqual(set(result[0][-1]), {"ap90"})

if __name__ == "__main__":
    unittest.main()
