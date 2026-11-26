"""The committed configuration and case book (benchmarks/compound-members*.json)."""

import unittest

from scripts.compound_members.evaluation import EvaluationConfiguration
from scripts.compound_members.scoring import CaseBook

from .support import CONFIG, has_database, shared_resources


class CommittedFilesTests(unittest.TestCase):
    def test_the_configuration_names_each_variant_once_and_the_baseline_first(self):
        names = [variant["name"] for variant in EvaluationConfiguration.load(CONFIG).variants]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], "baseline")

    def test_the_case_book_lists_each_headword_once_with_expected_members(self):
        book = CaseBook.load(EvaluationConfiguration.load(CONFIG).cases_path)
        headwords = [case.headword for case in book.cases]
        self.assertEqual(len(headwords), len(set(headwords)))
        for case in book.cases:
            self.assertIsInstance(case.expected, tuple, case.headword)


@unittest.skipUnless(has_database(), "lexicon database is not installed")
class EveryVariantBuildsTests(unittest.TestCase):
    def test_every_configured_variant_builds_and_answers(self):
        config = EvaluationConfiguration.load(CONFIG)
        factory = shared_resources().factory(config)
        for spec in config.variants:
            self.assertIsInstance(factory.build(spec).members("ālayavijñāna"), list, spec["name"])


if __name__ == "__main__":
    unittest.main()
