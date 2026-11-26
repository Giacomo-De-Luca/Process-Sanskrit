"""Lemma-mode analyses with the members left open (scripts/compound_members/analyses.py)."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compound_members.analyses import AnalysedPart, LibraryAnalyses

from .support import has_database


@unittest.skipUnless(has_database(), "lexicon database is not installed")
class RecordedAnalysisTests(unittest.TestCase):
    """Analysing once with the members left open, then filling in the library's,
    gives exactly what lemma mode gives: the premise of scoring every variant on
    one analysis."""

    WORDS = ("ālayavijñānam", "tajjātīyena", "svarūpam", "mahātmanā", "sukham", "upagacchati",
             "cittavṛttinirodhaḥ", "mahārathaḥ", "rājaputrāḥ", "ṛgvedasya", "maharṣiḥ")

    def test_recorded_parts_with_the_library_members_reproduce_lemma_mode(self):
        from process_sanskrit import process
        from process_sanskrit.functions.lemmaAnalysis import LemmaLexicon
        from process_sanskrit.utils.databaseSetup import session_scope

        with tempfile.TemporaryDirectory() as directory, session_scope() as session:
            lexicon = LemmaLexicon(session)
            analyses = LibraryAnalyses(Path(directory) / "store.jsonl", workers=0)
            analyses.analyse(self.WORDS)
            for word in self.WORDS:
                expected = [part for item in process(word, mode="lemmas", cached=False) for part in item["parts"]]
                recorded = []
                for part in analyses.parts(word):
                    shape = {"form": part.form, "lemmas": list(part.lemmas)}
                    members = lexicon.members(part.member_lemma) if part.member_lemma else []
                    if members:
                        shape["members"] = members
                    recorded.append(shape)
                self.assertEqual(recorded, expected, word)


@unittest.skipUnless(has_database(), "lexicon database is not installed")
class AnalysisStoreTests(unittest.TestCase):
    """The store is reused only by the same library, algorithm and database."""

    def store(self, directory, *rows, header=None):
        path = Path(directory) / "store.jsonl"
        lines = [json.dumps(header or LibraryAnalyses.signature())] + [json.dumps(row, ensure_ascii=False) for row in rows]
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_a_store_is_reloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.store(directory, ["rāma", [["rāma", ["rāma"], None]], None])
            analyses = LibraryAnalyses(path)
            self.assertEqual(analyses.parts("rāma"), (AnalysedPart("rāma", ("rāma",), None),))
            self.assertEqual(analyses.analyse(["rāma"]), 0)

    def test_a_store_with_another_signature_is_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.store(directory, ["rāma", [["rāma", ["rāma"], None]], None],
                              header={**LibraryAnalyses.signature(), "analysis_algorithm": "older"})
            analyses = LibraryAnalyses(path)
            self.assertTrue(analyses.discarded)
            self.assertEqual(len(analyses), 0)
            self.assertFalse(path.exists())

    def test_failed_words_are_retried_and_an_unfinished_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.store(directory, ["deva", None, "RuntimeError()"], ["rāma", [["rāma", ["rāma"], None]], None])
            with path.open("a") as handle:
                handle.write('["sukha", [["sukha"')
            analyses = LibraryAnalyses(path)
            self.assertEqual(len(analyses), 1)
            self.assertEqual(analyses.analyse(["deva", "rāma"]), 1)
            self.assertIn("deva", analyses.lemma_set("deva"))


if __name__ == "__main__":
    unittest.main()
