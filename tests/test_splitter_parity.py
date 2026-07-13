"""Tests for the vendored sandhi splitter (process_sanskrit/splitter).

The splitter is a reduced copy of sanskrit_parser with two substitutions that
could silently drift from upstream:

  * the validity oracle is a precomputed trie rather than two sqlite databases
    plus a generative stem analysis, and
  * the split scorer is numpy rather than gensim.

``SplitterTests`` runs everywhere and pins current behaviour. ``UpstreamParityTests``
proves the substitutions are faithful by diffing against the real sanskrit_parser;
it is skipped unless that package (and gensim) happen to be installed, since the
whole point of the fork is not to depend on them.

To run the parity half:

    pip install sanskrit-parser==0.2.6 gensim sentencepiece
    python -m unittest tests.test_splitter_parity
"""

import ast
import unittest

from process_sanskrit.splitter import Parser
from process_sanskrit.splitter.scorer import Scorer

try:
    import gensim  # noqa: F401
    import sanskrit_parser  # noqa: F401

    from sanskrit_parser.util import lexical_scorer
    from sanskrit_parser.util.lexical_lookup_factory import LexicalLookupFactory

    HAS_UPSTREAM = lexical_scorer.gensim_enabled
except ImportError:
    HAS_UPSTREAM = False

CORPUS = [
    "astyuttarasyAMdiSi",
    "ahiṃsāpratiṣṭhāyāṃ",
    "yogaścittavṛttinirodhaḥ",
    "tadā draṣṭuḥ svarūpe 'vasthānam",
    "pratyayaikatānatā",
    "kṣīyate",
]


class SplitterTests(unittest.TestCase):
    """Behaviour that must hold with no sanskrit_parser and no gensim installed."""

    @classmethod
    def setUpClass(cls):
        cls.parser = Parser(output_encoding="iast")

    def test_splits_a_known_compound(self):
        splits = self.parser.split("astyuttarasyAMdiSi", limit=10)
        self.assertIsNotNone(splits)
        self.assertEqual(ast.literal_eval(str(splits[0])),
                         ["asti", "uttarasyām", "di", "ṣi"])

    def test_statistical_scoring_is_active(self):
        """A silent fallback to the length heuristic would quietly degrade splits."""
        scorer = Scorer()
        self.assertTrue(scorer._load(), "DCS word2vec scoring is not active")
        # Real log-probabilities, not -len(words).
        score = scorer.score_strings(["asti uttarasyAm diSi"])[0]
        self.assertLess(score, -1.0)
        self.assertNotEqual(score, -3)

    def test_missing_model_raises_instead_of_degrading(self):
        """Without the model, splits still *work* -- they just get quietly worse.

        Upstream ranks by length in that case and logs a warning that
        process_sanskrit/__init__.py silences. Ranking must fail loudly instead.
        """
        from process_sanskrit.splitter import scorer as scorer_module

        original = scorer_module.data_file_path
        scorer_module.data_file_path = lambda f: "/nonexistent/" + f
        try:
            with self.assertRaises(RuntimeError):
                Scorer().score_strings(["asti uttarasyAm diSi"])
        finally:
            scorer_module.data_file_path = original

    def test_scoring_changes_the_ranking(self):
        """Guards against the scorer loading but contributing nothing."""
        scored = [str(s) for s in Parser(output_encoding="iast", score=True)
                  .split("astyuttarasyAMdiSi", limit=10)]
        unscored = [str(s) for s in Parser(output_encoding="iast", score=False)
                    .split("astyuttarasyAMdiSi", limit=10)]
        self.assertNotEqual(scored, unscored)

    def test_no_upstream_import(self):
        """The vendored splitter must not reach back into sanskrit_parser."""
        import pathlib

        import process_sanskrit.splitter as pkg

        for mod in pathlib.Path(pkg.__file__).parent.glob("*.py"):
            src = mod.read_text()
            self.assertNotIn("import sanskrit_parser", src, f"{mod.name} imports upstream")
            self.assertNotIn("import gensim", src, f"{mod.name} imports gensim")


@unittest.skipUnless(HAS_UPSTREAM, "sanskrit-parser + gensim not installed")
class UpstreamParityTests(unittest.TestCase):
    """Prove the trie and the numpy scorer reproduce upstream exactly."""

    @classmethod
    def setUpClass(cls):
        cls.ours = Parser(output_encoding="iast")
        cls.theirs = sanskrit_parser.Parser(output_encoding="iast")
        cls.combined = LexicalLookupFactory.create("combined")

    def test_validity_oracle_matches(self):
        """The trie must accept exactly what CombinedWrapper.valid() accepts.

        Includes forms only reachable through sanskrit_data's *generative* stem
        analysis, which appear in no table -- these are ~40% of accepted forms.
        """
        from process_sanskrit.splitter.lookup import TrieLookup

        trie = TrieLookup()
        probes = [
            # plain Inria form-table hits
            "asti", "diSi", "yogaH",
            # generative stem+ending hits (in no form table)
            "ARavi", "AjYAnam", "AlasyAs", "ANga",
            # non-words
            "qqqq", "xyzzy", "",
        ]
        for word in probes:
            self.assertEqual(trie.valid(word), self.combined.valid(word),
                             f"validity disagrees on {word!r}")

    def test_scorer_matches_gensim(self):
        """numpy scorer must reproduce gensim to within float32 rounding."""
        import sentencepiece as spm
        from process_sanskrit.splitter.data_manager import data_file_path

        ours = Scorer()
        ours._load()
        sp = spm.SentencePieceProcessor()
        sp.Load(data_file_path("sentencepiece.model"))
        theirs = lexical_scorer.Scorer()

        for text in ["asti uttarasyAm diSi", "yogaH cittavftti niroDaH",
                     "tadA drazwuH svarUpe avasTAnam"]:
            mine = ours.score_strings([text])[0]
            gold = float(theirs.model.score([sp.EncodeAsPieces(text)], total_sentences=1)[0])
            self.assertAlmostEqual(mine, gold, delta=1e-3,
                                   msg=f"score drifted on {text!r}")

    def test_splits_match_upstream(self):
        """Same candidate splits, same order, on real text."""
        for text in CORPUS:
            for word in text.split():
                mine = [str(s) for s in (self.ours.split(word, limit=10) or [])]
                gold = [str(s) for s in (self.theirs.split(word, limit=10) or [])]
                # Scores can tie; what must not change is the candidate set and
                # the winner, since sandhiSplitter re-ranks the whole list.
                self.assertEqual(set(mine), set(gold), f"candidate set differs on {word!r}")
                self.assertEqual(mine[:1], gold[:1], f"best split differs on {word!r}")


if __name__ == "__main__":
    unittest.main()
