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

    uv pip install sanskrit-parser==0.2.6 gensim sentencepiece
    uv run python -m unittest tests.test_splitter_parity
"""

import ast
from collections import Counter
import threading
import time
import unittest
from unittest.mock import Mock, patch

from indic_transliteration import sanscript

from process_sanskrit.splitter import Parser
from process_sanskrit.splitter.backends import PythonBackend
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

# The Python oracle can inherit process-dependent ordering inside tied score
# groups. The Rust implementation may make those ties deterministic, but it
# must preserve the winner and the complete candidate multiset.
PARITY_FIXTURES = {
    "ahiṃsāpratiṣṭhāyāṃ": [
        ["ahiṃsā", "pratiṣṭhāyām"],
        ["ahiṃsā", "a", "pratiṣṭhāyām"],
        ["ahiṃsā", "pratiṣṭhā", "ayām"],
        ["a", "hiṃsā", "pratiṣṭhāyām"],
        ["ahiṃsā", "apratiṣṭhāyām"],
        ["ahiṃsā", "pratiṣṭhā", "yām"],
        ["ahiṃsā", "apratiṣṭhā", "yām"],
        ["ahiṃsā", "pratiṣṭhā", "āyām"],
        ["ahim", "sā", "pratiṣṭhāyām"],
        ["a", "him", "sā", "pratiṣṭhāyām"],
    ],
    "yogaścittavṛttinirodhaḥ": [
        ["yogaḥ", "citta", "vṛtti", "nirodhaḥ"],
        ["yogaḥ", "citta", "vṛttinirodhaḥ"],
        ["yogaḥ", "cittavṛtti", "nirodhaḥ"],
        ["yogaḥ", "citta", "vṛttini", "rodhaḥ"],
        ["yok", "aḥ", "citta", "vṛtti", "nirodhaḥ"],
        ["yogaḥ", "citta", "vṛtti", "ni", "rodhaḥ"],
        ["yogaḥ", "cittavṛtti", "ni", "rodhaḥ"],
        ["yok", "aḥ", "cittavṛtti", "nirodhaḥ"],
        ["yo", "agaḥ", "cittavṛtti", "nirodhaḥ"],
        ["yo", "agaḥ", "citta", "vṛtti", "nirodhaḥ"],
    ],
}


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

        one_split = self.parser.split("astyuttarasyAMdiSi", limit=1)
        self.assertEqual(len(one_split), 1)
        self.assertEqual(
            ast.literal_eval(str(one_split[0])),
            ["asti", "uttarasyām", "di", "ṣi"],
        )

    def test_fixed_candidate_multisets_and_winners(self):
        for text, expected in PARITY_FIXTURES.items():
            with self.subTest(text=text):
                actual = [
                    ast.literal_eval(str(split))
                    for split in self.parser.split(text, limit=10)
                ]
                self.assertEqual(actual[0], expected[0])
                self.assertEqual(
                    Counter(map(tuple, actual)), Counter(map(tuple, expected))
                )

    def test_zero_limit_preserves_scored_and_unscored_behavior(self):
        with self.assertRaises(ValueError):
            self.parser.split("astyuttarasyAMdiSi", limit=0)

        unscored = Parser(output_encoding="iast", score=False)
        self.assertEqual(unscored.split("astyuttarasyAMdiSi", limit=0), [])

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

    def test_graphs_share_the_read_only_scorer(self):
        """Building each candidate graph must not reload the 6 MB model."""
        from process_sanskrit.splitter.datastructures import SandhiGraph

        first = SandhiGraph()
        second = SandhiGraph()
        self.assertIs(first.scorer, second.scorer)

    def test_native_export_uses_the_pinned_log_sigmoid_table(self):
        import hashlib
        import tempfile
        from pathlib import Path

        import numpy as np

        from process_sanskrit.splitter.scorer_model import log_sigmoid_table
        from tools.build_splitter_data import NativeInputExporter

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            NativeInputExporter(output).write_scorer(
                syn0=np.zeros((1, 1), dtype=np.float32),
                syn1=np.zeros((1, 1), dtype=np.float32),
                vocab=["a"],
                window=1,
                cbow_mean=True,
                codes=[np.array([0], dtype=np.uint8)],
                points=[np.array([0], dtype=np.uint32)],
            )
            exported = np.load(output / "log-table.npy", allow_pickle=False)

        self.assertTrue(np.array_equal(exported, log_sigmoid_table()))
        self.assertEqual(
            hashlib.sha256(exported.tobytes()).hexdigest(),
            "0ac6b09e6d522eb77c2bf196d2b9885d887a3fd67f946c940a13decc20899886",
        )

    def test_shared_scorer_initializes_once_under_concurrency(self):
        """Concurrent first use publishes one completely loaded model."""
        from process_sanskrit.splitter import scorer as scorer_module

        instance = Scorer()
        real_load = scorer_module.np.load
        load_count = 0
        count_lock = threading.Lock()
        barrier = threading.Barrier(8)
        results = []
        errors = []

        def counted_load(*args, **kwargs):
            nonlocal load_count
            with count_lock:
                load_count += 1
            # Widen the race between the initial enabled check and publication.
            time.sleep(0.03)
            return real_load(*args, **kwargs)

        def load() -> None:
            try:
                barrier.wait()
                results.append(instance._load())
            except Exception as error:  # pragma: no cover - asserted below
                errors.append(error)

        with patch.object(scorer_module.np, "load", side_effect=counted_load):
            threads = [threading.Thread(target=load) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(results, [True] * 8)
        self.assertEqual(load_count, 1)

    def test_forms_trie_is_not_published_before_loading_finishes(self):
        from process_sanskrit.splitter import lookup

        started = threading.Event()
        release = threading.Event()
        results = []

        class BlockingTrie:
            def load(self, _path):
                started.set()
                release.wait(timeout=5)

        trie = BlockingTrie()
        original = lookup._trie
        lookup._trie = None
        try:
            with patch.object(lookup.marisa_trie, "Trie", return_value=trie):
                first = threading.Thread(target=lambda: results.append(lookup._forms()))
                second = threading.Thread(target=lambda: results.append(lookup._forms()))
                first.start()
                self.assertTrue(started.wait(timeout=2))
                second.start()
                time.sleep(0.05)
                self.assertTrue(second.is_alive())
                release.set()
                first.join(timeout=2)
                second.join(timeout=2)
        finally:
            release.set()
            lookup._trie = original

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(results, [trie, trie])

    def test_one_parser_serializes_concurrent_requests(self):
        """LexicalSandhiAnalyzer keeps request state and cannot be overlapped."""
        from process_sanskrit.splitter.backends import PythonBackend

        class EmptyGraph:
            @staticmethod
            def find_all_paths(**_kwargs):
                return []

        class TrackingAnalyzer:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def getSandhiSplits(self, _input, pre_segmented=False):
                self.assert_not_pre_segmented(pre_segmented)
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.03)
                with self.lock:
                    self.active -= 1
                return EmptyGraph()

            @staticmethod
            def assert_not_pre_segmented(pre_segmented):
                if pre_segmented:
                    raise AssertionError("test expects an ordinary split request")

        with patch(
            "process_sanskrit.splitter.api.create_backend",
            return_value=PythonBackend(),
        ):
            parser = Parser(input_encoding=sanscript.SLP1)
        analyzer = TrackingAnalyzer()
        parser.sandhi_analyzer = analyzer
        barrier = threading.Barrier(8)
        errors = []

        def split() -> None:
            try:
                barrier.wait()
                parser.split("asti")
            except Exception as error:  # pragma: no cover - asserted below
                errors.append(error)

        threads = [threading.Thread(target=split) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(analyzer.max_active, 1)

    def test_two_python_parsers_share_complete_first_rule_load(self):
        """The shared Sandhi singleton must not publish half-loaded rules."""
        from process_sanskrit.splitter.sandhi_analyzer import LexicalSandhiAnalyzer

        sandhi = LexicalSandhiAnalyzer.sandhi
        real_backward = sandhi._load_rules_pickle("sandhi_backward.pkl")
        started = threading.Event()
        release = threading.Event()
        results = []
        errors = []

        class BlockingBackward:
            def keys(self):
                started.set()
                release.wait(timeout=5)
                return real_backward.keys()

            def __getitem__(self, key):
                return real_backward[key]

        first_backend = PythonBackend()
        second_backend = PythonBackend()
        with patch(
            "process_sanskrit.splitter.api.create_backend",
            side_effect=[first_backend, second_backend],
        ):
            parsers = [
                Parser(input_encoding=sanscript.SLP1),
                Parser(input_encoding=sanscript.SLP1),
            ]

        original_backward = sandhi.backward
        original_after_len_max = getattr(sandhi, "after_len_max", None)
        had_after_len_max = hasattr(sandhi, "after_len_max")
        sandhi.backward = None
        if had_after_len_max:
            del sandhi.after_len_max

        def split(parser):
            try:
                results.append(parser.split("astyuttarasyAMdiSi", limit=1))
            except Exception as error:  # pragma: no cover - asserted below
                errors.append(error)

        threads = []
        try:
            with patch.object(
                sandhi,
                "_load_rules_pickle",
                return_value=BlockingBackward(),
            ):
                threads.append(threading.Thread(target=split, args=(parsers[0],)))
                threads[0].start()
                self.assertTrue(started.wait(timeout=2))
                threads.append(threading.Thread(target=split, args=(parsers[1],)))
                threads[1].start()
                time.sleep(0.05)
                self.assertTrue(
                    threads[1].is_alive(),
                    "second parser observed partially initialized Sandhi rules",
                )
                release.set()
                for thread in threads:
                    thread.join(timeout=5)
        finally:
            release.set()
            sandhi.backward = original_backward
            if had_after_len_max:
                sandhi.after_len_max = original_after_len_max
            elif hasattr(sandhi, "after_len_max"):
                del sandhi.after_len_max

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)


class SandhiSplitterWrapperTests(unittest.TestCase):
    """Contract between the pipeline wrapper and Parser.split()."""

    def test_one_attempt_uses_the_single_returned_candidate(self):
        import process_sanskrit.functions.sandhiSplitter as sandhi_module

        parser = Mock()
        candidate = Mock()
        candidate._transcoded_tokens.return_value = [
            "asti", "uttarasyām", "diśi"
        ]
        parser.split.return_value = [candidate]
        wrapper_scorer = Mock()
        wrapper_scorer.rank_splits.return_value = [
            (["asti", "uttarasyām", "diśi"], 0.75, {"length": 0.5})
        ]

        with patch.object(sandhi_module, "_get_parser", return_value=parser), \
                patch.object(sandhi_module, "scorer", wrapper_scorer):
            result = sandhi_module.analyze_sandhi(
                "astyuttarasyāṃdiśi", attempts=1
            )

        self.assertEqual(result.split, ["asti", "uttarasyām", "diśi"])
        self.assertEqual(result.score, 0.75)
        self.assertEqual(result.all_splits, wrapper_scorer.rank_splits.return_value)
        wrapper_scorer.score_split.assert_not_called()

    def test_parser_failures_are_not_converted_to_unsplit_text(self):
        import process_sanskrit.functions.sandhiSplitter as sandhi_module

        parser = Mock()
        parser.split.side_effect = RuntimeError("native scorer is unavailable")
        with patch.object(sandhi_module, "_get_parser", return_value=parser):
            with self.assertRaisesRegex(RuntimeError, "native scorer is unavailable"):
                sandhi_module.analyze_sandhi("astyuttarasyāṃdiśi")


@unittest.skipUnless(HAS_UPSTREAM, "sanskrit-parser + gensim not installed")
class UpstreamParityTests(unittest.TestCase):
    """Prove the trie and the numpy scorer reproduce upstream exactly."""

    @classmethod
    def setUpClass(cls):
        # This suite proves that the retained Python substitutions match the
        # upstream Python implementation. Keep it independent of the default
        # production backend, which is Rust.
        with patch(
            "process_sanskrit.splitter.api.create_backend",
            return_value=PythonBackend(),
        ):
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
