"""Contracts for explicit Python and default native splitter backends."""

import ast
import inspect
import json
import os
import shutil
import tempfile
import threading
import unittest
import warnings
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

from indic_transliteration import sanscript

from process_sanskrit.splitter import Parser, Split
from process_sanskrit.splitter import backends

try:
    from process_sanskrit.splitter._native import NativeSplitter

    HAS_NATIVE = True
except ImportError:
    NativeSplitter = None
    HAS_NATIVE = False


class BackendSelectionTests(unittest.TestCase):
    def setUp(self):
        self.original = os.environ.get(backends.BACKEND_ENVIRONMENT_VARIABLE)
        backends._reset_backend_state_for_tests()

    def tearDown(self):
        if self.original is None:
            os.environ.pop(backends.BACKEND_ENVIRONMENT_VARIABLE, None)
        else:
            os.environ[backends.BACKEND_ENVIRONMENT_VARIABLE] = self.original
        backends._reset_backend_state_for_tests()

    def test_rust_is_the_default_without_automatic_fallback(self):
        os.environ.pop(backends.BACKEND_ENVIRONMENT_VARIABLE, None)
        native = Mock()
        with patch.object(backends, "_get_native_splitter", return_value=native):
            parser = Parser(input_encoding=sanscript.SLP1)
        self.assertEqual(parser._backend.name, "rust")
        self.assertIs(parser._backend._native, native)

    def test_python_must_be_selected_explicitly(self):
        os.environ[backends.BACKEND_ENVIRONMENT_VARIABLE] = "python"
        parser = Parser(output_encoding="iast")
        result = parser.split("astyuttarasyAMdiSi", limit=1)
        self.assertEqual(
            ast.literal_eval(str(result[0])),
            ["asti", "uttarasyām", "di", "ṣi"],
        )

    def test_invalid_backend_values_are_rejected_verbatim(self):
        for value in ("", "Rust", " rust", "auto"):
            with self.subTest(value=value):
                backends._reset_backend_state_for_tests()
                os.environ[backends.BACKEND_ENVIRONMENT_VARIABLE] = value
                with self.assertRaisesRegex(ValueError, "must be one of"):
                    Parser()

    def test_selection_is_captured_on_first_use(self):
        os.environ[backends.BACKEND_ENVIRONMENT_VARIABLE] = "python"
        first = Parser()
        os.environ[backends.BACKEND_ENVIRONMENT_VARIABLE] = "rust"
        second = Parser()
        self.assertEqual(first._backend.name, "python")
        self.assertEqual(second._backend.name, "python")

    def test_native_boundary_receives_and_returns_slp1(self):
        os.environ[backends.BACKEND_ENVIRONMENT_VARIABLE] = "rust"
        native = Mock()
        native.split_slp1.return_value = [["yogaH", "citta", "vftti"]]
        with patch.object(backends, "_get_native_splitter", return_value=native):
            parser = Parser(output_encoding="iast")
            result = parser.split("yogaścittavṛttiḥ", limit=7)
        native.split_slp1.assert_called_once_with(
            "yogaScittavfttiH", 7, True
        )
        self.assertEqual(
            ast.literal_eval(str(result[0])), ["yogaḥ", "citta", "vṛtti"]
        )

    def test_public_parser_and_split_shapes_are_unchanged(self):
        self.assertEqual(
            list(inspect.signature(Parser).parameters),
            [
                "strict_io",
                "input_encoding",
                "output_encoding",
                "score",
                "replace_ending_visarga",
            ],
        )
        self.assertEqual(
            list(inspect.signature(Parser.split).parameters),
            ["self", "input_string", "limit", "pre_segmented"],
        )
        self.assertIn("split", Split.__dataclass_fields__)

    def test_negative_limit_preserves_the_public_value_error(self):
        os.environ[backends.BACKEND_ENVIRONMENT_VARIABLE] = "rust"
        native = Mock()
        with patch.object(backends, "_get_native_splitter", return_value=native):
            parser = Parser(input_encoding=sanscript.SLP1)
            for pre_segmented in (False, True):
                with self.subTest(pre_segmented=pre_segmented):
                    with self.assertRaisesRegex(ValueError, "Stop argument for islice"):
                        parser.split(
                            "asti", limit=-1, pre_segmented=pre_segmented
                        )
        native.split_slp1.assert_not_called()


@unittest.skipUnless(HAS_NATIVE, "native extension has not been built")
class NativeBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = os.environ.get(backends.BACKEND_ENVIRONMENT_VARIABLE)
        os.environ[backends.BACKEND_ENVIRONMENT_VARIABLE] = "python"
        backends._reset_backend_state_for_tests()
        cls.python = Parser(output_encoding="iast")
        os.environ[backends.BACKEND_ENVIRONMENT_VARIABLE] = "rust"
        backends._reset_backend_state_for_tests()
        cls.rust = Parser(output_encoding="iast")

    @classmethod
    def tearDownClass(cls):
        if cls.original is None:
            os.environ.pop(backends.BACKEND_ENVIRONMENT_VARIABLE, None)
        else:
            os.environ[backends.BACKEND_ENVIRONMENT_VARIABLE] = cls.original
        backends._reset_backend_state_for_tests()

    @staticmethod
    def _parser_with_backend(parser, **kwargs):
        """Construct a public parser around one already-selected backend."""
        with patch(
            "process_sanskrit.splitter.api.create_backend",
            return_value=parser._backend,
        ):
            return Parser(**kwargs)

    def configured_parser_pair(self, **kwargs):
        return tuple(
            self._parser_with_backend(parser, **kwargs)
            for parser in (self.python, self.rust)
        )

    @staticmethod
    def _canonical_paths(results):
        if results is None:
            return None
        return [
            tuple(token.canonical() for token in result.split)
            for result in results
        ]

    @staticmethod
    def _output_paths(results):
        if results is None:
            return None
        return [tuple(result._transcoded_tokens()) for result in results]

    def assert_parser_parity(self, word, limit=10, score=True):
        self.python.score = score
        self.rust.score = score
        reference = self.python.split(word, limit=limit)
        native = self.rust.split(word, limit=limit)
        reference_strings = [str(value) for value in reference or []]
        native_strings = [str(value) for value in native or []]
        self.assertEqual(reference is None, native is None, word)
        self.assertCountEqual(reference_strings, native_strings, word)
        self.assertEqual(reference_strings[:1], native_strings[:1], word)

    def test_native_reports_its_build_profile(self):
        from process_sanskrit.splitter import _native

        self.assertIn(_native.BUILD_PROFILE, ("debug", "release"))

    def test_known_encodings_and_complex_cutoff_cases_match(self):
        for word in (
            "astyuttarasyAMdiSi",
            "योगश्चित्तवृत्तिनिरोधः",
            "pṛthagjanatvamityevamādibhedasamādānāḥ",
            "vidvadvākyavihitakarmaṇorjhaṭityupasthiteḥ",
            "dṛḍhatayopapāditayordvitīyapakṣatṛtīyapakṣayorekahetunaiva",
        ):
            for limit in (1, 10):
                with self.subTest(word=word, limit=limit):
                    self.assert_parser_parity(word, limit=limit)

    def test_requested_complex_compound_winners_are_pinned_by_limit(self):
        word = "pṛthagjanatvamityevamādibhedasamādānāḥ"
        expected = {
            1: (
                "pṛthak",
                "jana",
                "tvam",
                "iti",
                "evam",
                "ādi",
                "bheda",
                "samāḥ",
                "da",
                "anāḥ",
            ),
            10: (
                "pṛthak",
                "jana",
                "tvam",
                "iti",
                "evam",
                "ādi",
                "bheda",
                "samā",
                "dānāḥ",
            ),
        }

        for parser in (self.python, self.rust):
            for limit, expected_tokens in expected.items():
                with self.subTest(backend=parser._backend.name, limit=limit):
                    result = parser.split(word, limit=limit)
                    self.assertIsNotNone(result)
                    self.assertEqual(
                        tuple(result[0]._transcoded_tokens()), expected_tokens
                    )

    def test_pre_segmented_known_and_unknown_segments_match(self):
        text = "asti astyuttarasyAMdiSi"
        outcomes = []
        for parser in self.configured_parser_pair(
            input_encoding=sanscript.SLP1,
            output_encoding="iast",
        ):
            with self.assertLogs(
                "process_sanskrit.splitter.api", level="WARNING"
            ) as captured:
                result = parser.split(text, pre_segmented=True)
            outcomes.append(
                (
                    [value._transcoded_tokens() for value in result],
                    [record.getMessage() for record in captured.records],
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(
            outcomes[0],
            (
                [["asti", "asti", "uttarasyām", "diśi"]],
                ["Unknown pada astyuttarasyAMdiSi - will be split"],
            ),
        )

    def test_no_split_returns_none_and_emits_the_public_warning(self):
        outcomes = []
        for parser in self.configured_parser_pair(
            input_encoding=sanscript.SLP1,
            output_encoding="iast",
        ):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                result = parser.split("xyz")
            outcomes.append(
                (
                    result,
                    [
                        (type(item.message), str(item.message))
                        for item in captured
                    ],
                )
            )

        expected_warning = (
            UserWarning,
            "No splits found. Please check the input to ensure there are no typos.",
        )
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], (None, [expected_warning]))

    def test_public_misdeclared_non_ascii_slp1_remains_a_no_split(self):
        outcomes = []
        for parser in self.configured_parser_pair(
            input_encoding=sanscript.SLP1,
            output_encoding=sanscript.SLP1,
        ):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                result = parser.split("rāma", limit=1)
            outcomes.append(
                (
                    result,
                    [str(item.message) for item in captured],
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(
            outcomes[0],
            (
                None,
                [
                    "No splits found. Please check the input to ensure there are no typos."
                ],
            ),
        )

    def test_pre_segmented_non_ascii_slp1_preserves_python_failure_shape(self):
        outcomes = []
        for parser in self.configured_parser_pair(
            input_encoding=sanscript.SLP1,
            output_encoding=sanscript.SLP1,
        ):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    parser.split("rāma", limit=1, pre_segmented=True)
                except Exception as error:
                    outcome = (type(error), str(error))
                else:  # pragma: no cover - the reference currently raises
                    outcome = None
            outcomes.append(
                (outcome, [str(item.message) for item in captured])
            )

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][0][0], TypeError)
        self.assertIn("NoneType", outcomes[0][0][1])
        self.assertEqual(
            outcomes[0][1],
            ["No splits found. Please check the input to ensure there are no typos."],
        )

    def test_strict_io_and_ending_visarga_options_match(self):
        cases = (
            (False, None, [("rAmaH",)]),
            (False, "s", None),
            (False, "r", [("rAm", "aH")]),
            (True, None, [("rAmas",)]),
            (True, "s", [("rAmas",)]),
            (True, "r", [("rAmas",)]),
        )
        no_split_warning = (
            UserWarning,
            "No splits found. Please check the input to ensure there are no typos.",
        )

        for strict_io, replace_ending_visarga, expected in cases:
            with self.subTest(
                strict_io=strict_io,
                replace_ending_visarga=replace_ending_visarga,
            ):
                outcomes = []
                for parser in self.configured_parser_pair(
                    strict_io=strict_io,
                    input_encoding=sanscript.SLP1,
                    output_encoding=sanscript.SLP1,
                    replace_ending_visarga=replace_ending_visarga,
                ):
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        result = parser.split("rAmaH", limit=1)
                    outcomes.append(
                        (
                            self._output_paths(result),
                            [
                                (type(item.message), str(item.message))
                                for item in captured
                            ],
                        )
                    )

                self.assertEqual(outcomes[0], outcomes[1])
                self.assertEqual(outcomes[0][0], expected)
                self.assertEqual(
                    outcomes[0][1],
                    [no_split_warning] if expected is None else [],
                )

    def test_unknown_sentencepiece_surface_scores_match(self):
        sequences = [
            ["J"],
            ["Q"],
            ["J", "Q"],
            [
                "vidvat", "vAkya", "vihita", "karmaRA", "ur", "Jawiti",
                "upa", "sTites",
            ],
        ]
        reference = self.python._backend.score_slp1(sequences)
        native = self.rust._backend.score_slp1(sequences)
        for sequence, expected, actual in zip(sequences, reference, native):
            with self.subTest(sequence=sequence):
                self.assertAlmostEqual(actual, expected, delta=1e-3)

    def test_limit_and_scoring_contracts_match(self):
        for limit, scored in ((1, True), (10, True)):
            with self.subTest(limit=limit, scored=scored):
                self.assert_parser_parity(
                    "astyuttarasyAMdiSi", limit=limit, score=scored
                )

        # With score=False every equal-length path has the same NetworkX
        # weight. Python's candidate order, winner, and a subset cut through a
        # tie group all depend on process hash order. Compare against its
        # complete path multiset after applying the native deterministic key.
        word = "astyuttarasyAMdiSi"
        reference_all = self.python._backend.split_slp1(
            word, limit=1001, scored=False
        )
        canonical = sorted(reference_all, key=lambda path: (len(path), path))
        for limit in (10, 1001):
            with self.subTest(limit=limit, scored=False):
                native = self.rust._backend.split_slp1(
                    word, limit=limit, scored=False
                )
                expected = canonical[:limit] if limit <= 1000 else canonical
                self.assertEqual(native, expected)

    def test_scored_all_path_public_output_contract_matches(self):
        word = "astyuttarasyAMdiSi"
        parsers = self.configured_parser_pair(
            input_encoding=sanscript.SLP1,
            output_encoding="iast",
            score=True,
        )
        results = [parser.split(word, limit=1001) for parser in parsers]
        paths = [self._canonical_paths(result) for result in results]

        # Upstream scores the graph but then takes its all-simple-path branch,
        # whose same-length traversal order is process-dependent. Candidate
        # content and the public wrapper contract are the stable guarantees.
        self.assertEqual(Counter(paths[0]), Counter(paths[1]))
        self.assertGreater(len(paths[0]), 0)
        for parser, result in zip(parsers, results):
            self.assertTrue(all(isinstance(value, Split) for value in result))
            self.assertTrue(all(value.parser is parser for value in result))
            self.assertTrue(all(value.input_string == word for value in result))
            self.assertTrue(
                all(
                    all(isinstance(token, str) for token in value._transcoded_tokens())
                    for value in result
                )
            )

    def test_scored_all_path_errors_remain_fatal_at_public_boundary(self):
        word = "astyuttarasyAMdiSi"
        for parser in self.configured_parser_pair(
            input_encoding=sanscript.SLP1,
            score=True,
        ):
            with self.subTest(backend=parser._backend.name), patch.object(
                parser._backend,
                "split_slp1",
                side_effect=RuntimeError("scorer is unavailable"),
            ) as split_slp1, warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                with self.assertRaisesRegex(RuntimeError, "scorer is unavailable"):
                    parser.split(word, limit=1001)

            split_slp1.assert_called_once_with(
                "astyuttarasyAndiSi", limit=1001, scored=True
            )
            self.assertEqual(captured, [])

    def test_shared_native_parser_handles_concurrent_requests(self):
        words = [
            "astyuttarasyAMdiSi",
            "pṛthagjanatvamityevamādibhedasamādānāḥ",
            "yogaścittavṛttinirodhaḥ",
        ]
        expected = {word: str(self.rust.split(word, limit=1)[0]) for word in words}
        barrier = threading.Barrier(12)
        failures = []

        def split(index):
            try:
                word = words[index % len(words)]
                barrier.wait()
                actual = str(self.rust.split(word, limit=1)[0])
                if actual != expected[word]:
                    failures.append((word, actual))
            except Exception as error:  # pragma: no cover - asserted below
                failures.append(error)

        threads = [threading.Thread(target=split, args=(index,)) for index in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(failures, [])

    def test_corrupt_native_assets_are_rejected_before_use(self):
        source = Path(__file__).parents[1] / "process_sanskrit" / "splitter" / "data" / "native"
        for asset in ("forms.fst", "sentencepiece.model"):
            with self.subTest(asset=asset), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "native"
                shutil.copytree(source, target)
                corrupt = target / asset
                data = bytearray(corrupt.read_bytes())
                data[-1] ^= 0xFF
                corrupt.write_bytes(data)
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    NativeSplitter(str(target))

    def test_native_assets_load_from_a_unicode_install_path(self):
        source = Path(__file__).parents[1] / "process_sanskrit" / "splitter" / "data" / "native"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "saṃskṛta-資料"
            shutil.copytree(source, target)
            native = NativeSplitter(str(target))
            self.assertTrue(native.valid_slp1("asti"))

    @unittest.skipUnless(
        os.environ.get("PROCESS_SANSKRIT_FULL_NATIVE_PARITY") == "1",
        "set PROCESS_SANSKRIT_FULL_NATIVE_PARITY=1 for all complex compounds",
    )
    def test_all_complex_compounds_match(self):
        datasets = Path(__file__).parent / "datasets"
        words = []
        for name in (
            "sanskrit_compounds_benchmark.json",
            "sanskrit_compounds_benchmark2.json",
        ):
            groups = json.loads((datasets / name).read_text())["compounds"]
            words.extend(row["text"] for rows in groups.values() for row in rows)
        for word in dict.fromkeys(words):
            with self.subTest(word=word):
                self.assert_parser_parity(word)


if __name__ == "__main__":
    unittest.main()
