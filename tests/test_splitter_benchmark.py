"""Fast contract tests for the splitter-only benchmark harness."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_splitter import (
    BenchmarkConfiguration,
    CandidateSnapshot,
    ConfigurationError,
    CorpusLoader,
    CorrectnessGate,
    CorrectnessComparator,
    Distribution,
    RuntimeProvenance,
)


class BenchmarkConfigurationTests(unittest.TestCase):
    def _mapping(self):
        return {
            "schema_version": 1,
            "backends": ["python"],
            "reference_backend": "python",
            "datasets": [
                {
                    "path": "tests/datasets/one.json",
                    "categories": ["long"],
                }
            ],
            "extra_cases": [
                {
                    "text": "pṛthagjanatvamityevamādibhedasamādānāḥ",
                    "category": "requested_complex",
                    "source": "user-request",
                    "warm_repetitions": 5,
                }
            ],
            "length_buckets": [
                {"name": "short", "min_length": 0, "max_length": 9},
                {"name": "long", "min_length": 10},
            ],
            "splitter": {"limit": 10, "score": True},
            "execution": {
                "warm_repetitions": 1,
                "cold_repetitions": 3,
                "warmup_cases": ["yoga"],
                "cold_case": "pṛthagjanatvamityevamādibhedasamādānāḥ",
                "require_release_native": True,
            },
            "output": {
                "path": "build/benchmarks/report.json",
                "include_cases": True,
            },
        }

    def test_paths_are_resolved_from_repository_root(self):
        root = Path("/tmp/project")
        config = BenchmarkConfiguration.from_mapping(self._mapping(), root)

        self.assertEqual(
            config.datasets[0].path,
            root / "tests/datasets/one.json",
        )
        self.assertEqual(
            config.output_path,
            root / "build/benchmarks/report.json",
        )
        self.assertEqual(config.length_bucket(9).name, "short")
        self.assertEqual(config.length_bucket(10).name, "long")
        self.assertTrue(config.require_release_native)

    def test_overlapping_length_buckets_are_rejected(self):
        mapping = self._mapping()
        mapping["length_buckets"] = [
            {"name": "first", "min_length": 0, "max_length": 10},
            {"name": "second", "min_length": 10},
        ]

        with self.assertRaisesRegex(ConfigurationError, "overlap"):
            BenchmarkConfiguration.from_mapping(mapping, Path("/tmp/project"))


class CorpusLoaderTests(unittest.TestCase):
    def test_deduplicates_text_while_retaining_all_memberships(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = root / "datasets"
            datasets.mkdir()
            (datasets / "one.json").write_text(
                json.dumps(
                    {
                        "compounds": {
                            "long": [
                                {"text": "same", "source_file": "one"},
                                {"text": "only-one", "source_file": "one"},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (datasets / "two.json").write_text(
                json.dumps(
                    {
                        "compounds": {
                            "medium": [
                                {"text": "same", "source_file": "two"}
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            mapping = BenchmarkConfigurationTests()._mapping()
            mapping["datasets"] = [
                {"path": "datasets/one.json", "categories": ["long"]},
                {"path": "datasets/two.json", "categories": ["medium"]},
            ]
            mapping["extra_cases"] = [
                {
                    "text": "extra",
                    "category": "requested_complex",
                    "source": "user-request",
                    "warm_repetitions": 3,
                }
            ]
            config = BenchmarkConfiguration.from_mapping(mapping, root)

            corpus = CorpusLoader(config).load()

            self.assertEqual(corpus.loaded_records, 4)
            self.assertEqual(corpus.duplicate_records, 1)
            self.assertEqual(len(corpus.cases), 3)
            same = next(case for case in corpus.cases if case.text == "same")
            self.assertEqual(same.categories, ("long", "medium"))
            self.assertEqual(
                same.dataset_categories,
                ("one.json:long", "two.json:medium"),
            )
            extra = next(case for case in corpus.cases if case.text == "extra")
            self.assertEqual(extra.focus_repetitions, 3)

    def test_committed_corpus_includes_both_files_and_requested_case(self):
        root = Path(__file__).resolve().parents[1]
        config = BenchmarkConfiguration.load(
            root / "benchmarks/splitter-benchmark.json", root
        )

        self.assertEqual(config.backends, ("python", "rust"))
        self.assertEqual(config.reference_backend, "python")
        self.assertEqual(config.warm_repetitions, 2)
        self.assertTrue(config.require_release_native)
        corpus = CorpusLoader(config).load()

        self.assertEqual(corpus.loaded_records, 701)
        self.assertEqual(len(corpus.cases), 700)
        self.assertEqual(corpus.duplicate_records, 1)
        focus = [case for case in corpus.cases if case.focus_repetitions]
        self.assertEqual(len(focus), 1)
        self.assertEqual(
            focus[0].text,
            "pṛthagjanatvamityevamādibhedasamādānāḥ",
        )
        self.assertEqual(focus[0].focus_repetitions, 20)


class CandidateDigestTests(unittest.TestCase):
    def test_multiset_digest_ignores_order_but_preserves_duplicates(self):
        first = CandidateSnapshot.from_candidates(
            [["b"], ["a"], ["a"]]
        )
        reordered = CandidateSnapshot.from_candidates(
            [["a"], ["b"], ["a"]]
        )
        fewer = CandidateSnapshot.from_candidates([["a"], ["b"]])

        self.assertNotEqual(first.ordered_digest, reordered.ordered_digest)
        self.assertEqual(first.multiset_digest, reordered.multiset_digest)
        self.assertNotEqual(first.multiset_digest, fewer.multiset_digest)
        self.assertNotEqual(first.winner_digest, reordered.winner_digest)

    def test_none_is_distinct_from_an_empty_candidate_list(self):
        no_split = CandidateSnapshot.from_candidates(None)
        empty = CandidateSnapshot.from_candidates([])

        self.assertTrue(no_split.no_split)
        self.assertFalse(empty.no_split)
        self.assertNotEqual(no_split.ordered_digest, empty.ordered_digest)


class DistributionTests(unittest.TestCase):
    def test_reports_linear_percentiles(self):
        distribution = Distribution.from_values([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(distribution.samples, 4)
        self.assertEqual(distribution.mean, 2.5)
        self.assertEqual(distribution.p50, 2.5)
        self.assertAlmostEqual(distribution.p95, 3.85)
        self.assertEqual(distribution.maximum, 4.0)


class RuntimeProvenanceTests(unittest.TestCase):
    def test_consistency_includes_both_cold_and_warm_workers(self):
        runtime = {
            "module_sha256": "same-module",
            "asset_sha256": "same-assets",
        }
        cold_runtimes = [copy.deepcopy(runtime), copy.deepcopy(runtime)]

        self.assertTrue(
            RuntimeProvenance.consistent((*cold_runtimes, runtime))
        )

        changed_warm_runtime = copy.deepcopy(runtime)
        changed_warm_runtime["module_sha256"] = "changed-module"
        self.assertFalse(
            RuntimeProvenance.consistent(
                (*cold_runtimes, changed_warm_runtime)
            )
        )


class CorrectnessComparatorTests(unittest.TestCase):
    def test_separates_candidate_set_order_and_winner_parity(self):
        reference = {
            "same": CandidateSnapshot.from_candidates(
                [["winner"], ["a"], ["b"]]
            ),
        }
        target = {
            "same": CandidateSnapshot.from_candidates(
                [["winner"], ["b"], ["a"]]
            ),
        }

        comparison = CorrectnessComparator(reference, target).summarize()

        self.assertEqual(comparison["case_count"], 1)
        self.assertEqual(comparison["candidate_multiset_matches"], 1)
        self.assertEqual(comparison["ordered_candidate_matches"], 0)
        self.assertEqual(comparison["winner_matches"], 1)
        self.assertTrue(comparison["behavioral_parity"])
        self.assertFalse(comparison["ordered_parity"])
        self.assertFalse(comparison["exact_parity"])
        self.assertEqual(comparison["mismatches"][0]["text"], "same")

    def test_candidate_or_winner_drift_fails_behavioral_parity(self):
        reference = {
            "different-candidates": CandidateSnapshot.from_candidates(
                [["a"], ["b"]]
            ),
            "different-winner": CandidateSnapshot.from_candidates(
                [["winner"], ["runner-up"]]
            ),
        }
        target = {
            "different-candidates": CandidateSnapshot.from_candidates(
                [["a"], ["c"]]
            ),
            "different-winner": CandidateSnapshot.from_candidates(
                [["runner-up"], ["winner"]]
            ),
        }

        comparison = CorrectnessComparator(reference, target).summarize()

        self.assertFalse(comparison["behavioral_parity"])
        self.assertFalse(comparison["exact_parity"])

    def test_identical_exceptions_do_not_count_as_behavioral_parity(self):
        reference = {
            "error": CandidateSnapshot.from_error("ValueError: bad"),
        }
        target = {
            "error": CandidateSnapshot.from_error("ValueError: bad"),
        }

        comparison = CorrectnessComparator(reference, target).summarize()

        self.assertFalse(comparison["behavioral_parity"])
        self.assertFalse(comparison["exact_parity"])
        self.assertEqual(comparison["mismatches"][0]["reference_error"], "ValueError: bad")
        self.assertEqual(comparison["mismatches"][0]["target_error"], "ValueError: bad")


class CorrectnessGateTests(unittest.TestCase):
    @staticmethod
    def _clean_report():
        backend = {
            "cold": {
                "deterministic": True,
                "runtime_consistent": True,
                "result": {"error": None},
                "runtime": {"build_profile": "interpreted"},
            },
            "warm": {
                "runtime": {"build_profile": "interpreted"},
                "correctness": {
                    "error_count": 0,
                    "nondeterministic_count": 0,
                },
                "focus_cases": [
                    {
                        "deterministic": True,
                        "result": {"error": None},
                    }
                ],
            },
        }
        report = {
            "comparisons": {"rust": {"behavioral_parity": True}},
            "backends": {
                "python": copy.deepcopy(backend),
                "rust": copy.deepcopy(backend),
            },
        }
        report["backends"]["rust"]["cold"]["runtime"]["build_profile"] = "release"
        report["backends"]["rust"]["warm"]["runtime"]["build_profile"] = "release"
        report["backends"]["python"]["runtime_consistent"] = True
        report["backends"]["rust"]["runtime_consistent"] = True
        return report

    def test_accepts_error_free_deterministic_behavioral_parity(self):
        self.assertTrue(CorrectnessGate.passes(self._clean_report()))

    def test_rejects_errors_nondeterminism_and_behavioral_drift(self):
        mutations = {
            "behavioral mismatch": lambda report: report["comparisons"]["rust"].update(
                behavioral_parity=False
            ),
            "warm error": lambda report: report["backends"]["rust"]["warm"][
                "correctness"
            ].update(error_count=1),
            "warm nondeterminism": lambda report: report["backends"]["rust"]["warm"][
                "correctness"
            ].update(nondeterministic_count=1),
            "cold error": lambda report: report["backends"]["rust"]["cold"][
                "result"
            ].update(error="RuntimeError: bad"),
            "cold nondeterminism": lambda report: report["backends"]["rust"]["cold"].update(
                deterministic=False
            ),
            "cold/warm runtime mismatch": lambda report: report["backends"]["rust"].update(
                runtime_consistent=False
            ),
            "focus error": lambda report: report["backends"]["rust"]["warm"][
                "focus_cases"
            ][0]["result"].update(error="RuntimeError: bad"),
            "focus nondeterminism": lambda report: report["backends"]["rust"]["warm"][
                "focus_cases"
            ][0].update(deterministic=False),
            "debug native cold worker": lambda report: report["backends"]["rust"]["cold"][
                "runtime"
            ].update(build_profile="debug"),
            "debug native warm worker": lambda report: report["backends"]["rust"]["warm"][
                "runtime"
            ].update(build_profile="debug"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                report = self._clean_report()
                mutate(report)
                self.assertFalse(CorrectnessGate.passes(report))

    def test_can_allow_debug_native_for_exploratory_runs(self):
        report = self._clean_report()
        report["backends"]["rust"]["cold"]["runtime"]["build_profile"] = "debug"
        report["backends"]["rust"]["warm"]["runtime"]["build_profile"] = "debug"

        self.assertTrue(
            CorrectnessGate.passes(report, require_release_native=False)
        )


if __name__ == "__main__":
    unittest.main()
