#!/usr/bin/env python3
"""Measure import, processing speed, accuracy, and memory usage.

This benchmark intentionally compares required roots per case instead of a
single output digest because the Sanskrit parser permits multiple valid splits.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import resource
import runpy
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def _current_rss_mib() -> Optional[float]:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def cold_child() -> None:
    started = time.perf_counter()
    importlib.import_module("process_sanskrit")
    print(
        json.dumps(
            {
                "seconds": time.perf_counter() - started,
                "current_rss_mib": _current_rss_mib(),
                "peak_rss_mib": _peak_rss_mib(),
            }
        )
    )


def cold_runs(run_count: int) -> List[dict]:
    results = []
    for _ in range(run_count):
        command = [sys.executable, "-B", str(Path(__file__)), "--cold-child"]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    return results


def _flat_roots(result) -> set:
    roots = set()
    if isinstance(result, dict):
        roots.update(result)
        for values in result.values():
            roots.update(values)
    elif isinstance(result, (list, tuple, set)):
        for item in result:
            if isinstance(item, str):
                roots.add(item)
            elif isinstance(item, (list, tuple, set)):
                roots.update(value for value in item if isinstance(value, str))
    elif isinstance(result, str):
        roots.add(result)
    return roots


def suite_once() -> dict:
    from process_sanskrit import process
    from process_sanskrit.functions.sandhiSplitter import _get_parser
    from tests.datasets.testCases import test_cases

    # The old implementation constructed the parser during import. Explicitly
    # warm it here so lazy initialization is not counted as processing time.
    _get_parser()
    process("yoga", mode="parts", cached=False)

    timings = []
    accuracies = []
    cases = []
    for case in test_cases:
        started = time.perf_counter()
        output = process(case["input"], mode="parts", cached=False)
        timings.append((time.perf_counter() - started) * 1000)
        roots = _flat_roots(output)
        required = set(case["correct_split"])
        accuracy = len(required & roots) / len(required)
        accuracies.append(accuracy)
        cases.append(
            {
                "input": case["input"],
                "milliseconds": timings[-1],
                "required_roots": sorted(required),
                "returned_roots": sorted(roots),
                "accuracy": accuracy,
            }
        )

    ordered = sorted(timings)
    p95_position = 0.95 * (len(ordered) - 1)
    p95_lower = int(p95_position)
    p95_fraction = p95_position - p95_lower
    p95_upper = min(p95_lower + 1, len(ordered) - 1)
    p95 = ordered[p95_lower] + p95_fraction * (
        ordered[p95_upper] - ordered[p95_lower]
    )
    return {
        "case_count": len(cases),
        "mean_ms": statistics.mean(timings),
        "median_ms": statistics.median(timings),
        "p95_ms": p95,
        "max_ms": max(timings),
        "required_root_accuracy_pct": statistics.mean(accuracies) * 100,
        "current_rss_mib": _current_rss_mib(),
        "peak_rss_mib": _peak_rss_mib(),
        "cases": cases,
    }


def suite_runs(run_count: int) -> List[dict]:
    results = []
    for _ in range(run_count):
        command = [sys.executable, "-B", str(Path(__file__)), "--suite-child"]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    return results


def _timed(function, run_count: int) -> dict:
    function()
    values = []
    for _ in range(run_count):
        started = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "runs": run_count,
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": sorted(values)[int(0.95 * run_count) - 1],
    }


def microbenchmarks() -> dict:
    import regex
    from sqlalchemy import text

    from process_sanskrit import process
    from process_sanskrit.functions.SQLiteFind import SQLite_find_name
    from process_sanskrit.functions.dictionaryLookup import multidict
    from process_sanskrit.utils.databaseSetup import get_session

    session = get_session()

    def legacy_ambiguous_lookup():
        rows = session.execute(
            text("SELECT * FROM lgtab2 WHERE key=:word"),
            {"word": "viṃśatyai"},
        ).fetchall()
        outcome = []
        for _key, model, stem in rows:
            reference = session.execute(
                text(
                    "SELECT * FROM lgtab1 "
                    "WHERE stem=:stem AND model=:model"
                ),
                {"stem": stem, "model": model},
            ).fetchall()[0]
            # Keep the old lookup's parsing work in the comparison.
            regex.findall(r",(\p{L}+)", reference[2])[0]
            outcome.append(reference[3].split(":"))
        return outcome

    try:
        return {
            "exact_dictionary_yoga": _timed(
                lambda: multidict("yoga", "MW", session=session), 1000
            ),
            "ambiguous_morphology_legacy_n_plus_one": _timed(
                legacy_ambiguous_lookup, 500
            ),
            "ambiguous_morphology_indexed_join": _timed(
                lambda: SQLite_find_name("viṃśatyai", session=session), 500
            ),
            "warm_yoga": _timed(
                lambda: process("yoga", mode="parts", cached=False), 20
            ),
        }
    finally:
        session.close()


def cache_benchmarks() -> dict:
    """Measure cache miss, hit, memory, disk, and locked-write behavior."""
    from sqlalchemy import delete

    from process_sanskrit import process
    from process_sanskrit.functions.sandhiSplitter import _get_parser
    from process_sanskrit.utils.analysisCache import (
        ANALYSIS_ALGORITHM_VERSION,
        CacheKey,
        CacheRecord,
        analysis_cache_table,
        get_analysis_cache,
        lexicon_fingerprint,
        reset_analysis_cache,
    )

    text_to_process = "cittavṛttinirodhaḥ"
    _get_parser()
    process("yoga", mode="parts", cached=False)

    with tempfile.TemporaryDirectory(prefix="process-sanskrit-cache-") as directory:
        cache_path = Path(directory) / "analysis-cache.sqlite3"
        old_path = os.environ.get("PROCESS_SANSKRIT_CACHE_PATH")
        old_enabled = os.environ.get("PROCESS_SANSKRIT_CACHE_ENABLED")
        os.environ["PROCESS_SANSKRIT_CACHE_PATH"] = str(cache_path)
        os.environ["PROCESS_SANSKRIT_CACHE_ENABLED"] = "true"
        reset_analysis_cache()
        try:
            rss_before_cache = _current_rss_mib()
            bootstrap_started = time.perf_counter_ns()
            cache = get_analysis_cache()
            with cache.engine.connect():
                pass
            lexicon_fingerprint()
            bootstrap_ms = (
                time.perf_counter_ns() - bootstrap_started
            ) / 1_000_000

            # Warm parser/model state for this exact input. "Cold miss" here
            # means an absent cache row; one-time parser and cache bootstrap
            # costs are reported independently.
            process(text_to_process, mode="parts", cached=False)
            uncached_values = []
            miss_values = []
            uncached_outputs = []
            miss_outputs = []
            for _ in range(9):
                started = time.perf_counter_ns()
                uncached_outputs.append(
                    process(text_to_process, mode="parts", cached=False)
                )
                uncached_values.append(
                    (time.perf_counter_ns() - started) / 1_000_000
                )
                with cache.engine.begin() as connection:
                    connection.execute(delete(analysis_cache_table))
                started = time.perf_counter_ns()
                miss_outputs.append(
                    process(text_to_process, mode="parts", cached=True)
                )
                miss_values.append(
                    (time.perf_counter_ns() - started) / 1_000_000
                )

            hit_timings = []
            hit_output = None
            rss_before_hits = _current_rss_mib()
            for _ in range(100):
                started = time.perf_counter_ns()
                hit_output = process(text_to_process, mode="parts", cached=True)
                hit_timings.append(
                    (time.perf_counter_ns() - started) / 1_000_000
                )
            for _ in range(900):
                process(text_to_process, mode="parts", cached=True)
            rss_after = _current_rss_mib()

            locked_key = CacheKey.from_settings(
                normalized_input="locked-benchmark",
                analysis_kind="hybrid_morphology",
                algorithm_signature=ANALYSIS_ALGORITHM_VERSION,
                lexicon_fingerprint=lexicon_fingerprint(),
                settings={"attempts": 20, "score_threshold": 0.535},
            )
            lock_connection = sqlite3.connect(cache_path, timeout=0.1)
            lock_connection.execute("BEGIN IMMEDIATE")
            started = time.perf_counter_ns()
            cache.store(
                CacheRecord(
                    key=locked_key,
                    raw_input="locked-benchmark",
                    split=["locked", "benchmark"],
                )
            )
            locked_write_ms = (time.perf_counter_ns() - started) / 1_000_000
            lock_connection.rollback()
            lock_connection.close()

            uncached_median = statistics.median(uncached_values)
            miss_median = statistics.median(miss_values)
            hit_median = statistics.median(hit_timings)
            paired_overheads = [
                (miss - uncached) / uncached * 100
                for uncached, miss in zip(uncached_values, miss_values)
            ]
            miss_overhead_pct = statistics.median(paired_overheads)
            sizes = {}
            for suffix, name in (
                ("", "database_bytes"),
                ("-wal", "wal_bytes"),
                ("-shm", "shm_bytes"),
            ):
                candidate = Path(str(cache_path) + suffix)
                sizes[name] = candidate.stat().st_size if candidate.exists() else 0
            return {
                "input": text_to_process,
                "outputs_equal": all(
                    output == hit_output
                    for output in (*uncached_outputs, *miss_outputs)
                ),
                "uncached_ms": {
                    "runs": len(uncached_values),
                    "median": uncached_median,
                    "values": uncached_values,
                },
                "cache_miss_ms": {
                    "runs": len(miss_values),
                    "median": miss_median,
                    "values": miss_values,
                },
                "one_time_cache_bootstrap_ms": bootstrap_ms,
                "cache_miss_overhead_pct": miss_overhead_pct,
                "cache_miss_paired_overhead_values_pct": paired_overheads,
                "cache_hit_ms": {
                    "runs": len(hit_timings),
                    "median": hit_median,
                    "p95": sorted(hit_timings)[94],
                },
                "speedup": uncached_median / hit_median,
                "rss_before_cache_mib": rss_before_cache,
                "rss_before_1000_hits_mib": rss_before_hits,
                "rss_after_1000_hits_mib": rss_after,
                "rss_delta_after_1000_hits_mib": (
                    None
                    if rss_before_hits is None or rss_after is None
                    else rss_after - rss_before_hits
                ),
                "locked_write_ms": locked_write_ms,
                **sizes,
                "targets": {
                    "outputs_equal": all(
                        output == hit_output
                        for output in (*uncached_outputs, *miss_outputs)
                    ),
                    "speedup_at_least_5x": uncached_median / hit_median >= 5,
                    "miss_overhead_below_10_pct": miss_overhead_pct < 10,
                    "rss_delta_below_5_mib": (
                        None
                        if rss_before_hits is None or rss_after is None
                        else rss_after - rss_before_hits < 5
                    ),
                    "locked_write_below_500_ms": locked_write_ms < 500,
                },
            }
        finally:
            reset_analysis_cache()
            if old_path is None:
                os.environ.pop("PROCESS_SANSKRIT_CACHE_PATH", None)
            else:
                os.environ["PROCESS_SANSKRIT_CACHE_PATH"] = old_path
            if old_enabled is None:
                os.environ.pop("PROCESS_SANSKRIT_CACHE_ENABLED", None)
            else:
                os.environ["PROCESS_SANSKRIT_CACHE_ENABLED"] = old_enabled


def memory_child() -> None:
    database = ROOT / "process_sanskrit" / "resources" / "SQliteDB.sqlite"
    before = _current_rss_mib()
    connections = []
    for _ in range(5):
        connection = sqlite3.connect(
            "file:{}?mode=ro&immutable=1".format(database), uri=True
        )
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute(
            "SELECT SUM(LENGTH(cleaned_body)) FROM mw"
        ).fetchone()
        connections.append(connection)
    after = _current_rss_mib()
    print(
        json.dumps(
            {
                "five_connections_rss_mib": after,
                "five_connections_delta_mib": (
                    None if before is None or after is None else after - before
                ),
                "peak_rss_mib": _peak_rss_mib(),
            }
        )
    )


def references_child() -> None:
    namespace = runpy.run_path(
        str(ROOT / "process_sanskrit" / "utils" / "dictionary_references.py")
    )
    references = namespace["DICTIONARY_REFERENCES"]
    references["yoga"]
    print(
        json.dumps(
            {
                "rows": len(references),
                "current_rss_mib": _current_rss_mib(),
                "peak_rss_mib": _peak_rss_mib(),
            }
        )
    )


def memory_checks() -> dict:
    checks = {}
    for flag, name in (
        ("--memory-child", "five_connections"),
        ("--references-child", "dictionary_references"),
    ):
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__)), flag],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        checks[name] = json.loads(completed.stdout.strip().splitlines()[-1])
    return checks


def version_report() -> Dict[str, str]:
    report = {}
    distributions = {
        "gensim": "gensim",
        "sentencepiece": "sentencepiece",
        "numpy": "numpy",
        "scipy": "scipy",
        "marisa_trie": "marisa-trie",
        "networkx": "networkx",
        "sqlalchemy": "SQLAlchemy",
    }
    for package, distribution in distributions.items():
        try:
            report[package] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            report[package] = "absent"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cold-child", action="store_true")
    parser.add_argument("--suite-child", action="store_true")
    parser.add_argument("--memory-child", action="store_true")
    parser.add_argument("--references-child", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--no-enforce-targets",
        action="store_true",
        help="report cache target failures without returning a failing status",
    )
    args = parser.parse_args()

    if args.cold_child:
        cold_child()
        return
    if args.suite_child:
        print(json.dumps(suite_once(), ensure_ascii=False))
        return
    if args.memory_child:
        memory_child()
        return
    if args.references_child:
        references_child()
        return
    if args.cache_only:
        cache_report = cache_benchmarks()
        print(json.dumps(cache_report, ensure_ascii=False, indent=2))
        failed = [
            name
            for name, passed in cache_report["targets"].items()
            if passed is not True
        ]
        if failed and not args.no_enforce_targets:
            raise SystemExit("cache benchmark targets failed: " + ", ".join(failed))
        return

    report = {
        "environment": {
            "python": sys.version.split()[0],
            **version_report(),
        },
        "cold_import_runs": cold_runs(args.runs),
        "suite_runs": suite_runs(args.runs),
        "microbenchmarks": microbenchmarks(),
        "cache_benchmarks": cache_benchmarks(),
        "memory_checks": memory_checks(),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    failed = [
        name
        for name, passed in report["cache_benchmarks"]["targets"].items()
        if passed is not True
    ]
    if failed and not args.no_enforce_targets:
        raise SystemExit("cache benchmark targets failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
