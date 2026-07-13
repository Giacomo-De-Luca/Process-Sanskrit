"""Tests for the persistent, disk-backed Sanskrit analysis cache."""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import func, select, text
from sqlalchemy.pool import QueuePool

from process_sanskrit.utils.analysisCache import (
    ANALYSIS_ALGORITHM_VERSION,
    AnalysisCache,
    CacheConfig,
    CacheConfigurationError,
    CacheKey,
    CacheRecord,
    TaggedJSON,
    analysis_cache_table,
    get_analysis_cache,
    reset_analysis_cache,
    resolve_cache_enabled,
)


def _multiprocess_cache_writer(path: str, start_event, result_queue, index: int) -> None:
    """Independent worker used to race schema initialization and one insert."""
    try:
        config = CacheConfig(
            enabled=True,
            retention="prune",
            max_age_days=90,
            path=Path(path),
            busy_timeout_ms=500,
            pool_timeout_seconds=0.5,
        )
        cache = AnalysisCache(config)
        key = CacheKey.from_settings(
            normalized_input="concurrent",
            analysis_kind="hybrid_morphology",
            algorithm_signature=ANALYSIS_ALGORITHM_VERSION,
            lexicon_fingerprint="lexicon-v1",
            settings={"attempts": 20},
        )
        start_event.wait(5)
        cache.store(
            CacheRecord(
                key=key,
                raw_input=f"raw-{index}",
                split=[f"split-{index}"],
                grammar=[["root", ("Nom", "Sg")]],
            )
        )
        cache.close()
        result_queue.put(None)
    except BaseException as error:
        result_queue.put(repr(error))


class MutableClock:
    def __init__(self, value: int = 1_700_000_000):
        self.value = value

    def __call__(self) -> int:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += seconds


class AnalysisCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "analysis.sqlite3"
        self.clock = MutableClock()
        self.config = CacheConfig(
            enabled=True,
            retention="prune",
            max_age_days=90,
            path=self.path,
            busy_timeout_ms=100,
            pool_timeout_seconds=0.1,
            prune_interval_seconds=86_400,
            touch_interval_seconds=86_400,
            prune_batch_size=2,
        )
        self.cache = AnalysisCache(self.config, clock=self.clock)
        self.addCleanup(self.cache.close)

    def key(self, **changes) -> CacheKey:
        values = {
            "normalized_input": "yogena",
            "analysis_kind": "hybrid_morphology",
            "algorithm_signature": ANALYSIS_ALGORITHM_VERSION,
            "lexicon_fingerprint": "lexicon-v1",
            "settings": {"attempts": 20, "score_threshold": 0.535},
        }
        values.update(changes)
        return CacheKey.from_settings(**values)

    def record(self, **changes) -> CacheRecord:
        values = {
            "key": self.key(),
            "raw_input": "योगेन",
            "split": ["yoga", "ena"],
            "grammar": [["yoga", "m_a", [("Ins", "Sg")]]],
            "score": 0.625,
            "subscores": {"length": 0.4, "morphology": 0.225},
            "result_source": "statistical",
            "status": "success",
            "compute_ms": 123.5,
        }
        values.update(changes)
        return CacheRecord(**values)

    def test_defaults_and_environment_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            config = CacheConfig.from_environment()
        self.assertTrue(config.enabled)
        self.assertEqual(config.retention, "prune")
        self.assertEqual(config.max_age_days, 90)
        self.assertEqual(config.path.name, "analysis-cache.sqlite3")

        with patch.dict(
            os.environ,
            {"PROCESS_SANSKRIT_CACHE_RETENTION": "invalid"},
            clear=True,
        ):
            with self.assertRaises(CacheConfigurationError):
                CacheConfig.from_environment()

    def test_package_import_does_not_suppress_host_application_logging(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import logging; "
                "logging.getLogger().setLevel(logging.INFO); "
                "import process_sanskrit; "
                "print(logging.getLogger().level)",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(int(completed.stdout.strip()), logging.INFO)

    def test_tagged_json_round_trip_preserves_container_shapes(self):
        value = {
            "list": ["yoga", ("Ins", "Sg")],
            "tuple": (1, [2, None, True]),
            "float": 0.535,
        }
        decoded = TaggedJSON.loads(TaggedJSON.dumps(value))
        self.assertEqual(decoded, value)
        self.assertIsInstance(decoded["list"], list)
        self.assertIsInstance(decoded["list"][1], tuple)
        self.assertIsInstance(decoded["tuple"], tuple)
        self.assertIsNot(decoded, value)

        for invalid in (math.inf, math.nan, {1: "non-string key"}, object()):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaises((TypeError, ValueError)):
                    TaggedJSON.dumps(invalid)
        with self.assertRaises(ValueError):
            TaggedJSON.loads('{"__process_sanskrit_type__":"unknown","items":[]}')

    def test_schema_pool_and_first_writer_wins(self):
        first = self.cache.store(self.record())
        second = self.cache.store(
            self.record(raw_input="yogena", split=["different"], score=0.1)
        )

        self.assertEqual(first.split, ["yoga", "ena"])
        self.assertEqual(second.split, first.split)
        self.assertEqual(second.raw_input, "योगेन")
        self.assertIsInstance(self.cache.engine.pool, QueuePool)
        self.assertEqual(self.cache.engine.pool.size(), 1)
        self.assertEqual(self.cache.engine.pool._max_overflow, 1)

        with self.cache.engine.connect() as connection:
            row_count = connection.scalar(
                select(func.count()).select_from(analysis_cache_table)
            )
            pragmas = {
                "journal_mode": connection.exec_driver_sql(
                    "PRAGMA journal_mode"
                ).scalar(),
                "cache_size": connection.exec_driver_sql(
                    "PRAGMA cache_size"
                ).scalar(),
                "auto_vacuum": connection.exec_driver_sql(
                    "PRAGMA auto_vacuum"
                ).scalar(),
            }
        self.assertEqual(row_count, 1)
        self.assertEqual(pragmas["journal_mode"].lower(), "wal")
        self.assertEqual(pragmas["cache_size"], -2048)
        self.assertEqual(pragmas["auto_vacuum"], 2)

    def test_lookup_requires_matching_settings_algorithm_and_lexicon(self):
        self.cache.store(self.record())
        self.assertIsNotNone(self.cache.get(self.key()))
        self.assertIsNone(
            self.cache.get(self.key(settings={"attempts": 10}))
        )
        self.assertIsNone(
            self.cache.get(self.key(algorithm_signature="analysis-v2"))
        )
        self.assertIsNone(
            self.cache.get(self.key(lexicon_fingerprint="lexicon-v2"))
        )

    def test_reopening_evicts_records_from_superseded_algorithm_versions(self):
        ## the signature is part of the key, so a stale record is already
        ## unreadable -- but nothing would ever delete it, and the file would grow
        ## a full copy of the output of every version of the splitter ever shipped
        self.cache.store(self.record())
        self.cache.store(
            self.record(
                key=self.key(
                    normalized_input="stale",
                    algorithm_signature="hybrid-morphology-v0",
                )
            )
        )
        with self.cache.engine.connect() as connection:
            self.assertEqual(
                connection.scalar(
                    select(func.count()).select_from(analysis_cache_table)
                ),
                2,
            )
        self.cache.close()

        reopened = AnalysisCache(self.config, clock=self.clock)
        self.addCleanup(reopened.close)
        with reopened.engine.connect() as connection:
            signatures = connection.execute(
                select(analysis_cache_table.c.algorithm_signature)
            ).scalars().all()

        self.assertEqual(signatures, [ANALYSIS_ALGORITHM_VERSION])
        ## and the current record is still served
        self.assertIsNotNone(reopened.get(self.key()))

    def test_disabled_cache_never_creates_a_file(self):
        disabled_path = self.path.with_name("disabled.sqlite3")
        cache = AnalysisCache(
            CacheConfig(
                enabled=False,
                retention="prune",
                max_age_days=90,
                path=disabled_path,
            )
        )
        self.addCleanup(cache.close)
        self.assertIsNone(cache.get(self.key()))
        self.assertEqual(cache.store(self.record()).split, ["yoga", "ena"])
        self.assertFalse(disabled_path.exists())
        self.assertFalse(resolve_cache_enabled(False, configured_default=True))
        self.assertTrue(resolve_cache_enabled(True, configured_default=False))

    def test_prune_uses_last_access_and_keep_all_never_deletes(self):
        old = self.record()
        self.cache.store(old)
        self.clock.advance(89 * 86_400)
        recent_key = self.key(normalized_input="recent")
        self.cache.store(self.record(key=recent_key, raw_input="recent"))
        self.clock.advance(2 * 86_400)

        self.assertEqual(self.cache.prune(force=True), 1)
        self.assertIsNone(self.cache.get(old.key))
        self.assertIsNotNone(self.cache.get(recent_key))

        keep_path = self.path.with_name("keep.sqlite3")
        keep_clock = MutableClock()
        keep = AnalysisCache(
            CacheConfig(
                enabled=True,
                retention="keep_all",
                max_age_days=90,
                path=keep_path,
            ),
            clock=keep_clock,
        )
        self.addCleanup(keep.close)
        keep.store(self.record())
        keep_clock.advance(365 * 86_400)
        self.assertEqual(keep.prune(force=True), 0)
        self.assertIsNotNone(keep.get(self.key()))

    def test_prune_boundary_is_strictly_older_than_max_age(self):
        self.cache.store(self.record())
        self.clock.advance(90 * 86_400)
        self.assertEqual(self.cache.prune(force=True), 0)
        with self.cache.engine.connect() as connection:
            self.assertEqual(
                connection.scalar(
                    select(func.count()).select_from(analysis_cache_table)
                ),
                1,
            )
        self.clock.advance(1)
        self.assertEqual(self.cache.prune(force=True), 1)
        self.assertIsNone(self.cache.get(self.key()))

    def test_daily_touch_keeps_a_frequently_used_record(self):
        self.cache.store(self.record())
        self.clock.advance(89 * 86_400)
        self.assertIsNotNone(self.cache.get(self.key()))
        self.clock.advance(2 * 86_400)
        self.assertEqual(self.cache.prune(force=True), 0)
        self.assertIsNotNone(self.cache.get(self.key()))

    def test_recent_touch_survives_cache_reopen(self):
        reopen_path = self.path.with_name("reopen.sqlite3")
        first = AnalysisCache(
            CacheConfig(
                enabled=True,
                retention="prune",
                max_age_days=90,
                path=reopen_path,
            ),
            clock=self.clock,
        )
        first.store(self.record())
        self.clock.advance(89 * 86_400)
        self.assertIsNotNone(first.get(self.key()))
        first.close()

        reopened = AnalysisCache(
            CacheConfig(
                enabled=True,
                retention="prune",
                max_age_days=90,
                path=reopen_path,
            ),
            clock=self.clock,
        )
        self.addCleanup(reopened.close)
        self.clock.advance(2 * 86_400)
        self.assertEqual(reopened.prune(force=True), 0)
        self.assertIsNotNone(reopened.get(self.key()))

    def test_corrupt_payload_invalidates_only_its_row(self):
        good_key = self.key(normalized_input="good")
        bad_key = self.key(normalized_input="bad")
        self.cache.store(self.record(key=good_key, raw_input="good"))
        self.cache.store(self.record(key=bad_key, raw_input="bad"))
        with self.cache.engine.begin() as connection:
            connection.execute(
                analysis_cache_table.update()
                .where(analysis_cache_table.c.normalized_input == "bad")
                .values(split_payload=TaggedJSON.dumps("not-a-split-list"))
            )

        self.assertIsNone(self.cache.get(bad_key))
        self.assertIsNotNone(self.cache.get(good_key))
        with self.cache.engine.connect() as connection:
            remaining = connection.scalar(
                select(func.count()).select_from(analysis_cache_table)
            )
        self.assertEqual(remaining, 1)

    def test_corrupt_scalar_invalidates_only_its_row(self):
        good_key = self.key(normalized_input="good-scalar")
        bad_key = self.key(normalized_input="bad-scalar")
        self.cache.store(self.record(key=good_key, raw_input="good"))
        self.cache.store(self.record(key=bad_key, raw_input="bad"))
        with self.cache.engine.begin() as connection:
            connection.execute(
                analysis_cache_table.update()
                .where(analysis_cache_table.c.normalized_input == "bad-scalar")
                .values(last_accessed_at="not-an-integer")
            )

        self.assertIsNone(self.cache.get(bad_key))
        self.assertIsNotNone(self.cache.get(good_key))

    def test_corrupt_maintenance_metadata_fails_open(self):
        metadata_path = self.path.with_name("metadata.sqlite3")
        seed = AnalysisCache(
            CacheConfig(
                enabled=True,
                retention="prune",
                max_age_days=90,
                path=metadata_path,
            )
        )
        seed.store(self.record())
        seed.close()
        with sqlite3.connect(metadata_path) as connection:
            connection.execute(
                "UPDATE cache_metadata SET value='invalid' "
                "WHERE key='last_pruned_at'"
            )

        cache = AnalysisCache(
            CacheConfig(
                enabled=True,
                retention="prune",
                max_age_days=90,
                path=metadata_path,
            )
        )
        self.addCleanup(cache.close)
        self.assertIsNone(cache.get(self.key()))

    def test_concurrent_first_writers_leave_one_valid_record(self):
        barrier = threading.Barrier(6)
        outcomes = []

        def writer(index: int) -> None:
            barrier.wait()
            outcomes.append(
                self.cache.store(
                    self.record(raw_input=f"raw-{index}", split=[f"split-{index}"])
                )
            )

        threads = [threading.Thread(target=writer, args=(index,)) for index in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        canonical = self.cache.get(self.key())
        self.assertIsNotNone(canonical)
        self.assertTrue(all(outcome.split == canonical.split for outcome in outcomes))
        with self.cache.engine.connect() as connection:
            count = connection.scalar(
                select(func.count()).select_from(analysis_cache_table)
            )
        self.assertEqual(count, 1)

    def test_locked_database_fails_open_quickly(self):
        self.cache.store(self.record())
        raw = sqlite3.connect(self.path, timeout=0.1)
        self.addCleanup(raw.close)
        raw.execute("BEGIN IMMEDIATE")
        started = time.perf_counter()
        outcome = self.cache.store(
            self.record(key=self.key(normalized_input="locked"))
        )
        elapsed = time.perf_counter() - started
        raw.rollback()
        self.assertEqual(outcome.split, ["yoga", "ena"])
        self.assertLess(elapsed, 0.5)

    def test_pool_exhaustion_fails_open_quickly(self):
        engine = self.cache.engine
        with engine.connect(), engine.connect():
            started = time.perf_counter()
            self.assertIsNone(self.cache.get(self.key()))
            elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.5)

    @unittest.skipIf(os.name == "nt", "POSIX permission semantics required")
    def test_unwritable_path_fails_open_without_creating_a_file(self):
        unwritable_parent = self.path.parent / "unwritable"
        unwritable_parent.mkdir(mode=0o500)
        unwritable_parent.chmod(0o500)
        self.addCleanup(unwritable_parent.chmod, 0o700)
        unwritable_path = unwritable_parent / "analysis.sqlite3"
        cache = AnalysisCache(
            CacheConfig(
                enabled=True,
                retention="prune",
                max_age_days=90,
                path=unwritable_path,
            )
        )
        self.addCleanup(cache.close)

        self.assertIsNone(cache.get(self.key()))
        self.assertFalse(unwritable_path.exists())

    def test_corrupt_database_fails_open_without_replacing_it(self):
        corrupt_path = self.path.with_name("corrupt.sqlite3")
        corrupt_path.write_bytes(b"not a sqlite database")
        cache = AnalysisCache(
            CacheConfig(
                enabled=True,
                retention="prune",
                max_age_days=90,
                path=corrupt_path,
            )
        )
        self.addCleanup(cache.close)
        with self.assertLogs(
            "process_sanskrit.utils.analysisCache", level="WARNING"
        ) as captured:
            self.assertIsNone(cache.get(self.key()))
            self.assertIsNone(cache.get(self.key()))
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(corrupt_path.read_bytes(), b"not a sqlite database")

    def test_unrelated_database_fails_open_without_adding_cache_tables(self):
        unrelated_path = self.path.with_name("unrelated.sqlite3")
        with sqlite3.connect(unrelated_path) as connection:
            connection.execute("CREATE TABLE user_data (value TEXT NOT NULL)")
            connection.execute("INSERT INTO user_data VALUES ('preserve me')")

        cache = AnalysisCache(
            CacheConfig(
                enabled=True,
                retention="prune",
                max_age_days=90,
                path=unrelated_path,
            )
        )
        self.addCleanup(cache.close)

        self.assertIsNone(cache.get(self.key()))
        with sqlite3.connect(unrelated_path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            value = connection.execute("SELECT value FROM user_data").fetchone()[0]
        self.assertEqual(tables, {"user_data"})
        self.assertEqual(value, "preserve me")

    def test_special_characters_in_path_address_the_exact_file(self):
        special_path = self.path.with_name("cache?variant.sqlite3")
        cache = AnalysisCache(
            CacheConfig(
                enabled=True,
                retention="prune",
                max_age_days=90,
                path=special_path,
            )
        )
        self.addCleanup(cache.close)
        cache.store(self.record())

        self.assertEqual(Path(cache.engine.url.database), special_path)
        with sqlite3.connect(special_path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        self.assertIn("analysis_cache", tables)
        self.assertFalse(special_path.with_name("cache").exists())

    @unittest.skipIf(os.name == "nt", "POSIX permission semantics required")
    def test_existing_parent_directory_permissions_are_not_changed(self):
        existing_parent = self.path.parent / "shared-parent"
        existing_parent.mkdir(mode=0o755)
        existing_parent.chmod(0o755)
        cache = AnalysisCache(
            CacheConfig(
                enabled=True,
                retention="prune",
                max_age_days=90,
                path=existing_parent / "analysis.sqlite3",
            )
        )
        self.addCleanup(cache.close)
        cache.store(self.record())
        self.assertEqual(existing_parent.stat().st_mode & 0o777, 0o755)

    @unittest.skipIf(os.name == "nt", "POSIX permission semantics required")
    def test_new_cache_file_is_private(self):
        self.cache.store(self.record())
        self.assertEqual(self.path.stat().st_mode & 0o077, 0)

    def test_multiprocess_schema_and_insert_race_leaves_one_record(self):
        race_path = self.path.with_name("multiprocess.sqlite3")
        context = multiprocessing.get_context("spawn")
        start_event = context.Event()
        result_queue = context.Queue()
        workers = [
            context.Process(
                target=_multiprocess_cache_writer,
                args=(str(race_path), start_event, result_queue, index),
            )
            for index in range(4)
        ]
        for worker in workers:
            worker.start()
        start_event.set()
        for worker in workers:
            worker.join(timeout=10)
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertTrue(all(worker.exitcode == 0 for worker in workers))
        errors = [result_queue.get(timeout=2) for _ in workers]
        self.assertEqual(errors, [None] * len(workers))

        verification = AnalysisCache(
            CacheConfig(
                enabled=True,
                retention="prune",
                max_age_days=90,
                path=race_path,
            )
        )
        self.addCleanup(verification.close)
        with verification.engine.connect() as connection:
            count = connection.scalar(
                select(func.count()).select_from(analysis_cache_table)
            )
        self.assertEqual(count, 1)

    def test_query_plans_use_key_and_expiry_indexes(self):
        self.cache.store(self.record())
        with self.cache.engine.connect() as connection:
            key_plan = connection.execute(
                text(
                    "EXPLAIN QUERY PLAN SELECT * FROM analysis_cache "
                    "WHERE normalized_input=:text AND analysis_kind=:kind "
                    "AND algorithm_signature=:algorithm "
                    "AND lexicon_fingerprint=:lexicon AND settings_json=:settings"
                ),
                {
                    "text": self.key().normalized_input,
                    "kind": self.key().analysis_kind,
                    "algorithm": self.key().algorithm_signature,
                    "lexicon": self.key().lexicon_fingerprint,
                    "settings": self.key().settings_json,
                },
            ).fetchall()
            prune_plan = connection.execute(
                text(
                    "EXPLAIN QUERY PLAN SELECT id FROM analysis_cache "
                    "WHERE last_accessed_at < :cutoff "
                    "ORDER BY last_accessed_at, id LIMIT 10"
                ),
                {"cutoff": self.clock()},
            ).fetchall()
        self.assertTrue(any("INDEX" in row[3] for row in key_plan), key_plan)
        self.assertTrue(any("INDEX" in row[3] for row in prune_plan), prune_plan)


class ProcessCacheIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.cache_path = Path(self.temporary_directory.name) / "process.sqlite3"
        reset_analysis_cache()
        self.addCleanup(reset_analysis_cache)

    def test_process_hit_skips_split_and_inflection_but_not_rendering(self):
        from process_sanskrit.functions.hybridSplitter import HybridAnalysis
        from process_sanskrit.functions.process import process

        process_impl = process.__wrapped__.__wrapped__
        environment = {
            "PROCESS_SANSKRIT_CACHE_ENABLED": "true",
            "PROCESS_SANSKRIT_CACHE_RETENTION": "prune",
            "PROCESS_SANSKRIT_CACHE_PATH": str(self.cache_path),
        }
        analysis = HybridAnalysis(
            split=["yoga", "sūtra"],
            score=0.7,
            subscores={"length": 0.4, "morphology": 0.3},
            source="statistical",
            status="success",
        )
        grammar = [["yoga", "m_a", [("Nom", "Sg")]], "sūtra"]

        with patch.dict(os.environ, environment, clear=False), patch(
            "process_sanskrit.functions.hybridSplitter.analyze_hybrid",
            return_value=analysis,
        ) as split_mock, patch(
            "process_sanskrit.functions.inflect.inflect", return_value=grammar
        ) as inflect_mock, patch(
            "process_sanskrit.functions.process.dict_search",
            side_effect=lambda entries, *args, **kwargs: list(entries),
        ) as dictionary_mock, patch(
            "process_sanskrit.functions.process.clean_results",
            side_effect=lambda entries, mode, debug=False: {
                "mode": mode,
                "entries": entries,
            },
        ):
            first = process_impl(
                "yoga sūtra", mode="roots", session=object(), cached=True
            )
            second = process_impl(
                "yoga sūtra", mode="parts", session=object(), cached=True
            )

        self.assertEqual(first["mode"], "roots")
        self.assertEqual(second["mode"], "parts")
        self.assertEqual(split_mock.call_count, 1)
        self.assertEqual(inflect_mock.call_count, 1)
        self.assertEqual(dictionary_mock.call_count, 2)
        self.assertIsInstance(second["entries"][0][2][0], tuple)

    def test_real_rendering_parity_across_modes_and_dictionary_selection(self):
        from process_sanskrit.functions.hybridSplitter import HybridAnalysis
        from process_sanskrit.functions.process import process
        from process_sanskrit.utils.databaseSetup import get_session

        process_impl = process.__wrapped__.__wrapped__
        environment = {
            "PROCESS_SANSKRIT_CACHE_ENABLED": "true",
            "PROCESS_SANSKRIT_CACHE_RETENTION": "prune",
            "PROCESS_SANSKRIT_CACHE_PATH": str(self.cache_path),
        }
        analysis = HybridAnalysis(
            split=["citta", "vṛtti", "nirodha"],
            score=0.7,
            subscores={"morphology": 0.7},
            source="statistical",
            status="success",
        )
        grammar = ["citta", "vṛtti", "nirodha"]
        cases = (
            ("detailed", ("MW",)),
            ("roots", ("MW",)),
            ("parts", ("AP90",)),
        )
        session = get_session()
        self.addCleanup(session.close)

        with patch.dict(os.environ, environment, clear=False), patch(
            "process_sanskrit.functions.hybridSplitter.analyze_hybrid",
            return_value=analysis,
        ) as split_mock, patch(
            "process_sanskrit.functions.inflect.inflect", return_value=grammar
        ) as inflect_mock:
            uncached_outputs = [
                process_impl(
                    "cittavṛttinirodhaḥ",
                    *dictionaries,
                    mode=mode,
                    session=session,
                    cached=False,
                )
                for mode, dictionaries in cases
            ]
            cached_outputs = [
                process_impl(
                    "cittavṛttinirodhaḥ",
                    *dictionaries,
                    mode=mode,
                    session=session,
                    cached=True,
                )
                for mode, dictionaries in cases
            ]

        self.assertEqual(cached_outputs, uncached_outputs)
        self.assertEqual(split_mock.call_count, 4)
        self.assertEqual(inflect_mock.call_count, 4)

    def test_process_cached_false_never_initializes_global_cache(self):
        from process_sanskrit.functions.hybridSplitter import HybridAnalysis
        from process_sanskrit.functions.process import process

        process_impl = process.__wrapped__.__wrapped__
        environment = {
            "PROCESS_SANSKRIT_CACHE_ENABLED": "true",
            "PROCESS_SANSKRIT_CACHE_PATH": str(self.cache_path),
        }
        analysis = HybridAnalysis(
            split=["yoga", "sūtra"],
            score=0.7,
            subscores={},
            source="statistical",
            status="success",
        )
        with patch.dict(os.environ, environment, clear=False), patch(
            "process_sanskrit.functions.hybridSplitter.analyze_hybrid",
            return_value=analysis,
        ), patch(
            "process_sanskrit.functions.inflect.inflect", return_value=["yoga", "sūtra"]
        ), patch(
            "process_sanskrit.functions.process.dict_search",
            side_effect=lambda entries, *args, **kwargs: list(entries),
        ), patch(
            "process_sanskrit.functions.process.clean_results",
            side_effect=lambda entries, **kwargs: entries,
        ):
            process_impl("yoga sūtra", session=object(), cached=False)

        self.assertFalse(self.cache_path.exists())

    def test_process_cached_true_overrides_disabled_environment(self):
        from process_sanskrit.functions.hybridSplitter import HybridAnalysis
        from process_sanskrit.functions.process import process

        process_impl = process.__wrapped__.__wrapped__
        environment = {
            "PROCESS_SANSKRIT_CACHE_ENABLED": "false",
            "PROCESS_SANSKRIT_CACHE_PATH": str(self.cache_path),
        }
        analysis = HybridAnalysis(
            split=["yoga", "sūtra"],
            score=0.7,
            subscores={},
            source="statistical",
            status="success",
        )
        with patch.dict(os.environ, environment, clear=False), patch(
            "process_sanskrit.functions.hybridSplitter.analyze_hybrid",
            return_value=analysis,
        ) as split_mock, patch(
            "process_sanskrit.functions.inflect.inflect",
            return_value=["yoga", "sūtra"],
        ), patch(
            "process_sanskrit.functions.process.dict_search",
            side_effect=lambda entries, *args, **kwargs: list(entries),
        ), patch(
            "process_sanskrit.functions.process.clean_results",
            side_effect=lambda entries, **kwargs: entries,
        ):
            process_impl("yoga sūtra", session=object(), cached=True)
            process_impl("yoga sūtra", session=object(), cached=True)

        self.assertTrue(self.cache_path.exists())
        self.assertEqual(split_mock.call_count, 1)

    @unittest.skipUnless(hasattr(os, "fork"), "requires os.fork")
    def test_forked_child_reopens_global_cache_connection(self):
        environment = {
            "PROCESS_SANSKRIT_CACHE_ENABLED": "true",
            "PROCESS_SANSKRIT_CACHE_PATH": str(self.cache_path),
        }
        key = CacheKey.from_settings(
            normalized_input="forked",
            analysis_kind="hybrid_morphology",
            algorithm_signature=ANALYSIS_ALGORITHM_VERSION,
            lexicon_fingerprint="lexicon-v1",
            settings={"attempts": 20},
        )
        with patch.dict(os.environ, environment, clear=False):
            cache = get_analysis_cache()
            cache.store(
                CacheRecord(
                    key=key,
                    raw_input="forked",
                    split=["forked"],
                )
            )
            with cache.engine.connect() as connection:
                parent_record_pid = connection.connection._connection_record.info[
                    "pid"
                ]

            read_fd, write_fd = os.pipe()
            child_pid = os.fork()
            if child_pid == 0:
                os.close(read_fd)
                try:
                    inherited_engine_was_cleared = cache._engine is None
                    record = cache.get(key)
                    with cache.engine.connect() as connection:
                        record_pid = connection.connection._connection_record.info[
                            "pid"
                        ]
                    payload = {
                        "cleared": inherited_engine_was_cleared,
                        "record_pid": record_pid,
                        "process_pid": os.getpid(),
                        "split": record.split if record is not None else None,
                    }
                    os.write(write_fd, json.dumps(payload).encode("utf-8"))
                finally:
                    os.close(write_fd)
                    os._exit(0)

            os.close(write_fd)
            payload = json.loads(os.read(read_fd, 4096).decode("utf-8"))
            os.close(read_fd)
            _, status = os.waitpid(child_pid, 0)

        self.assertEqual(status, 0)
        self.assertTrue(payload["cleared"])
        self.assertNotEqual(payload["record_pid"], parent_record_pid)
        self.assertEqual(payload["record_pid"], payload["process_pid"])
        self.assertEqual(payload["split"], ["forked"])

    def test_direct_sandhi_cache_is_opt_in_and_detailed_calls_bypass_it(self):
        from process_sanskrit.functions.sandhiSplitter import (
            SplitResult,
            sandhi_splitter,
        )

        environment = {
            "PROCESS_SANSKRIT_CACHE_ENABLED": "true",
            "PROCESS_SANSKRIT_CACHE_PATH": str(self.cache_path),
        }
        result = SplitResult(
            split=["yoga", "sūtra"],
            score=0.7,
            subscores={"length": 0.4, "morphology": 0.3},
            all_splits=[
                (["yoga", "sūtra"], 0.7, {"length": 0.4, "morphology": 0.3})
            ],
        )
        with patch.dict(os.environ, environment, clear=False), patch(
            "process_sanskrit.functions.sandhiSplitter.analyze_sandhi",
            return_value=result,
        ) as analysis_mock:
            first = sandhi_splitter("yogasūtra", cached=True)
            second = sandhi_splitter("yogasūtra", cached=True)
            detailed = sandhi_splitter(
                "yogasūtra", cached=True, detailed_output=True
            )

        self.assertEqual(first, ["yoga", "sūtra"])
        self.assertEqual(second, first)
        self.assertEqual(detailed, (result.split, result.score, result.subscores, result.all_splits))
        self.assertEqual(analysis_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
