"""Web-worker lifecycle tests for the read-only SQLAlchemy lexicon engine."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import QueuePool

from process_sanskrit.utils.databaseSetup import (
    _create_read_only_engine,
    _reset_database_state,
    get_engine,
    get_scoped_session,
)


class DatabaseLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "lexicon.sqlite3"
        connection = sqlite3.connect(self.path)
        connection.execute("CREATE TABLE words (word TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO words VALUES ('yoga')")
        connection.commit()
        connection.close()
        _reset_database_state()
        self.addCleanup(_reset_database_state)

    def test_read_engine_uses_explicit_queue_pool_and_is_immutable(self):
        engine = _create_read_only_engine(
            str(self.path),
            pool_size=1,
            max_overflow=0,
            cache_kib=512,
            mmap_size=0,
        )
        self.addCleanup(engine.dispose)
        self.assertIsInstance(engine.pool, QueuePool)
        with engine.connect() as connection:
            self.assertEqual(
                connection.exec_driver_sql("SELECT word FROM words").scalar(),
                "yoga",
            )
            self.assertEqual(
                connection.exec_driver_sql("PRAGMA query_only").scalar(), 1
            )
            self.assertEqual(
                connection.exec_driver_sql("PRAGMA cache_size").scalar(), -512
            )
            with self.assertRaises(OperationalError):
                connection.exec_driver_sql("CREATE TABLE forbidden (id)")

    def test_concurrent_lazy_initialization_publishes_one_engine(self):
        barrier = threading.Barrier(8)
        engines = []

        def initialize() -> None:
            barrier.wait()
            engines.append(get_engine(str(self.path)))

        threads = [threading.Thread(target=initialize) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len({id(engine) for engine in engines}), 1)

        other = self.path.with_name("other.sqlite3")
        sqlite3.connect(other).close()
        with self.assertRaises(ValueError):
            get_engine(str(other))

    @unittest.skipUnless(hasattr(os, "fork"), "requires os.fork")
    def test_forked_child_replaces_inherited_dbapi_connection(self):
        engine = _create_read_only_engine(
            str(self.path),
            pool_size=1,
            max_overflow=0,
            cache_kib=512,
            mmap_size=0,
        )
        self.addCleanup(engine.dispose)
        with engine.connect() as connection:
            parent_pid = connection.connection._connection_record.info["pid"]

        read_fd, write_fd = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                with engine.connect() as connection:
                    payload = {
                        "record_pid": connection.connection._connection_record.info[
                            "pid"
                        ],
                        "process_pid": os.getpid(),
                        "word": connection.exec_driver_sql(
                            "SELECT word FROM words"
                        ).scalar(),
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
        self.assertNotEqual(payload["record_pid"], parent_pid)
        self.assertEqual(payload["record_pid"], payload["process_pid"])
        self.assertEqual(payload["word"], "yoga")

    @unittest.skipUnless(hasattr(os, "fork"), "requires os.fork")
    def test_forked_child_discards_global_engine_and_scoped_session(self):
        import process_sanskrit.utils.databaseSetup as database_setup

        get_engine(str(self.path))
        parent_scoped = get_scoped_session()
        parent_session = parent_scoped()
        parent_record_pid = (
            parent_session.connection().connection._connection_record.info["pid"]
        )

        read_fd, write_fd = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                inherited_state_was_cleared = (
                    database_setup._engine is None
                    and database_setup._session_factory is None
                    and database_setup._scoped_session is None
                )
                get_engine(str(self.path))
                child_scoped = get_scoped_session()
                child_session = child_scoped()
                record_pid = child_session.connection().connection._connection_record.info[
                    "pid"
                ]
                payload = {
                    "cleared": inherited_state_was_cleared,
                    "record_pid": record_pid,
                    "process_pid": os.getpid(),
                    "word": child_session.execute(
                        text("SELECT word FROM words")
                    ).scalar(),
                }
                child_scoped.remove()
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
        self.assertEqual(payload["word"], "yoga")


if __name__ == "__main__":
    unittest.main()
