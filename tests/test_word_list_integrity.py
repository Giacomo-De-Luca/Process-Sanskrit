"""Integrity of the derived ``word_list`` index.

``word_list`` is not source data: it is an index derived from the dictionary
tables that live in the same database.  The shipped v1.0.2 artifact was built
from only five of the seven dictionaries (``cae`` and ``ddsa`` were omitted),
and the library compensated at runtime with a 1.4 MB JSON overlay.  These tests
pin the rebuild that makes the overlay unnecessary, and the staleness signal
that keeps a five-dictionary lexicon from degrading silently.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from process_sanskrit.utils import dictionary_references
from process_sanskrit.utils.resourcePaths import get_database_path
from process_sanskrit.utils.wordListBuilder import WordListBuilder


def _dictionary_table(connection: sqlite3.Connection, name: str, headwords) -> None:
    """Create a table with the dictionary column signature and fill it."""
    connection.execute(
        f'CREATE TABLE "{name}" '
        '("keys_iast" TEXT, "components" TEXT, "lnum" REAL, "cleaned_body" TEXT)'
    )
    connection.executemany(
        f'INSERT INTO "{name}" VALUES (?, ?, ?, ?)',
        [(word, "", 0.0, "body") for word in headwords],
    )


class WordListBuilderTests(unittest.TestCase):
    """Behaviour of the rebuild, exercised on a small synthetic lexicon."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "lexicon.sqlite"
        connection = sqlite3.connect(self.database_path)
        # Two dictionaries with an overlapping headword, plus one word that is
        # attested only in the second -- the shape that produced the overlay's
        # "overrides" and "only_python" halves respectively.
        _dictionary_table(connection, "mw", ["shared", "mwonly"])
        _dictionary_table(connection, "ddsa", ["shared", "ddsaonly"])
        # An inflection-style table must not be mistaken for a dictionary.
        connection.execute(
            'CREATE TABLE "lgtab1" ("model" TEXT, "stem" TEXT, "refs" TEXT, "data" TEXT)'
        )
        connection.commit()
        self.connection = connection
        self.addCleanup(connection.close)

    def test_discovery_finds_dictionary_tables_and_nothing_else(self):
        self.assertEqual(
            WordListBuilder.discover_dictionaries(self.connection),
            ["ddsa", "mw"],
        )

    def test_build_indexes_every_dictionary(self):
        WordListBuilder.build(self.connection)
        rows = dict(
            self.connection.execute("SELECT keys_iast, dict_names FROM word_list")
        )
        self.assertEqual(
            {word: json.loads(names) for word, names in rows.items()},
            {
                "shared": ["ddsa", "mw"],
                "mwonly": ["mw"],
                "ddsaonly": ["ddsa"],
            },
        )

    def test_dictionary_names_are_stored_sorted(self):
        """The pre-overlay mapping stored sorted lists; callers compare by equality."""
        WordListBuilder.build(self.connection)
        names = self.connection.execute(
            "SELECT dict_names FROM word_list WHERE keys_iast = 'shared'"
        ).fetchone()[0]
        self.assertEqual(json.loads(names), sorted(json.loads(names)))

    def test_build_records_the_indexed_sources(self):
        WordListBuilder.build(self.connection)
        self.assertEqual(WordListBuilder.indexed_dictionaries(self.connection), {"ddsa", "mw"})

    def test_build_drops_the_legacy_duplicate_table(self):
        self.connection.execute(
            "CREATE TABLE dictionary_cross_references "
            "(keys_iast TEXT PRIMARY KEY, dict_names TEXT)"
        )
        WordListBuilder.build(self.connection)
        remaining = self.connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='dictionary_cross_references'"
        ).fetchone()
        self.assertIsNone(remaining)

    def test_rebuild_is_idempotent(self):
        WordListBuilder.build(self.connection)
        first = sorted(self.connection.execute("SELECT * FROM word_list"))
        WordListBuilder.build(self.connection)
        self.assertEqual(sorted(self.connection.execute("SELECT * FROM word_list")), first)

    def test_rebuild_repairs_a_five_dictionary_word_list(self):
        """The actual v1.0.2 bug: an index built from a subset of the dictionaries."""
        self.connection.execute(
            "CREATE TABLE word_list (keys_iast TEXT PRIMARY KEY, dict_names TEXT)"
        )
        self.connection.executemany(
            "INSERT INTO word_list VALUES (?, ?)",
            [("shared", json.dumps(["mw"])), ("mwonly", json.dumps(["mw"]))],
        )
        self.connection.commit()
        self.assertEqual(WordListBuilder.missing_dictionaries(self.connection), {"ddsa"})

        WordListBuilder.build(self.connection)

        self.assertEqual(WordListBuilder.missing_dictionaries(self.connection), set())
        self.assertEqual(
            json.loads(
                self.connection.execute(
                    "SELECT dict_names FROM word_list WHERE keys_iast = 'shared'"
                ).fetchone()[0]
            ),
            ["ddsa", "mw"],
        )
        # the ddsa-only headword was absent from the stale index entirely
        self.assertIsNotNone(
            self.connection.execute(
                "SELECT 1 FROM word_list WHERE keys_iast = 'ddsaonly'"
            ).fetchone()
        )

    def test_a_database_with_no_word_list_at_all_reads_as_stale(self):
        self.assertEqual(WordListBuilder.missing_dictionaries(self.connection), {"ddsa", "mw"})

    def test_a_freshly_built_index_is_not_stale(self):
        WordListBuilder.build(self.connection)
        self.assertEqual(WordListBuilder.missing_dictionaries(self.connection), set())

    def test_a_failed_build_leaves_the_previous_index_intact(self):
        WordListBuilder.build(self.connection)
        before = sorted(self.connection.execute("SELECT * FROM word_list"))
        with patch.object(
            WordListBuilder, "_collect", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                WordListBuilder.build(self.connection)
        self.assertEqual(sorted(self.connection.execute("SELECT * FROM word_list")), before)


class StaleDatabaseWarningTests(unittest.TestCase):
    """A stale external lexicon must announce itself rather than serve wrong lists."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "stale.sqlite"
        connection = sqlite3.connect(self.database_path)
        _dictionary_table(connection, "mw", ["shared"])
        _dictionary_table(connection, "ddsa", ["shared"])
        connection.execute(
            "CREATE TABLE word_list (keys_iast TEXT PRIMARY KEY, dict_names TEXT)"
        )
        connection.execute(
            "INSERT INTO word_list VALUES (?, ?)", ("shared", json.dumps(["mw"]))
        )
        connection.commit()
        connection.close()
        dictionary_references._reset_reference_state()
        self.addCleanup(dictionary_references._reset_reference_state)

    def test_a_stale_lexicon_warns_but_still_serves(self):
        with patch.dict(
            os.environ, {"PROCESS_SANSKRIT_DB_PATH": str(self.database_path)}
        ):
            with self.assertLogs(
                "process_sanskrit.utils.dictionary_references", level=logging.WARNING
            ) as captured:
                self.assertEqual(
                    dictionary_references.DICTIONARY_REFERENCES["shared"], ["mw"]
                )
        message = "\n".join(captured.output)
        self.assertIn("ddsa", message)
        self.assertIn("update-ps-database", message)

    def test_the_stale_warning_is_emitted_once_per_database(self):
        with patch.dict(
            os.environ, {"PROCESS_SANSKRIT_DB_PATH": str(self.database_path)}
        ):
            with self.assertLogs(
                "process_sanskrit.utils.dictionary_references", level=logging.WARNING
            ) as captured:
                for _ in range(5):
                    dictionary_references.DICTIONARY_REFERENCES["shared"]
        self.assertEqual(len(captured.output), 1)


@unittest.skipUnless(
    get_database_path().exists(), "packaged lexicon database is not installed"
)
class PackagedDatabaseTests(unittest.TestCase):
    """The shipped lexicon must index all seven dictionaries with no overlay."""

    # Ground truth from the pre-07b6678 in-memory mapping.
    EXPECTED_ENTRIES = 246955
    EXPECTED_DICTIONARIES = {"ap90", "bhs", "cae", "cped", "ddsa", "gra", "mw"}

    @classmethod
    def setUpClass(cls):
        cls.connection = sqlite3.connect(
            f"file:{get_database_path()}?mode=ro", uri=True
        )

    @classmethod
    def tearDownClass(cls):
        cls.connection.close()

    def test_the_packaged_index_covers_every_dictionary(self):
        self.assertEqual(
            set(WordListBuilder.discover_dictionaries(self.connection)),
            self.EXPECTED_DICTIONARIES,
        )
        self.assertEqual(WordListBuilder.missing_dictionaries(self.connection), set())

    def test_the_packaged_index_matches_the_historical_entry_count(self):
        self.assertEqual(len(dictionary_references.DICTIONARY_REFERENCES), self.EXPECTED_ENTRIES)

    def test_entries_the_overlay_used_to_repair_now_come_from_the_database(self):
        references = dictionary_references.DICTIONARY_REFERENCES
        # 'a' was an override: word_list had it, but without cae/ddsa.
        self.assertEqual(
            references["a"], ["ap90", "bhs", "cae", "ddsa", "gra", "mw"]
        )
        # '*ai' was only_python: attested in ddsa alone, so absent from word_list.
        self.assertEqual(references["*ai"], ["ddsa"])

    def test_the_legacy_duplicate_table_is_gone(self):
        self.assertIsNone(
            self.connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='dictionary_cross_references'"
            ).fetchone()
        )


if __name__ == "__main__":
    unittest.main()
