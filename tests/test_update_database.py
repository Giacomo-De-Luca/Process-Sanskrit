"""The database updater repairs configured external lexicons in place."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from process_sanskrit.setup import updateDB
from process_sanskrit.utils.wordListBuilder import WordListBuilder


def _dictionary_table(
    connection: sqlite3.Connection,
    name: str,
    headwords: list[str],
) -> None:
    connection.execute(
        f'CREATE TABLE "{name}" '
        '("keys_iast" TEXT, "components" TEXT, "lnum" REAL, "cleaned_body" TEXT)'
    )
    connection.executemany(
        f'INSERT INTO "{name}" VALUES (?, ?, ?, ?)',
        [(word, "", 0.0, "body") for word in headwords],
    )


class ExternalDatabaseUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = (
            Path(self.temporary_directory.name) / "external-lexicon.sqlite"
        )
        connection = sqlite3.connect(self.database_path)
        _dictionary_table(connection, "mw", ["shared", "mwonly"])
        _dictionary_table(connection, "ddsa", ["shared", "ddsaonly"])
        connection.commit()
        connection.close()

    def run_configured_update(self, database_path: Path | None = None) -> None:
        selected_path = self.database_path if database_path is None else database_path
        with patch.dict(
            os.environ,
            {"PROCESS_SANSKRIT_DB_PATH": str(selected_path)},
        ), patch.object(
            updateDB,
            "download_and_unzip",
            return_value=True,
        ) as download:
            updateDB.update_database()
        download.assert_not_called()

    def test_index_helper_repairs_a_database_without_word_list(self):
        self.assertTrue(updateDB.ensure_word_list_index(self.database_path))

        connection = sqlite3.connect(self.database_path)
        self.addCleanup(connection.close)
        rows = dict(connection.execute("SELECT keys_iast, dict_names FROM word_list"))
        self.assertEqual(
            {word: json.loads(names) for word, names in rows.items()},
            {
                "shared": ["ddsa", "mw"],
                "mwonly": ["mw"],
                "ddsaonly": ["ddsa"],
            },
        )

    def test_cli_repairs_configured_database_without_downloading(self):
        self.run_configured_update()
        connection = sqlite3.connect(self.database_path)
        self.addCleanup(connection.close)
        self.assertIsNotNone(
            connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'word_list'"
            ).fetchone()
        )

    def test_missing_configured_database_fails_without_downloading_or_creating(self):
        missing_path = Path(self.temporary_directory.name) / "missing.sqlite"
        with patch.dict(
            os.environ,
            {"PROCESS_SANSKRIT_DB_PATH": str(missing_path)},
        ), patch.object(updateDB, "download_and_unzip") as download:
            with self.assertRaises(SystemExit) as caught:
                updateDB.update_database()

        self.assertEqual(caught.exception.code, 1)
        download.assert_not_called()
        self.assertFalse(missing_path.exists())

    def test_unconfigured_cli_preserves_packaged_download_flow(self):
        with patch.dict(
            os.environ,
            {"PROCESS_SANSKRIT_DB_PATH": ""},
        ), patch.object(
            updateDB,
            "download_and_unzip",
            return_value=True,
        ) as download, patch.object(
            updateDB,
            "ensure_word_list_index",
            return_value=True,
        ) as ensure:
            updateDB.update_database()

        download.assert_called_once()
        ensure.assert_called_once()

    def test_empty_sqlite_file_is_rejected_as_not_a_lexicon(self):
        empty_path = Path(self.temporary_directory.name) / "empty.sqlite"
        sqlite3.connect(empty_path).close()

        with self.assertRaises(SystemExit) as caught:
            self.run_configured_update(empty_path)

        self.assertEqual(caught.exception.code, 1)
        connection = sqlite3.connect(empty_path)
        self.addCleanup(connection.close)
        self.assertEqual(
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
            ).fetchone()[0],
            0,
        )

    def test_missing_word_list_rebuilds_even_if_sources_claim_full_coverage(self):
        connection = sqlite3.connect(self.database_path)
        connection.execute("CREATE TABLE word_list_sources (name TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO word_list_sources VALUES (?)",
            [("ddsa",), ("mw",)],
        )
        connection.commit()
        connection.close()

        self.run_configured_update()

        connection = sqlite3.connect(self.database_path)
        self.addCleanup(connection.close)
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM word_list").fetchone()[0],
            3,
        )

    def test_complete_legacy_index_rebuilds_when_stub_metadata_is_missing(self):
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            "CREATE TABLE word_list (keys_iast TEXT PRIMARY KEY, dict_names TEXT)"
        )
        connection.executemany(
            "INSERT INTO word_list VALUES (?, ?)",
            [
                ("shared", '["ddsa", "mw"]'),
                ("mwonly", '["mw"]'),
                ("ddsaonly", '["ddsa"]'),
            ],
        )
        connection.execute("CREATE TABLE word_list_sources (name TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO word_list_sources VALUES (?)",
            [("ddsa",), ("mw",)],
        )
        connection.commit()
        connection.close()

        self.run_configured_update()

        connection = sqlite3.connect(self.database_path)
        self.addCleanup(connection.close)
        self.assertTrue(WordListBuilder.index_is_current(connection))
        self.assertEqual(
            connection.execute(
                f'SELECT value FROM "{WordListBuilder.METADATA_TABLE}" WHERE key = ?',
                (WordListBuilder.STUB_CLASSIFIER_KEY,),
            ).fetchone(),
            (str(WordListBuilder.STUB_CLASSIFIER_VERSION),),
        )

    def test_external_repair_atomically_replaces_the_database(self):
        inode_before = self.database_path.stat().st_ino
        old_reader = sqlite3.connect(
            f"{self.database_path.as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        self.addCleanup(old_reader.close)

        self.run_configured_update()

        self.assertNotEqual(self.database_path.stat().st_ino, inode_before)
        self.assertEqual(
            old_reader.execute("SELECT COUNT(*) FROM mw").fetchone()[0],
            2,
        )
        with self.assertRaises(sqlite3.OperationalError):
            old_reader.execute("SELECT COUNT(*) FROM word_list").fetchone()

        new_reader = sqlite3.connect(self.database_path)
        self.addCleanup(new_reader.close)
        self.assertEqual(
            new_reader.execute("SELECT COUNT(*) FROM word_list").fetchone()[0],
            3,
        )
        self.assertEqual(new_reader.execute("PRAGMA quick_check").fetchone()[0], "ok")

    def test_external_repair_preserves_legacy_tables(self):
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            "CREATE TABLE dictionary_cross_references "
            "(keys_iast TEXT PRIMARY KEY, dict_names TEXT)"
        )
        connection.execute(
            "INSERT INTO dictionary_cross_references VALUES (?, ?)",
            ("legacy", '["mw"]'),
        )
        connection.commit()
        connection.close()

        self.run_configured_update()

        connection = sqlite3.connect(self.database_path)
        self.addCleanup(connection.close)
        self.assertEqual(
            connection.execute(
                "SELECT dict_names FROM dictionary_cross_references "
                "WHERE keys_iast = 'legacy'"
            ).fetchone()[0],
            '["mw"]',
        )


if __name__ == "__main__":
    unittest.main()
