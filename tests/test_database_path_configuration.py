"""Configuration tests for an externally provisioned lexicon database."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from process_sanskrit.utils import databaseSetup
from process_sanskrit.utils import dictionary_references
from process_sanskrit.utils.resourcePaths import get_database_path


class DatabasePathConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = (
            Path(self.temporary_directory.name) / "lexicon # ? % configured.sqlite"
        )
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            "CREATE TABLE word_list (keys_iast TEXT PRIMARY KEY, dict_names TEXT)"
        )
        connection.execute(
            "INSERT INTO word_list VALUES (?, ?)",
            ("configuredword", json.dumps(["mw"])),
        )
        connection.commit()
        connection.close()
        databaseSetup._reset_database_state()
        dictionary_references._reset_reference_state()
        self.addCleanup(databaseSetup._reset_database_state)
        self.addCleanup(dictionary_references._reset_reference_state)

    def test_environment_path_overrides_packaged_database(self):
        with patch.dict(
            os.environ,
            {"PROCESS_SANSKRIT_DB_PATH": str(self.database_path)},
        ):
            self.assertEqual(get_database_path(), self.database_path.resolve())
            with databaseSetup.get_engine().connect() as connection:
                self.assertEqual(
                    connection.exec_driver_sql(
                        "SELECT keys_iast FROM word_list"
                    ).scalar(),
                    "configuredword",
                )

    def test_dictionary_references_use_the_same_configured_path(self):
        with patch.dict(
            os.environ,
            {"PROCESS_SANSKRIT_DB_PATH": str(self.database_path)},
        ):
            self.assertEqual(
                dictionary_references.DICTIONARY_REFERENCES["configuredword"],
                ["mw"],
            )

    def test_missing_configured_database_fails_without_fallback(self):
        missing = self.database_path.with_name("missing.sqlite")
        with patch.dict(
            os.environ,
            {"PROCESS_SANSKRIT_DB_PATH": str(missing)},
        ):
            self.assertEqual(get_database_path(), missing.resolve())
            with self.assertRaises(databaseSetup.DatabaseNotFoundError):
                databaseSetup.get_engine()
            with self.assertRaises(FileNotFoundError):
                dictionary_references.DICTIONARY_REFERENCES["configuredword"]

    def test_reference_connection_tracks_environment_path_changes(self):
        second_path = self.database_path.with_name("second.sqlite")
        connection = sqlite3.connect(second_path)
        connection.execute(
            "CREATE TABLE word_list (keys_iast TEXT PRIMARY KEY, dict_names TEXT)"
        )
        connection.execute(
            "INSERT INTO word_list VALUES (?, ?)",
            ("secondword", json.dumps(["cae"])),
        )
        connection.commit()
        connection.close()

        with patch.dict(
            os.environ,
            {"PROCESS_SANSKRIT_DB_PATH": str(self.database_path)},
        ):
            self.assertIn(
                "configuredword", dictionary_references.DICTIONARY_REFERENCES
            )
        with patch.dict(
            os.environ,
            {"PROCESS_SANSKRIT_DB_PATH": str(second_path)},
        ):
            self.assertEqual(
                dictionary_references.DICTIONARY_REFERENCES["secondword"],
                ["cae"],
            )


if __name__ == "__main__":
    unittest.main()
