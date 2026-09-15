"""The database updater installs and repairs lexicons without partial writes.

``AtomicDownloadTests`` pins that an interrupted or corrupt download never
leaves a truncated ``SQliteDB.sqlite`` behind and that failures carry their
cause; ``ExternalDatabaseUpdateTests`` pins in-place repair of a configured
external lexicon; ``PackagedUpdateTests`` pins the default packaged flow;
``LibraryEntryPointTests`` pins the importable ``process_sanskrit.update_database``
function and the console script that wraps it.  See
``documentation/database-setup.md``.
"""

from __future__ import annotations

import gzip
import importlib.resources
import io
import json
import os
import random
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterable, Optional
from unittest.mock import patch

import requests

from process_sanskrit.setup import updateDB
from process_sanskrit.utils.wordListBuilder import WordListBuilder


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


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


def _quiet_call(function, *args, **kwargs):
    """Run a chatty updater entry point with its console output captured."""
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


def _packaged_database_path() -> Path:
    return Path(
        str(
            importlib.resources.files("process_sanskrit").joinpath(
                "resources", updateDB.UNZIPPED_FILENAME
            )
        )
    ).resolve()


class _FakeResponse:
    """The subset of a streaming ``requests`` response the downloader uses."""

    def __init__(
        self,
        chunks: Iterable[bytes],
        *,
        content_length: Optional[int] = None,
        content_encoding: Optional[str] = None,
        drop_after: Optional[int] = None,
    ):
        self._chunks = list(chunks)
        self._drop_after = drop_after
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)
        if content_encoding is not None:
            self.headers["content-encoding"] = content_encoding

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        for index, chunk in enumerate(self._chunks):
            if self._drop_after is not None and index == self._drop_after:
                raise requests.exceptions.ChunkedEncodingError("connection dropped")
            yield chunk


class AtomicDownloadTests(unittest.TestCase):
    """Nothing reaches the final database path until it is complete."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        # The target does not exist yet: the downloader must create it.  The
        # temp root is resolved because the updater returns resolved paths and
        # macOS puts temp files behind a /var -> /private/var symlink.
        self.target = Path(self.temporary_directory.name).resolve() / "resources"
        self.final_path = self.target / updateDB.UNZIPPED_FILENAME
        # Incompressible so the archive spans several chunks and a short read
        # can be cut at a real chunk boundary.
        self.payload = b"SQLite format 3\x00" + random.Random(0).randbytes(64 * 1024)
        self.archive = gzip.compress(self.payload)
        self.chunks = [
            self.archive[offset : offset + 8192]
            for offset in range(0, len(self.archive), 8192)
        ]
        self.assertGreater(len(self.chunks), 1)

    def entries(self) -> list[str]:
        if not self.target.exists():
            return []
        return sorted(path.name for path in self.target.iterdir())

    def install(self) -> Path:
        return _quiet_call(
            updateDB.download_and_unzip,
            str(self.target),
            updateDB.ASSET_NAME,
            updateDB.DOWNLOAD_URL,
        )

    def download(self, response) -> Path:
        with patch("requests.get", return_value=response) as get:
            result = self.install()
        get.assert_called_once()
        return result

    def download_fails(self, response, *, mentions: str) -> updateDB.DatabaseUpdateError:
        """The download raises, names ``mentions``, and publishes nothing."""
        with patch("requests.get", return_value=response):
            with self.assertRaises(updateDB.DatabaseUpdateError) as caught:
                self.install()
        self.assertIn(mentions, str(caught.exception))
        self.assertEqual(self.entries(), [])
        return caught.exception

    def test_successful_download_publishes_only_the_final_file(self):
        response = _FakeResponse(self.chunks, content_length=len(self.archive))

        self.assertEqual(self.download(response), self.final_path)

        self.assertEqual(self.entries(), [updateDB.UNZIPPED_FILENAME])
        self.assertEqual(self.final_path.read_bytes(), self.payload)

    def test_a_missing_content_length_header_is_tolerated(self):
        self.download(_FakeResponse(self.chunks))
        self.assertEqual(self.final_path.read_bytes(), self.payload)

    @unittest.skipIf(os.name == "nt", "POSIX file modes")
    def test_published_database_gets_the_default_file_mode(self):
        ## mkstemp creates 0600 files.  A lexicon downloaded as root during an
        ## image build must still be readable by the unprivileged runtime user.
        previous_umask = os.umask(0o022)
        self.addCleanup(os.umask, previous_umask)

        self.download(_FakeResponse(self.chunks, content_length=len(self.archive)))

        self.assertEqual(stat.S_IMODE(self.final_path.stat().st_mode), 0o644)

    def test_an_existing_database_is_not_downloaded_again(self):
        self.target.mkdir()
        self.final_path.write_bytes(b"already installed")
        with patch("requests.get") as get:
            self.assertEqual(self.install(), self.final_path)
        get.assert_not_called()
        self.assertEqual(self.final_path.read_bytes(), b"already installed")

    def test_an_interrupt_during_unpacking_leaves_nothing_behind(self):
        ## Ctrl-C is a BaseException: cleanup must not depend on ``except
        ## Exception``.  This is the exact failure the old direct write had --
        ## a truncated final file that the next run accepted as installed.
        response = _FakeResponse(self.chunks, content_length=len(self.archive))
        with patch("requests.get", return_value=response), patch.object(
            updateDB.shutil, "copyfileobj", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.install()

        self.assertEqual(self.entries(), [])

    def test_a_failed_unpack_reports_its_cause_and_leaves_nothing_behind(self):
        response = _FakeResponse(self.chunks, content_length=len(self.archive))
        disk_full = OSError("disk full")
        with patch.object(updateDB.shutil, "copyfileobj", side_effect=disk_full):
            error = self.download_fails(response, mentions="disk full")

        self.assertIs(error.__cause__, disk_full)

    def test_a_dropped_connection_publishes_nothing(self):
        response = _FakeResponse(
            self.chunks, content_length=len(self.archive), drop_after=1
        )

        error = self.download_fails(response, mentions="connection dropped")

        self.assertIsInstance(error.__cause__, requests.exceptions.RequestException)

    def test_a_short_read_is_rejected_even_when_the_archive_decompresses(self):
        ## urllib3 1.x does not enforce Content-Length, so a body cut short at a
        ## gzip member boundary would otherwise install silently.
        response = _FakeResponse(
            self.chunks, content_length=len(self.archive) + 10
        )

        self.download_fails(response, mentions="ended after")

    def test_a_transfer_encoded_body_is_not_measured_against_content_length(self):
        ## Behind a proxy that gzips responses the header counts wire bytes
        ## while iter_content yields decoded ones; a complete download must not
        ## be rejected for that.
        response = _FakeResponse(
            self.chunks,
            content_length=len(self.archive) + 10,
            content_encoding="gzip",
        )

        self.download(response)

        self.assertEqual(self.final_path.read_bytes(), self.payload)

    def test_a_corrupt_archive_publishes_nothing(self):
        garbage = [b"this is not gzip data"]
        response = _FakeResponse(garbage, content_length=len(garbage[0]))

        error = self.download_fails(response, mentions="installing")

        self.assertIsInstance(error.__cause__, OSError)

    def test_a_refused_connection_publishes_nothing(self):
        refused = requests.exceptions.ConnectionError("refused")
        with patch("requests.get", side_effect=refused):
            with self.assertRaises(updateDB.DatabaseUpdateError) as caught:
                self.install()

        self.assertIn("refused", str(caught.exception))
        self.assertIs(caught.exception.__cause__, refused)
        self.assertEqual(self.entries(), [])


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

    def run_configured_update(
        self,
        database_path: Path | None = None,
        entry_point=updateDB.update_database,
    ):
        selected_path = self.database_path if database_path is None else database_path
        with patch.dict(
            os.environ,
            {"PROCESS_SANSKRIT_DB_PATH": str(selected_path)},
        ), patch.object(updateDB, "download_and_unzip") as download:
            result = _quiet_call(entry_point)
        download.assert_not_called()
        return result

    def sibling_entries(self) -> list[str]:
        return sorted(
            path.name for path in Path(self.temporary_directory.name).iterdir()
        )

    def test_index_helper_repairs_a_database_without_word_list(self):
        _quiet_call(updateDB.ensure_word_list_index, self.database_path)

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

    def test_update_repairs_configured_database_without_downloading(self):
        self.run_configured_update()
        connection = sqlite3.connect(self.database_path)
        self.addCleanup(connection.close)
        self.assertIsNotNone(
            connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'word_list'"
            ).fetchone()
        )

    def test_update_returns_the_configured_database_path(self):
        self.assertEqual(
            self.run_configured_update(), self.database_path.resolve()
        )

    def test_missing_configured_database_fails_without_downloading_or_creating(self):
        missing_path = Path(self.temporary_directory.name) / "missing.sqlite"
        with patch.dict(
            os.environ,
            {"PROCESS_SANSKRIT_DB_PATH": str(missing_path)},
        ), patch.object(updateDB, "download_and_unzip") as download:
            with self.assertRaises(updateDB.DatabaseUpdateError) as caught:
                _quiet_call(updateDB.update_database)

        self.assertIn(str(missing_path), str(caught.exception))
        download.assert_not_called()
        self.assertFalse(missing_path.exists())

    def test_cli_exits_nonzero_when_the_update_fails(self):
        missing_path = Path(self.temporary_directory.name) / "missing.sqlite"
        with self.assertRaises(SystemExit) as caught:
            self.run_configured_update(missing_path, entry_point=updateDB.main)

        self.assertEqual(caught.exception.code, 1)
        self.assertFalse(missing_path.exists())

    def test_cli_exits_zero_after_a_successful_update(self):
        self.assertIsNone(self.run_configured_update(entry_point=updateDB.main))

    def test_empty_sqlite_file_is_rejected_as_not_a_lexicon(self):
        empty_path = Path(self.temporary_directory.name) / "empty.sqlite"
        sqlite3.connect(empty_path).close()

        with self.assertRaises(updateDB.DatabaseUpdateError) as caught:
            self.run_configured_update(empty_path)

        self.assertIn("no dictionary tables", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, sqlite3.DatabaseError)
        connection = sqlite3.connect(empty_path)
        self.addCleanup(connection.close)
        self.assertEqual(
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
            ).fetchone()[0],
            0,
        )

    def test_a_failed_rebuild_leaves_no_temporary_sibling(self):
        failure = sqlite3.DatabaseError("simulated rebuild failure")
        with patch.object(WordListBuilder, "build", side_effect=failure):
            with self.assertRaises(updateDB.DatabaseUpdateError) as caught:
                self.run_configured_update()

        self.assertIs(caught.exception.__cause__, failure)
        self.assertIn("simulated rebuild failure", str(caught.exception))
        self.assertEqual(self.sibling_entries(), [self.database_path.name])
        connection = sqlite3.connect(self.database_path)
        self.addCleanup(connection.close)
        self.assertIsNone(
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'word_list'"
            ).fetchone()
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
        self.assertEqual(self.sibling_entries(), [self.database_path.name])

    @unittest.skipIf(os.name == "nt", "POSIX file modes")
    def test_external_repair_preserves_the_file_mode(self):
        self.database_path.chmod(0o640)

        self.run_configured_update()

        self.assertEqual(stat.S_IMODE(self.database_path.stat().st_mode), 0o640)

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


class PackagedUpdateTests(unittest.TestCase):
    """With no configured path the packaged database is installed and verified."""

    def run_packaged_update(self, *, download_error=None, index_error=None):
        expected = _packaged_database_path()
        with patch.dict(
            os.environ,
            {"PROCESS_SANSKRIT_DB_PATH": ""},
        ), patch.object(
            updateDB,
            "download_and_unzip",
            return_value=expected,
            side_effect=download_error,
        ) as download, patch.object(
            updateDB,
            "ensure_word_list_index",
            side_effect=index_error,
        ) as ensure:
            result = _quiet_call(updateDB.update_database)
        return result, download, ensure

    def test_unconfigured_update_targets_the_packaged_resources_directory(self):
        expected = _packaged_database_path()

        result, download, ensure = self.run_packaged_update()

        self.assertEqual(result, expected)
        download.assert_called_once()
        target_directory, asset_name, download_url = download.call_args.args
        self.assertEqual(Path(target_directory).resolve(), expected.parent)
        self.assertEqual(asset_name, updateDB.ASSET_NAME)
        self.assertEqual(download_url, updateDB.DOWNLOAD_URL)
        ensure.assert_called_once_with(expected)

    def test_a_failed_download_raises_before_the_index_is_checked(self):
        failure = updateDB.DatabaseUpdateError("simulated download failure")

        with self.assertRaises(updateDB.DatabaseUpdateError) as caught:
            self.run_packaged_update(download_error=failure)

        self.assertIs(caught.exception, failure)

    def test_a_failed_packaged_index_check_points_at_a_leftover_download(self):
        ## Older releases wrote the download straight to the final path, so a
        ## user interrupted mid-unpack may still hold a truncated file that the
        ## "already installed" check accepts.  The message must keep the cause
        ## and say what to do.
        failure = updateDB.DatabaseUpdateError("simulated index failure")

        with self.assertRaises(updateDB.DatabaseUpdateError) as caught:
            self.run_packaged_update(index_error=failure)

        message = str(caught.exception)
        self.assertIn("simulated index failure", message)
        self.assertIn("interrupted", message.lower())
        self.assertIn("delete", message.lower())
        self.assertIs(caught.exception.__cause__, failure)


class LibraryEntryPointTests(unittest.TestCase):
    def test_update_database_is_exported_from_the_package_root(self):
        import process_sanskrit

        self.assertIs(process_sanskrit.update_database, updateDB.update_database)
        self.assertIs(process_sanskrit.DatabaseUpdateError, updateDB.DatabaseUpdateError)

    def test_console_script_wraps_the_library_function(self):
        pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(
            'update-ps-database = "process_sanskrit.setup.updateDB:main"',
            pyproject,
        )

    def test_importing_the_package_does_not_import_the_http_client(self):
        ## The package root now imports the updater, so the HTTP stack has to be
        ## loaded lazily or every ``import process_sanskrit`` pays for it.  Only
        ## a fresh interpreter can show that: this process imported requests
        ## at the top of the module.
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, process_sanskrit; print('requests' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            cwd=REPOSITORY_ROOT,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
