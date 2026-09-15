"""Download, install, and repair the Process-Sanskrit lexicon database.

``update_database`` is the importable entry point, also exported as
``process_sanskrit.update_database``; ``main`` backs the ``update-ps-database``
console script.  Every file written here is staged next to its destination and
published with one atomic rename (``utils/atomicFiles.py``), so an interrupted
run never leaves a partial ``SQliteDB.sqlite`` behind.  Failures raise
``DatabaseUpdateError`` with the underlying error chained; only ``main`` turns
them into console output.  See ``documentation/database-setup.md``.
"""

from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Union

from process_sanskrit.utils.atomicFiles import atomic_replacement, scratch_file
from process_sanskrit.utils.resourcePaths import (
    DATABASE_FILENAME,
    DATABASE_PATH_ENV,
    get_database_path,
    resolve_configured_path,
)
from process_sanskrit.utils.wordListBuilder import WordListBuilder, WordListReport

# --- Configuration ---
# GitHub Release Info
REPO_OWNER = "Giacomo-De-Luca"
REPO_NAME = "Process-Sanskrit"
ASSET_NAME = f"{DATABASE_FILENAME}.gz"
RELEASE_TAG = "v1.0.2"
# --- End Configuration ---


def _unpacked_name(asset_name: str) -> str:
    """The on-disk name of a release asset once its gzip layer is removed."""
    return asset_name[:-3] if asset_name.endswith(".gz") else asset_name


UNZIPPED_FILENAME = _unpacked_name(ASSET_NAME)
DOWNLOAD_URL = (
    f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/"
    f"{RELEASE_TAG}/{ASSET_NAME}"
)


class DatabaseUpdateError(RuntimeError):
    """The lexicon could not be downloaded, is not a lexicon, or its index failed.

    The underlying network, filesystem, or SQLite error, when there is one, is
    chained as ``__cause__`` and quoted in the message.
    """


# --- Download ---


def _print_progress(downloaded: int, total: int) -> None:
    progress = int(50 * downloaded / total) if total else 0
    sys.stdout.write(
        f"\r[{'#' * progress}{'.' * (50 - progress)}] "
        f"{downloaded / (1024 * 1024):.2f} MB / {total / (1024 * 1024):.2f} MB"
    )
    sys.stdout.flush()


def _stream_to_file(response, destination: Path, *, label: str) -> None:
    """Write a streaming HTTP body to ``destination`` with a progress bar.

    A body shorter than the announced ``Content-Length`` is an error: urllib3 1.x
    does not enforce the header, and a body cut short at a gzip member boundary
    would otherwise decompress cleanly and install silently.  The check is
    skipped when the response carries a ``Content-Encoding``, because the header
    then counts wire bytes while ``iter_content`` yields decoded ones.
    """
    total_size = int(response.headers.get("content-length", 0))
    transfer_encoded = bool(response.headers.get("content-encoding"))
    downloaded_size = 0
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            handle.write(chunk)
            downloaded_size += len(chunk)
            _print_progress(downloaded_size, total_size)
    sys.stdout.write("\n")
    if total_size and not transfer_encoded and downloaded_size != total_size:
        raise DatabaseUpdateError(
            f"download of {label} ended after {downloaded_size} of "
            f"{total_size} bytes"
        )


def download_and_unzip(target_dir, asset_name, download_url) -> Path:
    """Install the release asset into ``target_dir`` and return the unpacked path.

    An already-present file is returned without a download.  Otherwise the
    archive is downloaded to a scratch sibling and unpacked into a staging
    sibling; only a complete, fully decompressed file is renamed onto the final
    path.

    Raises:
        DatabaseUpdateError: the request failed, the body was short, the target
            directory is not writable, or the archive could not be unpacked;
            the underlying error is chained.
    """
    import requests  # lazy: keeps ``import process_sanskrit`` free of the HTTP stack

    target_dir = resolve_configured_path(target_dir)
    destination = target_dir / _unpacked_name(asset_name)

    print(f"Target directory: {target_dir}")
    print(f"Download URL: {download_url}")
    print(f"Output file: {destination}")

    if destination.exists():
        print(f"File '{destination}' already exists. Skipping download.")
        return destination

    try:
        with atomic_replacement(destination, suffix=".downloading") as staged:
            print(f"Downloading '{asset_name}'...")
            with requests.get(download_url, stream=True, timeout=120) as response:
                response.raise_for_status()
                if not asset_name.endswith(".gz"):
                    _stream_to_file(response, staged, label=asset_name)
                else:
                    with scratch_file(
                        target_dir, prefix=f".{asset_name}.", suffix=".part"
                    ) as archive:
                        _stream_to_file(response, archive, label=asset_name)
                        print(f"Unpacking '{asset_name}'...")
                        with gzip.open(archive, "rb") as packed, staged.open(
                            "wb"
                        ) as unpacked:
                            shutil.copyfileobj(packed, unpacked)
    except DatabaseUpdateError:
        raise
    except requests.exceptions.RequestException as error:
        raise DatabaseUpdateError(
            f"Error downloading {download_url}: {error}"
        ) from error
    except Exception as error:
        raise DatabaseUpdateError(
            f"Error installing {asset_name} into {target_dir}: {error}"
        ) from error

    print(f"\nSuccess! Asset placed in '{destination}'.")
    return destination


# --- Derived index ---


def _open_existing_database(database_path: Path) -> sqlite3.Connection:
    """Open an existing database without sqlite3's create-if-missing fallback."""
    return sqlite3.connect(
        f"{database_path.as_uri()}?mode=ro",
        uri=True,
    )


def _rebuild_index(staged: Path, *, drop_legacy: bool) -> WordListReport:
    """Rebuild ``word_list`` on the staged copy and verify the result."""
    connection = sqlite3.connect(f"{staged.as_uri()}?mode=rw", uri=True)
    try:
        report = WordListBuilder.build(connection, drop_legacy=drop_legacy)
        integrity = connection.execute("PRAGMA quick_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise sqlite3.DatabaseError("rebuilt database failed PRAGMA quick_check")
    finally:
        connection.close()
    return report


def ensure_word_list_index(
    database_path: Union[str, Path],
    *,
    force: bool = False,
    preserve_legacy: bool = False,
) -> None:
    """Rebuild the derived word_list index unless it already covers every dictionary.

    The released artifact up to v1.0.2 indexed only five of the seven
    dictionaries, so an already-downloaded database is repaired in place here
    rather than forcing a fresh download of the whole file.  The rebuild runs on
    a sibling copy that atomically replaces the original.

    Raises:
        DatabaseUpdateError: the file is missing, is not a lexicon, or the
            rebuild failed; the underlying SQLite or OS error is chained.
    """
    database_path = resolve_configured_path(database_path)
    if not database_path.is_file():
        raise DatabaseUpdateError(f"Database not found at: {database_path}")

    source_connection = None
    try:
        source_connection = _open_existing_database(database_path)
        dictionaries = WordListBuilder.discover_dictionaries(source_connection)
        if not dictionaries:
            raise sqlite3.DatabaseError(
                "database contains no dictionary tables with the required schema"
            )
        if not force and WordListBuilder.index_is_current(source_connection):
            print("Dictionary index is up to date.")
            return

        missing = WordListBuilder.missing_dictionaries(source_connection)
        source_connection.close()
        source_connection = None

        if missing:
            reason = f"{', '.join(sorted(missing))} not covered by the current word_list"
        elif force:
            reason = "explicit external-database verification"
        else:
            reason = "the current index is missing or structurally invalid"
        print(f"Rebuilding the dictionary index: {reason}.")

        with atomic_replacement(database_path, suffix=".updating") as staged:
            shutil.copy2(database_path, staged)
            report = _rebuild_index(staged, drop_legacy=not preserve_legacy)
    except (OSError, sqlite3.Error) as error:
        raise DatabaseUpdateError(
            f"Error rebuilding the dictionary index of {database_path}: {error}"
        ) from error
    finally:
        if source_connection is not None:
            source_connection.close()

    print(
        f"Indexed {report.headwords} headwords across "
        f"{len(report.dictionaries)} dictionaries "
        f"({', '.join(report.dictionaries)})."
    )
    if report.dropped_tables:
        print(f"Dropped unused table(s): {', '.join(report.dropped_tables)}.")


# --- Entry points ---


def update_database() -> Path:
    """Install or repair the lexicon database and return its path.

    With ``PROCESS_SANSKRIT_DB_PATH`` unset, the packaged database is downloaded
    into ``process_sanskrit/resources/`` when it is missing and its derived
    ``word_list`` index is verified.  An installed, up-to-date packaged database
    returns at once, so repeating the call is cheap.

    With the variable set, the configured file must already exist and its index
    is always rebuilt on a sibling copy that atomically replaces it.  That is a
    full copy of the lexicon every time, so run it once when the database is
    provisioned rather than on every start-up.

    Progress is printed to stdout; wrap the call in
    ``process_sanskrit.suppress_all_output()`` to silence it.  The library never
    calls this on its own: a missing database raises ``DatabaseNotFoundError``
    from the first query instead.

    Raises:
        DatabaseUpdateError: the download failed, the file is not a lexicon, or
            its index could not be rebuilt; the underlying error is chained.
    """
    database_path = get_database_path()
    if os.getenv(DATABASE_PATH_ENV):
        print(f"Using configured database: {database_path}")
        if not database_path.is_file():
            raise DatabaseUpdateError(
                f"Configured database not found at: {database_path}. "
                f"Unset {DATABASE_PATH_ENV} to install the packaged database instead."
            )
        ensure_word_list_index(database_path, force=True, preserve_legacy=True)
    else:
        print(f"Determined target resource directory: {database_path.parent}")
        database_path = download_and_unzip(
            str(database_path.parent), ASSET_NAME, DOWNLOAD_URL
        )
        try:
            ensure_word_list_index(database_path)
        except DatabaseUpdateError as error:
            raise DatabaseUpdateError(
                f"{error}\nIf the file is left over from a download interrupted "
                "under an older release, delete it and run the update again."
            ) from error

    print("\nDatabase download/update process finished.")
    return database_path


def main() -> None:
    """Console-script entry point behind ``update-ps-database``."""
    print("Attempting to download/update the process-sanskrit database...")
    try:
        update_database()
    except DatabaseUpdateError as error:
        print(f"\n{error}", file=sys.stderr)
        print("\nDatabase download/update failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
