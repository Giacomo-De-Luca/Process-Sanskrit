import os
import sys
import requests
import gzip
import shutil
import sqlite3
import importlib.resources
import tempfile

from process_sanskrit.utils.wordListBuilder import WordListBuilder
from process_sanskrit.utils.resourcePaths import (
    DATABASE_PATH_ENV,
    resolve_configured_path,
)

# --- Configuration ---
# GitHub Release Info
REPO_OWNER = "Giacomo-De-Luca"
REPO_NAME = "Process-Sanskrit"
ASSET_NAME = "SQliteDB.sqlite.gz" 
RELEASE_TAG = "v1.0.2"
# Target Location within the package
TARGET_FOLDER_NAME = "resources"
# --- End Configuration ---

# Derive the final unzipped filename
if ASSET_NAME.endswith(".gz"):
    UNZIPPED_FILENAME = ASSET_NAME[:-3]
else:
    UNZIPPED_FILENAME = ASSET_NAME

DOWNLOAD_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{RELEASE_TAG}/{ASSET_NAME}"

def download_and_unzip(target_dir, asset_name, download_url):
    """Downloads and unzips the asset into the target directory."""
    os.makedirs(target_dir, exist_ok=True)
    downloaded_gz_path = os.path.join(target_dir, asset_name)
    unzipped_file_path = os.path.join(target_dir, UNZIPPED_FILENAME)

    print(f"Target directory: {target_dir}")
    print(f"Download URL: {download_url}")
    print(f"Output file: {unzipped_file_path}")

    # Check if file already exists
    if os.path.exists(unzipped_file_path):
        print(f"File '{unzipped_file_path}' already exists. Skipping download.")
        return True # Indicate success or skipped

    try:
        print(f"Downloading '{asset_name}'...")
        # Use verify=True by default for security.
        with requests.get(download_url, stream=True, timeout=120, verify=True) as response:
            response.raise_for_status() # Raise an exception for bad status codes

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded_size = 0

            print(f"Saving to '{downloaded_gz_path}'...")
            with open(downloaded_gz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = int(50 * downloaded_size / total_size) if total_size else 0
                    sys.stdout.write(f"\r[{'#' * progress}{'.' * (50 - progress)}] {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB")
                    sys.stdout.flush()
            print("\nDownload complete.")

        if asset_name.endswith(".gz"):
            print(f"Unzipping '{downloaded_gz_path}' to '{unzipped_file_path}'...")
            with gzip.open(downloaded_gz_path, 'rb') as f_in:
                with open(unzipped_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print("Unzipping complete.")
            print(f"Cleaning up '{downloaded_gz_path}'...")
            os.remove(downloaded_gz_path)
            print("Cleanup complete.")
        else:
             # If not gzipped, the downloaded file is the final file.
             if unzipped_file_path != downloaded_gz_path:
                 shutil.move(downloaded_gz_path, unzipped_file_path)
                 print(f"Moved '{downloaded_gz_path}' to '{unzipped_file_path}'.")

        print(f"\nSuccess! Asset placed in '{unzipped_file_path}'.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\nError during download: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nAn error occurred during download/unzip: {e}", file=sys.stderr)
    finally:
        # Ensure partial downloads are cleaned up on error
        if os.path.exists(downloaded_gz_path) and not os.path.exists(unzipped_file_path):
             try:
                 os.remove(downloaded_gz_path)
                 print(f"Removed partially downloaded file: {downloaded_gz_path}")
             except OSError as rm_err:
                 print(f"Error removing file {downloaded_gz_path} on error: {rm_err}", file=sys.stderr)
    return False # Indicate failure


def _open_existing_database(database_path):
    """Open an existing database without sqlite3's create-if-missing fallback."""
    return sqlite3.connect(
        f"{database_path.as_uri()}?mode=ro",
        uri=True,
    )


def _fsync_parent(path):
    """Persist an atomic directory-entry replacement where POSIX supports it."""
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_word_list_index(
    database_path,
    *,
    force=False,
    preserve_legacy=False,
):
    """Rebuild the derived word_list index if it does not cover every dictionary.

    The released artifact up to v1.0.2 indexed only five of the seven
    dictionaries, so an already-downloaded database is repaired in place here
    rather than forcing a fresh download of the whole file.
    """
    database_path = resolve_configured_path(database_path)
    if not database_path.is_file():
        print(
            f"\nDatabase not found at: {database_path}",
            file=sys.stderr,
        )
        return False

    source_connection = None
    temporary_path = None
    try:
        source_connection = _open_existing_database(database_path)
        dictionaries = WordListBuilder.discover_dictionaries(source_connection)
        if not dictionaries:
            raise sqlite3.DatabaseError(
                "database contains no dictionary tables with the required schema"
            )
        if not force and WordListBuilder.index_is_current(source_connection):
            print("Dictionary index is up to date.")
            return True

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

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{database_path.name}.",
            suffix=".updating",
            dir=database_path.parent,
        )
        os.close(descriptor)
        temporary_path = resolve_configured_path(temporary_name)
        shutil.copy2(database_path, temporary_path)

        connection = sqlite3.connect(
            f"{temporary_path.as_uri()}?mode=rw",
            uri=True,
        )
        try:
            report = WordListBuilder.build(
                connection,
                drop_legacy=not preserve_legacy,
            )
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise sqlite3.DatabaseError(
                    "rebuilt database failed PRAGMA quick_check"
                )
        finally:
            connection.close()

        print(
            f"Indexed {report.headwords} headwords across "
            f"{len(report.dictionaries)} dictionaries "
            f"({', '.join(report.dictionaries)})."
        )
        if report.dropped_tables:
            print(f"Dropped unused table(s): {', '.join(report.dropped_tables)}.")

        source_stat = database_path.stat()
        if hasattr(os, "chown"):
            try:
                os.chown(temporary_path, source_stat.st_uid, source_stat.st_gid)
            except PermissionError:
                # An unprivileged owner already creates the sibling file under
                # its own uid/gid, which is the desired deployment case.
                pass
        with temporary_path.open("rb") as rebuilt_file:
            os.fsync(rebuilt_file.fileno())
        os.replace(temporary_path, database_path)
        temporary_path = None
        _fsync_parent(database_path)
        return True
    except (OSError, sqlite3.Error) as error:
        print(f"\nError rebuilding the dictionary index: {error}", file=sys.stderr)
        return False
    finally:
        if source_connection is not None:
            source_connection.close()
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def update_database():
    """
    Command-line entry point function to download/update the database.
    Finds the installed package's resource directory.
    """
    print("Attempting to download/update the process-sanskrit database...")

    try:
        configured_path = os.getenv(DATABASE_PATH_ENV)
        if configured_path:
            database_path = resolve_configured_path(configured_path)
            print(f"Using configured database: {database_path}")
            if not database_path.is_file():
                print(
                    f"\nConfigured database not found at: {database_path}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            # With no external path, install or repair the packaged database.
            resource_dir_ref = importlib.resources.files('process_sanskrit').joinpath(TARGET_FOLDER_NAME)
            target_path = str(resource_dir_ref)
            print(f"Determined target resource directory: {target_path}")

            if not download_and_unzip(target_path, ASSET_NAME, DOWNLOAD_URL):
                print("\nDatabase download/update failed.", file=sys.stderr)
                sys.exit(1) # Exit with error code

            database_path = os.path.join(target_path, UNZIPPED_FILENAME)

        if not ensure_word_list_index(
            database_path,
            force=bool(configured_path),
            preserve_legacy=bool(configured_path),
        ):
            print("\nDatabase download/update failed.", file=sys.stderr)
            sys.exit(1)

        print("\nDatabase download/update process finished.")

    except ModuleNotFoundError:
         print(f"Error: Could not find the installed package 'process_sanskrit'. Is it installed correctly?", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    # Allow running this script directly for testing
    update_database()
