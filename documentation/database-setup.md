# Database setup: `update-ps-database` and `process_sanskrit.update_database()`

The lexicon (`SQliteDB.sqlite`, 150 MB compressed, 583 MB on disk) is not part
of the wheel. It is installed by an explicit step that exists in two equivalent
forms, both backed by `process_sanskrit/setup/updateDB.py`:

```bash
update-ps-database                      # shell; exits 1 on failure
```

```python
import process_sanskrit
process_sanskrit.update_database()      # returns the Path of the ready database
```

The console script is `main()`, a thin wrapper that prints a
`DatabaseUpdateError` to stderr and exits 1; everything else lives in
`update_database()`, and `main()` is the only place in the module that turns a
failure into console output.

## What `update_database()` does

| `PROCESS_SANSKRIT_DB_PATH` | Behaviour |
| --- | --- |
| unset | Downloads the release asset into the installed package's `resources/` if `SQliteDB.sqlite` is missing, then verifies the derived `word_list` index and rebuilds it only if stale. An installed, up-to-date database returns at once, so repeating the call is cheap. |
| set to an existing file | Never downloads. **Always** rebuilds that file's `word_list` index on a sibling copy and atomically swaps it in (see `database-location.md`). That is a full copy of the lexicon every time, so run it once when the database is provisioned, not on every start-up. |
| set to a missing file | Raises `DatabaseUpdateError`. There is no fallback to the packaged database. |

The path comes from `resourcePaths.get_database_path()`, the same resolver the
query engine and the dictionary-reference layer use, and the release asset name
is derived from `resourcePaths.DATABASE_FILENAME`, so the updater cannot
install to one place, or under one name, while queries read another.

### Errors

Every failure raises `process_sanskrit.DatabaseUpdateError` (a `RuntimeError`)
whose message quotes the underlying error and whose `__cause__` is that error:
a `requests` exception for a failed request, `OSError`/`EOFError` for a bad
archive or a full disk, `sqlite3.Error` for a file that is not a lexicon or a
rebuild that failed `PRAGMA quick_check`. The helpers `download_and_unzip` and
`ensure_word_list_index` raise the same type rather than printing and returning
a flag, so a caller that silenced stdout still learns what went wrong. The
updater never raises `DatabaseNotFoundError`; that one belongs to the query
path.

Progress and status are printed to stdout, as they are for the shell command.
Wrap the call in `process_sanskrit.suppress_all_output()` to silence it.

## No partial files, ever

Every file the updater produces is staged next to its destination and published
with a single `os.replace`, so a reader sees either the previous file or the
complete new one. The helpers live in `process_sanskrit/utils/atomicFiles.py`:

- `scratch_file(directory, prefix=, suffix=)` yields a `mkstemp` sibling and
  removes it in `finally`, so a `KeyboardInterrupt` during the unpack leaves
  nothing behind either.
- `atomic_replacement(destination, suffix=)` yields a scratch file that, if the
  block completes, takes the destination's mode and owner, is fsynced, renamed
  onto the destination, and has the directory entry fsynced. Every handle on
  the staging file must be closed before the block ends, or the rename fails on
  Windows.

Both the download and the index rebuild use `atomic_replacement`. Staging names
during a run, all in the destination's directory:

- `.SQliteDB.sqlite.gz.<random>.part` — the archive as it streams in
- `.SQliteDB.sqlite.<random>.downloading` — the decompressed database
- `.SQliteDB.sqlite.<random>.updating` — the copy being re-indexed

Two further guards close the gaps the rename alone cannot:

- **Short reads are rejected.** A body shorter than the announced
  `Content-Length` is a failure even when it decompresses cleanly. urllib3 1.x
  does not enforce the header, and a stream cut at a gzip member boundary would
  otherwise install silently. The check is skipped when the response carries a
  `Content-Encoding`, because the header then counts wire bytes while
  `iter_content` yields decoded ones; a gzipping proxy must not turn a complete
  download into an error.
- **File mode is deliberate.** `mkstemp` creates 0600 files. A replacement
  keeps the mode (and, when permitted, the owner) the file already had; a
  brand-new download gets the umask default, so a lexicon downloaded as root
  during an image build stays readable by the unprivileged runtime user.

### Leftovers from older releases

Releases before this one decompressed straight onto the final path. A run
interrupted mid-unpack left a truncated `SQliteDB.sqlite` that the "already
installed" check then accepted forever, and the next run failed inside the
index check with a confusing SQLite error. The updater cannot tell such a file
from a healthy one without a full integrity scan, so it does not try; instead
the error raised on the packaged path keeps the SQLite cause and adds that the
file should be deleted and the update run again.

## Why the library does not download on first use

`process()` raises `DatabaseNotFoundError` when the database is missing rather
than fetching it. This is deliberate:

- The engine is built lazily per process, and the Sanskrit Voyager backend runs
  several workers. A first-request download would have every worker fetch and
  unpack the same file at once.
- `site-packages` is often read-only (system Python, non-root containers, Nix).
  Failing once at install time, in a command run with the right permissions, is
  better than failing inside the first query.
- The package silences warnings at import and offers `suppress_all_output()`; a
  hidden multi-minute network stall inside a library call is the wrong default.
- CI runs the database-free suites on purpose.

NLTK and spaCy make the same choice. Every "database not found" error therefore
ends with the shared `DATABASE_SETUP_HINT` from `utils/resourcePaths.py`, which
names both the shell command and the Python function. The three "index is
stale" warnings keep their shorter `update-ps-database` wording on purpose.

## Import cost

The package root now imports `updateDB`, so `requests` is imported lazily inside
the download function. `import process_sanskrit` must not load the HTTP stack;
the `LibraryEntryPointTests` class in `tests/test_update_database.py` pins that
in a fresh interpreter.

## Tests

`tests/test_update_database.py` (database-free; runs in the CI "database-free
Python contracts" step together with `tests/test_database_path_configuration.py`):

- `AtomicDownloadTests` — fake streaming responses: success publishes only the
  final file; an existing file is not re-downloaded; a `KeyboardInterrupt` or
  `OSError` during unpack, a dropped connection, a short read, a corrupt
  archive, and a refused connection all leave the target directory empty and
  raise with the cause chained; a transfer-encoded body is not measured against
  `Content-Length`; the published file has the umask-default mode.
- `ExternalDatabaseUpdateTests` — in-place repair of a configured lexicon:
  atomic inode swap, preserved legacy tables and file mode, no temporary
  sibling after a failed rebuild, `DatabaseUpdateError` for a missing
  configured file and, with a chained `sqlite3` cause, for an empty one, and
  the CLI's exit codes.
- `PackagedUpdateTests` — the unconfigured flow targets the packaged
  `resources/` directory, a failed download propagates before the index check,
  and a failed packaged index check keeps its cause and mentions the
  leftover-file remedy.
- `LibraryEntryPointTests` — the function and error are exported from the
  package root, the console script points at `main`, and importing the package
  does not import `requests`.

`tests/test_database_path_configuration.py` pins that `get_engine`,
`requires_database`, and the dictionary-reference layer all name both setup
entry points when the database is missing.

## Releasing a new database

Bump `RELEASE_TAG` in `process_sanskrit/setup/updateDB.py`; see
`word-list-index.md` for what an already-downloaded database gets repaired in
place versus what needs a fresh download.
