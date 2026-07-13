# External database location

Process-Sanskrit normally reads `SQliteDB.sqlite` from the installed package's
`process_sanskrit/resources/` directory. Deployments that provision the lexicon
separately can set `PROCESS_SANSKRIT_DB_PATH` to an existing SQLite file:

```bash
export PROCESS_SANSKRIT_DB_PATH=/srv/process-sanskrit/SQliteDB.sqlite
uv run python -c "from process_sanskrit import process; print(process('yoga'))"
```

The value may be absolute, relative to the process working directory, or use
`~`; it is expanded and resolved before the first database connection. Set it
before importing or calling the library. Both the SQLAlchemy query engine and
the lazy dictionary-reference mapping use this resolver, so they cannot drift
to different lexicons.

The configured database is opened read-only and immutable during analysis. It
must already exist; a configured but missing path fails explicitly rather than
falling back to the packaged database.

`update-ps-database` honours the same environment variable. When
`PROCESS_SANSKRIT_DB_PATH` names an existing database, the command rebuilds its
derived `word_list` index on a sibling copy, validates that copy, and atomically
replaces the configured file. It does not download or modify the packaged
database. Existing immutable readers retain the old inode while new workers
open the repaired one. A configured but missing path fails explicitly; unset
the variable when the intended target is the normal packaged database. The
database must be readable and its parent directory must be writable with enough
space for a temporary full-size copy.

## Path resolution is memoized

`get_database_path()` sits on the per-word dictionary-reference lookup path, and
both of its resolution steps — `Path.resolve()` and `importlib.resources` — hit
the filesystem (~26 µs per call, against a ~0.2 µs cache hit for the lookup it
guards). `_resolve_database_path` therefore memoizes on the raw environment
value, so the filesystem work runs once per distinct setting rather than once
per word.

Reading the environment variable is still done on every call, so changing
`PROCESS_SANSKRIT_DB_PATH` at runtime still selects a different lexicon without
an explicit reset — behaviour pinned by
`tests/test_database_path_configuration.py::test_reference_connection_tracks_environment_path_changes`.
What is cached is the *resolution* of a given value, not the choice of value. A
changed environment value is a different cache key, so it misses by
construction. The only case the key cannot see is the same value now resolving
elsewhere (a retargeted symlink, a recreated directory); for that, call
`reset_database_path_cache()`. The `_reset_database_state()` and
`_reset_reference_state()` helpers already do. The `os.register_at_fork` hook
retains this path cache because it holds no file descriptors. It closes the
module-owned SQLAlchemy session and engine in the parent immediately before the
fork, then the parent and child reopen independent read-only engines lazily.
No SQLite connection created in the parent is used or closed by the child.

## Known issue: the engine does not follow a runtime path change

`get_engine()` caches `_engine` on first use and never re-reads the environment,
whereas the dictionary-reference layer resolves the path per lookup. Changing
`PROCESS_SANSKRIT_DB_PATH` *after* the engine is built therefore moves the
reference layer to the new lexicon while root and inflection queries keep
serving the old one — silently, in the same process. This predates the
memoization and is not introduced by it. Until it is resolved, set the variable
before first use, or call both `databaseSetup._reset_database_state()` and
`dictionary_references._reset_reference_state()` after changing it.

The analysis cache is a separate writable SQLite file. Configure it with
`PROCESS_SANSKRIT_CACHE_PATH`, or disable it with
`PROCESS_SANSKRIT_CACHE_ENABLED=false`; never point the cache at the lexicon.
