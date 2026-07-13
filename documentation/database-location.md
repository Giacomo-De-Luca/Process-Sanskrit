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

The configured database is opened read-only and immutable. It must already
exist; a configured but missing path fails explicitly rather than falling back
to the packaged database. `update-ps-database` is unchanged and still installs
the released database into the package resources directory.

The analysis cache is a separate writable SQLite file. Configure it with
`PROCESS_SANSKRIT_CACHE_PATH`, or disable it with
`PROCESS_SANSKRIT_CACHE_ENABLED=false`; never point the cache at the lexicon.
