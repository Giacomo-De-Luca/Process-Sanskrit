"""Filesystem locations shared by the Process-Sanskrit runtime."""

from __future__ import annotations

import importlib.resources
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


DATABASE_PATH_ENV = "PROCESS_SANSKRIT_DB_PATH"


@lru_cache(maxsize=8)
def _resolve_database_path(configured_path: Optional[str]) -> Path:
    """Resolve a configured value to an absolute path.

    ``Path.resolve`` and ``importlib.resources`` both hit the filesystem, and
    ``get_database_path`` sits on the per-word dictionary lookup path.  Memoizing
    on the raw environment value keeps that work to once per distinct setting
    while still letting a changed environment select a different database.

    Returning a stable *instance*, not merely a stable value, is load-bearing:
    the resolved path is part of the ``_lookup_for_path`` cache key in
    ``dictionary_references``, and a shared instance keeps that key's hash
    memoized.  Do not turn this into a defensive copy.
    """
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    database_resource = importlib.resources.files("process_sanskrit").joinpath(
        "resources",
        "SQliteDB.sqlite",
    )
    with importlib.resources.as_file(database_resource) as database_path:
        return Path(database_path).resolve()


def get_database_path() -> Path:
    """Return the configured lexicon path, or the packaged default path."""
    return _resolve_database_path(os.getenv(DATABASE_PATH_ENV))


def reset_database_path_cache() -> None:
    """Discard memoized path resolutions.

    A changed environment value needs no explicit reset: it is a different cache
    key, so it misses by construction.  This exists for the one case the key
    cannot see — the same configured value now resolving somewhere else, as with
    a retargeted symlink or a recreated directory.
    """
    _resolve_database_path.cache_clear()


__all__ = [
    "DATABASE_PATH_ENV",
    "get_database_path",
    "reset_database_path_cache",
]
