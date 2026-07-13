"""Filesystem locations shared by the Process-Sanskrit runtime."""

from __future__ import annotations

import importlib.resources
import os
from pathlib import Path


DATABASE_PATH_ENV = "PROCESS_SANSKRIT_DB_PATH"


def get_database_path() -> Path:
    """Return the configured lexicon path, or the packaged default path."""
    configured_path = os.getenv(DATABASE_PATH_ENV)
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    database_resource = importlib.resources.files("process_sanskrit").joinpath(
        "resources",
        "SQliteDB.sqlite",
    )
    with importlib.resources.as_file(database_resource) as database_path:
        return Path(database_path).resolve()


__all__ = ["DATABASE_PATH_ENV", "get_database_path"]
