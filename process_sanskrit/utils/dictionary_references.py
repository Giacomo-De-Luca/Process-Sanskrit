"""Low-memory dictionary cross-reference mapping.

The historical implementation embedded roughly 247,000 entries in one Python
literal.  Compiling that module caused a very large transient memory spike.
This module keeps the same mapping-style interface while reading unchanged
entries from the indexed ``word_list`` SQLite table.  A small generated overlay
preserves entries that differ from the historical mapping.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from process_sanskrit.utils.resourcePaths import get_database_path

_RESOURCES = Path(__file__).resolve().parents[1] / "resources"
_OVERLAY = _RESOURCES / "dictionary_references_overlay.json"
_thread_state = threading.local()


def _cache_size() -> int:
    value = os.getenv("PROCESS_SANSKRIT_REFERENCE_CACHE_SIZE", "32768")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(
            "PROCESS_SANSKRIT_REFERENCE_CACHE_SIZE must be an integer"
        ) from error
    if parsed <= 0:
        raise ValueError(
            "PROCESS_SANSKRIT_REFERENCE_CACHE_SIZE must be greater than zero"
        )
    return parsed


@lru_cache(maxsize=1)
def _overlay() -> Dict[str, object]:
    with _OVERLAY.open("r", encoding="utf-8") as overlay_file:
        return json.load(overlay_file)


def _connection(database_path: Optional[Path] = None) -> sqlite3.Connection:
    selected_path = (
        get_database_path()
        if database_path is None
        else Path(database_path).expanduser().resolve()
    )
    connection = getattr(_thread_state, "reference_connection", None)
    connection_path = getattr(_thread_state, "reference_connection_path", None)
    if connection is not None and connection_path != selected_path:
        connection.close()
        connection = None
    if connection is None:
        if not selected_path.exists():
            raise FileNotFoundError(
                f"Dictionary database not found at {selected_path}. "
                "Run 'update-ps-database' first."
            )
        connection = sqlite3.connect(
            f"{selected_path.as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA cache_size=-2048")
        _thread_state.reference_connection = connection
        _thread_state.reference_connection_path = selected_path
    return connection


@lru_cache(maxsize=_cache_size())
def _lookup_for_path(word: str, database_path: Path) -> Optional[Tuple[str, ...]]:
    overlay = _overlay()
    overrides = overlay["overrides"]
    if word in overrides:
        return tuple(overrides[word])

    only_python = overlay["only_python"]
    if word in only_python:
        return tuple(only_python[word])

    row = _connection(database_path).execute(
        "SELECT dict_names FROM word_list WHERE keys_iast = ?",
        (word,),
    ).fetchone()
    if row is None:
        return None
    return tuple(json.loads(row[0]))


def _lookup(word: str) -> Optional[Tuple[str, ...]]:
    return _lookup_for_path(word, get_database_path())


def _reset_reference_state() -> None:
    """Close thread-local state and clear lookups after configuration changes."""
    connection = getattr(_thread_state, "reference_connection", None)
    if connection is not None:
        connection.close()
    for attribute in ("reference_connection", "reference_connection_path"):
        if hasattr(_thread_state, attribute):
            delattr(_thread_state, attribute)
    _lookup_for_path.cache_clear()


class DictionaryReferences(Mapping):
    """A read-only, mapping-compatible view of dictionary references."""

    def __getitem__(self, word: str) -> List[str]:
        result = _lookup(word)
        if result is None:
            raise KeyError(word)
        return list(result)

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and _lookup(word) is not None

    def __iter__(self) -> Iterator[str]:
        for (word,) in _connection(get_database_path()).execute(
            "SELECT keys_iast FROM word_list ORDER BY keys_iast"
        ):
            yield word
        yield from _overlay()["only_python"]

    def __len__(self) -> int:
        database_rows = _connection(get_database_path()).execute(
            "SELECT COUNT(*) FROM word_list"
        ).fetchone()[0]
        return database_rows + len(_overlay()["only_python"])


DICTIONARY_REFERENCES = DictionaryReferences()


__all__ = ["DICTIONARY_REFERENCES", "DictionaryReferences"]
