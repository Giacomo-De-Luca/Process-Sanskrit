"""Low-memory dictionary cross-reference mapping.

Maps an IAST headword to the dictionaries that attest it.  The historical
implementation embedded roughly 247,000 entries in one Python literal, which
cost ~558 MB of transient RSS to compile.  This module keeps the same
mapping-style interface while reading the entries from the indexed ``word_list``
SQLite table.

``word_list`` is derived from the dictionary tables in the same database; see
``utils/wordListBuilder.py``.  An index that does not cover every dictionary
present in the database under-reports which dictionaries attest a word, so a
stale one is reported rather than served silently.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from process_sanskrit.utils.resourcePaths import (
    get_database_path,
    reset_database_path_cache,
    resolve_configured_path,
)
from process_sanskrit.utils.wordListBuilder import WordListBuilder

logger = logging.getLogger(__name__)

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


def _warn_if_stale(connection: sqlite3.Connection, database_path: Path) -> None:
    """Report a ``word_list`` that does not cover every dictionary present.

    Emitted once per connection, so once per database per thread rather than
    once per lookup.
    """
    try:
        missing = WordListBuilder.missing_dictionaries(connection)
    except sqlite3.Error:  # pragma: no cover - a database too broken to inspect
        return
    if not missing:
        return
    logger.warning(
        "The word_list index in %s does not record coverage of: %s. Dictionary "
        "references may be incomplete for words attested in those dictionaries, "
        "and words attested only there will not resolve at all. Run "
        "'update-ps-database' to rebuild the index.",
        database_path,
        ", ".join(sorted(missing)),
    )


def _connection(database_path: Optional[Path] = None) -> sqlite3.Connection:
    selected_path = (
        get_database_path()
        if database_path is None
        else resolve_configured_path(database_path)
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
        _warn_if_stale(connection, selected_path)
    return connection


@lru_cache(maxsize=_cache_size())
def _lookup_for_path(word: str, database_path: Path) -> Optional[Tuple[str, ...]]:
    row = _connection(database_path).execute(
        "SELECT dict_names FROM word_list WHERE keys_iast = ?",
        (word,),
    ).fetchone()
    if row is None:
        return None
    return tuple(json.loads(row[0]))


def _lookup(word: str) -> Optional[Tuple[str, ...]]:
    return _lookup_for_path(word, get_database_path())


@lru_cache(maxsize=None)
def _stubs_for_path(database_path: Path) -> frozenset:
    """The flagged non-words, held in memory: a few hundred keys, read once.

    Small enough to load whole, and every compound cut consults it, so a query per
    lookup would be pure overhead.
    """
    return frozenset(WordListBuilder.stub_headwords(_connection(database_path)))


def _stubs() -> frozenset:
    return _stubs_for_path(get_database_path())


def _reset_reference_state() -> None:
    """Close thread-local state and clear lookups after configuration changes."""
    connection = getattr(_thread_state, "reference_connection", None)
    if connection is not None:
        connection.close()
    for attribute in ("reference_connection", "reference_connection_path"):
        if hasattr(_thread_state, attribute):
            delattr(_thread_state, attribute)
    _lookup_for_path.cache_clear()
    _stubs_for_path.cache_clear()
    reset_database_path_cache()


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

    def __len__(self) -> int:
        return _connection(get_database_path()).execute(
            "SELECT COUNT(*) FROM word_list"
        ).fetchone()[0]

    def is_stub(self, word: str) -> bool:
        """Whether this headword is an apparatus artefact rather than a word.

        Such a headword stays in the index -- it is still worth looking up a
        spelling one has actually read -- but it is not something a text can be
        built out of, so the compound splitter must not cut a word on it.  See
        ``utils/wordListBuilder.py`` for what qualifies.
        """
        return word in _stubs()


DICTIONARY_REFERENCES = DictionaryReferences()


__all__ = ["DICTIONARY_REFERENCES", "DictionaryReferences"]
