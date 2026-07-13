"""Rebuild the derived ``word_list`` index from the dictionary tables.

``word_list`` maps an IAST headword to the dictionaries that attest it.  It is
*derived* data: everything it contains is recoverable from the dictionary tables
in the same database.  The v1.0.2 artifact shipped an index built from only five
of the seven dictionaries -- ``cae`` and ``ddsa`` were omitted -- which the
library papered over at runtime with a JSON overlay.  Rebuilding the index from
every dictionary table reproduces the historical mapping exactly and makes that
overlay unnecessary.

The set of dictionaries is discovered from the schema rather than hardcoded, so
adding a dictionary to the database and rebuilding is enough to index it.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class WordListReport:
    """What a rebuild did, for callers that want to print or assert on it."""

    dictionaries: List[str]
    headwords: int
    dropped_tables: List[str]


class WordListBuilder:
    """Builds, inspects and repairs the ``word_list`` index."""

    #: A dictionary table is any table carrying the dictionary column signature.
    #: This excludes the inflection tables (lgtab/vlgtab), the index itself, and
    #: the split cache, none of which have all three columns.
    DICTIONARY_COLUMNS = frozenset({"keys_iast", "cleaned_body", "components"})

    #: Records which dictionaries the current index was built from, so staleness
    #: is an O(1) check instead of a scan of a quarter-million JSON values.
    SOURCES_TABLE = "word_list_sources"

    INDEX_TABLE = "word_list"
    INDEX_COLUMNS = frozenset({"keys_iast", "dict_names"})

    #: An unused byte-identical copy of ``word_list`` shipped in v1.0.2.
    LEGACY_TABLES = ("dictionary_cross_references",)

    @classmethod
    def discover_dictionaries(cls, connection: sqlite3.Connection) -> List[str]:
        """Return the dictionary tables present in the database, sorted."""
        found = []
        for (name,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ):
            columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{name}")')}
            if cls.DICTIONARY_COLUMNS <= columns:
                found.append(name)
        return found

    @classmethod
    def indexed_dictionaries(cls, connection: sqlite3.Connection) -> Set[str]:
        """Return the dictionaries the current index claims to cover."""
        if not cls._table_exists(connection, cls.SOURCES_TABLE):
            return set()
        return {
            name
            for (name,) in connection.execute(f'SELECT name FROM "{cls.SOURCES_TABLE}"')
        }

    @classmethod
    def missing_dictionaries(cls, connection: sqlite3.Connection) -> Set[str]:
        """Dictionaries present in the database but absent from the index.

        A non-empty result means the index is stale: lookups will under-report
        the dictionaries attesting a word, and headwords attested *only* by a
        missing dictionary will not resolve at all.
        """
        return set(cls.discover_dictionaries(connection)) - cls.indexed_dictionaries(
            connection
        )

    @classmethod
    def index_is_current(cls, connection: sqlite3.Connection) -> bool:
        """Return whether a structurally valid index covers the exact schema."""
        dictionaries = set(cls.discover_dictionaries(connection))
        if not dictionaries or not cls._table_exists(connection, cls.INDEX_TABLE):
            return False
        columns = {
            row[1]
            for row in connection.execute(
                f'PRAGMA table_info("{cls.INDEX_TABLE}")'
            )
        }
        if not cls.INDEX_COLUMNS <= columns:
            return False
        return cls.indexed_dictionaries(connection) == dictionaries

    @classmethod
    def build(
        cls,
        connection: sqlite3.Connection,
        drop_legacy: bool = True,
        vacuum: bool = False,
    ) -> WordListReport:
        """Rebuild the index from every dictionary table in the database.

        The swap runs in one transaction, so an interrupted rebuild leaves the
        previous index in place rather than a half-written one.
        """
        dictionaries = cls.discover_dictionaries(connection)
        mapping = cls._collect(connection, dictionaries)

        dropped: List[str] = []
        with connection:  # commit on success, roll back on exception
            connection.execute(f'DROP TABLE IF EXISTS "{cls.INDEX_TABLE}"')
            connection.execute(
                f'CREATE TABLE "{cls.INDEX_TABLE}" '
                "(keys_iast TEXT PRIMARY KEY, dict_names TEXT)"
            )
            connection.executemany(
                f'INSERT INTO "{cls.INDEX_TABLE}" VALUES (?, ?)',
                (
                    (word, json.dumps(sorted(names)))
                    for word, names in mapping.items()
                ),
            )

            connection.execute(f'DROP TABLE IF EXISTS "{cls.SOURCES_TABLE}"')
            connection.execute(
                f'CREATE TABLE "{cls.SOURCES_TABLE}" (name TEXT PRIMARY KEY)'
            )
            connection.executemany(
                f'INSERT INTO "{cls.SOURCES_TABLE}" VALUES (?)',
                ((name,) for name in dictionaries),
            )

            if drop_legacy:
                for table in cls.LEGACY_TABLES:
                    if cls._table_exists(connection, table):
                        connection.execute(f'DROP TABLE "{table}"')
                        dropped.append(table)

        if vacuum:
            # Outside the transaction: VACUUM cannot run inside one.  Only worth
            # the rewrite when producing the release artifact.
            connection.execute("VACUUM")

        return WordListReport(
            dictionaries=dictionaries,
            headwords=len(mapping),
            dropped_tables=dropped,
        )

    @classmethod
    def _collect(
        cls, connection: sqlite3.Connection, dictionaries: List[str]
    ) -> Dict[str, List[str]]:
        """Map each headword to the dictionaries attesting it."""
        mapping: Dict[str, List[str]] = defaultdict(list)
        for dictionary in dictionaries:
            for (headword,) in connection.execute(
                f'SELECT DISTINCT keys_iast FROM "{dictionary}" '
                "WHERE keys_iast IS NOT NULL"
            ):
                mapping[headword].append(dictionary)
        return mapping

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (name,),
            ).fetchone()
            is not None
        )


__all__ = ["WordListBuilder", "WordListReport"]
