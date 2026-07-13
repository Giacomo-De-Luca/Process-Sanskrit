"""Rebuild the ``word_list`` index in the lexicon database.

``word_list`` maps an IAST headword to the dictionaries attesting it.  It is
derived data -- everything in it is recoverable from the dictionary tables in
the same database -- but it was shipped as a baked table, and the v1.0.2
artifact was built from only five of the seven dictionaries: ``cae`` and
``ddsa`` were omitted.  The library papered over the gap at runtime with a
1.4 MB JSON overlay, which is now gone.

Use this to regenerate the release artifact before uploading it.  End users do
not need it: ``update-ps-database`` repairs an already-downloaded database in
place via the same builder.

    uv run python tools/build_word_list.py                     # packaged database
    uv run python tools/build_word_list.py --database PATH     # another copy
    uv run python tools/build_word_list.py --vacuum            # shrink for release

``--vacuum`` rewrites the whole 596 MB file to reclaim the space freed by
dropping the unused ``dictionary_cross_references`` duplicate.  It is slow and
needs roughly twice the database size in free disk, so it is off by default and
worth running only for the artifact you intend to publish.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from process_sanskrit.utils.resourcePaths import get_database_path
from process_sanskrit.utils.wordListBuilder import WordListBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="lexicon to rebuild (default: the packaged/configured database)",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="VACUUM afterwards to reclaim freed space (slow; for release builds)",
    )
    arguments = parser.parse_args()

    database_path = arguments.database or get_database_path()
    if not database_path.exists():
        parser.error(f"database not found: {database_path}")

    connection = sqlite3.connect(database_path)
    try:
        before = WordListBuilder.missing_dictionaries(connection)
        if before:
            print(f"Stale index: missing {', '.join(sorted(before))}")

        report = WordListBuilder.build(connection, vacuum=arguments.vacuum)

        print(f"Indexed {report.headwords} headwords from {database_path}")
        print(f"Dictionaries: {', '.join(report.dictionaries)}")
        if report.dropped_tables:
            print(f"Dropped: {', '.join(report.dropped_tables)}")
        print(f"Missing after rebuild: {WordListBuilder.missing_dictionaries(connection) or 'none'}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
