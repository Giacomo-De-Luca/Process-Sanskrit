"""Indexed morphology lookups for nouns and verbs."""

import regex
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text

from process_sanskrit.utils.paradigm import NOMINAL_CASES, VERBAL_PERSONS, tags_for


_NAME_QUERY = text(
    """
    SELECT t2.key, t2.model, t2.stem, t1.refs, t1.data
    FROM lgtab2 AS t2
    JOIN lgtab1 AS t1
      ON t1.stem = t2.stem AND t1.model = t2.model
    WHERE t2.key = :word
      AND t1.rowid = (
          SELECT MIN(candidate.rowid)
          FROM lgtab1 AS candidate
          WHERE candidate.stem = t2.stem AND candidate.model = t2.model
      )
    ORDER BY t2.rowid
    """
)

## Every paradigm a verb form belongs to, in the order the historical rule ranked
## them: the *last* vlgtab2 row for the form, then its *first* vlgtab1 paradigm.
## That rule kept only the first row, so `SQLite_find_verb` keeps it first.
_VERB_QUERY = text(
    """
    SELECT t2.key, t2.model, t2.stem, t1.refs, t1.data
    FROM vlgtab2 AS t2
    JOIN vlgtab1 AS t1
      ON t1.stem = t2.stem AND t1.model = t2.model
    WHERE t2.key = :verb
    ORDER BY t2.rowid DESC, t1.rowid
    """
)


## A stem is listed in lgtab2 under its own name (key == stem) as well as under
## every form it inflects to; matching both columns is what distinguishes a
## prātipadika ("śūnya") from an inflected form of one ("śūnyāya").
_STEM_QUERY = text(
    "SELECT 1 FROM lgtab2 WHERE key = :stem AND stem = :stem LIMIT 1"
)

_PARADIGM_QUERY = text(
    "SELECT data FROM lgtab1 WHERE stem = :stem AND model = :model LIMIT 1"
)


def _name_rows(word, session):
    try:
        return session.execute(_NAME_QUERY, {"word": word}).fetchall()
    except Exception:
        return []


## These two deliberately do NOT swallow every exception the way the lookups
## above do.  A missing `session` is a programming error, and laundering the
## resulting AttributeError into "no such row" makes the taddhita deriver report
## a *stale database* -- telling the user to re-download 583 MB to fix a dropped
## argument.  Only genuine database errors are caught.


def SQLite_stem_exists(stem, session=None):
    """True when `stem` is itself a nominal stem in the inflection tables."""
    try:
        return session.execute(_STEM_QUERY, {"stem": stem}).fetchone() is not None
    except SQLAlchemyError:
        return False


def SQLite_paradigm(stem, model, session=None):
    """Return the inflected forms stored for one (stem, model), or None."""
    try:
        row = session.execute(
            _PARADIGM_QUERY, {"stem": stem, "model": model}
        ).fetchone()
    except SQLAlchemyError:
        return None
    return row[0].split(":") if row and row[0] else None


def SQLite_find_name(name, session=None):
    """Find nominal morphology using one indexed join statement."""
    lookup_name = name
    rows = _name_rows(lookup_name, session)

    if not rows and lookup_name:
        if lookup_name[-1] == "ṃ":
            lookup_name = lookup_name[:-1] + "m"
            rows = _name_rows(lookup_name, session)
        elif lookup_name[-1] == "m":
            lookup_name = lookup_name[:-1] + "ṃ"
            rows = _name_rows(lookup_name, session)

    outcome = []

    for _key, model, stem, refs, inflection_data in rows:
        if not stem:
            continue
        inflection_words = inflection_data.split(":")
        outcome.append(
            [
                paradigm_headword(stem, refs),
                model,
                tags_for(inflection_words, lookup_name, rows=NOMINAL_CASES),
                inflection_words,
                lookup_name,
            ]
        )

    return outcome


def paradigm_headword(stem, refs):
    """The headword a paradigm row is filed under: its first reference, else its stem.

    A row names its headword in `refs` (`63409,gam`) -- for a verb the root,
    while the stem is the present stem the row inflects (`gacch`).  No verb row
    names two different roots.
    """
    reference_match = regex.search(r",(\p{L}+)", refs or "")
    return reference_match.group(1) if reference_match else stem


def SQLite_find_verb(verb, session=None):
    """Find verbal morphology: one entry per root the form belongs to.

    2.2% of the forms in the verb table belong to more than one root, at two
    levels of the data: several vlgtab2 rows for the form (`kariṣyati`: stems
    `kṛ`, `kṝ`), or several vlgtab1 paradigms for one stem and model
    (`abhajyata`: `bhaj`, `bhañj`).  Each root is reported once, tagged from the
    paradigm it is read from.

    The first entry is the one the historical last-match rule returned -- the
    last vlgtab2 row and its first paradigm -- so a form of one root gives
    exactly what it always gave, and a caller that reads only the first entry
    sees what it always saw.  A later root is read from the first of its rows
    whose paradigm has the form, or from its first row when none does
    (`juhumahe`: `hu` from its present, not from a perfect that lacks the form).
    None when the form is not in the table.
    """
    try:
        rows = session.execute(_VERB_QUERY, {"verb": verb}).fetchall()
    except Exception:
        rows = []

    entries = {}
    first_root = None
    for _key, model, stem, refs, inflection_data in rows:
        if not stem:
            continue
        root = paradigm_headword(stem, refs)
        inflection_words = inflection_data.split(":")
        entry = [
            root,
            model,
            tags_for(inflection_words, verb, rows=VERBAL_PERSONS),
            inflection_words,
            verb,
        ]
        if root not in entries:
            entries[root] = entry
            first_root = first_root or root
        ## the first root stays as the old rule read it, tagged or not
        elif root != first_root and entries[root][2] is None and entry[2] is not None:
            entries[root] = entry
    return list(entries.values()) or None


def optimized_find_name(name, session=None):
    """Backward-compatible alias for the indexed nominal lookup."""
    return SQLite_find_name(name, session=session)


def optimized_find_verb(verb, session=None):
    """Backward-compatible alias for the indexed verbal lookup."""
    return SQLite_find_verb(verb, session=session)


__all__ = [
    "SQLite_find_name",
    "SQLite_find_verb",
    "paradigm_headword",
    "SQLite_paradigm",
    "SQLite_stem_exists",
    "optimized_find_name",
    "optimized_find_verb",
]
