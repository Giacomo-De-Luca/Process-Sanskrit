"""Indexed morphology lookups for nouns and verbs."""

import regex
from sqlalchemy.sql import text


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

_VERB_QUERY = text(
    """
    SELECT t2.key, t2.model, t2.stem, t1.refs, t1.data
    FROM vlgtab2 AS t2
    JOIN vlgtab1 AS t1
      ON t1.stem = t2.stem AND t1.model = t2.model
    WHERE t2.rowid = (
        SELECT candidate.rowid
        FROM vlgtab2 AS candidate
        WHERE candidate.key = :verb
        ORDER BY candidate.rowid DESC
        LIMIT 1
    )
    ORDER BY t1.rowid
    LIMIT 1
    """
)


def _name_rows(word, session):
    try:
        return session.execute(_NAME_QUERY, {"word": word}).fetchall()
    except Exception:
        return []


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
    row_titles = ["Nom", "Acc", "Inst", "Dat", "Abl", "Gen", "Loc", "Voc"]
    column_titles = ["Sg", "Du", "Pl"]

    for _key, model, stem, refs, inflection_data in rows:
        if not stem:
            continue
        reference_matches = regex.findall(r",(\p{L}+)", refs or "")
        word_reference = reference_matches[0] if reference_matches else stem
        inflection_words = inflection_data.split(":")
        indices = [
            index
            for index, inflected_word in enumerate(inflection_words)
            if inflected_word == lookup_name
        ]
        row_column_names = (
            [
                (row_titles[index // 3], column_titles[index % 3])
                for index in indices
            ]
            if indices
            else None
        )
        outcome.append(
            [
                word_reference,
                model,
                row_column_names,
                inflection_words,
                lookup_name,
            ]
        )

    return outcome


def SQLite_find_verb(verb, session=None):
    """Find verbal morphology while preserving the historical last-match rule."""
    try:
        row = session.execute(_VERB_QUERY, {"verb": verb}).fetchone()
    except Exception:
        row = None

    if row is None:
        return None

    _key, model, stem, refs, inflection_data = row
    if not stem:
        return None

    reference_match = regex.search(r",(\p{L}+)", refs or "")
    if reference_match and stem != reference_match.group(1):
        stem = reference_match.group(1)

    inflection_words = inflection_data.split(":")
    indices = [
        index
        for index, inflected_word in enumerate(inflection_words)
        if inflected_word == verb
    ]
    row_titles = ["First", "Second", "Third"]
    column_titles = ["Sg", "Du", "Pl"]
    row_column_names = (
        [
            (row_titles[index // 3], column_titles[index % 3])
            for index in indices
        ]
        if indices
        else None
    )
    return [[stem, model, row_column_names, inflection_words, verb]]


def optimized_find_name(name, session=None):
    """Backward-compatible alias for the indexed nominal lookup."""
    return SQLite_find_name(name, session=session)


def optimized_find_verb(verb, session=None):
    """Backward-compatible alias for the indexed verbal lookup."""
    return SQLite_find_verb(verb, session=session)


__all__ = [
    "SQLite_find_name",
    "SQLite_find_verb",
    "optimized_find_name",
    "optimized_find_verb",
]
