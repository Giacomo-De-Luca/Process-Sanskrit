"""Reading a stored paradigm.

Every inflection table in the database is a flat, colon-separated list of forms
laid out as *rows x numbers*, read across: index // 3 picks the row, index % 3
picks the number.  Nominal tables have eight rows (the cases), verbal tables have
three (the persons); nothing else about the layout differs.

The same "find every cell this surface form fills, then name those cells" step is
needed by the nominal lookup, the verbal lookup and the taddhita deriver, so it
lives here once rather than three times.
"""

from typing import List, Optional, Sequence, Tuple

## the eight cases of a nominal paradigm, in stored order
NOMINAL_CASES: Tuple[str, ...] = (
    "Nom",
    "Acc",
    "Inst",
    "Dat",
    "Abl",
    "Gen",
    "Loc",
    "Voc",
)

## CSL stores 3sg/3du/3pl, 2sg/2du/2pl, then 1sg/1du/1pl.
VERBAL_PERSONS: Tuple[str, ...] = ("Third", "Second", "First")

## every table is three columns wide, whatever its rows are
NUMBERS: Tuple[str, ...] = ("Sg", "Du", "Pl")


def tags_for(
    forms: Sequence[str],
    surface: str,
    rows: Sequence[str] = NOMINAL_CASES,
) -> Optional[List[Tuple[str, str]]]:
    """Name every cell of `forms` that `surface` fills.

    One surface form routinely fills several cells -- an ā-stem's *-te* is both
    the Voc. Sg. and all three duals -- so this returns a list, and None (not an
    empty list) when the form does not occur in the table at all, which is what
    the callers have always distinguished on.
    """
    tags = [
        (rows[index // len(NUMBERS)], NUMBERS[index % len(NUMBERS)])
        for index, form in enumerate(forms)
        if form == surface
    ]
    return tags or None


def is_verbal_paradigm(forms: Optional[Sequence[str]]) -> bool:
    """True when `forms` is a verbal table: three persons by three numbers.

    The layout is the one thing that tells the two kinds of table apart without
    reading a label.  A nominal table has eight rows, and an indeclinable stores
    its single form, so neither can be mistaken for the nine cells here.
    """
    return forms is not None and len(forms) == len(VERBAL_PERSONS) * len(NUMBERS)


def is_verbal_entry(entry) -> bool:
    """True for a morphology entry read from a verbal table.

    `entry` is a morphology entry, `[lemma, type, tags, forms, surface]`, before
    or after the dictionary lookup has extended it.  An unresolved token -- a
    string, or the three-field entry built from one -- carries no table and is
    never verbal.
    """
    return isinstance(entry, list) and len(entry) > 4 and is_verbal_paradigm(entry[3])


def is_one_letter_root(entry) -> bool:
    """True for a verbal analysis whose root is a single letter (`i`, `ṛ`).

    `process()` drops one-letter lemmas on its split path as scanner noise, and
    these two roots are the exception it has to make: without it `tatra eti`
    loses its verb.  The forms table also files `i` and `ṛ` as nouns (the letter
    names), which *are* noise, so the test is the verbal table rather than the
    lemma.
    """
    return is_verbal_entry(entry) and len(entry[0]) == 1


__all__ = [
    "NOMINAL_CASES",
    "NUMBERS",
    "VERBAL_PERSONS",
    "is_one_letter_root",
    "is_verbal_entry",
    "is_verbal_paradigm",
    "tags_for",
]
