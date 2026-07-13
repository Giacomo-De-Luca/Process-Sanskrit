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

## the three persons of a verbal paradigm, in stored order
VERBAL_PERSONS: Tuple[str, ...] = ("First", "Second", "Third")

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


__all__ = ["NOMINAL_CASES", "NUMBERS", "VERBAL_PERSONS", "tags_for"]
