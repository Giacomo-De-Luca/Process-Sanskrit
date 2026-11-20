"""Records of *how* an analysis was reached.

The analysis functions return flat lists of entries, and three facts are lost in
the flattening that `mode="lemmas"` needs:

* which prefix `root_any_word` stripped, and what was left -- it stamps the whole
  word over the remainder's own surface (see documentation/prefix-segmentation.md);
* which pieces the fallback scanner `root_compounds` matched;
* how many characters the scanner could not match at all, which it steps over
  without saying so.

Each function takes an optional trace and fills it in as it goes.  A trace only
observes: with or without one, the function returns exactly the same thing.
Pinned by tests/test_word_cascade.py.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class PrefixTrace:
    """The prefixes stripped off a word before its stem was found.

    `prefixes` names them outermost first, each under the spelling its own entry
    is filed under (`prati` for a stripped `praty`).  `remainder` is the stem's
    surface as it was matched, before the whole word was stamped over it.

    Falsy when nothing was stripped.
    """

    prefixes: List[str] = field(default_factory=list)
    remainder: Optional[str] = None

    def record(self, prefix: str, remainder: str) -> None:
        """Note one stripped prefix.

        The stripper recurses, so the innermost prefix is recorded first and each
        later call adds one further out; the remainder that counts is the first
        one seen, which is what the innermost strip left.
        """
        self.prefixes.insert(0, prefix)
        if self.remainder is None:
            self.remainder = remainder

    def __bool__(self) -> bool:
        return bool(self.prefixes)


@dataclass
class ScanPiece:
    """One stretch of a token that the scanner matched.

    `entries` is what the stretch resolved to: morphology entries, or the bare
    surface when only a dictionary knows it.  `prefixes` is what was stripped
    from the stretch to resolve it.
    """

    surface: str
    entries: list
    prefixes: PrefixTrace


@dataclass
class ScanTrace:
    """What the fallback scanner made of a token the lexicon does not know."""

    pieces: List[ScanPiece] = field(default_factory=list)
    skipped: int = 0

    def skip(self) -> None:
        """A character the scanner could match nothing at and stepped over."""
        self.skipped += 1

    def piece(self, surface: str, entries: list, prefixes: PrefixTrace) -> None:
        self.pieces.append(ScanPiece(surface, list(entries), prefixes))

    def withhold(self, unwanted: Callable[[object], bool]) -> None:
        """Drop the entries `unwanted` names, and any piece left with none.

        Withholding is a judgement on what a piece resolved to, not a failure to
        match, so it does not count as a skip.
        """
        for piece in self.pieces:
            piece.entries = [entry for entry in piece.entries if not unwanted(entry)]
        self.pieces = [piece for piece in self.pieces if piece.entries]


__all__ = ["PrefixTrace", "ScanPiece", "ScanTrace"]
