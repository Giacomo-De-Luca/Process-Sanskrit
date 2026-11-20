"""Resolve split words and compound pieces before dictionary lookup."""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.functions.compoundAnalysis import root_compounds
from process_sanskrit.utils.analysisTrace import PrefixTrace, ScanTrace
from process_sanskrit.utils.paradigm import is_one_letter_root


prefixes = ['sva', 'anu', 'sam', 'pra', 'upa', 'vi', 'nis', 'abhi', 'ni', 'pari', 'prati', 'parā', 'ava', 'adhi', 'api', 'ati', 'ud', 'dvi', 'su', 'dur', 'duḥ']


@dataclass
class TokenAnalysis:
    """What `Inflector` made of one split token -- or of two it read together.

    `tokens` are the split tokens the analysis consumed: one, or a prefix token
    and the word after it.  `entries` is exactly what `Inflector.analyze` puts
    in its flat list for them.  `source` says where they came from:

    * LOOKUP -- the lexicon resolved the token; `prefixes` says what was stripped;
    * SCANNER -- the lexicon did not, and the fallback scanner found pieces
      inside it; `scan` has the pieces and the characters it skipped;
    * UNRESOLVED -- nothing was found, and the token itself is the only entry.
    """

    LOOKUP = "lookup"
    SCANNER = "scanner"
    UNRESOLVED = "unresolved"

    tokens: Tuple[str, ...]
    entries: list
    source: str
    prefixes: Optional[PrefixTrace] = None
    scan: Optional[ScanTrace] = None


class Inflector:
    """Share one morphology fallback and memo across a sequence of words."""

    PREFIX_SPELLINGS = {"sam": ("sam", "saṃ"), "vi": ("vi", "vy")}

    ## The splitter cuts the prefix of nirgacchati off in its pausa form, `niḥ`,
    ## and `niḥ` looked up as a word is the noun `ni`.  Read the token back as
    ## the prefix it is.  This is where the `nir-` rule lives on purpose: making
    ## `nir` strippable in root_any_word instead pre-empts the dictionary and the
    ## -tā/-tva deriver (nirvikāratvāt -> nis + vikāratva).
    SPLIT_TOKEN_SPELLINGS = {"niḥ": "nis"}

    def __init__(self, *, debug=False, session=None, memo=None):
        self.debug = debug
        self.session = session
        self.memo = {} if memo is None else memo

    def _lookup(self, word, *, compound=False, trace=None):
        started = time.perf_counter() if self.debug else None
        lookup = root_compounds if compound else root_any_word
        options = {"inflection": True} if compound else {}
        result = lookup(
            word, session=self.session, _memo=self.memo, _trace=trace, **options
        )
        if self.debug:
            print(f"{lookup.__name__}({word}) took {time.perf_counter() - started:.6f} seconds")
        return result

    def _fragments(self, word, scan=None):
        """Scan a token the lexicon does not know for the words inside it.

        The scanner may not introduce a one-letter verb root.  It cuts an unknown
        token wherever something matches, and `i` has short forms that match the
        tail of almost any causative: `arcayet` leaves `ayet`, an optative of `i`.
        A root that resolves a whole token never comes through here, which is
        what lets `process()` keep `i` and `ṛ` without readmitting these.

        The same pieces are withheld from `scan`.  That is a judgement on what a
        piece resolved to, not a character the scanner failed to match, so it
        does not count as a skip.
        """
        fragments = self._lookup(word, compound=True, trace=scan)
        if scan is not None:
            scan.withhold(is_one_letter_root)
        return [piece for piece in fragments if not is_one_letter_root(piece)]

    @staticmethod
    def _without_hyphens(entries):
        """Strip the hyphens the forms table writes inside some lemmas."""
        cleaned = []
        for entry in entries:
            if isinstance(entry, list):
                entry[0] = entry[0].replace('-', '')
                cleaned.append(entry)
            else:
                cleaned.append(entry.replace('-', ''))
        return cleaned

    def resolve(self, words):
        """One `TokenAnalysis` per token read, in order.

        `analyze` is the flattening of this; the grouping is what
        `mode="lemmas"` needs and the flat list no longer shows.
        """
        words = [self.SPLIT_TOKEN_SPELLINGS.get(word, word) for word in words]
        analyses = []
        i = 0
        while i < len(words):
            word = words[i]
            if word in prefixes and i + 1 < len(words):
                next_word = words[i + 1]
                if self.debug:
                    print(f"Found prefix: {word}, next word: {next_word}")
                rooted = None
                stripped = None
                for prefix in self.PREFIX_SPELLINGS.get(word, (word,)):
                    stripped = PrefixTrace()
                    rooted = self._lookup(prefix + next_word, trace=stripped)
                    if rooted:
                        break
                if rooted:
                    analyses.append(
                        TokenAnalysis((word, next_word), rooted, TokenAnalysis.LOOKUP, stripped)
                    )
                    i += 2
                    continue

            stripped = PrefixTrace()
            rooted = self._lookup(word, trace=stripped)
            if rooted:
                analyses.append(TokenAnalysis((word,), rooted, TokenAnalysis.LOOKUP, stripped))
            else:
                scan = ScanTrace()
                rooted = self._fragments(word, scan)
                if rooted:
                    analyses.append(
                        TokenAnalysis((word,), rooted, TokenAnalysis.SCANNER, None, scan)
                    )
                else:
                    # The scanner returns [] on a complete miss, and so does
                    # _fragments when every piece it found was withheld. Preserve
                    # the token so dictionary lookup can report an unresolved
                    # entry to the caller.
                    analyses.append(
                        TokenAnalysis((word,), [word], TokenAnalysis.UNRESOLVED, None, scan)
                    )
            i += 1

        for analysis in analyses:
            analysis.entries = self._without_hyphens(analysis.entries)
        return analyses

    def analyze(self, words):
        return [entry for analysis in self.resolve(words) for entry in analysis.entries]


def inflect(splitted_text, debug=False, session=None, _memo=None):
    return Inflector(debug=debug, session=session, memo=_memo).analyze(splitted_text)
