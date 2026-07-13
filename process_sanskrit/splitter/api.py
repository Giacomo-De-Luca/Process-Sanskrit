"""Split-only Parser, API-compatible with sanskrit_parser.Parser.split().

``Parser.parse()`` (dependency parsing via VakyaGraph) is deliberately absent --
see NOTICE.md. ``split()`` keeps upstream's signature and return type, and
``Split`` still stringifies to a list literal in the requested encoding.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Sequence

from indic_transliteration import sanscript

from .backends import create_backend
from .sanskrit_base import SanskritNormalizedString, SanskritObject

logger = logging.getLogger(__name__)


class Parser:
    def __init__(self, strict_io: bool = False, input_encoding: str = None,
                 output_encoding: str = sanscript.SLP1,
                 score: bool = True,
                 replace_ending_visarga: str = None):
        self.strict_io = strict_io
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding
        self.score = score
        self.replace_ending_visarga = replace_ending_visarga
        self._backend = create_backend()
        # Retain the useful implementation detail for explicit Python-reference
        # users. Native request state is local, so there is no analyzer object.
        self._sandhi_analyzer = getattr(self._backend, "analyzer", None)

    @property
    def sandhi_analyzer(self):
        """Compatibility view of the explicit Python backend's analyzer."""
        return getattr(self._backend, "analyzer", self._sandhi_analyzer)

    @sandhi_analyzer.setter
    def sandhi_analyzer(self, analyzer):
        self._sandhi_analyzer = analyzer
        if hasattr(self._backend, "analyzer"):
            self._backend.analyzer = analyzer

    def _maybe_pre_segment(self, input_string: str, pre_segmented: bool):
        if not pre_segmented:
            s = SanskritNormalizedString(
                input_string,
                encoding=self.input_encoding,
                strict_io=self.strict_io,
                replace_ending_visarga=self.replace_ending_visarga,
            )
            logger.info("Input String in SLP1: %s", s.canonical())
            return s

        s = []
        for seg in input_string.split(" "):
            o = SanskritObject(seg, encoding=self.input_encoding,
                               strict_io=self.strict_io, replace_ending_visarga='r')
            canonical = o.canonical()
            if not canonical.isascii() or not self._backend.valid_slp1(canonical):
                # Possible sakaranta: retry with ending visarga as 's'
                o = SanskritObject(seg, encoding=self.input_encoding,
                                   strict_io=self.strict_io, replace_ending_visarga='s')
                canonical = o.canonical()
            if not canonical.isascii() or not self._backend.valid_slp1(canonical):
                logger.warning("Unknown pada %s - will be split", seg)
                _s = list(self.split(seg, pre_segmented=False, limit=1))[0]
                s.extend(_s.split)
            else:
                s.append(o)
        return s

    def split(self, input_string: str, limit: int = 10, pre_segmented: bool = False):
        if limit < 0:
            raise ValueError(
                "Stop argument for islice() must be None or an integer: "
                "0 <= x <= sys.maxsize."
            )
        prepared = self._maybe_pre_segment(input_string, pre_segmented)
        if pre_segmented:
            tokens = [token.canonical() for token in prepared]
            # The Python graph scores even its sole pre-segmented path, so a
            # broken model must remain fatal here too.
            if self.score:
                self._backend.score_slp1([tokens])
                if limit == 0:
                    raise ValueError("not enough values to unpack (expected 2, got 0)")
            paths = [] if limit == 0 else [tokens]
        else:
            canonical = prepared.canonical()
            # The retained Python parser historically treats malformed public
            # input that survives normalization (for example non-ASCII text
            # misdeclared as SLP1) as an ordinary no-split. Keep that facade
            # contract while the private native API remains strict.
            paths = (
                self._backend.split_slp1(
                    canonical, limit=limit, scored=self.score
                )
                if canonical.isascii()
                else None
            )
        if paths is None:
            warnings.warn(
                "No splits found. Please check the input to ensure there are no typos."
            )
            return None
        splits = [
            [SanskritObject(token, encoding=sanscript.SLP1) for token in path]
            for path in paths
        ]
        return [Split(self, input_string, split) for split in splits]


@dataclass
class Split:
    parser: Parser
    input_string: str
    split: Sequence[SanskritObject]

    def __repr__(self):
        return f'Split({self.input_string}) = {self.split}'

    def __str__(self):
        return str(self._transcoded_tokens())

    def _transcoded_tokens(self):
        """Return output strings without a stringify-and-parse round trip."""
        out = [t.transcoded(self.parser.output_encoding, self.parser.strict_io)
               for t in self.split]
        return out
