"""Split-only Parser, API-compatible with sanskrit_parser.Parser.split().

``Parser.parse()`` (dependency parsing via VakyaGraph) is deliberately absent --
see NOTICE.md. ``split()`` keeps upstream's signature and return type so
``Split`` still stringifies to a list literal in the requested encoding, which is
what functions/sandhiSplitter.py parses back with ast.literal_eval.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Sequence

from indic_transliteration import sanscript

from .sandhi_analyzer import LexicalSandhiAnalyzer
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
        self.sandhi_analyzer = LexicalSandhiAnalyzer()

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
            if not self.sandhi_analyzer.forms.valid(o.canonical()):
                # Possible sakaranta: retry with ending visarga as 's'
                o = SanskritObject(seg, encoding=self.input_encoding,
                                   strict_io=self.strict_io, replace_ending_visarga='s')
            if not self.sandhi_analyzer.forms.valid(o.canonical()):
                logger.warning("Unknown pada %s - will be split", seg)
                _s = list(self.split(seg, pre_segmented=False, limit=1))[0]
                s.extend(_s.split)
            else:
                s.append(o)
        return s

    def split(self, input_string: str, limit: int = 10, pre_segmented: bool = False):
        s = self._maybe_pre_segment(input_string, pre_segmented)
        graph = self.sandhi_analyzer.getSandhiSplits(s, pre_segmented=pre_segmented)
        if graph is None:
            warnings.warn(
                "No splits found. Please check the input to ensure there are no typos."
            )
            return None
        splits = graph.find_all_paths(max_paths=limit, sort=True, score=self.score)
        return [Split(self, input_string, split) for split in splits]


@dataclass
class Split:
    parser: Parser
    input_string: str
    split: Sequence[SanskritObject]

    def __repr__(self):
        return f'Split({self.input_string}) = {self.split}'

    def __str__(self):
        out = [t.transcoded(self.parser.output_encoding, self.parser.strict_io)
               for t in self.split]
        return str(out)
