"""The Digital Corpus of Sanskrit as gold for compound members.

DCS annotates every member of a compound as a word of its own, with its lemma
and its unsandhied form, and marks it `Case=Cpd`.  `ConlluReader` reads the
CoNLL-U dump into `Sentence` / `Token` / `Segment`; a `GoldWord` is a run of
members and the head they belong to.

`DcsEvidence` then finds where each wanted headword occurs: *whole*, as the
lemma of a segment (`sukha`, also inside a compound), or *split*, as a run of
segments whose forms join to it by sandhi (`tad` + `jātīya` -> `tajjātīya`,
`antaḥ` read as `antar` + `gata` -> `antargata`).  The split occurrences are
the gold for that headword's members.

The reading conventions follow the DCS reader of the split-ranker build
(`scripts/build_split_ranker.py` on the `split-ranker` branch): a parsed
chapter is read only when its plain twin is missing, U+FFB1 stands for Vedic
`ḷ`, and the abstract suffixes `-tva` / `-tā`, which DCS writes as words, are
folded onto their base.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import islice, product
from pathlib import Path
from typing import Collection, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .joiner import SandhiJoiner
from .lexicon import JoinSpellings, Spelling


@dataclass(frozen=True)
class Segment:
    """One DCS word: its form in the token, lemma, unsandhied form, and two flags."""

    form: str
    lemma: str
    unsandhied: str
    compound: bool
    participle: bool


@dataclass(frozen=True)
class GoldWord:
    """The members of one compound with its head, or one simple word."""

    segments: Tuple[Segment, ...]

    @property
    def lemmas(self) -> Tuple[str, ...]:
        return tuple(segment.lemma for segment in self.segments)

    @property
    def is_compound(self) -> bool:
        return len(self.segments) > 1

    @property
    def complete(self) -> bool:
        """False when the token ends inside the compound (its head is in a later token)."""
        return not self.segments[-1].compound


@dataclass(frozen=True)
class Token:
    """A whitespace token of the sentence and the DCS words it covers."""

    surface: str
    segments: Tuple[Segment, ...]

    def words(self) -> Tuple[GoldWord, ...]:
        words: List[GoldWord] = []
        current: List[Segment] = []
        for segment in self.segments:
            current.append(segment)
            if not segment.compound:
                words.append(GoldWord(tuple(current)))
                current = []
        if current:
            words.append(GoldWord(tuple(current)))
        return tuple(words)


@dataclass(frozen=True)
class Sentence:
    sent_id: str
    text: str
    work: str
    tokens: Tuple[Token, ...]


class ConlluReader:
    """The DCS CoNLL-U dump, one folder per work."""

    _UNSANDHIED = re.compile(r"Unsandhied=([^|]+)")
    SUFFIX_LEMMAS = frozenset({"tva", "tā"})

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def files(self) -> List[Path]:
        plain = sorted(self.directory.glob("*/*.conllu"))
        present = {str(path) for path in plain}
        parsed = [
            path for path in sorted(self.directory.glob("*/*.conllu_parsed"))
            if str(path)[: -len("_parsed")] not in present
        ]
        return plain + parsed

    def sentences(self) -> Iterator[Sentence]:
        for path in self.files():
            with path.open(encoding="utf-8") as handle:
                yield from self.parse(handle, path.parent.name)

    @staticmethod
    def clean(line: str) -> str:
        return line.rstrip("\n").replace("ﾱ", "ḷ")

    @classmethod
    def parse(cls, lines: Iterable[str], work: str) -> Iterator[Sentence]:
        text: Optional[str] = None
        sent_id = ""
        rows: List[List[str]] = []
        for raw in lines:
            line = cls.clean(raw)
            if line.startswith("# text = "):
                text, sent_id, rows = line[9:], "", []
            elif line.startswith("# sent_id = "):
                sent_id = line[12:].strip()
            elif line.startswith("#"):
                continue
            elif not line.strip():
                if text is not None and rows:
                    yield Sentence(sent_id, text, work, cls._tokens(rows))
                text, rows = None, []
            elif text is not None:
                columns = line.split("\t")
                if len(columns) >= 10:
                    rows.append(columns)
        if text is not None and rows:
            yield Sentence(sent_id, text, work, cls._tokens(rows))

    @classmethod
    def _segment(cls, columns: Sequence[str]) -> Segment:
        features = columns[5].split("|")
        unsandhied = cls._UNSANDHIED.search(columns[9])
        return Segment(
            columns[1], columns[2], unsandhied.group(1) if unsandhied else columns[1],
            "Case=Cpd" in features, "VerbForm=Part" in features,
        )

    @classmethod
    def _fold_suffixes(cls, segments: List[Segment]) -> Tuple[Segment, ...]:
        """`viṣaya` + `tvāt` (lemma `tva`) as the one word `viṣayatvāt`, lemma `viṣayatva`."""
        folded: List[Segment] = []
        for segment in segments:
            if segment.lemma in cls.SUFFIX_LEMMAS and folded and folded[-1].compound:
                base = folded.pop()
                segment = Segment(
                    base.form + segment.form, base.lemma + segment.lemma,
                    base.unsandhied + segment.unsandhied, segment.compound, segment.participle,
                )
            folded.append(segment)
        return tuple(folded)

    @classmethod
    def _tokens(cls, rows: List[List[str]]) -> Tuple[Token, ...]:
        ranges: Dict[int, Tuple[int, str]] = {}
        for columns in rows:
            first, dash, last = columns[0].partition("-")
            if dash and first.isdigit() and last.isdigit():
                ranges[int(first)] = (int(last), columns[1])
        tokens: List[Token] = []
        covering: Optional[Tuple[int, str]] = None
        segments: List[Segment] = []
        for columns in rows:
            if not columns[0].isdigit():
                continue
            index = int(columns[0])
            if covering is None and index in ranges:
                covering = ranges[index]
                segments = []
            segment = cls._segment(columns)
            if covering is None:
                tokens.append(Token(columns[1], cls._fold_suffixes([segment])))
                continue
            segments.append(segment)
            if index >= covering[0]:
                tokens.append(Token(covering[1], cls._fold_suffixes(segments)))
                covering = None
        return tuple(tokens)


@dataclass
class HeadwordEvidence:
    """Where one headword occurs in DCS.

    `splits` counts each way DCS cut it: (DCS lemmas, forms joined) -> count.
    `surfaces` counts the tokens it occurs in, whole or split; `exact_surfaces`
    those that are nothing but this word (`ālayavijñānasya`), the forms a user
    could type for it.
    """

    whole: int = 0
    splits: Counter = field(default_factory=Counter)
    surfaces: Counter = field(default_factory=Counter)
    exact_surfaces: Counter = field(default_factory=Counter)

    @property
    def occurrences(self) -> int:
        return self.whole + sum(self.splits.values())

    def gold_splits(self, limit: int = 3) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        return [split for split, _ in self.splits.most_common(limit)]

    def as_dict(self, surfaces: int = 10) -> dict:
        return {
            "whole": self.whole,
            "splits": [[list(lemmas), list(forms), count] for (lemmas, forms), count in self.splits.most_common(5)],
            "surfaces": self.surfaces.most_common(surfaces),
        }


class DcsEvidence:
    """The occurrences in DCS of a set of wanted headwords."""

    ## nominal endings of an a-stem participle, longest first: a participle
    ## heading a compound has DCS's root as its lemma (`bhūṣitā`, lemma `bhūṣ`),
    ## so its stem is read off its form.
    PARTICIPLE_ENDINGS = (
        "ābhyām", "ebhyaḥ", "ābhyaḥ", "ānām", "āyāḥ", "āyām", "ābhiḥ",
        "asya", "āyai", "ayoḥ", "āsu", "eṣu", "āni", "aiḥ", "ena", "āya", "ayā",
        "aḥ", "am", "āt", "au", "āḥ", "ān", "ām", "e", "ā",
    )

    MAX_FORMS = 16

    def __init__(self, joiner: SandhiJoiner, wanted: Collection[str], max_run: int = 5):
        self.joiner = joiner
        self.max_run = max_run
        self._wanted: Dict[str, Tuple[str, ...]] = Spelling.keys_of(wanted)
        self._evidence: Dict[str, HeadwordEvidence] = {}

    def get(self, headword: str) -> Optional[HeadwordEvidence]:
        return self._evidence.get(headword)

    def items(self):
        return self._evidence.items()

    def __len__(self) -> int:
        return len(self._evidence)

    def _entry(self, headword: str) -> HeadwordEvidence:
        if headword not in self._evidence:
            self._evidence[headword] = HeadwordEvidence()
        return self._evidence[headword]

    def add(self, sentence: Sentence) -> None:
        for token in sentence.tokens:
            words = token.words()
            for word in words:
                self._add_word(token.surface, word, alone=len(words) == 1)

    def _add_word(self, surface: str, word: GoldWord, alone: bool) -> None:
        segments = word.segments
        for segment in segments:
            for headword in self._wanted.get(Spelling.key(segment.lemma), ()):
                evidence = self._entry(headword)
                evidence.whole += 1
                evidence.surfaces[surface] += 1
                if alone and len(segments) == 1:
                    evidence.exact_surfaces[surface] += 1
        if len(segments) < 2:
            return
        found = set()
        for start in range(len(segments) - 1):
            for end in range(start + 1, min(len(segments), start + self.max_run)):
                run = segments[start:end + 1]
                for forms in self._run_forms(run):
                    for joined in self.joiner.join(forms):
                        for headword in self._wanted.get(Spelling.key(joined), ()):
                            if (headword, start, end) in found:
                                continue
                            found.add((headword, start, end))
                            evidence = self._entry(headword)
                            evidence.splits[(tuple(segment.lemma for segment in run), forms)] += 1
                            evidence.surfaces[surface] += 1
                            if alone and word.complete and start == 0 and end == len(segments) - 1:
                                evidence.exact_surfaces[surface] += 1

    @classmethod
    def _run_forms(cls, run: Sequence[Segment]) -> List[Tuple[str, ...]]:
        """The spellings a run of segments is joined from, DCS's own spelling first.

        Each member as DCS spells it or in a pausa reading (`JoinSpellings`),
        then the head's stem: its lemma, or for a participle the stem read
        off its form.  At most `MAX_FORMS` combinations.
        """
        members = [JoinSpellings.of(segment.unsandhied) for segment in run[:-1]]
        head = run[-1]
        if head.compound:
            heads = [head.unsandhied]
        else:
            heads = [head.lemma]
            if head.participle:
                heads += [stem for stem in cls.participle_stems(head.unsandhied) if stem != head.lemma]
        combinations = (choice + (stem,) for choice in product(*members) for stem in heads)
        return list(islice(combinations, cls.MAX_FORMS))

    @classmethod
    def participle_stems(cls, form: str) -> List[str]:
        for ending in cls.PARTICIPLE_ENDINGS:
            if form.endswith(ending) and len(form) > len(ending) + 1:
                return [form[: -len(ending)] + "a"]
        return []
