"""How a variant's members are scored.

* `MatchRule`           the search's matching rule, as compound-members-brief.md
                        states it (which lines a query matches; BM25 then ranks them);
* `MemberScorer`        a headword's members against the segments DCS gives it,
                        by one of two yardsticks;
* `CaseBook`            the hand-checked cases of benchmarks/compound-members-cases.json;
* `RetrievalSimulation` the rule run over the library's analyses of DCS
                        occurrences: does a query reach the occurrences of its word?
* `RunningTextImpact`   how often a variant's members reach the index in running
                        text, and how they are judged.
"""

import json
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from process_sanskrit.utils.lexicalResources import pausa_readings

from .analyses import AnalysedPart, LibraryAnalyses
from .dcs import HeadwordEvidence
from .joiner import SandhiJoiner
from .lexicon import Spelling


class MatchRule:
    """A line is indexed as every lemma and every member of each of its parts.
    A query is analysed the same way, and each of its parts is required: one of
    its lemmas, or all of its members."""

    def __init__(self, members_of: Callable[[str], Sequence[str]]):
        self.members_of = members_of

    def _members(self, part: AnalysedPart) -> Sequence[str]:
        return self.members_of(part.member_lemma) if part.member_lemma else ()

    def terms(self, parts: Iterable[AnalysedPart]) -> FrozenSet[str]:
        terms: Set[str] = set()
        for part in parts:
            terms.update(part.lemmas)
            terms.update(self._members(part))
        return frozenset(terms)

    def part_matches(self, part: AnalysedPart, terms: FrozenSet[str]) -> bool:
        if any(lemma in terms for lemma in part.lemmas):
            return True
        members = self._members(part)
        return bool(members) and all(member in terms for member in members)

    def matches(self, query: Sequence[AnalysedPart], terms: FrozenSet[str]) -> bool:
        return bool(query) and all(self.part_matches(part, terms) for part in query)


class MemberScorer:
    """Judges a headword's members against the ways DCS cut it.

    Two yardsticks say when a member stands for a DCS segment:

    * `index`: it is a term the index holds for the segment when the
      compound is written split, a lemma lemma mode gives the segment's form
      as the splitter hands it over, in pausa (`mahā` for a segment DCS
      lemmatises `mahant`; not `rājan` for a split `rāja`, which is indexed as
      `rāja`/`rāj`).  Such members let one form of a compound reach another.
    * `lemma`: it is the segment's citation lemma (DCS's, with its `-ant` for
      the dictionaries' `-at`: `mahant` = `mahat`).  Such members also reach
      the member written as a word of its own (`tad … jātīya`).

    Judgments:

    * `exact`       one member per segment, each standing for it;
    * `coarse`      members that join runs of segments (`sarvābharaṇa` + `bhūṣita`
                    for `sarva` + `ābharaṇa` + `bhūṣita`); `finer` the reverse;
    * `wrong`       anything else, against each of the most frequent DCS cuts;
    * `missing`     no members, although DCS cuts the headword;
    * `unverified`  members for a headword DCS never cuts (a lexicalised
                    compound such as `ālayavijñāna`, or no compound at all);
    * `no_members`  no members, and DCS never cuts it;
    * `absent`      the headword does not occur in DCS.
    """

    EXACT, COARSE, FINER, WRONG = "exact", "coarse", "finer", "wrong"
    MISSING, UNVERIFIED, NO_MEMBERS, ABSENT = "missing", "unverified", "no_members", "absent"
    ERRORS = (WRONG, MISSING)
    YARDSTICKS = ("index", "lemma")
    _RANK = {EXACT: 0, COARSE: 1, FINER: 1, WRONG: 2}

    def __init__(self, joiner: SandhiJoiner, piece_lemmas: Callable[[str], Iterable[str]],
                 gold_splits: int = 3, yardstick: str = "index"):
        if yardstick not in self.YARDSTICKS:
            raise ValueError(f"unknown yardstick {yardstick!r}; expected one of {self.YARDSTICKS}")
        self.joiner = joiner
        self.piece_lemmas = piece_lemmas
        self.gold_splits = gold_splits
        self.yardstick = yardstick

    def judge(self, members: Sequence[str], evidence: Optional[HeadwordEvidence]) -> str:
        if evidence is None or evidence.occurrences == 0:
            return self.ABSENT
        splits = evidence.gold_splits(self.gold_splits)
        if not splits:
            return self.UNVERIFIED if members else self.NO_MEMBERS
        if not members:
            return self.MISSING
        keys = tuple(Spelling.key(member) for member in members)
        return min((self._judge_split(keys, lemmas, forms) for lemmas, forms in splits), key=self._RANK.get)

    def _judge_split(self, members: Tuple[str, ...], lemmas: Tuple[str, ...], forms: Tuple[str, ...]) -> str:
        pieces = tuple(zip(lemmas, forms))
        if len(members) == len(pieces) and all(self._stands_for(m, p) for m, p in zip(members, pieces)):
            return self.EXACT
        if len(members) < len(pieces) and self._tiles(members, pieces, coarse=True):
            return self.COARSE
        if len(members) > len(pieces) and self._tiles(members, pieces, coarse=False):
            return self.FINER
        return self.WRONG

    def _stands_for(self, member: str, piece: Tuple[str, str]) -> bool:
        lemma, form = piece
        return member in self.accepted(lemma, form)

    @staticmethod
    def at_stem(lemma: str) -> str:
        """DCS writes at-stems as -ant (`mahant`, `bhagavant`); the dictionaries as -at."""
        return lemma[:-3] + "at" if lemma.endswith("ant") else lemma

    @lru_cache(maxsize=None)
    def accepted(self, lemma: str, form: str) -> FrozenSet[str]:
        """The keys of the members that stand for a segment with this DCS lemma and form."""
        if self.yardstick == "lemma":
            return frozenset({Spelling.key(lemma), Spelling.key(self.at_stem(lemma))})
        found = {Spelling.key(form)}
        for reading in pausa_readings(form):
            found.update(Spelling.key(lemma_) for lemma_ in self.piece_lemmas(reading))
        return frozenset(found)

    def _joins(self, forms: Sequence[str]) -> FrozenSet[str]:
        return frozenset(Spelling.key(joined) for joined in self.joiner.join(list(forms)))

    def _tiles(self, members: Tuple[str, ...], pieces: Tuple[Tuple[str, str], ...], coarse: bool) -> bool:
        """Whether one side cuts into consecutive runs, each joining to one item of the other side."""
        long, short = (pieces, members) if coarse else (members, pieces)

        @lru_cache(maxsize=None)
        def fits(start_long: int, start_short: int) -> bool:
            if start_short == len(short):
                return start_long == len(long)
            for end in range(start_long + 1, len(long) - (len(short) - start_short - 1) + 1):
                if self._run_matches(long[start_long:end], short[start_short], coarse) and fits(end, start_short + 1):
                    return True
            return False

        return fits(0, 0)

    def _run_matches(self, run, item, coarse: bool) -> bool:
        if coarse:  # a run of DCS segments against one member
            if len(run) == 1:
                return self._stands_for(item, run[0])
            return item in self._joins([form for _, form in run])
        lemma, form = item  # a run of members against one DCS segment
        if len(run) == 1:
            return self._stands_for(run[0], item)
        return bool(self._joins(run) & {Spelling.key(lemma), Spelling.key(form)})


@dataclass(frozen=True)
class Case:
    headword: str
    expected: Tuple[str, ...]
    acceptable: Tuple[Tuple[str, ...], ...]
    point: str
    note: str


class CaseBook:
    """Hand-checked headwords and the members search needs for them."""

    CORRECT, ACCEPTABLE, MISSED, SPURIOUS, WRONG = "correct", "acceptable", "missed", "spurious", "wrong"
    OUTCOMES = (CORRECT, ACCEPTABLE, MISSED, SPURIOUS, WRONG)

    def __init__(self, cases: Sequence[Case]):
        self.cases = tuple(cases)

    @classmethod
    def load(cls, path: Path) -> "CaseBook":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([
            Case(case["headword"], tuple(case["expected"]),
                 tuple(tuple(option) for option in case.get("acceptable", ())),
                 case.get("point", ""), case.get("note", ""))
            for case in raw["cases"]
        ])

    def outcome(self, case: Case, members: Sequence[str]) -> str:
        members = tuple(members)
        if members == case.expected:
            return self.CORRECT
        if members in case.acceptable:
            return self.ACCEPTABLE
        if not members:
            return self.MISSED
        return self.SPURIOUS if not case.expected else self.WRONG

    def score(self, members_of: Callable[[str], Sequence[str]]) -> dict:
        outcomes = []
        totals: Counter = Counter()
        for case in self.cases:
            members = list(members_of(case.headword))
            outcome = self.outcome(case, members)
            totals[outcome] += 1
            outcomes.append({"headword": case.headword, "members": members, "expected": list(case.expected),
                             "outcome": outcome, "point": case.point})
        return {"totals": {name: totals[name] for name in self.OUTCOMES}, "cases": outcomes}


class RetrievalSimulation:
    """The search's rule over the library's analyses of where a headword occurs in DCS.

    For each sampled headword, two kinds of query:

    * the headword typed as a stem (`ālayavijñāna`), as a user types a lemma;
    * the forms it most often takes in DCS (`ālayavijñānasya`).

    A query is scored against the tokens the headword occurs in (whole or
    split, inflected or inside a longer compound), each weighted by its count.
    Only the difference between two variants on the same sample is reported:
    `compare` gives the occurrences one variant reaches and the other does
    not, weighted back to the strata the sample was drawn from, with a
    bootstrap interval.
    """

    def __init__(self, analyses: LibraryAnalyses, lines_per_headword: int, query_forms: int):
        self.analyses = analyses
        self.lines_per_headword = lines_per_headword
        self.query_forms = query_forms

    def words(self, sample: Mapping[str, HeadwordEvidence]) -> Set[str]:
        """Every word the simulation needs analysed."""
        words: Set[str] = set()
        for headword, evidence in sample.items():
            words.add(headword)
            words.update(surface for surface, _ in evidence.surfaces.most_common(self.lines_per_headword))
            words.update(surface for surface, _ in evidence.exact_surfaces.most_common(self.query_forms))
        return words

    def reached(self, members_of: Callable[[str], Sequence[str]],
                sample: Mapping[str, HeadwordEvidence]) -> Dict[str, Tuple[int, int, int, int]]:
        """Per headword: (stem hits, stem total, form hits, form total), in DCS occurrences."""
        rule = MatchRule(members_of)
        reached = {}
        for headword, evidence in sample.items():
            lines = [(surface, count, rule.terms(self.analyses.parts(surface)))
                     for surface, count in evidence.surfaces.most_common(self.lines_per_headword)]
            query = self.analyses.parts(headword)
            stem_hits = sum(count for _, count, terms in lines if rule.matches(query, terms))
            stem_total = sum(count for _, count, _ in lines)
            form_hits = form_total = 0
            for form, _ in evidence.exact_surfaces.most_common(self.query_forms):
                form_query = self.analyses.parts(form)
                others = [(count, terms) for surface, count, terms in lines if surface != form]
                form_hits += sum(count for count, terms in others if rule.matches(form_query, terms))
                form_total += sum(count for count, _ in others)
            reached[headword] = (stem_hits, stem_total, form_hits, form_total)
        return reached

    @staticmethod
    def compare(reached: Mapping[str, Tuple[int, ...]], baseline: Mapping[str, Tuple[int, ...]],
                weights: Mapping[str, float], strata: Mapping[str, str], resamples: int = 1000,
                seed: int = 13) -> dict:
        """Occurrences gained and lost against the baseline, per query kind.

        `net` is weighted to the strata; `interval` is a 95% bootstrap
        interval of it, resampling headwords within their stratum.
        """
        result = {}
        by_stratum: Dict[str, List[str]] = {}
        for headword, stratum in strata.items():
            if weights.get(headword):
                by_stratum.setdefault(stratum, []).append(headword)
        rng = random.Random(seed)
        for kind, index in (("stem", 0), ("form", 2)):
            delta = {h: reached[h][index] - baseline[h][index] for h in reached}
            gained = sum(d for d in delta.values() if d > 0)
            lost = -sum(d for d in delta.values() if d < 0)
            net = sum(weights.get(h, 0.0) * d for h, d in delta.items())
            draws = []
            for _ in range(resamples):
                total = 0.0
                for headwords in by_stratum.values():
                    for _ in headwords:
                        pick = headwords[rng.randrange(len(headwords))]
                        total += weights[pick] * delta[pick]
                draws.append(total)
            draws.sort()
            interval = (draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]) if draws else (0.0, 0.0)
            result[kind] = {"gained_in_sample": gained, "lost_in_sample": lost,
                            "net_weighted": round(net), "interval": [round(interval[0]), round(interval[1])]}
        return result


class RunningTextImpact:
    """A variant's members in the parts of a sample of running text.

    The sample is DCS sentences, each token analysed on its own as lemma mode
    analyses running text token by token.  Counts parts (weighted by how
    often their token occurs) that get members, by the judgment of those
    members, and separately the parts whose members differ from the baseline's.
    """

    def __init__(self, analyses: LibraryAnalyses, scorer: MemberScorer, evidence):
        self.analyses = analyses
        self.scorer = scorer
        self.evidence = evidence

    def score(self, members_of: Callable[[str], Sequence[str]], baseline: Callable[[str], Sequence[str]],
              tokens: Mapping[str, int]) -> dict:
        judged: Counter = Counter()
        changed: Counter = Counter()
        total = 0
        for surface, count in tokens.items():
            total += count
            for part in self.analyses.parts(surface):
                if part.member_lemma is None:
                    continue
                members = list(members_of(part.member_lemma))
                found = self.evidence.get(part.member_lemma)
                if members:
                    judged[self.scorer.judge(members, found)] += count
                before = list(baseline(part.member_lemma))
                if members != before:
                    changed[f"{self.scorer.judge(before, found)}->{self.scorer.judge(members, found)}"] += count
        return {"tokens": total, "parts_with_members": dict(judged), "changed_vs_baseline": dict(changed)}
