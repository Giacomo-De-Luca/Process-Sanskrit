"""How a variant is scored (scripts/compound_members/scoring.py)."""

import unittest

from scripts.compound_members.analyses import AnalysedPart
from scripts.compound_members.evaluation import EvaluationConfiguration
from scripts.compound_members.joiner import SandhiJoiner
from scripts.compound_members.scoring import CaseBook, MatchRule, MemberScorer, RetrievalSimulation, RunningTextImpact

from .support import CONFIG, FakeAnalyses, evidence_of


class MatchRuleTests(unittest.TestCase):
    """The rule as the brief states it: a line is indexed under every lemma and
    member of its parts; each query part needs one of its lemmas, or all of its
    members."""

    members = {"ālayavijñāna": ["ālaya", "vijñāna"], "sukha": []}
    rule = MatchRule(lambda lemma: MatchRuleTests.members.get(lemma, []))

    def test_the_line_is_indexed_under_lemmas_and_members(self):
        parts = [AnalysedPart("ālayavijñānam", ("ālayavijñāna",), "ālayavijñāna")]
        self.assertEqual(self.rule.terms(parts), {"ālayavijñāna", "ālaya", "vijñāna"})

    def test_a_whole_query_reaches_the_split_line_through_its_members(self):
        query = [AnalysedPart("ālayavijñāna", ("ālayavijñāna",), "ālayavijñāna")]
        line = self.rule.terms([AnalysedPart("ālaya", ("ālaya",), "ālaya"),
                                AnalysedPart("vijñānasya", ("vijñāna",), "vijñāna")])
        self.assertTrue(self.rule.matches(query, line))

    def test_every_member_is_required(self):
        query = [AnalysedPart("ālayavijñāna", ("ālayavijñāna",), "ālayavijñāna")]
        self.assertFalse(self.rule.matches(query, {"ālaya"}))

    def test_every_query_part_is_required_and_its_lemmas_are_alternatives(self):
        query = [AnalysedPart("tat", ("tad", "tat"), None), AnalysedPart("jātīyena", ("jātīya",), "jātīya")]
        self.assertTrue(self.rule.matches(query, {"tat", "jātīya"}))
        self.assertFalse(self.rule.matches(query, {"jātīya"}))

    def test_a_part_without_lemmas_or_members_matches_nothing(self):
        self.assertFalse(self.rule.matches([AnalysedPart("xyz", (), None)], {"xyz"}))


class MemberScorerTests(unittest.TestCase):
    """Judging members against the segments DCS gives a headword, by both yardsticks."""

    PIECE_LEMMAS = {"mahā": {"maha", "mahā"}, "tad": {"tad", "tat"}, "tat": {"tad", "tat"},
                    "bhūṣita": {"bhūṣita"}, "rāja": {"rāja", "rāj"}}

    def scorer(self, yardstick="index"):
        return MemberScorer(SandhiJoiner.shared(), lambda form: self.PIECE_LEMMAS.get(form, set()),
                            yardstick=yardstick)

    def test_exact_members(self):
        evidence = evidence_of((["ālaya", "vijñāna"], ["ālaya", "vijñāna"]))
        self.assertEqual(self.scorer().judge(["ālaya", "vijñāna"], evidence), MemberScorer.EXACT)

    def test_the_index_yardstick_takes_what_a_split_piece_is_indexed_under(self):
        mahā = evidence_of((["mahant", "ratha"], ["mahā", "ratha"]))
        self.assertEqual(self.scorer().judge(["mahā", "ratha"], mahā), MemberScorer.EXACT)
        self.assertEqual(self.scorer().judge(["mahat", "ratha"], mahā), MemberScorer.WRONG)
        rāja = evidence_of((["rājan", "putra"], ["rāja", "putra"]))
        self.assertEqual(self.scorer().judge(["rāja", "putra"], rāja), MemberScorer.EXACT)
        self.assertEqual(self.scorer().judge(["rājan", "putra"], rāja), MemberScorer.WRONG)

    def test_the_lemma_yardstick_takes_the_citation_lemma(self):
        mahā = evidence_of((["mahant", "ratha"], ["mahā", "ratha"]))
        self.assertEqual(self.scorer("lemma").judge(["mahat", "ratha"], mahā), MemberScorer.EXACT)
        self.assertEqual(self.scorer("lemma").judge(["mahā", "ratha"], mahā), MemberScorer.WRONG)
        rāja = evidence_of((["rājan", "putra"], ["rāja", "putra"]))
        self.assertEqual(self.scorer("lemma").judge(["rājan", "putra"], rāja), MemberScorer.EXACT)

    def test_coarser_and_finer_members(self):
        fine = evidence_of((["sarva", "ābharaṇa", "bhūṣ"], ["sarva", "ābharaṇa", "bhūṣita"]))
        self.assertEqual(self.scorer().judge(["sarvābharaṇa", "bhūṣita"], fine), MemberScorer.COARSE)
        coarse = evidence_of((["sarvābharaṇa", "bhūṣ"], ["sarvābharaṇa", "bhūṣita"]))
        self.assertEqual(self.scorer().judge(["sarva", "ābharaṇa", "bhūṣita"], coarse), MemberScorer.FINER)

    def test_wrong_missing_unverified_and_absent(self):
        evidence = evidence_of((["tad", "jātīya"], ["tad", "jātīya"]))
        self.assertEqual(self.scorer().judge(["taj", "jātīya"], evidence), MemberScorer.WRONG)
        self.assertEqual(self.scorer().judge(["tad", "jātīya"], evidence), MemberScorer.EXACT)
        self.assertEqual(self.scorer().judge([], evidence), MemberScorer.MISSING)
        whole_only = evidence_of(whole=5)
        self.assertEqual(self.scorer().judge(["su", "kha"], whole_only), MemberScorer.UNVERIFIED)
        self.assertEqual(self.scorer().judge([], whole_only), MemberScorer.NO_MEMBERS)
        self.assertEqual(self.scorer().judge(["su", "kha"], None), MemberScorer.ABSENT)


class CaseBookTests(unittest.TestCase):
    def test_outcomes(self):
        book = CaseBook.load(EvaluationConfiguration.load(CONFIG).cases_path)
        case = next(case for case in book.cases if case.headword == "rājaputra")
        self.assertEqual(book.outcome(case, ["rājan", "putra"]), CaseBook.CORRECT)
        self.assertEqual(book.outcome(case, ["rāja", "putra"]), CaseBook.ACCEPTABLE)
        self.assertEqual(book.outcome(case, []), CaseBook.MISSED)
        self.assertEqual(book.outcome(case, ["rāj", "aputra"]), CaseBook.WRONG)
        sukha = next(case for case in book.cases if case.headword == "sukha")
        self.assertEqual(book.outcome(sukha, ["su", "kha"]), CaseBook.SPURIOUS)
        self.assertEqual(book.outcome(sukha, []), CaseBook.CORRECT)


class RetrievalSimulationTests(unittest.TestCase):
    """A stem query reaches split occurrences only through members."""

    ANALYSES = FakeAnalyses({
        "ālayavijñāna": (AnalysedPart("ālayavijñāna", ("ālayavijñāna",), "ālayavijñāna"),),
        "ālayavijñānasya": (AnalysedPart("ālaya", ("ālaya",), "ālaya"),
                            AnalysedPart("vijñānasya", ("vijñāna",), "vijñāna")),
        "ālayavijñānam": (AnalysedPart("ālayavijñānam", ("ālayavijñāna",), "ālayavijñāna"),),
    })
    SAMPLE = {"ālayavijñāna": evidence_of(whole=4, surfaces={"ālayavijñānasya": 3, "ālayavijñānam": 1},
                                          exact={"ālayavijñānasya": 3})}

    def test_reached_occurrences(self):
        simulation = RetrievalSimulation(self.ANALYSES, lines_per_headword=30, query_forms=3)
        without = simulation.reached(lambda lemma: [], self.SAMPLE)
        members = {"ālayavijñāna": ["ālaya", "vijñāna"]}
        with_members = simulation.reached(lambda lemma: members.get(lemma, []), self.SAMPLE)
        ## the stem query reaches the whole token by lemma; the split one only by members
        self.assertEqual(without["ālayavijñāna"][:2], (1, 4))
        self.assertEqual(with_members["ālayavijñāna"][:2], (4, 4))
        ## the form query `ālayavijñānasya` (split) reaches `ālayavijñānam` only by members
        self.assertEqual(without["ālayavijñāna"][2:], (0, 1))
        self.assertEqual(with_members["ālayavijñāna"][2:], (1, 1))

    def test_compare_weights_the_difference_to_the_strata(self):
        reached = {"a": (3, 4, 0, 0), "b": (1, 1, 0, 0)}
        baseline = {"a": (1, 4, 0, 0), "b": (1, 1, 0, 0)}
        compared = RetrievalSimulation.compare(reached, baseline, {"a": 10.0, "b": 10.0},
                                               {"a": "s", "b": "s"}, resamples=50)
        self.assertEqual(compared["stem"]["gained_in_sample"], 2)
        self.assertEqual(compared["stem"]["lost_in_sample"], 0)
        self.assertEqual(compared["stem"]["net_weighted"], 20)
        low, high = compared["stem"]["interval"]
        self.assertLessEqual(low, 20)
        self.assertGreaterEqual(high, 20)


class RunningTextImpactTests(unittest.TestCase):
    def test_parts_with_members_and_changes_are_counted_by_token_frequency(self):
        analyses = FakeAnalyses({"sukham": (AnalysedPart("sukham", ("sukha",), "sukha"),),
                                 "ca": (AnalysedPart("ca", ("ca",), None),)})
        scorer = MemberScorer(SandhiJoiner.shared(), lambda form: set())
        impact = RunningTextImpact(analyses, scorer, {"sukha": evidence_of(whole=7)})
        result = impact.score(lambda lemma: ["su", "kha"], lambda lemma: [], {"sukham": 2, "ca": 5})
        self.assertEqual(result["tokens"], 7)
        self.assertEqual(result["parts_with_members"], {"unverified": 2})
        self.assertEqual(result["changed_vs_baseline"], {"no_members->unverified": 2})


if __name__ == "__main__":
    unittest.main()
