r"""
Compound cuts are preferred when their first member is morphologically analysable.

`dict_word_iterative` walks a compound longest-first and scores every cut that
lands on a dictionary key.  Dictionary presence alone is weak evidence: a large
part of the lexicon is not made of *words* at all but of Monier-Williams
cross-reference stubs -- headwords whose whole body is a pointer elsewhere:

    tanni   "<s>tanni</s> <s>°nnī</s> variant reading for <s>°nvī</s>, q.v."

`tanni` is a dictionary key, but it heads no inflection table.  It is not
something a text can contain.  Yet in `tat-nirodha` ("the cessation of that") it
covers *more characters* than the true cut, and the old scoring gave both a
perfect 1.0 -- dictionary presence + a valid remainder + a length bonus, capped
at 1.0.  Since the walk is longest-first and a tie does not displace the
incumbent, the stub won:

    process("tannirodha", mode="roots")
    -> ['tanni', ('rodha', 'rudh')]          # tad + nirodha destroyed

The fix is a *preference*, not a filter: a cut whose first member has an
inflection table scores above one that does not, so `tan`/`tad` outranks `tanni`.
It must stay a preference, because plenty of genuine compound members are
unanalysable -- roughly a third of the lexicon's keys head no paradigm.
`niṣyanda` ("flowing", a BHS headword with no inflection table of its own) is the
witness for that half of the contract: it must still be found as the first member
of `niṣyanda-guṇa`, since no analysable rival competes with it there.

Both halves are load-bearing.  Turning the preference into a requirement fragments
every compound built on an unanalysable headword; dropping it lets the stubs win.

The signal has to be *lexical*, and the near miss is worth recording, because it
is the obvious thing to reach for.  "Heads no inflection table" does NOT separate
a stub from a word:

    tanni   (5) vs tan    (3)   -- only `tan` inflects    -> the SHORTER must win
    gacchat (7) vs gaccha (6)   -- only `gaccha` inflects -> the LONGER must win

Those are the same shape, so any weight big enough to beat `tanni` also beats
`gacchat` and hands the participle gacchatā ("by the one going") to gaccha + tā.
`gacchat` and `niṣyanda` are real words that merely lack a paradigm; `tanni` is not
a word at all.  So the flag is derived once, in the word-list build, from what the
dictionary *says* -- and the splitter only has to consult it.
"""

import unittest

from process_sanskrit.functions.compoundAnalysis import (
    dict_word_iterative,
    evaluate_compound_split,
    root_compounds,
)
from process_sanskrit.functions.process import process
from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.utils.databaseSetup import session_scope
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES


class LexiconAssumptions(unittest.TestCase):
    """The fixtures below only test anything if the lexicon still looks like this."""

    def test_tanni_is_a_flagged_stub_that_stays_searchable(self):
        ## still a key -- the penalty demotes it, it does not delete it
        self.assertIn("tanni", DICTIONARY_REFERENCES)
        self.assertTrue(DICTIONARY_REFERENCES.is_stub("tanni"))

    def test_the_true_first_members_are_not_flagged(self):
        for word in ("tan", "tad", "nirodha", "rodha"):
            self.assertFalse(DICTIONARY_REFERENCES.is_stub(word), word)

    def test_real_words_without_a_paradigm_are_not_flagged(self):
        ## the whole point: these have no inflection table either, and must NOT be
        ## treated like tanni -- penalising them hands gacchatā to gaccha + tā
        for word in ("niṣyanda", "gacchat"):
            self.assertIn(word, DICTIONARY_REFERENCES)
            with session_scope() as session:
                self.assertIsNone(root_any_word(word, session=session))
            self.assertFalse(DICTIONARY_REFERENCES.is_stub(word), word)


class RealWordsOutrankStubs(unittest.TestCase):
    def test_scoring_prefers_the_real_word_over_the_stub(self):
        with session_scope() as session:
            stub = evaluate_compound_split("tanni", "rodha", session=session)
            true_cut = evaluate_compound_split("tan", "nirodha", session=session)
            self.assertGreater(
                true_cut,
                stub,
                "tan+nirodha must outrank the cross-reference stub tanni+rodha",
            )

    def test_a_stub_cut_is_demoted_but_still_acceptable(self):
        ## it must stay above dict_word_iterative's 0.6 gate, so a stub can still
        ## be matched when nothing else fits
        with session_scope() as session:
            self.assertGreaterEqual(
                evaluate_compound_split("tanni", "rodha", session=session), 0.6
            )

    def test_walk_does_not_stop_on_the_stub(self):
        with session_scope() as session:
            match = dict_word_iterative("tannirodha", session=session)
        self.assertIsNotNone(match)
        self.assertNotEqual(match[0], "tanni")

    def test_root_compounds_keeps_nirodha_whole(self):
        with session_scope() as session:
            self.assertNotIn("tanni", root_compounds("tannirodha", session=session))


class PipelineOutput(unittest.TestCase):
    def test_tannirodha(self):
        roots = process("tannirodha", mode="roots")
        flat = [r[0] if isinstance(r, tuple) else r for r in roots]
        self.assertNotIn("tanni", flat)
        self.assertIn("nirodha", flat)

    def test_the_compound_that_reported_the_regression(self):
        roots = process("ananyathāviparyāsatannirodhāryagocaraiḥ", mode="roots")
        flat = [r[0] if isinstance(r, tuple) else r for r in roots]
        self.assertNotIn("tanni", flat)
        self.assertIn("nirodha", flat)
        ## the rest of the compound must survive the change
        self.assertIn("viparyāsa", flat)


class PreferenceIsNotAFilter(unittest.TestCase):
    """An unanalysable headword still wins when nothing analysable competes."""

    def test_unanalysable_headword_still_heads_a_compound(self):
        with session_scope() as session:
            self.assertEqual(
                root_compounds("niṣyandaguṇa", session=session),
                ["niṣyanda", "guṇa"],
            )

    def test_unanalysable_headword_survives_the_pipeline(self):
        self.assertEqual(process("niṣyandaguṇa", mode="roots"), ["niṣyanda", "guṇa"])

    def test_an_unanalysable_cut_still_clears_the_acceptance_gate(self):
        ## the penalty ranks; it must never push an otherwise-good cut below the
        ## 0.6 gate in dict_word_iterative, or the walk fragments the word
        with session_scope() as session:
            score = evaluate_compound_split("niṣyanda", "guṇa", session=session)
        self.assertGreaterEqual(score, 0.6)


if __name__ == "__main__":
    unittest.main()
