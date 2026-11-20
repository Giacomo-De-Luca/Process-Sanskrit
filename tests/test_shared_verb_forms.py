"""A verb form that belongs to several roots: every root in lemma mode, one elsewhere.

2,127 of the 98,441 forms in the verb table (2.2%) belong to more than one
root, and the lookup used to return one: the last matching `vlgtab2` row and
its first paradigm ("the historical last-match rule").  The roots sit at two
levels of the data, and `SQLite_find_verb` now reads both:

* several `vlgtab2` rows for the form -- `kariṣyati`: stems `kṛ` and `kṝ`;
  `ucyate`: models `pre-4m` (`uc`) and `pre-p` (`vac`);
* several `vlgtab1` paradigms for one stem and model -- `abhajyata`: stem
  `abhajy`, model `ipf-p`, paradigms of `bhaj` and `bhañj`.  767 of the 2,127
  forms reach a root only this way.

The entry the old rule returned comes first.  Lemma mode reports every root
(`every_verb_root`); `roots`, `parts` and detailed output report the first only,
exactly as before, because `clean_results` re-joins a prefix with one root and
would leave the others as pieces of their own (`saṃkariṣyati` -> `saṃkṝ`,
`kṛ`).  See documentation/lemma-mode.md ("Verb forms of several roots").
"""

import unittest
from unittest.mock import patch

from process_sanskrit import process
from process_sanskrit.functions import hybridSplitter
from process_sanskrit.functions.SQLiteFind import SQLite_find_verb
from process_sanskrit.functions.inflect import Inflector, TokenAnalysis
from process_sanskrit.functions.lemmaAnalysis import LemmaAnalyzer
from process_sanskrit.functions.rootAnyWord import (
    _EVERY_VERB_ROOT,
    _direct_roots,
    every_verb_root,
    root_any_word,
)
from process_sanskrit.utils.databaseSetup import session_scope
from process_sanskrit.utils.paradigm import VERBAL_PERSONS, tags_for
from process_sanskrit.utils.resourcePaths import get_database_path


def lemma_parts(word):
    result = process(word, mode="lemmas", cached=False, input_scheme="IAST")
    return [(part["form"], part["lemmas"]) for item in result for part in item["parts"]]


class _WithSession(unittest.TestCase):
    def setUp(self):
        context = session_scope()
        self.session = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class LookupTests(_WithSession):
    """`SQLite_find_verb` returns one entry per distinct root, the old answer first."""

    def entries(self, form):
        return SQLite_find_verb(form, session=self.session)

    def roots(self, form):
        return [entry[0] for entry in self.entries(form)]

    def test_several_index_rows(self):
        """`kṝ` is what the last-match rule returned, so it stays first."""
        self.assertEqual(self.roots("kariṣyati"), ["kṝ", "kṛ"])
        self.assertEqual(self.roots("cakāra"), ["kṝ", "kṛ"])
        self.assertEqual(self.roots("ucyate"), ["uc", "vac"])

    def test_several_paradigms_of_one_stem_and_model(self):
        """Both roots of `abhajyata` hang off the same index row."""
        self.assertEqual(self.roots("abhajyata"), ["bhaj", "bhañj"])

    def test_each_entry_is_read_from_its_own_paradigm(self):
        """`abhakṣata` is a third singular of `bhakṣ` and a third plural of `bhaj`."""
        entries = self.entries("abhakṣata")
        self.assertEqual(
            [entry[0:3] for entry in entries],
            [["bhakṣ", "ipf-1m", [("Third", "Sg")]], ["bhaj", "aor-m", [("Third", "Pl")]]],
        )
        for entry in entries:
            with self.subTest(root=entry[0]):
                self.assertEqual(entry[2], tags_for(entry[3], "abhakṣata", rows=VERBAL_PERSONS))
                self.assertEqual(entry[4], "abhakṣata")

    def test_a_later_root_is_read_from_a_paradigm_that_has_the_form(self):
        """`juhumahe`: the first `hu` row is a perfect whose table lacks the form.

        The first root keeps the row the old rule picked, tagged or not.
        """
        self.assertEqual(
            [entry[0:3] for entry in self.entries("juhumahe")],
            [["hve", "prf-m", None], ["hu", "pre-3m", [("First", "Pl")]]],
        )

    def test_a_form_of_one_root_gives_one_entry(self):
        """`gacchati` has two index rows (stems `gacch`, `gam`) but one root."""
        entries = self.entries("gacchati")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][0:3], ["gam", "pre-1a", [("Third", "Sg")]])
        self.assertEqual(entries[0][4], "gacchati")

    def test_an_unknown_form_gives_nothing(self):
        self.assertIsNone(SQLite_find_verb("xyzabc", session=self.session))


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class ReadingPolicyTests(_WithSession):
    """`every_verb_root` decides how many roots the morphology lookups report."""

    def roots(self, word, memo):
        return [entry[0] for entry in root_any_word(word, session=self.session, _memo=memo)]

    def test_the_first_root_only_by_default(self):
        self.assertEqual(self.roots("kariṣyati", {}), ["kṝ"])

    def test_every_root_inside_the_block(self):
        with every_verb_root():
            self.assertEqual(self.roots("kariṣyati", {}), ["kṝ", "kṛ"])

    def test_the_block_restores_the_default_even_when_it_raises(self):
        with self.assertRaises(RuntimeError):
            with every_verb_root():
                raise RuntimeError
        self.assertEqual(self.roots("kariṣyati", {}), ["kṝ"])

    def test_one_memo_serves_both_settings(self):
        """The memo keeps every root; the setting decides how many come back."""
        memo = {}
        with every_verb_root():
            self.assertEqual(self.roots("kariṣyati", memo), ["kṝ", "kṛ"])
        self.assertEqual(self.roots("kariṣyati", memo), ["kṝ"])
        with every_verb_root():
            self.assertEqual(self.roots("kariṣyati", memo), ["kṝ", "kṛ"])

    def test_a_hit_reads_the_same_whichever_setting_filled_the_memo(self):
        """By repr, tag types included: a miss returns the entries as read (tag
        tuples), a hit a thawed copy (tag lists), exactly as before."""
        filled_outside, filled_inside = {}, {}
        miss = _direct_roots("gacchati", self.session, filled_outside)
        with every_verb_root():
            _direct_roots("gacchati", self.session, filled_inside)
        hit = _direct_roots("gacchati", self.session, filled_outside)
        self.assertEqual(repr(_direct_roots("gacchati", self.session, filled_inside)), repr(hit))
        self.assertIsInstance(miss[-1][2][0], tuple)
        self.assertIsInstance(hit[-1][2][0], list)

    def test_the_hybrid_split_of_lemma_mode_reads_the_first_root_only(self):
        """It must be the split the other modes use, as `split_source="hybrid"` promises."""
        seen = []
        original = hybridSplitter.analyze_hybrid

        def recording(*args, **kwargs):
            seen.append(_EVERY_VERB_ROOT.get())
            return original(*args, **kwargs)

        with patch.object(hybridSplitter, "analyze_hybrid", recording):
            LemmaAnalyzer(session=self.session, cached=False, split_source="hybrid").analyze(
                "xyzabc", input_scheme="IAST"
            )
        self.assertEqual(seen, [False])

    def test_nominal_readings_come_first_and_are_never_cut(self):
        """`cakāra` is also a noun (the letter c); only the verb gains a root."""
        self.assertEqual(self.roots("cakāra", {}), ["cakāra", "kṝ"])
        with every_verb_root():
            self.assertEqual(self.roots("cakāra", {}), ["cakāra", "kṝ", "kṛ"])


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class LemmaModeTests(unittest.TestCase):
    def test_every_root_is_a_lemma(self):
        self.assertEqual(lemma_parts("kariṣyati"), [("kariṣyati", ["kṝ", "kṛ"])])
        self.assertEqual(lemma_parts("dhīyate"), [("dhīyate", ["dhī", "dhā"])])
        self.assertEqual(lemma_parts("abhajyata"), [("abhajyata", ["bhaj", "bhañj"])])

    def test_every_root_gets_its_joined_key(self):
        self.assertEqual(
            lemma_parts("saṃkariṣyati"),
            [("sam", ["sam"]), ("kariṣyati", ["kṝ", "kṛ", "saṃkṝ", "saṃkṛ"])],
        )

    def test_a_word_of_a_one_letter_root_and_another_has_both(self):
        """`agāt` is an aorist of `i` and of `gā`; a whole word keeps its one-letter root."""
        self.assertEqual(lemma_parts("dṛṣṭvāgāt")[-1], ("agāt", ["aga", "i", "gā"]))

    def test_the_scanner_reads_every_root_and_still_withholds_one_letter_ones(self):
        """`agāt` as a piece of a token the lexicon does not know, read by the scanner."""
        with session_scope() as session, every_verb_root():
            (analysis,) = Inflector(session=session, memo={}).resolve(["dṛṣṭvāgāt"])
        self.assertEqual(analysis.source, TokenAnalysis.SCANNER)
        lemmas = [entry[0] for entry in analysis.scan.pieces[-1].entries]
        self.assertIn("gā", lemmas)
        self.assertNotIn("i", lemmas)


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class OtherModesTests(unittest.TestCase):
    """`roots`, `parts` and detailed output give exactly what they gave before."""

    def test_roots_mode_keeps_the_first_root(self):
        self.assertEqual(process("kariṣyati", mode="roots", cached=False), ["kṝ"])
        self.assertEqual(process("abhajyata", mode="roots", cached=False), ["bhaj"])

    def test_a_prefix_is_still_rejoined_with_its_one_root(self):
        """With every root, the second came back as a piece of its own: `saṃkṝ`, `kṛ`."""
        self.assertEqual(process("saṃkariṣyati", mode="roots", cached=False), ["saṃkṝ"])
        self.assertEqual(process("samucyate", mode="roots", cached=False), ["samuc"])

    def test_the_n_rule_sees_the_list_it_always_saw(self):
        """`vena` is filed under `van` and `ven`; with both, `van` was rewritten."""
        self.assertEqual(
            process("rudrakopabhavatvena", mode="roots", cached=False)[-1], ("va", "vena", "van")
        )

    def test_detailed_output_has_one_verb_entry(self):
        verbs = [
            entry for entry in process("abhakṣata", cached=False)
            if isinstance(entry[1], str) and entry[1].startswith("verb,")
        ]
        self.assertEqual([(entry[0], entry[2]) for entry in verbs], [("bhakṣ", [("Third", "Sg")])])

    def test_a_lemma_mode_call_leaves_them_as_they_were(self):
        for word in ("kariṣyati", "saṃkariṣyati", "dṛṣṭvāgāt"):
            with self.subTest(word=word):
                before = process(word, mode="roots", cached=False)
                lemma_parts(word)
                self.assertEqual(process(word, mode="roots", cached=False), before)


if __name__ == "__main__":
    unittest.main()
