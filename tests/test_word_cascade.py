"""The single-word cascade and the traces lemma mode reads from it.

`process()` resolves a single word through a fixed sequence -- forms table,
dictionary, `-tā`/`-tva` deriver -- before it falls back to splitting.  That
sequence lives in `WordCascade`, so `mode="lemmas"` and the older modes share one
ordering instead of two copies of it.

Lemma mode also needs three facts the flat entry list no longer carries: which
prefix was stripped, which split token an entry came from, and whether the
fallback scanner skipped characters.  Each is recorded in an optional trace.  The
contract pinned here is that a trace only *observes*: every function returns
exactly what it returned without one.
"""

import copy
import unittest

from process_sanskrit.functions.compoundAnalysis import root_compounds
from process_sanskrit.functions.inflect import Inflector, TokenAnalysis, inflect
from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.functions.wordCascade import WordCascade, WordResolution
from process_sanskrit.utils.analysisTrace import PrefixTrace, ScanTrace
from process_sanskrit.utils.databaseSetup import session_scope
from process_sanskrit.utils.resourcePaths import get_database_path


class _WithSession(unittest.TestCase):
    def setUp(self):
        context = session_scope()
        self.session = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class CascadeStageTests(_WithSession):
    def resolve(self, word):
        return WordCascade(session=self.session, memo={}).resolve(word)

    def test_forms_table_resolves_an_inflected_word(self):
        resolution = self.resolve("gacchati")
        self.assertEqual(resolution.stage, WordResolution.MORPHOLOGY)
        self.assertEqual(resolution.entries[-1][0], "gam")

    def test_inserted_headword_is_reported_separately(self):
        """`gacchati` is a headword of the Pali dictionary; the cascade says so."""
        resolution = self.resolve("gacchati")
        self.assertIsNotNone(resolution.headword)
        self.assertEqual(resolution.headword[0], "gacchati")
        self.assertIs(resolution.entries[0], resolution.headword)

    def test_no_headword_when_the_form_is_not_one(self):
        self.assertIsNone(self.resolve("pratyeti").headword)

    def test_stripped_prefix_is_traced(self):
        resolution = self.resolve("pratyeti")
        self.assertEqual(resolution.stage, WordResolution.MORPHOLOGY)
        self.assertEqual(resolution.prefixes.prefixes, ["prati"])
        self.assertEqual(resolution.prefixes.remainder, "eti")

    def test_dictionary_resolves_what_the_forms_table_lacks(self):
        resolution = self.resolve("gatvā")
        self.assertEqual(resolution.stage, WordResolution.DICTIONARY)
        self.assertEqual(resolution.entries[0][0], "gatvā")

    def test_deriver_runs_after_the_dictionary(self):
        resolution = self.resolve("niṣyandatā")
        self.assertEqual(resolution.stage, WordResolution.DERIVED)
        self.assertEqual(resolution.entries[0][0], "niṣyandatā")

    def test_unresolved_word_is_handed_on(self):
        resolution = self.resolve("śrutvā")
        self.assertEqual(resolution.stage, WordResolution.UNRESOLVED)
        self.assertFalse(resolution.resolved)
        self.assertEqual(resolution.text, "śrutvā")
        self.assertEqual(resolution.entries, [])

    def test_final_letter_fix_is_part_of_the_resolution(self):
        """A final sandhi letter is read back before anything is looked up.

        Whoever splits an unresolved word next must split this text, not the
        one that was passed in.
        """
        self.assertEqual(self.resolve("punar").text, "punaḥ")
        self.assertEqual(self.resolve("deśabandhaś").text, "deśabandhaḥ")


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class PrefixTraceThroughTheStripperTests(_WithSession):
    def trace(self, word):
        trace = PrefixTrace()
        root_any_word(word, session=self.session, _memo={}, _trace=trace)
        return trace

    def test_one_prefix(self):
        trace = self.trace("upagacchati")
        self.assertEqual((trace.prefixes, trace.remainder), (["upa"], "gacchati"))

    def test_sandhi_spelling_is_reported_under_its_own_name(self):
        trace = self.trace("vyavaharati")
        self.assertEqual((trace.prefixes, trace.remainder), (["vi", "ava"], "harati"))

    def test_three_prefixes_read_outermost_first(self):
        trace = self.trace("abhisamāgacchati")
        self.assertEqual((trace.prefixes, trace.remainder), (["abhi", "sam", "ā"], "gacchati"))

    def test_direct_hit_leaves_the_trace_empty(self):
        self.assertFalse(self.trace("gacchati"))

    def test_failed_lookup_leaves_the_trace_empty(self):
        self.assertFalse(self.trace("ḍḍḍḍ"))


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class ScanTraceThroughTheScannerTests(_WithSession):
    def scan(self, token):
        trace = ScanTrace()
        root_compounds(token, inflection=True, session=self.session, _memo={}, _trace=trace)
        return trace

    def test_pieces_are_recorded_with_their_surface(self):
        scan = self.scan("deśabandhaś")
        self.assertEqual([piece.surface for piece in scan.pieces], ["deśa", "bandhaś"])
        self.assertEqual([piece.entries[-1][0] for piece in scan.pieces], ["deśa", "bandha"])
        self.assertEqual(scan.skipped, 0)

    def test_skipped_characters_are_counted(self):
        self.assertGreaterEqual(self.scan("xyzabc").skipped, 2)

    def test_inflector_withholds_without_counting_a_skip(self):
        analysis = Inflector(session=self.session).resolve(["arcayet"])[0]
        self.assertEqual(analysis.source, TokenAnalysis.SCANNER)
        self.assertEqual([piece.surface for piece in analysis.scan.pieces], ["arc"])
        self.assertEqual(analysis.scan.skipped, 0)


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class TracesOnlyObserveTests(_WithSession):
    WORDS = ("gacchati", "upagacchati", "vyavaharati", "saṃgacchati", "virati",
             "anumitārthe", "pratisaṃvedanā", "ḍḍḍḍ", "api")
    TOKENS = ("deśabandhaś", "xyzabc", "arcayet", "śrutvā", "cittavṛtti", "ḍḍḍḍ")

    def test_root_any_word_returns_the_same_with_a_trace(self):
        for word in self.WORDS:
            with self.subTest(word=word):
                plain = root_any_word(word, session=self.session, _memo={})
                traced = root_any_word(word, session=self.session, _memo={}, _trace=PrefixTrace())
                self.assertEqual(plain, traced)

    def test_root_compounds_returns_the_same_with_a_trace(self):
        for token in self.TOKENS:
            with self.subTest(token=token):
                plain = root_compounds(token, inflection=True, session=self.session, _memo={})
                traced = root_compounds(
                    token, inflection=True, session=self.session, _memo={}, _trace=ScanTrace()
                )
                self.assertEqual(plain, traced)

    ## what `inflect` returned for these tokens before `resolve` existed, as
    ## (lemma, matched surface) pairs; a bare string is an unresolved token
    FLAT = {
        ("yogaḥ", "cittavṛtti", "nirodhaḥ"): [
            ("yoga", "yogaḥ"), ("yogas", "yogaḥ"), "cittavṛtti", ("nirodha", "nirodhaḥ")],
        ("upa", "āsate"): [("upa", "upa"), ("ās", "upaāsate")],
        ("niḥ", "gacchati"): [("nis", "nis"), ("gam", "nisgacchati")],
        ("deśabandhaś",): [("deśa", "deśa"), ("bandha", "bandhaḥ")],
        ("ḍḍḍḍ",): ["ḍḍḍḍ"],
        ("arcayet",): ["arc"],
    }

    @staticmethod
    def brief(entries):
        return [(entry[0], entry[4]) if isinstance(entry, list) else entry for entry in entries]

    def test_flat_analysis_is_what_it_was(self):
        for tokens, expected in self.FLAT.items():
            with self.subTest(tokens=tokens):
                self.assertEqual(self.brief(inflect(list(tokens), session=self.session)), expected)

    def test_resolve_groups_the_same_entries_by_token(self):
        for tokens, expected in self.FLAT.items():
            with self.subTest(tokens=tokens):
                grouped = Inflector(session=self.session).resolve(list(tokens))
                flattened = [entry for analysis in grouped for entry in analysis.entries]
                self.assertEqual(self.brief(flattened), expected)
                consumed = [token for analysis in grouped for token in analysis.tokens]
                self.assertEqual(len(consumed), len(tokens))

    def test_resolve_reports_which_tokens_an_analysis_consumed(self):
        grouped = Inflector(session=self.session).resolve(["upa", "āsate", "tatra"])
        self.assertEqual([analysis.tokens for analysis in grouped], [("upa", "āsate"), ("tatra",)])
        self.assertEqual(grouped[0].prefixes.prefixes, ["upa"])
        self.assertEqual(grouped[0].prefixes.remainder, "āsate")

    def test_older_modes_render_what_they_rendered_before_the_move(self):
        """Detailed output of single-word-path words, pinned from before the cascade moved."""
        from process_sanskrit import process

        expected = {
            "gacchati": ["gacchati", "gam"],
            "upagacchati": ["upagacchati", "upa", "gam"],
            "virati": ["virati", "vi", "vi", "ratī"],
            "gatvā": ["gatvā"],
            "niṣyandatā": ["niṣyandatā"],
        }
        for word, lemmas in expected.items():
            with self.subTest(word=word):
                self.assertEqual([entry[0] for entry in process(word, cached=False)], lemmas)

    def test_cascade_does_not_share_its_entries_with_a_later_call(self):
        """`clean_results` rewrites the list it is given; the next resolution must not see that."""
        from process_sanskrit import process

        first = copy.deepcopy(WordCascade(session=self.session, memo={}).resolve("saṃgacchati").entries)
        process("saṃgacchati", mode="roots", cached=False)
        second = WordCascade(session=self.session, memo={}).resolve("saṃgacchati").entries
        self.assertEqual([entry[0] for entry in second], [entry[0] for entry in first])


if __name__ == "__main__":
    unittest.main()
