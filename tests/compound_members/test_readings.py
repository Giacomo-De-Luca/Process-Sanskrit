"""The variants of member reading the evaluation compares (scripts/compound_members/readings.py).

Each option of documentation/compound-members-brief.md, on headwords whose
dictionary entries were checked by hand, built from the library's classes as
the evaluation builds them.  `conservative` must be exactly what lemma mode
ships (`LemmaLexicon.members`), and `baseline` exactly the rule before it.
"""

import random
import unittest

from process_sanskrit.utils.compoundMembers import AffixFilter
from scripts.compound_members.evaluation import EvaluationConfiguration
from scripts.compound_members.readings import AffixProfiles, prevailing_with_ties

from .support import CONFIG, has_database, shared_resources


class AffixProfilesTests(unittest.TestCase):
    """The filter profiles of the configuration."""

    @classmethod
    def setUpClass(cls):
        cls.profiles = AffixProfiles(EvaluationConfiguration.load(CONFIG).filter_profiles,
                                     is_word={"samādhi", "vi"}.__contains__)

    def test_the_extended_profile_is_the_library_filter(self):
        library, extended = AffixFilter.default(), self.profiles.affixes("extended")
        self.assertEqual((extended.prefixes, extended.suffixes, extended.runs),
                         (library.prefixes, library.suffixes, library.runs))
        ## like lemma mode, it asks whether a run of preverbs is a word
        self.assertEqual(extended.merge(["samādhi", "saṃyuktaka"]), ["samādhi", "saṃyuktaka"])
        self.assertEqual(extended.merge(["vi", "jñāna"]), ["vijñāna"])

    def test_word_runs_need_a_word_test(self):
        with self.assertRaises(ValueError):
            AffixProfiles(EvaluationConfiguration.load(CONFIG).filter_profiles).affixes("extended")

    def test_the_brief_profile_cuts_after_su_and_sa(self):
        ## the conflict the brief's "must not change" list runs into
        brief = self.profiles.hyphen_marks("brief")
        self.assertEqual(brief.members("su-kha/", "sukha"), ["su", "kha"])
        self.assertEqual(brief.members("sa-lila", "salila"), ["sa", "lila"])
        self.assertEqual(self.profiles.hyphen_marks("extended").members("su-kha/", "sukha"), [])


class TiePolicyTests(unittest.TestCase):
    def test_a_tie_between_a_compound_and_no_compound_gives_the_compound(self):
        from collections import Counter

        self.assertEqual(prevailing_with_ties(Counter({(): 1, ("muktā", "hāra"): 1})), ["muktā", "hāra"])
        self.assertEqual(prevailing_with_ties(Counter({("a", "b"): 1, ("c", "d"): 1})), [])
        self.assertEqual(prevailing_with_ties(Counter({(): 2, ("a", "b"): 1})), [])


@unittest.skipUnless(has_database(), "lexicon database is not installed")
class VariantTests(unittest.TestCase):
    """Each option, on headwords whose dictionary entries were checked by hand."""

    @classmethod
    def setUpClass(cls):
        cls.config = EvaluationConfiguration.load(CONFIG)
        cls.factory = shared_resources().factory(cls.config)

    def members(self, spec, headword):
        return self.factory.build({"name": "test", **spec}).members(headword)

    def configured(self, name):
        return self.factory.build(next(spec for spec in self.config.variants if spec["name"] == name))

    def test_the_baseline_is_the_rule_before_the_change(self):
        from process_sanskrit.functions.dictionaryLookup import headword_components
        from process_sanskrit.utils.componentMarks import ComponentMarks

        baseline = self.configured("baseline")
        session = self.factory.forms.session
        for headword in ("ālayavijñāna", "tajjātīya", "svarūpa", "sukha", "maharṣi", "mahātman", "muktāhāra"):
            self.assertEqual(baseline.members(headword),
                             ComponentMarks.prevailing_members(headword_components(headword, session=session)),
                             headword)

    def test_conservative_is_what_lemma_mode_ships(self):
        from process_sanskrit.functions.lemmaAnalysis import LemmaLexicon

        conservative = self.configured("conservative")
        lexicon = LemmaLexicon(self.factory.forms.session)
        named = ["ālayavijñāna", "tajjātīya", "manobhava", "maharṣi", "mahātman", "yudhiṣṭhira", "ātmaja",
                 "dharmakṣetra", "agnīṣoma", "sukha", "svarūpa"]
        drawn = random.Random(13).sample(sorted(self.factory.tables.headwords()), 3000)
        for headword in named + drawn:
            self.assertEqual(conservative.members(headword), lexicon.members(headword), headword)

    def test_strict_fallback_reads_edgerton_only_for_words_monier_williams_lacks(self):
        spec = {"fallback": {"dictionaries": ["bhs", "gra"], "when": "mw_absent", "filters": "extended"}}
        self.assertEqual(self.members(spec, "ālayavijñāna"), ["ālaya", "vijñāna"])
        self.assertEqual(self.members(spec, "sukha"), [])

    def test_loose_fallback_with_the_brief_filters_cuts_sukha(self):
        spec = {"fallback": {"dictionaries": ["bhs", "gra"], "when": "mw_no_members", "filters": "brief"}}
        self.assertEqual(self.members(spec, "sukha"), ["su", "kha"])
        extended = {"fallback": {"dictionaries": ["bhs", "gra"], "when": "mw_no_members", "filters": "extended"}}
        self.assertEqual(self.members(extended, "sukha"), [])

    def test_word_hyphens(self):
        spec = {"mw_hyphens": True}
        self.assertEqual(self.members(spec, "ṛgyajus"), ["ṛg", "yajus"])
        self.assertEqual(self.members(spec, "abja"), ["ab", "ja"])
        for headword in ("pratiprasava", "anucara", "upādāna", "bilavat", "vikalpa", "sukha"):
            self.assertEqual(self.members(spec, headword), [], headword)

    def test_a_minimum_length_keeps_short_etymological_pieces_together(self):
        ## Monier-Williams writes the etymology of simple words with `-`:
        ## `nṛ-pa`, `dvi-ja`.  With `min_letters` 3 neither is cut; `viśva-karman` is.
        spec = {"mw_hyphens": {"left": "word", "min_letters": 3}}
        self.assertEqual(self.members({"mw_hyphens": True}, "nṛpa"), ["nṛ", "pa"])
        self.assertEqual(self.members(spec, "nṛpa"), [])
        self.assertEqual(self.members(spec, "dvija"), [])
        self.assertEqual(self.members(spec, "viśvakarman"), ["viśva", "karman"])

    def test_a_word_hyphen_after_a_piece_that_is_no_headword(self):
        ## `sarvābharaṇa` is in no dictionary: by default the cut needs a word on
        ## both sides, so the brief's example only gains members when it may not.
        self.assertEqual(self.members({"mw_hyphens": True}, "sarvābharaṇabhūṣita"), [])
        self.assertEqual(self.members({"mw_hyphens": {"left": "any"}}, "sarvābharaṇabhūṣita"),
                         ["sarvābharaṇa", "bhūṣita"])

    def test_ties(self):
        self.assertEqual(self.members({"ties": True}, "muktāhāra"), ["muktā", "hāra"])
        self.assertEqual(self.members({"ties": True}, "sukha"), [])

    def test_affixes_in_monier_williams_members(self):
        ## the brief's rule (a suffix or a prefix is not a member), applied to
        ## Monier-Williams' own `—`: `tat—tva`, `dhī—mat`, `dur—yodhana`
        spec = {"mw_affixes": "extended"}
        self.assertEqual(self.members({}, "tattva"), ["tat", "tva"])
        for headword in ("tattva", "dhīmat", "tadvat", "duryodhana"):
            self.assertEqual(self.members(spec, headword), [], headword)
        self.assertEqual(self.members(spec, "svarūpa"), ["sva", "rūpa"])

    def test_the_narrow_member_affixes_keep_what_dcs_and_lemma_mode_treat_as_words(self):
        ## -vat and -maya are words to DCS, and lemma mode splits `balavataḥ` into them
        narrow = {"mw_affixes": "member-affixes"}
        for headword in ("tattva", "dhīmat", "duryodhana"):
            self.assertEqual(self.members(narrow, headword), [], headword)
        self.assertEqual(self.members(narrow, "balavat"), self.members({}, "balavat"))
        self.assertEqual(self.members(narrow, "viprarṣi"), self.members({}, "viprarṣi"))

    def test_compound_forms_with_and_without_the_guard(self):
        for mode in ("xref", "xref_guarded"):
            spec = {"lemma_form": mode}
            self.assertEqual(self.members(spec, "tajjātīya"), ["tad", "jātīya"], mode)
            self.assertEqual(self.members(spec, "manobhava"), ["manas", "bhava"], mode)
            ## homographs that are words of their own are left alone
            self.assertEqual(self.members(spec, "mahāratha"), ["mahā", "ratha"], mode)
            self.assertEqual(self.members(spec, "dharmakṣetra"), ["dharma", "kṣetra"], mode)
            self.assertEqual(self.members(spec, "paraloka"), ["para", "loka"], mode)
        ## `ātma` is only "in compound for ātman", but a split `ātma` is indexed as `ātma`
        self.assertEqual(self.members({"lemma_form": "xref"}, "ātmajñāna")[0], "ātman")
        self.assertEqual(self.members({"lemma_form": "xref_guarded"}, "ātmajñāna")[0], "ātma")

    def test_merged_vowel_repair(self):
        junction = {"merged_vowel": "junction"}
        self.assertEqual(self.members(junction, "nānārtha"), ["nānā", "artha"])
        ## a tie (one plain entry, one ci—trāyudha) gives no members to repair
        self.assertEqual(self.members(junction, "citrāyudha"), [])
        self.assertEqual(self.members({**junction, "ties": True}, "citrāyudha"), ["citra", "āyudha"])
        self.assertEqual(self.members(junction, "mahātman"), ["mahā", "tman"])
        always = {"merged_vowel": "always"}
        self.assertEqual(self.members(always, "mahātman"), ["mahā", "ātman"])
        self.assertEqual(self.members(always, "mahākāla"), ["mahā", "akāla"])

    def test_an_unrepaired_pair_is_kept_or_dropped(self):
        self.assertEqual(self.members({"merged_vowel": "junction"}, "vitṛṣṇa"), ["vi", "tṛṣṇa"])
        self.assertEqual(self.members({"merged_vowel": "junction", "unrepaired": "drop"}, "vitṛṣṇa"), [])
        with self.assertRaises(ValueError):
            self.members({"unrepaired": "drop"}, "vitṛṣṇa")


if __name__ == "__main__":
    unittest.main()
