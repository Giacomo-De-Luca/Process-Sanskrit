"""Compound members in lemma mode (`utils/compoundMembers.py`, rule 9 of
documentation/lemma-mode.md).

Members come from Monier-Williams' `—` by majority vote, as before, and now
also:

* from Edgerton's and Grassmann's `-` for a headword Monier-Williams lacks;
* re-cut where sandhi hides the boundary and a side is no word (`maha` +
  `rṣi` -> `maha` + `ṛṣi`, `nānā` + `rtha` -> `nānā` + `artha`); a pair no
  reading of the junction makes words of keeps the dictionary's cut;
* in citation form where Monier-Williams says a spelling is "in compound for"
  another word (`taj` -> `tad`) and lemma mode gives that word for the piece.

The expected values are those of the `conservative` variant measured in
documentation/compound-members-evaluation.md.
"""

import subprocess
import sys
import unittest

from process_sanskrit.utils.componentMarks import ComponentMarks
from process_sanskrit.utils.compoundMembers import (
    AffixFilter,
    CompoundForms,
    CompoundMembers,
    HyphenMarks,
    JunctionSandhi,
    MemberRepair,
)
from process_sanskrit.utils.lexicalResources import pausa_readings
from process_sanskrit.utils.preverbSandhi import PreverbSandhi
from process_sanskrit.utils.resourcePaths import get_database_path


def has_database():
    return get_database_path().exists()


class PausaReadingsTests(unittest.TestCase):
    """The spellings a compound-initial form has as a split piece, in pausa."""

    def test_readings(self):
        self.assertIn("manaḥ", pausa_readings("mano"))
        self.assertIn("ṛk", pausa_readings("ṛg"))
        self.assertIn("tat", pausa_readings("taj"))
        self.assertIn("ṣaṭ", pausa_readings("ṣaḍ"))
        self.assertIn("jagat", pausa_readings("jagan"))

    def test_the_form_comes_first(self):
        self.assertEqual(pausa_readings("mano")[0], "mano")
        self.assertEqual(pausa_readings("mahā")[0], "mahā")
        self.assertNotIn("mahat", pausa_readings("mahā"))


class PreverbSpellingsTests(unittest.TestCase):
    spellings = PreverbSandhi.spellings()

    def test_runs_of_preverbs_and_their_sandhi_spellings(self):
        for spelling in ("upā", "prot", "vyā", "abhy", "niṣ", "anv", "atipra", "samud", "vy", "uc", "pi"):
            self.assertIn(spelling, self.spellings, spelling)

    def test_a_final_i_or_u_becomes_its_semivowel_before_a_vowel(self):
        self.assertEqual(PreverbSandhi._before_next("abhi", "abhi"), {"abhi", "abhy"})
        self.assertEqual(PreverbSandhi._before_next("pratī", "i"), {"pratī", "praty"})
        self.assertEqual(PreverbSandhi._before_next("anu", "anu"), {"anu", "anv"})
        self.assertEqual(PreverbSandhi._before_next("abhyū", "ū"), {"abhyū", "abhyv"})

    def test_content_words_are_not_preverbs(self):
        for word in ("sarva", "mahā", "deva", "ālaya", "rāja"):
            self.assertNotIn(word, self.spellings, word)


class AffixFilterTests(unittest.TestCase):
    affixes = AffixFilter.default()

    def test_a_prefix_is_glued_to_what_follows_and_a_suffix_to_what_precedes(self):
        self.assertEqual(self.affixes.merge(["a", "kalpya"]), ["akalpya"])
        self.assertEqual(self.affixes.merge(["daurmanasya", "tā"]), ["daurmanasyatā"])
        self.assertEqual(self.affixes.merge(["su", "kha"]), ["sukha"])

    def test_members_are_left_alone(self):
        self.assertEqual(self.affixes.merge(["ālaya", "vijñāna"]), ["ālaya", "vijñāna"])

    def test_a_run_of_preverbs_that_is_a_word_is_a_member(self):
        ## `sam` + `ā` + `adhi` spells `samādhi`, `vi` + `pra` spells `vipra`
        affixes = AffixFilter.default(is_word={"samādhi", "vipra", "vi"}.__contains__)
        self.assertEqual(affixes.merge(["samādhi", "saṃyuktaka"]), ["samādhi", "saṃyuktaka"])
        self.assertEqual(affixes.merge(["vipra", "rājia"]), ["vipra", "rājia"])
        self.assertEqual(self.affixes.merge(["samādhi", "saṃyuktaka"]), ["samādhisaṃyuktaka"])

    def test_a_single_preverb_is_a_prefix_even_when_it_is_a_word(self):
        affixes = AffixFilter.default(is_word={"vi", "upa"}.__contains__)
        self.assertEqual(affixes.merge(["vi", "jñāna"]), ["vijñāna"])
        self.assertEqual(affixes.merge(["upa", "kāra"]), ["upakāra"])


class HyphenMarksTests(unittest.TestCase):
    """Edgerton's and Grassmann's `-`: members, unless a piece is an affix or a stub."""

    reading = HyphenMarks()

    def members(self, components, headword):
        return self.reading.members(components, headword)

    def test_members_are_cut_at_the_hyphen(self):
        self.assertEqual(self.members("ālaya-vijñāna", "ālayavijñāna"), ["ālaya", "vijñāna"])
        self.assertEqual(self.members("i/ndra-jūta", "indrajūta"), ["indra", "jūta"])
        self.assertEqual(self.members("āyuḥ-saṃskāra", "āyuḥsaṃskāra"), ["āyuḥ", "saṃskāra"])

    def test_stubs_suffixes_prefixes_and_particles_give_no_members(self):
        for components, headword in (("-ālāpaka", "ālāpaka"), ("odhṛta-", "odhṛta"),
                                     ("daurmanasya-tā", "daurmanasyatā"), ("a-kalpya", "akalpya"),
                                     ("an-adhyavasita", "anadhyavasita"), ("anubṛṃhaṇa-tā, °ṇā", "anubṛṃhaṇatā"),
                                     ("su-kha/", "sukha"), ("sa-lila", "salila"), ("aty-ā/ya", "atyāya"),
                                     ("nir-āmiṣa", "nirāmiṣa")):
            self.assertEqual(self.members(components, headword), [], components)

    def test_glue_letters_and_bracketed_context(self):
        self.assertEqual(self.members("tatha-r-iva", "tathariva"), [])
        self.assertEqual(self.members("akṣuṇṇa(-vyākaraṇa)", "akṣuṇṇa"), [])
        self.assertEqual(self.members("agni-varṇa(-ratna)", "agnivarṇa"), ["agni", "varṇa"])


class JunctionSandhiTests(unittest.TestCase):
    """The readings of two members that undo the sandhi where they meet."""

    junctions = JunctionSandhi()

    def readings(self, left, right, elided=False):
        return {(reading.left, reading.right) for reading in self.junctions.readings(left, right, elided)}

    def test_a_merged_vowel_at_the_mark(self):
        self.assertIn(("nānā", "artha"), self.readings("nānā", "rtha"))
        self.assertIn(("mahā", "arṇava"), self.readings("mahā", "rṇava"))
        self.assertTrue({("maha", "ṛṣi"), ("mahā", "ṛṣi")} <= self.readings("maha", "rṣi"))

    def test_a_mark_that_stands_early(self):
        distances = {(reading.left, reading.right): reading.distance
                     for reading in self.junctions.readings("kā", "mārtha")}
        self.assertEqual(distances[("kāma", "artha")], 1)
        self.assertIn(("daśa", "avasthā"), self.readings("da", "śāvasthā"))

    def test_never_behind_the_mark(self):
        ## the `ya` of `sūrya` is not `ī` + `a`, the `va` of `sarva` not `u` + `a`
        self.assertNotIn(("sūri", "aprabha"), self.readings("sūrya", "prabha"))
        self.assertNotIn(("saru", "artuka"), self.readings("sarva", "rtuka"))

    def test_a_semivowel_before_the_mark(self):
        self.assertIn(("madhu", "āsava"), self.readings("madhv", "āsava"))
        self.assertIn(("abhi", "upapādana"), self.readings("abhy", "upapādana"))
        self.assertIn(("tri", "aha"), self.readings("try", "aha"))

    def test_an_elided_a_only_where_an_avagraha_marks_it(self):
        self.assertIn(("manas", "anuga"), self.readings("mano", "nuga", elided=True))
        self.assertIn(("araṇye", "anuvākya"), self.readings("araṇye", "nuvākya", elided=True))
        self.assertIn(("uru", "abja"), self.readings("uru", "bja", elided=True))
        self.assertNotIn(("namas", "avṛdha"), self.readings("namo", "vṛdha"))

    def test_s_and_n_made_retroflex_by_the_member_before(self):
        self.assertIn(("tri", "stubh"), self.readings("tri", "ṣṭubh"))
        self.assertIn(("yudhi", "sthira"), self.readings("yudhi", "ṣṭhira"))
        self.assertIn(("tri", "nāman"), self.readings("tri", "ṇāman"))
        self.assertIn(("tri", "savana"), self.readings("tri", "ṣavaṇa"))

    def test_ch(self):
        self.assertIn(("gāyatra", "chandas"), self.readings("gāyatra", "cchandas"))
        self.assertIn(("bṛhat", "śravas"), self.readings("bṛhac", "chravas"))
        self.assertIn(("dik", "śabda"), self.readings("dik", "chabda"))

    def test_no_junction_no_reading(self):
        self.assertEqual(self.readings("guṇa", "vatī"), set())
        self.assertEqual(self.readings("sama", "prabha"), set())
        self.assertEqual(self.readings("", "cchandas"), set())
        self.assertEqual(self.readings("mano", "", elided=True), set())

    def test_the_window_is_two_letters_after_the_mark(self):
        distances = {(reading.left, reading.right): reading.distance
                     for reading in self.junctions.readings("ci", "trāyudha")}
        self.assertEqual(distances[("citra", "āyudha")], 2)
        ## `a—dhvāpanna`: the merged `ā` is three letters after the mark
        self.assertNotIn(("adhva", "āpanna"), self.readings("a", "dhvāpanna"))


class MemberRepairTests(unittest.TestCase):
    """Which reading of a pair replaces it, over a made-up lexicon: word -> dictionaries listing it."""

    WORDS = {
        "mahā": 7, "ṛṣi": 4, "sura": 4, "surā": 6, "nānā": 6, "artha": 5, "kā": 4, "kāma": 6, "vi": 6,
        "kṛta": 6, "akṛta": 4, "pattra": 2, "daśa": 2, "daśā": 6, "avasthā": 5, "gandha": 5,
        "gandhaka": 3, "ilikā": 2, "mano": 2, "anuga": 5,
    }

    def repair(self, members, elided=(), words=WORDS, **options):
        return MemberRepair(words.__contains__, lambda word: words.get(word, 0), **options).repair(members, elided)

    def test_the_non_word_side_is_read_again(self):
        self.assertEqual(self.repair(["maha", "rṣi"]), ["mahā", "ṛṣi"])
        self.assertEqual(self.repair(["nānā", "rtha"]), ["nānā", "artha"])

    def test_a_side_that_is_a_word_keeps_its_spelling(self):
        self.assertEqual(self.repair(["sura", "rṣi"]), ["sura", "ṛṣi"])
        self.assertEqual(self.repair(["maha", "rṣi"], words={**self.WORDS, "maha": 5}), ["maha", "ṛṣi"])
        ## no vowel is put before a right side that is a word
        self.assertEqual(self.repair(["pattrā", "kṛta"]), ["pattrā", "kṛta"])

    def test_an_avagraha(self):
        self.assertEqual(self.repair(["mano", "nuga"], elided={1}), ["mano", "anuga"])
        self.assertEqual(self.repair(["mano", "nuga"], elided={1}, words={**self.WORDS, "manas": 6}),
                         ["mano", "anuga"])
        self.assertEqual(self.repair(["mano", "nuga"]), ["mano", "nuga"])

    def test_an_early_mark_needs_words_more_dictionaries_list(self):
        self.assertEqual(self.repair(["kā", "mārtha"]), ["kāma", "artha"])
        ## `ilikā` is in two dictionaries
        self.assertEqual(self.repair(["gandha", "kelikā"]), ["gandha", "kelikā"])
        ## the left member needs two
        words = {"ka": 6, "akṣa": 5}
        self.assertEqual(self.repair(["ka", "ṭākṣa"], words={**words, "kaṭa": 1}), ["ka", "ṭākṣa"])
        self.assertEqual(self.repair(["ka", "ṭākṣa"], words={**words, "kaṭa": 2}), ["kaṭa", "akṣa"])

    def test_an_avagraha_between_two_words(self):
        ## `bhūyo—'kṣara`: `kṣara` is a word, but the mark says `akṣara`
        words = {**self.WORDS, "bhūyo": 1, "kṣara": 6, "akṣara": 5, "jaya": 5}
        self.assertEqual(self.repair(["bhūyo", "kṣara"], elided={1}, words=words), ["bhūyo", "akṣara"])
        self.assertEqual(self.repair(["bhūyo", "kṣara"], words=words), ["bhūyo", "kṣara"])
        ## a pair of words no reading helps is kept, also when dropping
        self.assertEqual(self.repair(["vi", "jaya"], elided={1}, words=words, unrepaired="drop"), ["vi", "jaya"])

    def test_three_members(self):
        words = {**self.WORDS, "su": 7, "abhirāma": 4, "agni": 7, "stoma": 5, "yājin": 4}
        self.assertEqual(self.repair(["su", "mano", "bhirāma"], elided={2}, words=words),
                         ["su", "mano", "abhirāma"])
        ## a right member read again is the left member of the next pair
        self.assertEqual(self.repair(["agni", "ṣṭoma", "yājin"], words=words), ["agni", "stoma", "yājin"])

    def test_fewer_long_vowels_first(self):
        self.assertEqual(self.repair(["da", "śāvasthā"]), ["daśa", "avasthā"])

    def test_a_pair_no_reading_makes_words_of(self):
        self.assertEqual(self.repair(["vi", "tṛṣṇa"]), ["vi", "tṛṣṇa"])
        self.assertEqual(self.repair(["vi", "tṛṣṇa"], unrepaired="drop"), [])
        with self.assertRaises(ValueError):
            MemberRepair(self.WORDS.__contains__, self.WORDS.get, unrepaired="guess")

    def test_a_pair_of_words_is_left_alone(self):
        self.assertEqual(self.repair(["kā", "kṛta"]), ["kā", "kṛta"])


class CompoundMembersTests(unittest.TestCase):
    """`CompoundMembers` over made-up dictionaries: the order of its steps."""

    class TableForms(CompoundForms):
        """`CompoundForms` over a dict instead of a database."""

        def __init__(self, targets):
            self.targets = targets

        def target(self, form):
            return self.targets.get(form)

    WORDS = {"mano": 2, "manas": 6, "anuga": 5, "ālaya": 5, "vijñāna": 6, "yudhi": 1, "sthira": 5}

    def reading(self, components, in_monier_williams):
        return CompoundMembers(
            self.TableForms({"mano": "manas"}),
            components=lambda dictionary, headword: components.get(dictionary, {}).get(headword, []),
            is_word=self.WORDS.__contains__, breadth=lambda word: self.WORDS.get(word, 0),
            in_monier_williams=in_monier_williams.__contains__,
            piece_lemmas=lambda piece: {"manas"} if piece == "manaḥ" else set(),
        )

    def test_the_repair_reads_the_avagraha_and_runs_before_the_citation_form(self):
        members = self.reading({"mw": {"manonuga": [("mano—'nuga", 1)]}}, {"manonuga"})
        self.assertEqual(members.members("manonuga"), ["manas", "anuga"])

    def test_the_fallback_answers_only_for_a_headword_monier_williams_lacks(self):
        components = {"bhs": {"ālayavijñāna": [("ālaya-vijñāna", 1)]}}
        self.assertEqual(self.reading(components, set()).members("ālayavijñāna"), ["ālaya", "vijñāna"])
        self.assertEqual(self.reading(components, {"ālayavijñāna"}).members("ālayavijñāna"), [])

    def test_the_fallback_members_are_repaired_too(self):
        components = {"gra": {"yudhiṣṭhira": [("yudhi-ṣṭhira", 1)]}}
        self.assertEqual(self.reading(components, set()).members("yudhiṣṭhira"), ["yudhi", "sthira"])


class VoteTests(unittest.TestCase):
    """`prevailing_members` takes the reader of the dictionary whose analyses it counts."""

    def test_monier_williams_by_default(self):
        self.assertEqual(ComponentMarks.prevailing_members([("sva—rūpa", 14), ("sva-rūpa", 1)]), ["sva", "rūpa"])

    def test_another_reader(self):
        reading = HyphenMarks()
        analyses = [("ālaya-vijñāna", 1)]
        self.assertEqual(ComponentMarks.prevailing_members(analyses), [])
        self.assertEqual(
            ComponentMarks.prevailing_members(analyses, reader=lambda c: reading.members(c, "ālayavijñāna")),
            ["ālaya", "vijñāna"],
        )

    def test_votes_are_counted_per_reading(self):
        votes = ComponentMarks.votes([("su—kha/", 1), ("sukha/", 9), ("sukha", 1)])
        self.assertEqual(votes[()], 10)
        self.assertEqual(votes[("su", "kha")], 1)


class ElisionTests(unittest.TestCase):
    """`ComponentMarks.elisions`: the members an avagraha marks as having lost an `a`."""

    def test_an_avagraha_after_the_mark(self):
        self.assertEqual(ComponentMarks.elisions([("mano—'nuga", 1), ("mano—nuga/", 2)], ["mano", "nuga"]), {1})
        self.assertEqual(ComponentMarks.elisions([("adho—'ṃśuka", 1), ("x—y", 1)], ["adho", "ṃśuka"]), {1})
        self.assertEqual(ComponentMarks.elisions([("ā—vi—'ṣkṛta", 1)], ["ā", "vi", "ṣkṛta"]), {2})

    def test_none_without_one(self):
        self.assertEqual(ComponentMarks.elisions([("namo—vṛdha/", 1)], ["namo", "vṛdha"]), frozenset())
        self.assertEqual(ComponentMarks.elisions([("mano—'nuga", 1)], ["manas", "anuga"]), frozenset())
        self.assertEqual(ComponentMarks.elisions([], []), frozenset())


class CompoundFormsExpansionTests(unittest.TestCase):
    def test_a_degree_sign_stands_for_the_start_of_the_stub(self):
        self.assertEqual(CompoundForms.expand("nabhaḥ", "°bhas"), "nabhas")
        self.assertEqual(CompoundForms.expand("catuś", "°tu/r"), "catur")
        self.assertEqual(CompoundForms.expand("nagnī", "°na"), "nagna")
        self.assertEqual(CompoundForms.expand("taj", "ta/d"), "tad")
        self.assertIsNone(CompoundForms.expand("x", "°"))


@unittest.skipUnless(has_database(), "lexicon database is not installed")
class CompoundFormsTests(unittest.TestCase):
    """Monier-Williams' "in compound for X" entries, read per homograph."""

    @classmethod
    def setUpClass(cls):
        from process_sanskrit.utils.databaseSetup import session_scope

        with session_scope() as session:
            cls.forms = CompoundForms(session)
            cls.targets = {stub: cls.forms.target(stub) for stub in
                           ("taj", "mano", "ṛg", "mahā", "maha", "para", "pura", "dharma", "sena", "ātma")}

    def test_a_stub_is_followed(self):
        self.assertEqual(self.targets["taj"], "tad")
        self.assertEqual(self.targets["mano"], "manas")
        self.assertEqual(self.targets["ṛg"], "ṛc")
        self.assertEqual(self.targets["ātma"], "ātman")

    def test_a_word_of_its_own_is_not(self):
        ## `para` has 31 entries, one "for °ras"; `mahā` is also "a cow"; `maha`
        ## is "great" and "a feast", and "in compound for mahā" before ṛ
        for homograph in ("mahā", "maha", "para", "pura", "dharma", "sena"):
            self.assertIsNone(self.targets[homograph], homograph)


@unittest.skipUnless(has_database(), "lexicon database is not installed")
class MembersTests(unittest.TestCase):
    """`LemmaLexicon.members`, on headwords checked by hand."""

    @classmethod
    def setUpClass(cls):
        from process_sanskrit.functions.lemmaAnalysis import LemmaLexicon
        from process_sanskrit.utils.databaseSetup import session_scope

        cls.context = session_scope()
        cls.lexicon = LemmaLexicon(cls.context.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def assertMembers(self, expected):
        for headword, members in expected.items():
            self.assertEqual(self.lexicon.members(headword), members, headword)

    def test_what_must_not_change(self):
        self.assertMembers({
            "sukha": [], "bhāvana": [], "ayoga": [], "vikalpa": [], "pratiprasava": [],
            "yogānuśāsana": [], "athātas": [], "svarūpa": ["sva", "rūpa"],
            "cittavṛtti": ["citta", "vṛtti"], "sukṛta": ["su", "kṛta"], "tadartha": ["tad", "artha"],
        })

    def test_edgerton_and_grassmann_answer_for_headwords_monier_williams_lacks(self):
        self.assertMembers({
            "ālayavijñāna": ["ālaya", "vijñāna"], "granthadhāraṇī": ["grantha", "dhāraṇī"],
            "āyuḥsaṃskāra": ["āyus", "saṃskāra"],
        })

    def test_a_run_of_preverbs_that_is_a_word_is_a_member_there(self):
        ## Edgerton `samādhi-saṃyuktaka`, `vyādhi-prajñāyate`; Grassmann `vipra-rā/jia`
        self.assertMembers({
            "samādhisaṃyuktaka": ["samādhi", "saṃyuktaka"], "vyādhiprajñāyate": ["vyādhi", "prajñāyate"],
            "viprarājia": ["vipra", "rājia"],
        })

    def test_but_not_for_headwords_monier_williams_has(self):
        ## Edgerton `sa-lila`, Grassmann `su-kha/`
        self.assertMembers({"salila": [], "sukha": []})

    def test_members_in_citation_form(self):
        self.assertMembers({
            "tajjātīya": ["tad", "jātīya"], "manobhava": ["manas", "bhava"], "ṛgveda": ["ṛc", "veda"],
            "jagannātha": ["jagat", "nātha"], "tapodhana": ["tapas", "dhana"], "ṣaḍguṇa": ["ṣaṣ", "guṇa"],
            "svastyayana": ["svasti", "ayana"],
        })

    def test_no_citation_form_for_a_word_of_its_own_or_one_lemma_mode_does_not_give(self):
        ## a split `ātma` is indexed as `ātma`; `rāja` is also a word ("king")
        self.assertMembers({
            "mahāratha": ["mahā", "ratha"], "dharmakṣetra": ["dharma", "kṣetra"], "paraloka": ["para", "loka"],
            "ātmaja": ["ātma", "ja"], "rājaputra": ["rāja", "putra"],
        })

    def test_a_merged_vowel_is_read_again_at_the_mark(self):
        ## `maha` is Monier-Williams' own "in compound for mahā before ṛ", so it stays
        self.assertMembers({
            "maharṣi": ["maha", "ṛṣi"], "surarṣi": ["sura", "ṛṣi"], "bharatarṣabha": ["bharata", "ṛṣabha"],
            "nānārtha": ["nānā", "artha"], "yathārha": ["yathā", "arha"], "mahārṇava": ["mahā", "arṇava"],
            "manonuga": ["manas", "anuga"],
        })

    def test_an_avagraha_marks_an_elided_a_also_between_two_words(self):
        ## `bhūyo—'kṣara`: `kṣara` is a word, but the mark says `akṣara`
        self.assertMembers({
            "bhūyokṣara": ["bhūyas", "akṣara"], "gośva": ["go", "aśva"], "ayogra": ["ayas", "agra"],
            "ayośana": ["ayas", "aśana"], "urubja": ["uru", "abja"],
            "sumanobhirāma": ["su", "manas", "abhirāma"],
        })

    def test_a_mark_that_stands_early_is_moved(self):
        self.assertMembers({"kāmārtha": ["kāma", "artha"], "daśāvasthā": ["daśa", "avasthā"]})

    def test_consonants_at_the_junction(self):
        self.assertMembers({
            "triṣṭubh": ["tri", "stubh"], "agniṣṭoma": ["agni", "stoma"], "gāyatracchandas": ["gāyatra", "chandas"],
        })

    def test_a_pair_no_reading_makes_words_of_keeps_the_dictionary_cut(self):
        ## the cut the dictionary marks, where the side that is no headword is
        ## an unlisted form (`tṛṣṇa`, `vatī`) or a longer compound (`cittajñāna`)
        self.assertMembers({
            "vitṛṣṇa": ["vi", "tṛṣṇa"], "guṇavatī": ["guṇa", "vatī"], "samaprabha": ["sama", "prabha"],
            "aśvamedhaka": ["aśva", "medhaka"], "paracittajñāna": ["para", "cittajñāna"],
            "mṛdumadhyādhimātra": ["mṛdu", "madhyādhimātra"], "gandhakelikā": ["gandha", "kelikā"],
            "manojavitva": ["manas", "javitva"],
        })

    def test_a_merged_vowel_the_rules_cannot_see_stays(self):
        ## known-errors.md, section 2: `tman` is itself a word
        self.assertMembers({"mahātman": ["mahā", "tman"]})


@unittest.skipUnless(has_database(), "lexicon database is not installed")
class LemmaModeTests(unittest.TestCase):
    """The members reach `process(mode="lemmas")`."""

    def parts(self, text):
        from process_sanskrit import process

        return [part for item in process(text, mode="lemmas", cached=False) for part in item["parts"]]

    def test_members_of_a_whole_compound(self):
        self.assertEqual(self.parts("tajjātīyena"),
                         [{"form": "tajjātīyena", "lemmas": ["tajjātīya"], "members": ["tad", "jātīya"]}])
        self.assertEqual(self.parts("ālayavijñāna"),
                         [{"form": "ālayavijñāna", "lemmas": ["ālayavijñāna"], "members": ["ālaya", "vijñāna"]}])

    def test_an_inflected_compound_keeps_the_dictionary_cut(self):
        ## Yoga Sūtra 1.15: `vitṛṣṇa` is `vi—tṛṣṇa`, never `vitṛṣ` + `na`
        (part,) = self.parts("vitṛṣṇasya")
        self.assertEqual((part["lemmas"], part["members"]), (["vitṛṣṇa"], ["vi", "tṛṣṇa"]))


@unittest.skipUnless(has_database(), "lexicon database is not installed")
class FootprintTests(unittest.TestCase):
    """Members read only the database: no sandhi rules of the Python splitter."""

    def test_the_python_splitter_is_not_loaded(self):
        code = (
            "import sys\n"
            "from process_sanskrit.functions.lemmaAnalysis import LemmaLexicon\n"
            "from process_sanskrit.utils.databaseSetup import session_scope\n"
            "with session_scope() as session:\n"
            "    lexicon = LemmaLexicon(session)\n"
            "    for headword in ('maharṣi', 'kāmārtha', 'manobhava', 'tajjātīya'):\n"
            "        assert lexicon.members(headword), headword\n"
            "print('process_sanskrit.splitter.sandhi' in sys.modules)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "False")


if __name__ == "__main__":
    unittest.main()
