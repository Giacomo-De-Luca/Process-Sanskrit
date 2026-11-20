"""`process(text, mode="lemmas")`: the analysis as a search-index query.

One item per whitespace token of the input, in order.  Each item lists the
*parts* of the word, which are all required; each part lists its alternative
lemmas:

    {"word": "upagacchati", "parts": [
        {"form": "upa",      "lemmas": ["upa"]},
        {"form": "gacchati", "lemmas": ["gam", "upagam"]}]}

`upagam` is the compound the preverb and the verb make, the Monier-Williams
headword: the lemma that ties the two parts together.

`mode="roots"` cannot say this: `['gacchati', 'gam']` (two readings of one word)
looks the same as `['bhāvaya', 'id']` (two pieces of one word), a sentence comes
back as one flat list, and an unknown word comes back as a guess.

See documentation/lemma-mode.md.  `cached=False` unless the cache is under test.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import text

from process_sanskrit import process
from process_sanskrit.functions import sandhiSplitter
from process_sanskrit.functions.SQLiteFind import paradigm_headword
from process_sanskrit.functions.lemmaAnalysis import LemmaAnalyzer, LemmaLexicon
from process_sanskrit.utils.analysisCache import reset_analysis_cache
from process_sanskrit.utils.databaseSetup import session_scope
from process_sanskrit.utils.lexicalResources import SANSKRIT_PREFIXES
from process_sanskrit.utils.resourcePaths import get_database_path
from tests.datasets.yogaSutra import ys


def lemmas(text, **options):
    return process(text, mode="lemmas", cached=False, **options)


def parts(word, **options):
    """The parts of a single word, as `(form, lemmas)` pairs."""
    result = lemmas(word, **options)
    assert len(result) == 1, result
    return [(part["form"], part["lemmas"]) for part in result[0]["parts"]]


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class SpecificationExampleTests(unittest.TestCase):
    def test_three_word_example(self):
        self.assertEqual(
            lemmas("yogaścittavṛttinirodhaḥ upagacchati xyzabc"),
            [
                {"word": "yogaścittavṛttinirodhaḥ", "parts": [
                    {"form": "yogaḥ", "lemmas": ["yoga", "yogas"]},
                    {"form": "cittavṛtti", "lemmas": ["cittavṛtti"],
                     "members": ["citta", "vṛtti"]},
                    {"form": "nirodhaḥ", "lemmas": ["nirodha"]},
                ]},
                {"word": "upagacchati", "parts": [
                    {"form": "upa", "lemmas": ["upa"]},
                    {"form": "gacchati", "lemmas": ["gam", "upagam"]},
                ]},
                {"word": "xyzabc", "parts": [{"form": "xyzabc", "lemmas": []}]},
            ],
        )


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class AlternativesAndPartsTests(unittest.TestCase):
    """Readings of one word share a part; pieces of one word are parts."""

    def test_headword_is_an_alternative_listed_last(self):
        self.assertEqual(parts("gacchati"), [("gacchati", ["gam", "gacchati"])])

    def test_every_reading_is_an_alternative_of_one_part(self):
        self.assertEqual(parts("te"), [("te", ["ta", "tā", "tad", "yuṣmad", "te"])])

    def test_nominative_headword_stays_beside_the_stem(self):
        self.assertEqual(parts("samādhiḥ"), [("samādhiḥ", ["samādhi", "samādhiḥ"])])

    def test_dictionary_only_word(self):
        self.assertEqual(parts("gatvā"), [("gatvā", ["gatvā"])])

    def test_derived_abstract_noun(self):
        self.assertEqual(parts("niṣyandatā"), [("niṣyandatā", ["niṣyandatā"])])

    def test_compound_members_are_parts(self):
        self.assertEqual(
            parts("deśabandhaś"), [("deśa", ["deśa"]), ("bandhaḥ", ["bandha"])]
        )

    def test_lexicalised_compound_is_one_part_with_members(self):
        result = lemmas("cittavṛtti")[0]["parts"]
        self.assertEqual(
            result,
            [{"form": "cittavṛtti", "lemmas": ["cittavṛtti"], "members": ["citta", "vṛtti"]}],
        )

    def test_a_word_with_a_rare_compound_homograph_has_no_members(self):
        """`bhāvana` is not `bhā` + `vana` ("forest of rays"), in any form."""
        self.assertEqual(
            lemmas("tadarthabhāvanam")[0]["parts"],
            [
                {"form": "tadartha", "lemmas": ["tadartha"], "members": ["tad", "artha"]},
                {"form": "bhāvanam", "lemmas": ["bhāvana"]},
            ],
        )
        self.assertEqual(lemmas("sukha")[0]["parts"], [{"form": "sukha", "lemmas": ["sukha"]}])


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class PreverbTests(unittest.TestCase):
    """A preverb is an ordinary part before the verb, and carries itself only.

    The part after it also carries the compound they make (`upagam`); the
    compounds are pinned in `JoinedPreverbKeyTests`.
    """

    def test_preverb_and_verb(self):
        self.assertEqual(parts("upagacchati"), [("upa", ["upa"]), ("gacchati", ["gam", "upagam"])])

    def test_preverb_in_a_sandhi_spelling(self):
        self.assertEqual(parts("pratyeti"), [("prati", ["prati"]), ("eti", ["i", "pratī"])])

    def test_stacked_preverbs(self):
        self.assertEqual(
            parts("vyavaharati"),
            [("vi", ["vi"]), ("ava", ["ava"]), ("harati", ["hṛ", "vyavahṛ"])],
        )

    def test_verb_is_not_rejoined_with_its_preverb(self):
        """`roots` mode gives `saṃgam`; a search for `gam` must still find this."""
        self.assertEqual(
            parts("saṃgacchati"), [("sam", ["sam"]), ("gacchati", ["gam", "saṃgam"])]
        )

    def test_preverb_cut_off_by_the_splitter(self):
        self.assertEqual(
            parts("nirgacchati"), [("nis", ["nis"]), ("gacchati", ["gam", "nirgam"])]
        )

    def test_preverb_token_left_on_its_own_carries_no_homograph(self):
        """Nothing joins `sam` to `arcayet`, so `sam` is looked up as a word (`sam`, `sa`)."""
        self.assertEqual(parts("samarcayet")[0], ("sam", ["sam"]))
        self.assertEqual(parts("avabhāsayet")[0], ("ava", ["ava"]))

    def test_whole_word_only_the_pali_dictionary_has_does_not_win(self):
        """Neither directly nor through the noun re-join."""
        self.assertEqual(parts("upagacchati"), [("upa", ["upa"]), ("gacchati", ["gam", "upagam"])])
        self.assertEqual(
            parts("anubuddha"), [("anu", ["anu"]), ("buddha", ["buddha", "anubuddha"])]
        )

    def test_whole_word_in_monier_williams_wins_over_the_split(self):
        self.assertEqual(parts("virati"), [("virati", ["virati"])])
        self.assertEqual(parts("avirati"), [("avirati", ["avirati"])])

    def test_attested_prefixed_noun_is_rejoined_under_its_outer_prefix(self):
        self.assertEqual(
            parts("pratisaṃvedanā"),
            [("prati", ["prati"]), ("samvedanā", ["saṃvedana", "pratisaṃvedana"])],
        )

    def test_unattested_prefixed_noun_stays_split(self):
        self.assertEqual(
            parts("anumitārthe"),
            [("anu", ["anu"]), ("mitārthe", ["mitārtha", "anumitārtha"])],
        )


def all_lemmas(item):
    return [lemma for part in item["parts"] for lemma in part["lemmas"]]


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class TypedHeadwordTests(unittest.TestCase):
    """A word typed as a Monier-Williams or Apte headword is one of its own lemmas.

    The cascade rewrites a final letter before any dictionary is asked, so a
    root typed on its own used to come back as something else (`vad` -> `vat`).
    """

    def test_a_root_typed_on_its_own_includes_itself_listed_last(self):
        for word in ("as", "dṛś", "vad", "yuj"):
            with self.subTest(word=word):
                found = parts(word, input_scheme="IAST")
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0][1][-1], word)

    def test_an_indeclinable_in_its_sandhi_spelling_includes_itself(self):
        """`punar` used to be `puna`, `pu`, `punaḥ` only."""
        for word in ("punar", "antar"):
            with self.subTest(word=word):
                self.assertIn(word, all_lemmas(lemmas(word, input_scheme="IAST")[0]))

    def test_a_word_analysed_as_several_parts_becomes_one(self):
        """`sev` was `se` + `u`."""
        self.assertEqual(parts("sev", input_scheme="IAST"), [("sev", ["sev"])])

    def test_words_that_already_include_themselves_are_unchanged(self):
        self.assertEqual(parts("gacchati"), [("gacchati", ["gam", "gacchati"])])
        self.assertEqual(parts("te"), [("te", ["ta", "tā", "tad", "yuṣmad", "te"])])
        self.assertEqual(parts("virati"), [("virati", ["virati"])])

    def test_every_root_of_the_verb_table_comes_back_among_its_own_lemmas(self):
        """The regression pin over the data: 161 of the 628 roots did not.

        The roots spelled with `jj` are left to the test below.
        """
        with session_scope() as session:
            roots = [root for root in verb_table_roots(session) if "jj" not in root]
            analyzer = LemmaAnalyzer(session=session, cached=False)
            missed = [
                root for root in roots
                if root not in all_lemmas(analyzer.analyze(root, input_scheme="IAST")[0])
            ]
        self.assertGreater(len(roots), 600)
        self.assertEqual(missed, [])

    # Text cleaning used to cut every `jj` as `j j`, so these arrived as two
    # words, neither of them the word typed.  A word that resolves whole now
    # keeps its `jj` (documentation/double-j.md).
    def test_a_root_spelled_with_jj_comes_back_as_itself(self):
        for root in ("lajj", "majj", "bhrajj", "ujjh"):
            with self.subTest(root=root):
                self.assertEqual(parts(root, input_scheme="IAST")[0][0], root)


def verb_table_roots(session):
    """Every root the verb table files a paradigm under."""
    rows = session.execute(text("SELECT stem, refs FROM vlgtab1"))
    return sorted({paradigm_headword(stem, refs) for stem, refs in rows})


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class JoinedPreverbKeyTests(unittest.TestCase):
    """The part after a run of preverbs also carries the compound they make with its lemma.

    A bag-of-words index loses the fact that `prati` and `i` are one word; the
    compound `pratī` keeps it.  It is the Monier-Williams headword whose parse
    it records, else the preverbs and the lemma joined by `PreverbSandhi`, and
    never contains `+`, which a search tokenizer cuts at.
    """

    def test_preverb_cut_off_by_the_splitter_and_left_on_its_own(self):
        self.assertEqual(parts("samarcayet"), [("sam", ["sam"]), ("arc", ["arc", "samarc"])])

    def test_every_lemma_of_the_stem_gets_a_key(self):
        self.assertEqual(
            parts("anubhavati"),
            [("anu", ["anu"]), ("bhavati", [
                "bhavatī", "bhavat", "bhū", "anubhavatī", "anubhavat", "anubhū",
            ])],
        )

    def test_a_standalone_preverb_is_a_word_and_joins_nothing(self):
        self.assertEqual(parts("prati"), [("prati", ["prati"])])
        self.assertEqual(
            [all_lemmas(item) for item in lemmas("prati eti")], [["prati"], ["i", "eti"]]
        )

    def test_a_whole_word_has_nothing_to_join(self):
        self.assertEqual(parts("virati"), [("virati", ["virati"])])

    def test_two_preverbs_cut_off_by_the_splitter_are_one_run(self):
        """The inflector reads the two preverbs together; the inner one is no stem."""
        self.assertEqual(
            parts("tadvyavaharati"),
            [("tat", ["tad"]), ("vi", ["vi"]), ("ava", ["ava"]), ("harati", ["hṛ", "vyavahṛ"])],
        )
        self.assertEqual(
            parts("upasamarcayet"),
            [("upa", ["upa"]), ("sam", ["sam"]), ("arc", ["arc", "upasamarc"])],
        )
        self.assertEqual(parts("cāpyupasaṃharati")[-1], ("harati", ["hṛ", "upasaṃhṛ"]))

    def test_the_particle_api_is_tied_to_nothing(self):
        """In every held-out token where it was keyed, `api` was a word of its own."""
        for text in ("nāpyasti", "apyevam", "punarapyuktam"):
            with self.subTest(text=text):
                found = lemmas(text, input_scheme="IAST")[0]
                self.assertIn("api", all_lemmas(found))
                self.assertFalse(
                    [lemma for lemma in all_lemmas(found) if lemma.startswith(("apy", "apī", "ape"))]
                )

    def test_no_key_crosses_a_segment_of_a_pre_split_compound(self):
        """Each segment is a word: `upa-gacchati` is `upa` and `gacchati`."""
        self.assertEqual(parts("upa-gacchati"), [("upa", ["upa"]), ("gacchati", ["gam", "gacchati"])])

    def test_every_part_after_preverbs_in_running_text_carries_their_compound(self):
        """Over the Yoga Sūtra: the compound of the run and the part's first lemma."""
        with session_scope() as session:
            analyzer = LemmaAnalyzer(session=session, cached=False)
            analyzer.analyze(" ".join(line.strip() for line in ys), input_scheme="IAST")
            keyed = 0
            for found in analyzer._parts_of.values():
                run = []
                for part in found:
                    self.assertFalse([lemma for lemma in part.lemmas if "+" in lemma], part)
                    if part.preverb:
                        run.append(part.lemmas[0])
                        continue
                    if run and part.lemmas and "api" not in part.lemmas:
                        keyed += 1
                        self.assertNotIn(part.form, SANSKRIT_PREFIXES, found)
                        for key in analyzer.builder.preverb_keys(run, part.lemmas[0]):
                            self.assertIn(key, part.lemmas, found)
                    run = []
        self.assertGreater(keyed, 0)


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class SuppletiveRootTests(unittest.TestCase):
    """The verb table files the present of `dṛś` under `paś`; both are lemmas."""

    def test_present_of_drs(self):
        self.assertEqual(parts("paśyati"), [("paśyati", ["paś", "dṛś"])])

    def test_after_a_preverb_both_roots_get_a_key(self):
        self.assertEqual(
            parts("anupaśyati"),
            [("anu", ["anu"]), ("paśyati", ["paś", "dṛś", "anupaś", "anudṛś"])],
        )

    def test_an_unrelated_verb_is_unchanged(self):
        self.assertEqual(parts("gacchati"), [("gacchati", ["gam", "gacchati"])])


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class CommonFormsReachTheirRootTests(unittest.TestCase):
    """Sixteen common forms of four roots: those that reach the root must keep reaching it.

    A floor, not a ceiling: a form that starts reaching its root is a gain.  The
    forms that do not reach it yet are participles, absolutives, infinitives and
    causatives, which the lexicon does not link to a root (see
    documentation/lemma-mode-handoff.md, "Later").
    """

    REACHED = {
        "gam": "gacchati gacchanti agacchat gacchet gacchatu gamiṣyati jagāma agamat gamyate",
        "kṛ": "karoti kurvanti akarot kuryāt karotu kariṣyati cakāra akārṣīt kriyate kṛta kurvan",
        "bhū": "bhavati bhavanti abhavat bhavet bhavatu bhaviṣyati babhūva abhūt bhūyate bhūta",
        "dṛś": "paśyati paśyanti apaśyat paśyet paśyatu drakṣyati dadarśa adrākṣīt dṛśyate",
    }

    def test_forms_that_reach_their_root_keep_reaching_it(self):
        for root, forms in self.REACHED.items():
            items = lemmas(forms, input_scheme="IAST")
            missed = [item["word"] for item in items if root not in all_lemmas(item)]
            with self.subTest(root=root):
                self.assertEqual(missed, [])


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class UnknownWordTests(unittest.TestCase):
    """Pieces are kept unless the scanner skipped two or more characters."""

    def test_made_up_word_has_no_lemmas(self):
        for word in ("xyzabc", "blorptu", "zzkrm"):
            with self.subTest(word=word):
                self.assertEqual(parts(word, input_scheme="IAST"), [(word, [])])

    def test_unknown_verb_keeps_the_stem_the_scanner_found(self):
        """A junk tail costs an index little; losing `śrut` or `bhāvaya` costs more."""
        self.assertEqual([lemma for _, lemma in parts("śrutvā")][0], ["śrut"])
        self.assertEqual([lemma for _, lemma in parts("bhāvayet")][0], ["bhāvaya"])

    def test_no_word_ever_comes_back_without_a_part(self):
        for word in ("xyzabc", "ḍḍḍḍ", "zzkrm"):
            with self.subTest(word=word):
                result = lemmas(word, input_scheme="IAST")
                self.assertEqual(len(result), 1)
                self.assertTrue(result[0]["parts"])


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class WordBoundaryTests(unittest.TestCase):
    """One item per whitespace token, and nothing is truncated."""

    def test_one_item_per_whitespace_token(self):
        text = "tatra | tatra eti ."
        result = lemmas(text)
        self.assertEqual([item["word"] for item in result], text.split())

    def test_punctuation_token_is_an_item_without_parts(self):
        result = lemmas("tatra | eti")
        self.assertEqual(result[1], {"word": "|", "parts": []})

    def test_repeated_word_is_reported_each_time(self):
        """`roots` mode folds `tatra tatra` into one `tatra`."""
        result = lemmas("tatra tatra")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], result[1])
        self.assertIsNot(result[0], result[1])
        self.assertIsNot(result[0]["parts"], result[1]["parts"])

    def test_long_passage_is_not_truncated(self):
        text = " ".join(line.strip() for line in ys[:15])
        self.assertGreater(len(text), 400)
        result = lemmas(text)
        self.assertEqual([item["word"] for item in result], text.split())
        self.assertTrue(result[-1]["parts"][-1]["lemmas"])

    def test_long_single_word_is_analysed_whole(self):
        """Every character of a 120-character word is in some part."""
        word = "dharma" * 19 + "kṣetre"
        self.assertGreater(len(word), 100)
        result = lemmas(word)
        self.assertEqual(len(result), 1)
        found = result[0]["parts"]
        self.assertEqual("".join(part["form"] for part in found), word)
        self.assertEqual(found[-1]["lemmas"], ["dharmakṣetra"])

    def test_token_that_cleans_to_two_words_keeps_both(self):
        self.assertEqual(
            [form for form, _ in parts("te'pi")], ["te", "api"]
        )

    def test_avagraha_on_its_own_token(self):
        result = lemmas("so 'nupalambhena")
        self.assertEqual(len(result), 2)
        self.assertIn("anupalambha", result[1]["parts"][-1]["lemmas"])

    def test_spaced_avagraha_undoes_the_o_of_the_word_before(self):
        """`ko 'pi` is `kaḥ api`.  Cleaned token by token, `ko` came back as `ku`."""
        spaced = lemmas("ko 'pi")
        self.assertEqual([item["word"] for item in spaced], ["ko", "'pi"])
        self.assertEqual(spaced[0]["parts"][0]["form"], "kaḥ")
        self.assertIn("ka", spaced[0]["parts"][0]["lemmas"])
        self.assertEqual(spaced[0]["parts"], lemmas("ko'pi")[0]["parts"][:1])
        self.assertIn("manas", lemmas("mano 'pi")[0]["parts"][0]["lemmas"])

    def test_quotation_marks_are_not_read_as_avagrahas(self):
        """Opening or closing, mid-text or around the whole of it.

        `ucyate` is a form of `uc` and the passive of `vac`; the verb table files
        it under both (tests/test_shared_verb_forms.py).
        """
        for text in ("iti 'yoga ucyate'", "'yoga ucyate'", "‘yoga ucyate’"):
            with self.subTest(text=text):
                result = lemmas(text, input_scheme="IAST")
                self.assertEqual(result[-2]["parts"], [{"form": "yoga", "lemmas": ["yoga"]}])
                self.assertEqual(
                    result[-1]["parts"], [{"form": "ucyate", "lemmas": ["uc", "vac"]}]
                )

    def test_separator_and_pattern_characters_alone_are_not_words(self):
        for mark in ("-", "+", "*", "**", "%", "_"):
            with self.subTest(mark=mark):
                self.assertEqual(lemmas(f"yoga {mark} dharma")[1], {"word": mark, "parts": []})
                self.assertEqual(lemmas(mark, input_scheme="IAST"), [{"word": mark, "parts": []}])

    def test_pattern_characters_in_running_text_are_not_patterns(self):
        """A stray `%` would otherwise bring back every headword there is."""
        result = lemmas("tatra 50 % eti dharm_ kṣetre")
        self.assertEqual(len(result), 6)
        self.assertEqual(result[4]["parts"], [{"form": "dharm", "lemmas": []}])
        self.assertLess(sum(len(part["lemmas"]) for item in result for part in item["parts"]), 20)

    def test_word_the_splitter_cannot_handle_still_comes_back(self):
        """A token of a thousand characters makes the splitter give up; the call must not."""
        text = "tatra " + "dharmakṣetre" * 90 + " eti"
        result = lemmas(text)
        self.assertEqual(len(result), 3)
        self.assertTrue(result[1]["parts"])
        self.assertEqual(result[2]["parts"][0]["lemmas"][0], "i")

    def test_token_the_splitter_rejects_is_an_unknown_part(self):
        result = lemmas("yoga ऽ dharma", input_scheme="IAST")
        self.assertEqual([item["word"] for item in result], ["yoga", "ऽ", "dharma"])
        self.assertEqual(result[1]["parts"], [{"form": "ऽ", "lemmas": []}])

    def test_empty_input(self):
        self.assertEqual(lemmas(""), [])
        self.assertEqual(lemmas("   "), [])


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class SpecialInputTests(unittest.TestCase):
    def test_pre_split_compound_is_one_word(self):
        self.assertEqual(
            parts("dharma-kṣetra"),
            [("dharma", ["dharma", "dharman"]), ("kṣetra", ["kṣetra"])],
        )

    def test_trailing_star_finds_the_headword(self):
        self.assertEqual(parts("gatvā*"), [("gatvā", ["gatvā"])])

    def test_pattern_lists_the_matching_headwords(self):
        form, found = parts("dharm_")[0]
        self.assertEqual(form, "dharm_")
        self.assertIn("dharma", found)

    def test_devanagari_input(self):
        self.assertEqual(
            lemmas("गच्छति"),
            [{"word": "gacchati", "parts": [{"form": "gacchati", "lemmas": ["gam", "gacchati"]}]}],
        )

    def test_dictionary_names_do_not_change_the_lemmas(self):
        self.assertEqual(
            process("gacchati", "ap90", mode="lemmas", cached=False), lemmas("gacchati")
        )

    def test_result_is_plain_json(self):
        result = lemmas("yogaścittavṛttinirodhaḥ upagacchati")
        self.assertEqual(json.loads(json.dumps(result, ensure_ascii=False)), result)


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class OtherModesAreUntouchedTests(unittest.TestCase):
    def test_roots_after_lemmas(self):
        for word in ("saṃgacchati", "upagacchati", "virati", "gacchati", "deśabandhaś"):
            with self.subTest(word=word):
                before = process(word, mode="roots", cached=False)
                lemmas(word)
                self.assertEqual(process(word, mode="roots", cached=False), before)


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class SplitSourceTests(unittest.TestCase):
    """The split is one seam, so the two sources can be compared."""

    def analyze(self, text, split_source):
        with session_scope() as session:
            return LemmaAnalyzer(session=session, cached=False, split_source=split_source).analyze(
                text, input_scheme="IAST"
            )

    def test_statistical_split_lets_the_junk_rule_see_the_scanner(self):
        self.assertEqual(self.analyze("xyzabc", "statistical")[0]["parts"][0]["lemmas"], [])

    def test_hybrid_split_is_really_a_different_split(self):
        """The hybrid lets the scanner replace the split unseen: its junk gets through."""
        self.assertEqual(self.analyze("xyzabc", "hybrid")[0]["parts"][0]["lemmas"], ["ab"])
        self.assertNotEqual(
            self.analyze("ekātmatevāsmitā", "hybrid"), self.analyze("ekātmatevāsmitā", "statistical")
        )

    def test_unknown_split_source_is_refused(self):
        with self.assertRaises(ValueError):
            self.analyze("yoga", "guess")


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.cache_path = Path(self.temporary_directory.name) / "lemmas.sqlite3"
        reset_analysis_cache()
        self.addCleanup(reset_analysis_cache)

    def test_cached_split_is_reused_and_gives_the_same_result(self):
        text = "yogaścittavṛttinirodhaḥ deśabandhaś"
        expected = lemmas(text)
        with patch.dict(os.environ, {"PROCESS_SANSKRIT_CACHE_PATH": str(self.cache_path)}), patch.object(
            sandhiSplitter, "analyze_sandhi", wraps=sandhiSplitter.analyze_sandhi
        ) as split:
            cold = process(text, mode="lemmas", cached=True)
            self.assertTrue(split.called)
            split.side_effect = AssertionError("cache hit recomputed the split")
            warm = process(text, mode="lemmas", cached=True)
        self.assertEqual(cold, expected)
        self.assertEqual(warm, expected)


@unittest.skipUnless(get_database_path().exists(), "lexicon database is not installed")
class LexiconTests(unittest.TestCase):
    """`LemmaLexicon` against the installed dictionaries."""

    def setUp(self):
        context = session_scope()
        self.lexicon = LemmaLexicon(context.__enter__())
        self.addCleanup(context.__exit__, None, None, None)

    def test_members_of_a_compound_headword(self):
        self.assertEqual(self.lexicon.members("cittavṛtti"), ["citta", "vṛtti"])

    def test_no_members_when_the_prevailing_reading_is_no_compound(self):
        """`ayoga` is mostly `a-yoga` (9 entries), once `ayo—ga`: no members."""
        self.assertEqual(self.lexicon.members("ayoga"), [])

    def test_no_members_for_a_simple_headword(self):
        self.assertEqual(self.lexicon.members("yoga"), [])

    def test_a_rare_compound_homograph_gives_no_members(self):
        """One entry each (`su—kha`, `bhā—vana` "forest of rays") against many plain ones."""
        for headword in ("sukha", "bhāvana", "tatas", "sarvathā", "duḥkha"):
            with self.subTest(headword=headword):
                self.assertEqual(self.lexicon.members(headword), [])

    def test_a_compound_keeps_its_members_beside_a_stray_plain_entry(self):
        self.assertEqual(self.lexicon.members("sukṛta"), ["su", "kṛta"])

    def test_sanskrit_headword_is_stricter_than_headword(self):
        self.assertTrue(self.lexicon.has_sanskrit_headword("virati"))
        self.assertFalse(self.lexicon.has_sanskrit_headword("upagacchati"))
        self.assertEqual(self.lexicon.headword("upagacchati"), "upagacchati")

    def test_headword_folds_the_sam_spelling(self):
        self.assertEqual(self.lexicon.headword("samyoga"), "saṃyoga")
        self.assertIsNone(self.lexicon.headword("ḍḍḍḍ"))

    def test_compound_verbs_are_read_from_the_parses_monier_williams_records(self):
        self.assertEqual(self.lexicon.compound_verbs(["prati"], "i"), ["pratī"])
        self.assertEqual(self.lexicon.compound_verbs(["vi", "ava"], "hṛ"), ["vyavahṛ"])
        self.assertEqual(self.lexicon.compound_verbs(["sam"], "gam"), ["saṃgam"])

    def test_compound_verbs_read_our_preverb_spellings_as_monier_williams_does(self):
        """We name `ud` and `nir-` `ut` and `nis`; Monier-Williams `ud` and `nir`."""
        self.assertEqual(self.lexicon.compound_verbs(["ut"], "gam"), ["udgam"])
        self.assertEqual(self.lexicon.compound_verbs(["nis"], "gam"), ["nirgam"])
        self.assertEqual(self.lexicon.compound_verbs(["saṃ"], "gam"), ["saṃgam"])

    def test_no_compound_verb_where_monier_williams_records_none(self):
        self.assertEqual(self.lexicon.compound_verbs(["upa", "sam"], "arc"), [])
        self.assertEqual(self.lexicon.compound_verbs(["upa"], "iti"), [])

    def test_joined_headword_must_be_a_sanskrit_headword(self):
        self.assertEqual(self.lexicon.joined_headword("sam", "vedana"), "saṃvedana")
        self.assertIsNone(self.lexicon.joined_headword("anu", "buddha"))
        self.assertIsNone(self.lexicon.joined_headword("sam", "upekṣa"))

    def test_pattern_matches(self):
        self.assertIn("dharma", self.lexicon.headwords_matching("dharm_"))
        self.assertEqual(self.lexicon.headwords_matching("ḍḍḍḍ_"), [])


if __name__ == "__main__":
    unittest.main()
