"""Building lemma-mode parts from analyses; needs no lexicon.

`mode="lemmas"` returns, per input word, a list of *parts* that are all required,
each carrying a list of alternative lemmas.  Everything here runs on hand-built
entries and a fake lexicon, so the rules are pinned independently of the data:

* a stripped preverb is a part of its own, and carries the preverb only -- never
  the homographs the forms table returns beside it (`sam` -> `sa`);
* a verb is never re-joined with its preverb; an attested prefixed noun is;
* a whole word Monier-Williams or Apte has wins over its preverb split;
* the whole-word headword is one more lemma of a one-part word, listed last;
* the fallback scanner's pieces are kept unless it skipped two or more
  characters -- a junk tail costs a search index little, a lost stem a lot;
* a word *typed* as a Monier-Williams or Apte headword is one of its own lemmas
  (`vad`, which the cascade reads as `vat`);
* the part after a run of preverbs also carries the compound they make with
  each of its lemmas: the Monier-Williams headword whose parse it records
  (`prati` + `i` -> `pratī`), else the preverbs and the lemma joined by
  `PreverbSandhi` (`upa` + `sam` + `arc` -> `upasamarc`);
* a verb the table files under another root brings that root along (`paś`,
  `dṛś`).

See documentation/lemma-mode.md.
"""

import unittest

from process_sanskrit.functions.inflect import TokenAnalysis
from process_sanskrit.functions.lemmaAnalysis import LemmaBuilder, LemmaPart, ScannerPolicy
from process_sanskrit.functions.wordCascade import WordResolution
from process_sanskrit.utils.analysisTrace import PrefixTrace, ScanTrace
from process_sanskrit.utils.componentMarks import ComponentMarks
from process_sanskrit.utils.paradigm import is_one_letter_root


VERBAL = ["a"] * 9
NOMINAL = ["a"] * 24


def verb(lemma, surface):
    return [lemma, "verb, present tense", [("Third", "Sg")], VERBAL, surface]


def noun(lemma, surface):
    return [lemma, "masculine noun/adjective ending in -a", [("Nom", "Sg")], NOMINAL, surface]


def particle(lemma):
    return [lemma, "indeclinable (avyaya)", [("Nom", "Sg")], [lemma], lemma]


def stamped(entry, whole):
    """An entry as `root_any_word` leaves a remainder's: its surface is the whole word."""
    return entry[:4] + [whole]


def stripped(prefixes, remainder):
    """A trace as `root_any_word` leaves it: innermost prefix recorded first."""
    trace = PrefixTrace()
    for index, prefix in enumerate(reversed(prefixes)):
        trace.record(prefix, remainder if index == 0 else "ignored")
    return trace


class FakeLexicon:
    """The builder's only window on the dictionaries, replaced here by tables."""

    def __init__(self, *, sanskrit=(), headwords=(), members=None, joins=None, compound_verbs=None):
        self.compound_verb_table = compound_verbs or {}
        self.sanskrit = set(sanskrit)
        self.headwords = set(headwords)
        self.member_table = members or {}
        self.joins = joins or {}
        self.join_requests = []

    def has_sanskrit_headword(self, word):
        return word in self.sanskrit

    def headword(self, word):
        return word if word in self.headwords else None

    def members(self, lemma):
        return list(self.member_table.get(lemma, []))

    def compound_verbs(self, preverbs, stem):
        return list(self.compound_verb_table.get((tuple(preverbs), stem), []))

    def joined_headword(self, prefix, stem):
        self.join_requests.append((prefix, stem))
        return self.joins.get((prefix, stem))


def shapes(parts):
    return [part.as_dict() for part in parts]


def pairs(parts):
    return [(part.form, part.lemmas) for part in parts]


def preverb(name):
    """A preverb part as the builder makes one."""
    return LemmaPart(name, [name], preverb=True)


class PrefixTraceTests(unittest.TestCase):
    def test_empty_trace_is_falsy(self):
        self.assertFalse(PrefixTrace())

    def test_prefixes_read_outermost_first(self):
        trace = PrefixTrace()
        trace.record("ā", "gacchati")
        trace.record("sam", "āgacchati")
        trace.record("abhi", "samāgacchati")
        self.assertEqual(trace.prefixes, ["abhi", "sam", "ā"])

    def test_remainder_is_the_innermost_one(self):
        trace = PrefixTrace()
        trace.record("ā", "gacchati")
        trace.record("sam", "āgacchati")
        self.assertEqual(trace.remainder, "gacchati")


class ScanTraceTests(unittest.TestCase):
    def test_skips_are_counted(self):
        scan = ScanTrace()
        scan.skip()
        scan.skip()
        self.assertEqual(scan.skipped, 2)

    def test_pieces_keep_their_order_and_surface(self):
        scan = ScanTrace()
        scan.piece("deśa", ["deśa"], PrefixTrace())
        scan.piece("bandhaḥ", [noun("bandha", "bandhaḥ")], PrefixTrace())
        self.assertEqual([piece.surface for piece in scan.pieces], ["deśa", "bandhaḥ"])

    def test_withholding_empties_a_piece_without_counting_a_skip(self):
        """`arcayet`: the tail `ayet` is a form of `i` and is withheld, not skipped."""
        scan = ScanTrace()
        scan.piece("arc", ["arc"], PrefixTrace())
        scan.piece("ayet", [verb("i", "ayet")], PrefixTrace())
        scan.withhold(is_one_letter_root)
        self.assertEqual([piece.surface for piece in scan.pieces], ["arc"])
        self.assertEqual(scan.skipped, 0)


class ScannerPolicyTests(unittest.TestCase):
    @staticmethod
    def scan(*surfaces, skipped=0):
        trace = ScanTrace()
        for surface in surfaces:
            trace.piece(surface, [surface], PrefixTrace())
        for _ in range(skipped):
            trace.skip()
        return trace

    def test_one_covering_piece_is_accepted(self):
        self.assertTrue(ScannerPolicy().accepts(self.scan("dṛṣṭvā")))

    def test_several_pieces_are_accepted_whatever_their_length(self):
        self.assertTrue(ScannerPolicy().accepts(self.scan("deśa", "bandhaḥ")))
        self.assertTrue(ScannerPolicy().accepts(self.scan("bhāvaya", "id")))
        self.assertTrue(ScannerPolicy().accepts(self.scan("śrut", "vā")))

    def test_one_skipped_character_is_tolerated(self):
        """An ending the scanner could not read: `śāmyat` still gives `śam`."""
        self.assertTrue(ScannerPolicy().accepts(self.scan("śāmya", skipped=1)))

    def test_two_skipped_characters_reject_the_token(self):
        self.assertFalse(ScannerPolicy().accepts(self.scan("ab", skipped=4)))
        self.assertFalse(ScannerPolicy().accepts(self.scan("ba", skipped=2)))

    def test_nothing_found_rejects_the_token(self):
        self.assertFalse(ScannerPolicy().accepts(self.scan()))

    def test_tolerance_is_configurable(self):
        self.assertFalse(ScannerPolicy(max_skipped=0).accepts(self.scan("śāmya", skipped=1)))
        self.assertTrue(ScannerPolicy(max_skipped=2).accepts(self.scan("ba", skipped=2)))


class ComponentMarkMemberTests(unittest.TestCase):
    def test_members_are_cut_on_the_compound_mark_only(self):
        self.assertEqual(ComponentMarks.members("citta—vṛtti"), ["citta", "vṛtti"])

    def test_a_prefix_boundary_makes_no_members(self):
        self.assertEqual(ComponentMarks.members("ni-rodha"), [])

    def test_accent_marks_are_dropped(self):
        self.assertEqual(ComponentMarks.members("deva/—datta"), ["deva", "datta"])

    def test_a_prefix_boundary_inside_a_member_is_not_a_cut(self):
        self.assertEqual(ComponentMarks.members("a-dharma—kṣetra"), ["adharma", "kṣetra"])

    def test_missing_components_make_no_members(self):
        self.assertEqual(ComponentMarks.members(None), [])
        self.assertEqual(ComponentMarks.members(""), [])


class ComponentMarkLetterTests(unittest.TestCase):
    def test_letters_drop_accents_marks_and_spaces(self):
        self.assertEqual(ComponentMarks.letters("i/ndra-jūta"), "indrajūta")
        self.assertEqual(ComponentMarks.letters("agni—tapta/"), "agnitapta")
        self.assertEqual(ComponentMarks.letters("agni—tapta/, °ā"), "agnitaptaā")

    def test_letters_are_composed_first(self):
        ## a decomposed ū (u + combining macron) would lose its macron as a non-letter
        decomposed = "jūta"
        self.assertEqual(ComponentMarks.letters(decomposed), "jūta")


class PrevailingMemberTests(unittest.TestCase):
    """The reading most of a headword's entries give decides its members.

    Each analysis comes with the number of dictionary entries that give it.
    The counts below are Monier-Williams'.
    """

    def test_a_rare_compound_homograph_loses_to_the_plain_word(self):
        """`sukha` is one `su—kha` entry against ten plain ones."""
        analyses = [("sukha", 1), ("sukha/", 9), ("su—kha/", 1)]
        self.assertEqual(ComponentMarks.prevailing_members(analyses), [])

    def test_a_stray_plain_entry_does_not_hide_a_compound(self):
        """`sukṛta` is eleven `su—kṛta` entries, accented or not, against one plain."""
        analyses = [("sukṛta", 1), ("su—kṛta", 4), ("su—kṛta/", 7)]
        self.assertEqual(ComponentMarks.prevailing_members(analyses), ["su", "kṛta"])

    def test_a_prefix_reading_is_no_compound(self):
        """`vikalpa`: seventeen `vi-kalpa` entries outvote the bird compound `vi—kalpa`."""
        analyses = [("vi-kalpa", 17), ("vi—kalpa", 1)]
        self.assertEqual(ComponentMarks.prevailing_members(analyses), [])

    def test_prefix_and_plain_readings_vote_together(self):
        """`ayoga`: `a-yoga` and `ayoga` are both no compound, and outvote `ayo—ga`."""
        analyses = [("a-yoga", 9), ("ayoga", 1), ("ayo—ga", 1)]
        self.assertEqual(ComponentMarks.prevailing_members(analyses), [])

    def test_a_tie_gives_no_members(self):
        analyses = [("gaṅgāṣṭaka", 1), ("gaṅ—gāṣṭaka", 1)]
        self.assertEqual(ComponentMarks.prevailing_members(analyses), [])

    def test_a_compound_with_no_rival_keeps_its_members(self):
        self.assertEqual(
            ComponentMarks.prevailing_members([("citta—vṛtti", 3)]), ["citta", "vṛtti"]
        )

    def test_no_analyses_give_no_members(self):
        self.assertEqual(ComponentMarks.prevailing_members([]), [])


class LookupPartTests(unittest.TestCase):
    """`LemmaBuilder.from_lookup`: one lookup of one span, with its prefix trace."""

    def test_alternatives_share_one_part(self):
        parts = LemmaBuilder(FakeLexicon()).from_lookup(
            [noun("yoga", "yogaḥ"), noun("yogas", "yogaḥ")]
        )
        self.assertEqual(shapes(parts), [{"form": "yogaḥ", "lemmas": ["yoga", "yogas"]}])

    def test_alternatives_are_deduplicated_in_order(self):
        entries = [noun(lemma, "te") for lemma in ("ta", "ta", "tā", "tad", "tad", "yuṣmad")]
        parts = LemmaBuilder(FakeLexicon()).from_lookup(entries)
        self.assertEqual(parts[0].lemmas, ["ta", "tā", "tad", "yuṣmad"])

    def test_preverb_is_a_part_before_the_verb(self):
        entries = [particle("upa"), verb("gam", "upagacchati")]
        parts = LemmaBuilder(FakeLexicon()).from_lookup(entries, stripped(["upa"], "gacchati"))
        self.assertEqual(
            shapes(parts),
            [{"form": "upa", "lemmas": ["upa"]}, {"form": "gacchati", "lemmas": ["gam"]}],
        )

    def test_preverb_part_carries_no_homograph(self):
        """`sam` comes back from the forms table with the noun `sa` beside it."""
        entries = [particle("sam"), noun("sa", "sam"), verb("gam", "samgacchati")]
        parts = LemmaBuilder(FakeLexicon()).from_lookup(entries, stripped(["sam"], "gacchati"))
        self.assertEqual(parts[0].as_dict(), {"form": "sam", "lemmas": ["sam"]})

    def test_verb_is_never_rejoined_with_its_preverb(self):
        lexicon = FakeLexicon(joins={("sam", "gam"): "saṃgam"})
        entries = [particle("sam"), verb("gam", "samgacchati")]
        parts = LemmaBuilder(lexicon).from_lookup(entries, stripped(["sam"], "gacchati"))
        self.assertEqual([part.lemmas for part in parts], [["sam"], ["gam"]])
        self.assertEqual(lexicon.join_requests, [])

    def test_stacked_preverbs_are_one_part_each(self):
        entries = [particle("vi"), particle("ava"), verb("hṛ", "vyavaharati")]
        parts = LemmaBuilder(FakeLexicon()).from_lookup(
            entries, stripped(["vi", "ava"], "harati")
        )
        self.assertEqual(
            [(part.form, part.lemmas) for part in parts],
            [("vi", ["vi"]), ("ava", ["ava"]), ("harati", ["hṛ"])],
        )

    def test_whole_word_in_a_sanskrit_dictionary_wins_over_the_split(self):
        entries = [particle("vi"), noun("ratī", "virati")]
        parts = LemmaBuilder(FakeLexicon(sanskrit={"virati"})).from_lookup(
            entries, stripped(["vi"], "rati")
        )
        self.assertEqual(shapes(parts), [{"form": "virati", "lemmas": ["virati"]}])

    def test_whole_word_elsewhere_does_not_win(self):
        """A headword only the Pali dictionary has is not a Sanskrit lemma."""
        entries = [particle("upa"), verb("gam", "upagacchati")]
        parts = LemmaBuilder(FakeLexicon(sanskrit=set())).from_lookup(
            entries, stripped(["upa"], "gacchati")
        )
        self.assertEqual([part.lemmas for part in parts], [["upa"], ["gam"]])

    def test_attested_prefixed_noun_is_rejoined(self):
        lexicon = FakeLexicon(joins={("sam", "vedana"): "saṃvedana"})
        entries = [particle("prati"), particle("sam"), noun("vedana", "pratisaṃvedanā")]
        parts = LemmaBuilder(lexicon).from_lookup(entries, stripped(["prati", "sam"], "vedanā"))
        self.assertEqual(
            [(part.form, part.lemmas) for part in parts],
            [("prati", ["prati"]), ("samvedanā", ["saṃvedana"])],
        )

    def test_unattested_prefixed_noun_stays_split(self):
        entries = [particle("anu"), noun("mitārtha", "anumitārthe")]
        parts = LemmaBuilder(FakeLexicon()).from_lookup(entries, stripped(["anu"], "mitārthe"))
        self.assertEqual(
            [(part.form, part.lemmas) for part in parts],
            [("anu", ["anu"]), ("mitārthe", ["mitārtha"])],
        )

    def test_dictionary_only_piece_is_a_headword(self):
        lexicon = FakeLexicon(headwords={"cittavṛtti"}, members={"cittavṛtti": ["citta", "vṛtti"]})
        parts = LemmaBuilder(lexicon).from_lookup(["cittavṛtti"])
        self.assertEqual(
            shapes(parts),
            [{"form": "cittavṛtti", "lemmas": ["cittavṛtti"], "members": ["citta", "vṛtti"]}],
        )

    def test_unknown_piece_has_no_lemmas(self):
        parts = LemmaBuilder(FakeLexicon()).from_lookup(["ḍḍḍḍ"])
        self.assertEqual(shapes(parts), [{"form": "ḍḍḍḍ", "lemmas": []}])

    def test_members_key_is_absent_when_there_are_none(self):
        self.assertNotIn("members", LemmaPart("yogaḥ", ["yoga"]).as_dict())

    def test_members_are_given_only_to_a_part_with_one_reading(self):
        """With two lemmas nothing would say which of them the members belong to."""
        lexicon = FakeLexicon(members={"ayoga": ["ayo", "ga"]})
        one = LemmaBuilder(lexicon).from_lookup([noun("ayoga", "ayogaḥ")])
        two = LemmaBuilder(lexicon).from_lookup([noun("ayoga", "ayogaḥ"), noun("ayogas", "ayogaḥ")])
        self.assertEqual(one[0].members, ["ayo", "ga"])
        self.assertEqual(two[0].members, [])

    def test_entries_are_not_modified(self):
        entries = [particle("sam"), noun("sa", "sam"), verb("gam", "samgacchati")]
        before = [list(entry) for entry in entries]
        LemmaBuilder(FakeLexicon()).from_lookup(entries, stripped(["sam"], "gacchati"))
        self.assertEqual(entries, before)

    def test_stripped_prefixes_are_marked_as_preverbs(self):
        """`join_preverbs` reads the mark; `as_dict` never exports it."""
        entries = [particle("vi"), particle("ava"), verb("hṛ", "vyavaharati")]
        parts = LemmaBuilder(FakeLexicon()).from_lookup(entries, stripped(["vi", "ava"], "harati"))
        self.assertEqual([part.preverb for part in parts], [True, True, False])
        self.assertEqual(parts[0].as_dict(), {"form": "vi", "lemmas": ["vi"]})

    def test_outer_prefix_of_a_rejoined_noun_is_a_preverb(self):
        lexicon = FakeLexicon(joins={("sam", "vedana"): "saṃvedana"})
        entries = [particle("prati"), particle("sam"), noun("vedana", "pratisaṃvedanā")]
        parts = LemmaBuilder(lexicon).from_lookup(entries, stripped(["prati", "sam"], "vedanā"))
        self.assertEqual([part.preverb for part in parts], [True, False])

    def test_a_word_resolved_whole_has_no_preverb(self):
        whole = LemmaBuilder(FakeLexicon(sanskrit={"virati"})).from_lookup(
            [particle("vi"), noun("ratī", "virati")], stripped(["vi"], "rati")
        )
        unstripped = LemmaBuilder(FakeLexicon()).from_lookup([particle("prati")])
        self.assertEqual([part.preverb for part in whole + unstripped], [False, False])

    def test_a_remainder_that_is_itself_a_prefix_is_a_preverb_of_its_own(self):
        """The inflector reads `upa` + `sam` together: `sam` is the inner preverb.

        Its entries carry the whole word, as a stem's do, homograph `sa`
        included; read as a stem it took `sa` along and ended the run of
        preverbs before the verb.
        """
        entries = [particle("upa"), stamped(particle("sam"), "upasam"), noun("sa", "upasam")]
        parts = LemmaBuilder(FakeLexicon()).from_lookup(entries, stripped(["upa"], "sam"))
        self.assertEqual(
            [(part.form, part.lemmas, part.preverb) for part in parts],
            [("upa", ["upa"], True), ("sam", ["sam"], True)],
        )

    def test_the_particle_api_is_not_a_preverb(self):
        """Stripped like a prefix, but in running text it is the particle `api`."""
        entries = [particle("api"), verb("as", "apiasti")]
        parts = LemmaBuilder(FakeLexicon()).from_lookup(entries, stripped(["api"], "asti"))
        self.assertEqual(
            [(part.form, part.lemmas, part.preverb) for part in parts],
            [("api", ["api"], False), ("asti", ["as"], False)],
        )


class ResolutionPartTests(unittest.TestCase):
    """`LemmaBuilder.from_resolution`: a word the single-word cascade resolved."""

    @staticmethod
    def looked_up(entry, components=None):
        return entry + [components or entry[0], {"mw": {entry[0]: ["..."]}}]

    def test_headword_is_one_more_lemma_listed_last(self):
        headword = ["gacchati", "gacchati", {"cped": {"gacchati": ["..."]}}]
        root = self.looked_up(verb("gam", "gacchati"))
        resolution = WordResolution(
            "gacchati", WordResolution.MORPHOLOGY, [headword, root], headword, PrefixTrace()
        )
        parts = LemmaBuilder(FakeLexicon()).from_resolution(resolution)
        self.assertEqual(shapes(parts), [{"form": "gacchati", "lemmas": ["gam", "gacchati"]}])

    def test_headword_is_dropped_when_the_word_has_several_parts(self):
        headword = ["upagacchati", "upagacchati", {"cped": {"upagacchati": ["..."]}}]
        entries = [headword, self.looked_up(particle("upa")),
                   self.looked_up(verb("gam", "upagacchati"))]
        resolution = WordResolution(
            "upagacchati", WordResolution.MORPHOLOGY, entries, headword,
            stripped(["upa"], "gacchati"),
        )
        parts = LemmaBuilder(FakeLexicon()).from_resolution(resolution)
        self.assertEqual([part.lemmas for part in parts], [["upa"], ["gam"]])

    def test_dictionary_stage_is_one_part(self):
        entry = ["gatvā", "gatvā/", {"mw": {"gatvā": ["..."]}}]
        resolution = WordResolution(
            "gatvā", WordResolution.DICTIONARY, [entry], None, PrefixTrace()
        )
        parts = LemmaBuilder(FakeLexicon(headwords={"gatvā"})).from_resolution(resolution)
        self.assertEqual(shapes(parts), [{"form": "gatvā", "lemmas": ["gatvā"]}])

    def test_derived_stage_is_one_part_without_members(self):
        entry = self.looked_up(noun("niṣyandatā", "niṣyandatā"), components="ni—ṣyanda")
        resolution = WordResolution(
            "niṣyandatā", WordResolution.DERIVED, [entry], None, PrefixTrace()
        )
        ## a lexicon that would supply members, to show it is not asked
        lexicon = FakeLexicon(members={"niṣyandatā": ["ni", "ṣyanda"]})
        parts = LemmaBuilder(lexicon).from_resolution(resolution)
        self.assertEqual(shapes(parts), [{"form": "niṣyandatā", "lemmas": ["niṣyandatā"]}])


class TokenPartTests(unittest.TestCase):
    """`LemmaBuilder.from_tokens`: the split path, one analysis per split token."""

    def test_lookup_token(self):
        analysis = TokenAnalysis(("yogaḥ",), [noun("yoga", "yogaḥ")], TokenAnalysis.LOOKUP, PrefixTrace())
        parts = LemmaBuilder(FakeLexicon()).from_tokens([analysis])
        self.assertEqual(shapes(parts), [{"form": "yogaḥ", "lemmas": ["yoga"]}])

    def test_scanner_pieces_become_parts(self):
        scan = ScanTrace()
        scan.piece("deśa", ["deśa"], PrefixTrace())
        scan.piece("bandhaḥ", [noun("bandha", "bandhaḥ")], PrefixTrace())
        analysis = TokenAnalysis(("deśabandhaś",), [], TokenAnalysis.SCANNER, None, scan)
        parts = LemmaBuilder(FakeLexicon(headwords={"deśa"})).from_tokens([analysis])
        self.assertEqual(
            [(part.form, part.lemmas) for part in parts],
            [("deśa", ["deśa"]), ("bandhaḥ", ["bandha"])],
        )

    def test_rejected_scan_is_one_part_without_lemmas(self):
        scan = ScanTrace()
        scan.piece("ab", ["ab"], PrefixTrace())
        for _ in range(4):
            scan.skip()
        analysis = TokenAnalysis(("xyzabc",), [], TokenAnalysis.SCANNER, None, scan)
        parts = LemmaBuilder(FakeLexicon(headwords={"ab"})).from_tokens([analysis])
        self.assertEqual(shapes(parts), [{"form": "xyzabc", "lemmas": []}])

    def test_preverb_token_the_splitter_cut_off_carries_no_homograph(self):
        """`sam` + `arcayet`: nothing joined them, so `sam` was looked up as a word."""
        analysis = TokenAnalysis(
            ("sam",), [particle("sam"), noun("sa", "sam")], TokenAnalysis.LOOKUP, PrefixTrace()
        )
        parts = LemmaBuilder(FakeLexicon()).from_tokens([analysis])
        self.assertEqual(shapes(parts), [{"form": "sam", "lemmas": ["sam"]}])
        self.assertTrue(parts[0].preverb)

    def test_token_that_only_looks_like_a_preverb_keeps_its_readings(self):
        """The rule is for tokens that are prefixes, not for every short token."""
        analysis = TokenAnalysis(
            ("tat",), [noun("tad", "tat"), noun("tat", "tat")], TokenAnalysis.LOOKUP, PrefixTrace()
        )
        parts = LemmaBuilder(FakeLexicon()).from_tokens([analysis])
        self.assertEqual(parts[0].lemmas, ["tad", "tat"])
        self.assertFalse(parts[0].preverb)

    def test_lone_api_token_carries_itself_and_is_no_preverb(self):
        analysis = TokenAnalysis(("api",), [particle("api")], TokenAnalysis.LOOKUP, PrefixTrace())
        parts = LemmaBuilder(FakeLexicon()).from_tokens([analysis])
        self.assertEqual([(part.form, part.lemmas, part.preverb) for part in parts], [("api", ["api"], False)])

    def test_two_tokens_read_together_are_split_at_the_prefix(self):
        """`upa` + `āsate`, joined by the inflector: the form is never `upaāsate`."""
        analysis = TokenAnalysis(
            ("upa", "āsate"), [particle("upa"), verb("ās", "upaāsate")],
            TokenAnalysis.LOOKUP, stripped(["upa"], "āsate"),
        )
        parts = LemmaBuilder(FakeLexicon()).from_tokens([analysis])
        self.assertEqual(
            [(part.form, part.lemmas) for part in parts], [("upa", ["upa"]), ("āsate", ["ās"])]
        )
        self.assertEqual([part.preverb for part in parts], [True, False])

    def test_unresolved_token_is_one_part_without_lemmas(self):
        analysis = TokenAnalysis(("ḍḍḍḍ",), ["ḍḍḍḍ"], TokenAnalysis.UNRESOLVED, None, ScanTrace())
        parts = LemmaBuilder(FakeLexicon()).from_tokens([analysis])
        self.assertEqual(shapes(parts), [{"form": "ḍḍḍḍ", "lemmas": []}])


class TypedHeadwordTests(unittest.TestCase):
    """`LemmaBuilder.with_typed_headword`: the word as typed, when it is a headword.

    The cascade rewrites a final letter before any dictionary is asked, so a
    root typed on its own came back as something else (`vad` -> `vat`).
    """

    def test_one_part_gains_the_typed_word_listed_last(self):
        builder = LemmaBuilder(FakeLexicon(sanskrit={"vad"}))
        parts = builder.with_typed_headword("vad", [LemmaPart("vat", ["vat"])])
        self.assertEqual(shapes(parts), [{"form": "vat", "lemmas": ["vat", "vad"]}])

    def test_several_parts_give_way_to_one_part_for_the_whole_word(self):
        builder = LemmaBuilder(FakeLexicon(sanskrit={"sev"}))
        parts = builder.with_typed_headword(
            "sev", [LemmaPart("se", ["sa", "se"]), LemmaPart("u", ["u"])]
        )
        self.assertEqual(shapes(parts), [{"form": "sev", "lemmas": ["sev"]}])

    def test_the_whole_word_part_has_its_members(self):
        lexicon = FakeLexicon(sanskrit={"cittavṛtti"}, members={"cittavṛtti": ["citta", "vṛtti"]})
        parts = LemmaBuilder(lexicon).with_typed_headword(
            "cittavṛtti", [LemmaPart("citta", ["citta"]), LemmaPart("vṛtti", ["vṛtti"])]
        )
        self.assertEqual(
            shapes(parts),
            [{"form": "cittavṛtti", "lemmas": ["cittavṛtti"], "members": ["citta", "vṛtti"]}],
        )

    def test_a_word_already_among_the_lemmas_is_unchanged(self):
        builder = LemmaBuilder(FakeLexicon(sanskrit={"gacchati"}))
        before = [LemmaPart("gacchati", ["gam", "gacchati"])]
        self.assertEqual(builder.with_typed_headword("gacchati", before), before)

    def test_a_headword_only_other_dictionaries_have_changes_nothing(self):
        """`upagacchati` is a Pali headword; its preverb split stands."""
        builder = LemmaBuilder(FakeLexicon(headwords={"upagacchati"}))
        before = [preverb("upa"), LemmaPart("gacchati", ["gam"])]
        self.assertEqual(builder.with_typed_headword("upagacchati", before), before)

    def test_a_word_with_no_parts_gets_one(self):
        builder = LemmaBuilder(FakeLexicon(sanskrit={"vad"}))
        self.assertEqual(shapes(builder.with_typed_headword("vad", [])), [{"form": "vad", "lemmas": ["vad"]}])

    def test_the_parts_read_are_not_modified(self):
        part = LemmaPart("vat", ["vat"])
        LemmaBuilder(FakeLexicon(sanskrit={"vad"})).with_typed_headword("vad", [part])
        self.assertEqual(part.lemmas, ["vat"])


def JOIN(parts, **tables):
    return LemmaBuilder(FakeLexicon(**tables)).join_preverbs(parts)


class JoinedPreverbKeyTests(unittest.TestCase):
    """`LemmaBuilder.join_preverbs`: a run of preverbs is tied to the part after it."""

    def test_the_key_is_the_monier_williams_headword_of_the_compound(self):
        parts = JOIN(
            [preverb("ut"), LemmaPart("gacchati", ["gam"])],
            compound_verbs={(("ut",), "gam"): ["udgam"]},
        )
        self.assertEqual(pairs(parts), [("ut", ["ut"]), ("gacchati", ["gam", "udgam"])])

    def test_without_one_the_key_is_the_compound_the_sandhi_rules_make(self):
        parts = JOIN([preverb("prati"), LemmaPart("eti", ["i"])])
        self.assertEqual(pairs(parts), [("prati", ["prati"]), ("eti", ["i", "pratī"])])

    def test_of_several_headwords_the_one_the_rules_spell_is_taken(self):
        """Monier-Williams records `vi` + `dhī` under `vidhī` and `vidīdhī`."""
        parts = JOIN(
            [preverb("vi"), LemmaPart("dhīyate", ["dhī"])],
            compound_verbs={(("vi",), "dhī"): ["vidhī", "vidīdhī"]},
        )
        self.assertEqual(parts[1].lemmas, ["dhī", "vidhī"])

    def test_otherwise_every_headword_is_a_key(self):
        parts = JOIN(
            [preverb("palā"), LemmaPart("ayate", ["i"])],
            compound_verbs={(("palā",), "i"): ["palāy", "palāyay"]},
        )
        self.assertEqual(parts[1].lemmas, ["i", "palāy", "palāyay"])

    def test_no_key_contains_a_separator(self):
        """A search tokenizer cuts at `+`; the key must survive as one term."""
        parts = JOIN([preverb("upa"), preverb("sam"), LemmaPart("arc", ["arc"])])
        self.assertEqual(parts[2].lemmas, ["arc", "upasamarc"])

    def test_stacked_preverbs_make_one_key(self):
        parts = JOIN(
            [preverb("vi"), preverb("ava"), LemmaPart("harati", ["hṛ"])]
        )
        self.assertEqual(
            pairs(parts), [("vi", ["vi"]), ("ava", ["ava"]), ("harati", ["hṛ", "vyavahṛ"])]
        )

    def test_every_lemma_of_the_stem_gets_a_key_after_the_lemmas(self):
        parts = JOIN([preverb("anu"), LemmaPart("bhavati", ["bhavat", "bhū"])])
        self.assertEqual(parts[1].lemmas, ["bhavat", "bhū", "anubhavat", "anubhū"])

    def test_a_key_two_lemmas_share_is_listed_once(self):
        parts = JOIN([preverb("prati"), LemmaPart("ī", ["i", "ī"])])
        self.assertEqual(parts[1].lemmas, ["i", "ī", "pratī"])

    def test_a_preverb_with_nothing_after_it_adds_nothing(self):
        before = [LemmaPart("yogaḥ", ["yoga"]), preverb("sam")]
        self.assertEqual(JOIN(before), before)

    def test_a_word_that_is_not_a_preverb_joins_nothing(self):
        """A standalone `prati` is a word, not a preverb."""
        before = [LemmaPart("prati", ["prati"]), LemmaPart("eti", ["i"])]
        self.assertEqual(JOIN(before), before)

    def test_each_run_joins_the_part_after_it(self):
        parts = JOIN(
            [preverb("upa"), LemmaPart("xa", ["xa"]), preverb("anu"), LemmaPart("ya", ["ya"])]
        )
        self.assertEqual(
            [part.lemmas for part in parts], [["upa"], ["xa", "upaxa"], ["anu"], ["ya", "anuya"]]
        )

    def test_a_part_without_lemmas_takes_no_key_and_ends_the_run(self):
        before = [preverb("sam"), LemmaPart("ḍḍḍ", []), LemmaPart("yogaḥ", ["yoga"])]
        self.assertEqual(JOIN(before), before)

    def test_the_particle_api_takes_no_key_and_ends_the_run(self):
        before = [preverb("sam"), LemmaPart("api", ["api"]), LemmaPart("asti", ["as"])]
        self.assertEqual(JOIN(before), before)

    def test_the_particle_is_told_by_its_lemma_not_its_spelling(self):
        """`root_any_word` reads `āpi` as the particle `api`."""
        before = [preverb("sam"), LemmaPart("āpi", ["api"])]
        self.assertEqual(JOIN(before), before)

    def test_members_stay_with_the_stem(self):
        """A key is the stem with its preverbs, not a rival reading of it."""
        parts = JOIN(
            [preverb("anu"), LemmaPart("cittavṛtti", ["cittavṛtti"], ["citta", "vṛtti"])]
        )
        self.assertEqual(
            parts[1].as_dict(),
            {"form": "cittavṛtti", "lemmas": ["cittavṛtti", "anucittavṛtti"],
             "members": ["citta", "vṛtti"]},
        )

    def test_the_parts_read_are_not_modified(self):
        """Parts are memoised per word and shared between its repeats."""
        stem = LemmaPart("eti", ["i"])
        JOIN([preverb("prati"), stem])
        self.assertEqual(stem.lemmas, ["i"])


class FinishedWordTests(unittest.TestCase):
    """`LemmaBuilder.finished`: the typed headword is applied before the preverb keys."""

    def test_a_word_replaced_by_its_typed_headword_has_nothing_to_join(self):
        builder = LemmaBuilder(FakeLexicon(sanskrit={"pratyeti"}))
        parts = builder.finished("pratyeti", [preverb("prati"), LemmaPart("eti", ["i"])])
        self.assertEqual(shapes(parts), [{"form": "pratyeti", "lemmas": ["pratyeti"]}])

    def test_otherwise_the_preverbs_are_joined(self):
        builder = LemmaBuilder(FakeLexicon())
        parts = builder.finished("pratyeti", [preverb("prati"), LemmaPart("eti", ["i"])])
        self.assertEqual(pairs(parts), [("prati", ["prati"]), ("eti", ["i", "pratī"])])


class SuppletiveRootTests(unittest.TestCase):
    """A verb the table files under another root brings that root along."""

    def lemmas(self, entries, prefixes=None):
        return [part.lemmas for part in LemmaBuilder(FakeLexicon()).from_lookup(entries, prefixes)]

    def test_the_mapped_root_follows_the_filed_one(self):
        """The verb table files `paśyati`, the present of `dṛś`, under `paś`."""
        self.assertEqual(self.lemmas([verb("paś", "paśyati")]), [["paś", "dṛś"]])

    def test_it_follows_immediately_whatever_comes_after(self):
        self.assertEqual(
            self.lemmas([verb("paś", "x"), verb("gam", "x")]), [["paś", "dṛś", "gam"]]
        )

    def test_after_a_stripped_preverb(self):
        self.assertEqual(
            self.lemmas([particle("anu"), verb("paś", "anupaśyati")], stripped(["anu"], "paśyati")),
            [["anu"], ["paś", "dṛś"]],
        )

    def test_a_noun_of_the_same_spelling_is_not_mapped(self):
        """The forms table also has a feminine noun `paś`."""
        self.assertEqual(self.lemmas([noun("paś", "paśam")]), [["paś"]])

    def test_a_root_already_present_is_not_repeated(self):
        self.assertEqual(
            self.lemmas([verb("paś", "x"), verb("dṛś", "x")]), [["paś", "dṛś"]]
        )

    def test_an_unrelated_verb_is_unchanged(self):
        self.assertEqual(self.lemmas([verb("gam", "gacchati")]), [["gam"]])


if __name__ == "__main__":
    unittest.main()
