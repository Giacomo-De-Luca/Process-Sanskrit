"""The DCS reader and the evidence it gives for a headword (scripts/compound_members/dcs.py)."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.compound_members.dcs import ConlluReader, DcsEvidence, Segment
from scripts.compound_members.joiner import SandhiJoiner

from .support import has_database, sample_sentences


class ConlluReaderTests(unittest.TestCase):
    def setUp(self):
        self.sentences = sample_sentences()

    def test_reads_every_sentence_with_its_id_and_work(self):
        self.assertEqual([sentence.sent_id for sentence in self.sentences], ["478558", "900001_1", "900002"])
        self.assertEqual(self.sentences[0].work, "Yogasūtrabhāṣya")

    def test_a_multiword_token_keeps_its_surface_and_its_segments(self):
        token = self.sentences[0].tokens[3]
        self.assertEqual(token.surface, "sthitinibandhanīti")
        self.assertEqual([segment.lemma for segment in token.segments], ["sthiti", "nibandhana", "iti"])

    def test_compound_members_are_grouped_with_their_head(self):
        words = self.sentences[0].tokens[3].words()
        self.assertEqual([word.lemmas for word in words], [("sthiti", "nibandhana"), ("iti",)])
        self.assertTrue(words[0].is_compound)
        self.assertFalse(words[1].is_compound)

    def test_a_plain_token_is_one_word_of_one_segment(self):
        token = self.sentences[0].tokens[0]
        self.assertEqual(token.surface, "pravṛttir")
        self.assertEqual(token.segments, (Segment("pravṛttir", "pravṛtti", "pravṛttiḥ", False, False),))

    def test_a_participle_is_marked(self):
        self.assertTrue(self.sentences[0].tokens[1].segments[0].participle)

    def test_an_abstract_suffix_is_folded_onto_its_base(self):
        words = self.sentences[1].tokens[2].words()
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].lemmas, ("viṣayatva",))
        self.assertEqual(words[0].segments[0].unsandhied, "viṣayatvāt")

    def test_the_vedic_l_stand_in_is_restored(self):
        segment = self.sentences[1].tokens[3].segments[0]
        self.assertEqual((segment.form, segment.lemma), ("kḷptam", "kḷp"))

    def test_a_parsed_chapter_is_read_only_without_its_plain_twin(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "Work"
            work.mkdir()
            for name in ("a.conllu", "a.conllu_parsed", "b.conllu_parsed"):
                (work / name).write_text("")
            self.assertEqual([path.name for path in ConlluReader(Path(directory)).files()],
                             ["a.conllu", "b.conllu_parsed"])


@unittest.skipUnless(has_database(), "lexicon database is not installed")
class EvidenceTests(unittest.TestCase):
    """A headword is found in DCS whole (its lemma) or split (segments that join to it)."""

    WANTED = {"ālayavijñāna", "tajjātīya", "sthiti", "vijñāna", "viṣayatva", "sarvābharaṇabhūṣita",
              "antargata", "sarvābharaṇa"}

    @classmethod
    def setUpClass(cls):
        cls.evidence = DcsEvidence(SandhiJoiner.shared(), cls.WANTED)
        cls.short = DcsEvidence(SandhiJoiner.shared(), cls.WANTED, max_run=2)
        for sentence in sample_sentences():
            cls.evidence.add(sentence)
            cls.short.add(sentence)

    def test_a_split_compound_is_evidence_for_the_headword_it_spells(self):
        found = self.evidence.get("ālayavijñāna")
        self.assertEqual(list(found.splits), [(("ālaya", "vijñāna"), ("ālaya", "vijñāna"))])
        self.assertEqual(found.surfaces, Counter({"ālayavijñānasya": 1}))
        self.assertEqual(found.exact_surfaces, Counter({"ālayavijñānasya": 1}))

    def test_consonant_sandhi_between_segments(self):
        self.assertEqual(list(self.evidence.get("tajjātīya").splits),
                         [(("tad", "jātīya"), ("tad", "jātīya"))])

    def test_a_participle_head_is_joined_through_its_stem(self):
        ## DCS gives the participle its root as lemma (`bhūṣ`); the join reads the stem off the form
        found = self.evidence.get("sarvābharaṇabhūṣita")
        self.assertEqual(list(found.splits), [(("sarva", "ābharaṇa", "bhūṣ"), ("sarva", "ābharaṇa", "bhūṣita"))])
        self.assertEqual(self.evidence.get("sarvābharaṇa").whole, 0)
        self.assertEqual(sum(self.evidence.get("sarvābharaṇa").splits.values()), 1)

    def test_a_visarga_member_is_joined_as_its_stem(self):
        self.assertEqual(list(self.evidence.get("antargata").splits), [(("antar", "gam"), ("antar", "gata"))])

    def test_runs_longer_than_max_run_are_not_read(self):
        self.assertIsNone(self.short.get("sarvābharaṇabhūṣita"))
        self.assertIsNotNone(self.short.get("sarvābharaṇa"))

    def test_a_lemma_is_whole_evidence_also_inside_a_compound(self):
        self.assertEqual(self.evidence.get("sthiti").whole, 1)
        self.assertEqual(self.evidence.get("vijñāna").whole, 1)
        self.assertEqual(self.evidence.get("viṣayatva").whole, 1)

    def test_unwanted_headwords_are_not_recorded(self):
        self.assertIsNone(self.evidence.get("iti"))


if __name__ == "__main__":
    unittest.main()
