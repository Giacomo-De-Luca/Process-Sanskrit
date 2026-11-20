"""`PreverbSandhi`: preverbs joined to a stem the way the dictionaries spell them.

Lemma mode keys a preverb compound by the Monier-Williams headword whose parse
it records (`prati+i` -> `pratī`).  When Monier-Williams has none, the key is
the preverbs and the stem joined by these rules, so that it is still a word a
search can match.  Needs no database.
"""

import unittest

from process_sanskrit.utils.preverbSandhi import PreverbSandhi


class VowelTests(unittest.TestCase):
    def test_like_vowels_merge_long(self):
        self.assertEqual(PreverbSandhi.join("prati", "i"), "pratī")
        self.assertEqual(PreverbSandhi.join("adhi", "i"), "adhī")
        self.assertEqual(PreverbSandhi.join("anu", "ukṣ"), "anūkṣ")
        self.assertEqual(PreverbSandhi.join("ava", "as"), "avās")

    def test_a_before_i_and_u(self):
        self.assertEqual(PreverbSandhi.join("upa", "i"), "upe")
        self.assertEqual(PreverbSandhi.join("ā", "i"), "e")
        self.assertEqual(PreverbSandhi.join("pra", "uh"), "proh")

    def test_a_before_edh_is_vrddhi(self):
        """P. 6.1.89; before any other e or o the preverb's a is dropped."""
        self.assertEqual(PreverbSandhi.join("upa", "edh"), "upaidh")
        self.assertEqual(PreverbSandhi.join("pra", "ūh"), "proh")
        self.assertEqual(PreverbSandhi.join_all(["sam", "upa"], "edh"), "samupaidh")
        self.assertEqual(PreverbSandhi.join("pra", "ej"), "prej")

    def test_a_before_r_is_ar_for_a_preverb(self):
        self.assertEqual(PreverbSandhi.join("upa", "ṛ"), "upār")

    def test_i_and_u_before_another_vowel(self):
        self.assertEqual(PreverbSandhi.join("prati", "ā"), "pratyā")
        self.assertEqual(PreverbSandhi.join("vi", "ava"), "vyava")
        self.assertEqual(PreverbSandhi.join("anu", "i"), "anvi")
        self.assertEqual(PreverbSandhi.join("abhi", "as"), "abhyas")

    def test_the_privative_a_before_a_vowel_is_an(self):
        self.assertEqual(PreverbSandhi.join("a", "anta"), "ananta")
        self.assertEqual(PreverbSandhi.join("a", "tad"), "atad")


class ConsonantTests(unittest.TestCase):
    def test_sam(self):
        self.assertEqual(PreverbSandhi.join("sam", "gam"), "saṃgam")
        self.assertEqual(PreverbSandhi.join("sam", "hṛ"), "saṃhṛ")
        self.assertEqual(PreverbSandhi.join("sam", "pad"), "sampad")
        self.assertEqual(PreverbSandhi.join("sam", "bhū"), "sambhū")
        self.assertEqual(PreverbSandhi.join("sam", "arc"), "samarc")
        self.assertEqual(PreverbSandhi.join("saṃ", "gam"), "saṃgam")

    def test_ud(self):
        self.assertEqual(PreverbSandhi.join("ut", "gam"), "udgam")
        self.assertEqual(PreverbSandhi.join("ud", "pad"), "utpad")
        self.assertEqual(PreverbSandhi.join("ut", "sthā"), "utthā")
        self.assertEqual(PreverbSandhi.join("ut", "ā"), "udā")
        self.assertEqual(PreverbSandhi.join("ut", "hṛ"), "uddhṛ")
        self.assertEqual(PreverbSandhi.join("ut", "car"), "uccar")
        self.assertEqual(PreverbSandhi.join("ut", "jval"), "ujjval")
        self.assertEqual(PreverbSandhi.join("ut", "likh"), "ullikh")
        self.assertEqual(PreverbSandhi.join("ut", "nī"), "unnī")
        self.assertEqual(PreverbSandhi.join("ut", "ḍī"), "uḍḍī")

    def test_nis(self):
        self.assertEqual(PreverbSandhi.join("nis", "gam"), "nirgam")
        self.assertEqual(PreverbSandhi.join("nis", "ā"), "nirā")
        self.assertEqual(PreverbSandhi.join("nis", "kṛ"), "niṣkṛ")
        self.assertEqual(PreverbSandhi.join("nis", "car"), "niścar")
        self.assertEqual(PreverbSandhi.join("nis", "tṝ"), "nistṝ")
        self.assertEqual(PreverbSandhi.join("nis", "sṛ"), "niḥsṛ")
        self.assertEqual(PreverbSandhi.join("nis", "kṣip"), "niḥkṣip")

    def test_n_after_r_is_retroflex(self):
        self.assertEqual(PreverbSandhi.join("pra", "nam"), "praṇam")
        self.assertEqual(PreverbSandhi.join("pari", "nī"), "pariṇī")
        self.assertEqual(PreverbSandhi.join("upa", "nī"), "upanī")

    def test_n_after_nis_is_retroflex(self):
        self.assertEqual(PreverbSandhi.join("nis", "nam"), "nirṇam")

    def test_s_after_i_or_u_is_retroflex_except_before_a_second_preverb(self):
        self.assertEqual(PreverbSandhi.join("abhi", "sañj"), "abhiṣañj")
        self.assertEqual(PreverbSandhi.join("adhi", "sthā"), "adhiṣṭhā")
        self.assertEqual(PreverbSandhi.join("anu", "sṛ"), "anusṛ")
        self.assertEqual(PreverbSandhi.join_all(["abhi", "sam"], "gam"), "abhisaṃgam")

    def test_ch_doubles_after_a_vowel(self):
        self.assertEqual(PreverbSandhi.join("ā", "chad"), "ācchad")

    def test_a_preverb_before_a_consonant_is_kept_as_it_is(self):
        self.assertEqual(PreverbSandhi.join("upa", "gam"), "upagam")
        self.assertEqual(PreverbSandhi.join("prati", "pad"), "pratipad")


class ChainTests(unittest.TestCase):
    def test_a_chain_is_joined_from_the_right(self):
        """Each rule is chosen by the preverb on its left: `sam` + `uccar`."""
        self.assertEqual(PreverbSandhi.join_all(["sam", "ud"], "car"), "samuccar")
        self.assertEqual(PreverbSandhi.join_all(["vi", "ava"], "hṛ"), "vyavahṛ")
        self.assertEqual(PreverbSandhi.join_all(["upa", "sam"], "arc"), "upasamarc")
        self.assertEqual(PreverbSandhi.join_all(["prati", "ā"], "gam"), "pratyāgam")
        self.assertEqual(PreverbSandhi.join_all(["sam", "ā"], "dhā"), "samādhā")

    def test_empty_pieces(self):
        self.assertEqual(PreverbSandhi.join_all([], "gam"), "gam")
        self.assertEqual(PreverbSandhi.join("upa", ""), "upa")


if __name__ == "__main__":
    unittest.main()
