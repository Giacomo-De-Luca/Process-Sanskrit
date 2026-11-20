"""`CompoundVerbIndex`: the preverb compounds Monier-Williams records, by their parse.

Runs on a small in-memory `mw` table, so the spelling rules and the cache are
pinned without the lexicon database.  The lookups against the real data are in
`LexiconTests` of `test_lemma_mode.py`.
"""

import unittest

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from process_sanskrit.utils.compoundVerbs import CompoundVerbIndex

ROWS = [
    # keys_iast, lnum, cleaned_body
    ("pratī", 2.0, '<s>pratī</s> (prati-√ i) <info verb="pre" parse="prati+i"/>'),
    ("pratī", 1.0, "<s>pratī</s> in compound for prati"),
    ("nirgam", 3.0, '<info verb="pre" parse="nir+gam"/>'),
    ("niḥsṛ", 4.0, '<info verb="pre" parse="nis+sf"/>'),
    ("udgam", 5.0, '<info verb="pre" parse="ud+gam"/>'),
    ("pratyāgam", 6.0, '<info verb="pre" cp="P" parse="prati+A+gam"/>'),
    ("pratyagra", 7.0, '<info verb="pre" parse="praty+agra"/>'),
    ("antargam", 8.0, '<info verb="pre" parse="antaH+gam"/>'),
    ("vidīdhī", 10.0, '<info verb="pre" parse="vi+DI"/>'),
    ("vidhī", 9.0, '<info verb="pre" parse="vi+DI"/>'),
]


def database(rows=ROWS):
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE mw (keys_iast TEXT, components TEXT, lnum REAL, cleaned_body TEXT)"))
        for key, lnum, body in rows:
            connection.execute(
                text("INSERT INTO mw VALUES (:k, :k, :n, :b)"), {"k": key, "n": lnum, "b": body}
            )
    return engine


class LookupTests(unittest.TestCase):
    def setUp(self):
        CompoundVerbIndex.reset()
        self.engine = database()
        self.session = Session(self.engine)
        self.addCleanup(self.session.close)
        self.index = CompoundVerbIndex(self.session)

    def test_a_parse_names_its_headword(self):
        self.assertEqual(self.index.headwords(["prati"], "i"), ["pratī"])
        self.assertEqual(self.index.headwords(["prati", "ā"], "gam"), ["pratyāgam"])

    def test_monier_williams_spellings_of_a_preverb_read_as_ours(self):
        self.assertEqual(self.index.headwords(["nis"], "gam"), ["nirgam"])
        self.assertEqual(self.index.headwords(["nis"], "sṛ"), ["niḥsṛ"])
        self.assertEqual(self.index.headwords(["prati"], "agra"), ["pratyagra"])
        self.assertEqual(self.index.headwords(["antar"], "gam"), ["antargam"])

    def test_our_spellings_of_a_preverb_read_as_monier_williams(self):
        self.assertEqual(self.index.headwords(["ut"], "gam"), ["udgam"])
        self.assertEqual(self.index.headwords(["ud"], "gam"), ["udgam"])

    def test_several_headwords_come_in_dictionary_order(self):
        self.assertEqual(self.index.headwords(["vi"], "dhī"), ["vidhī", "vidīdhī"])

    def test_nothing_recorded_is_an_empty_list(self):
        self.assertEqual(self.index.headwords(["upa", "sam"], "arc"), [])
        self.assertEqual(self.index.headwords([], "gam"), [])
        self.assertEqual(self.index.headwords(["upa"], ""), [])


class CacheTests(unittest.TestCase):
    def setUp(self):
        CompoundVerbIndex.reset()

    def headwords(self, engine, preverbs, root):
        with Session(engine) as session:
            return CompoundVerbIndex(session).headwords(preverbs, root)

    def test_each_database_has_its_own_table(self):
        """Every in-memory database has the URL `sqlite://`; they must not share one."""
        first = database()
        second = database([("upagam", 1.0, '<info verb="pre" parse="upa+gam"/>')])
        self.assertEqual(self.headwords(first, ["prati"], "i"), ["pratī"])
        self.assertEqual(self.headwords(second, ["prati"], "i"), [])
        self.assertEqual(self.headwords(second, ["upa"], "gam"), ["upagam"])

    def test_the_table_is_read_once_and_again_after_a_reset(self):
        engine = database()
        self.assertEqual(self.headwords(engine, ["prati"], "i"), ["pratī"])
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM mw"))
        self.assertEqual(self.headwords(engine, ["prati"], "i"), ["pratī"])
        CompoundVerbIndex.reset()
        self.assertEqual(self.headwords(engine, ["prati"], "i"), [])


if __name__ == "__main__":
    unittest.main()
