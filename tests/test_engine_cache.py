"""`EngineCache`: data read from a database once per engine, and its users.

Runs on small in-memory databases.  Every in-memory database has the URL
`sqlite://`, which is why the caches are keyed on the engine and not on the
URL: two of them must never share an entry.

* `EngineCacheTests`      the cache itself: per engine, read once, cleared, weak keys;
* `CompoundFormsCacheTests`, `PieceLemmasCacheTests`
                          the two tests `CacheTests` of test_compound_verbs.py
                          has for `CompoundVerbIndex`;
* `LemmaLexiconMembersCacheTests`
                          the members of a headword, shared by every lexicon on one engine;
* `ReferenceResetTests`   `_reset_reference_state` empties all of them.
"""

import gc
import unittest
import weakref
from unittest.mock import patch

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from process_sanskrit.functions.lemmaAnalysis import LemmaLexicon, PieceLemmas
from process_sanskrit.utils import dictionary_references
from process_sanskrit.utils.compoundMembers import CompoundForms
from process_sanskrit.utils.compoundVerbs import CompoundVerbIndex
from process_sanskrit.utils.engineCache import EngineCache
from tests.test_compound_verbs import database

## Monier-Williams rows that make a compound form, as `CompoundForms` reads them
TAJ = ("taj", 1.0, "<s>taj</s> <ab>in compound or Comparative degree</ab> for <s>tad</s>")
MANO = ("mano", 1.0, "<s>mano</s> <ab>in compound or Comparative degree</ab> for <s>manas</s>")
PRATI_I = ("pratī", 1.0, '<info verb="pre" parse="prati+i"/>')


def lemma_database(rows):
    """An in-memory database with a `lemmas(word, lemma)` table."""
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE lemmas (word TEXT, lemma TEXT)"))
        for word, lemma in rows:
            connection.execute(text("INSERT INTO lemmas VALUES (:w, :l)"), {"w": word, "l": lemma})
    return engine


def lemmas_from_table(piece_lemmas, word):
    """Stands in for `PieceLemmas._lemmas`, which needs the lexicon database."""
    rows = piece_lemmas.session.execute(text("SELECT lemma FROM lemmas WHERE word = :w"), {"w": word})
    return frozenset(lemma for (lemma,) in rows)


class EngineCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = EngineCache()
        self.reads = []

    def read(self, value):
        def reader():
            self.reads.append(value)
            return value
        return reader

    def test_a_table_is_read_once_per_engine(self):
        engine = create_engine("sqlite://")
        with Session(engine) as session:
            self.assertEqual(self.cache.table(session, self.read("first")), "first")
            self.assertEqual(self.cache.table(session, self.read("again")), "first")
        with Session(engine) as other_session:
            self.assertEqual(self.cache.table(other_session, self.read("again")), "first")
        self.assertEqual(self.reads, ["first"])

    def test_each_engine_has_its_own_entry(self):
        first, second = create_engine("sqlite://"), create_engine("sqlite://")
        with Session(first) as one, Session(second) as two:
            self.assertEqual(self.cache.table(one, self.read("first")), "first")
            self.assertEqual(self.cache.table(two, self.read("second")), "second")
        self.assertEqual(self.reads, ["first", "second"])

    def test_a_memo_is_one_dict_per_engine(self):
        first, second = create_engine("sqlite://"), create_engine("sqlite://")
        with Session(first) as one, Session(first) as also_one, Session(second) as two:
            self.cache.memo(one)["word"] = "lemma"
            self.assertEqual(self.cache.memo(also_one), {"word": "lemma"})
            self.assertEqual(self.cache.memo(two), {})

    def test_a_value_of_none_is_kept_too(self):
        engine = create_engine("sqlite://")
        with Session(engine) as session:
            self.assertIsNone(self.cache.table(session, self.read(None)))
            self.assertIsNone(self.cache.table(session, self.read("again")))
        self.assertEqual(self.reads, [None])

    def test_clear_forgets_every_engine(self):
        engine = create_engine("sqlite://")
        with Session(engine) as session:
            self.cache.table(session, self.read("first"))
            self.cache.clear()
            self.assertEqual(len(self.cache), 0)
            self.assertEqual(self.cache.table(session, self.read("second")), "second")

    def test_clear_all_also_empties_a_cache_created_later(self):
        later = EngineCache()
        engine = create_engine("sqlite://")
        with Session(engine) as session:
            later.table(session, self.read("first"))
            EngineCache.clear_all()
            self.assertEqual(len(later), 0)
            self.assertEqual(later.table(session, self.read("second")), "second")

    def test_a_dropped_engine_drops_its_entry(self):
        engine = create_engine("sqlite://")
        with Session(engine) as session:
            self.cache.table(session, self.read("first"))
        self.assertEqual(len(self.cache), 1)
        dropped = weakref.ref(engine)
        del engine, session
        gc.collect()
        self.assertIsNone(dropped())
        self.assertEqual(len(self.cache), 0)


class CompoundFormsCacheTests(unittest.TestCase):
    def setUp(self):
        CompoundForms.reset()

    def target(self, engine, form):
        with Session(engine) as session:
            return CompoundForms(session).target(form)

    def test_each_database_has_its_own_table(self):
        first, second = database([TAJ]), database([MANO])
        self.assertEqual(self.target(first, "taj"), "tad")
        self.assertIsNone(self.target(second, "taj"))
        self.assertEqual(self.target(second, "mano"), "manas")

    def test_the_table_is_read_once_and_again_after_a_reset(self):
        engine = database([TAJ])
        self.assertEqual(self.target(engine, "taj"), "tad")
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM mw"))
        self.assertEqual(self.target(engine, "taj"), "tad")
        CompoundForms.reset()
        self.assertIsNone(self.target(engine, "taj"))


class PieceLemmasCacheTests(unittest.TestCase):
    def setUp(self):
        PieceLemmas.reset()
        patcher = patch.object(PieceLemmas, "_lemmas", lemmas_from_table)
        patcher.start()
        self.addCleanup(patcher.stop)

    def lemmas(self, engine, word):
        with Session(engine) as session:
            return PieceLemmas(session)(word)

    def test_each_database_has_its_own_answers(self):
        first, second = lemma_database([("manaḥ", "manas")]), lemma_database([("manaḥ", "mana")])
        self.assertEqual(self.lemmas(first, "manaḥ"), {"manas"})
        self.assertEqual(self.lemmas(second, "manaḥ"), {"mana"})

    def test_an_answer_is_worked_out_once_and_again_after_a_reset(self):
        engine = lemma_database([("manaḥ", "manas")])
        self.assertEqual(self.lemmas(engine, "manaḥ"), {"manas"})
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM lemmas"))
        self.assertEqual(self.lemmas(engine, "manaḥ"), {"manas"})
        PieceLemmas.reset()
        self.assertEqual(self.lemmas(engine, "manaḥ"), frozenset())


class CountingMembers:
    """Stands in for `CompoundMembers`: counts the headwords it is asked about."""

    def __init__(self, asked):
        self.asked = asked

    def members(self, headword):
        self.asked.append(headword)
        return [headword[:2], headword[2:]]


class LemmaLexiconMembersCacheTests(unittest.TestCase):
    """Members depend on the database only, so every lexicon on one engine shares them."""

    def setUp(self):
        EngineCache.clear_all()
        self.asked = []
        patcher = patch.object(LemmaLexicon, "_compound_members", lambda lexicon: CountingMembers(self.asked))
        patcher.start()
        self.addCleanup(patcher.stop)

    def members(self, engine, lemma):
        with Session(engine) as session:
            return LemmaLexicon(session).members(lemma)

    def test_lexicons_on_one_engine_work_a_headword_out_once(self):
        engine = create_engine("sqlite://")
        self.assertEqual(self.members(engine, "sukha"), ["su", "kha"])
        self.assertEqual(self.members(engine, "sukha"), ["su", "kha"])
        self.assertEqual(self.asked, ["sukha"])

    def test_each_engine_works_it_out_again(self):
        self.members(create_engine("sqlite://"), "sukha")
        self.members(create_engine("sqlite://"), "sukha")
        self.assertEqual(self.asked, ["sukha", "sukha"])

    def test_a_caller_cannot_change_the_kept_members(self):
        engine = create_engine("sqlite://")
        self.members(engine, "sukha").append("changed")
        self.assertEqual(self.members(engine, "sukha"), ["su", "kha"])


class ReferenceResetTests(unittest.TestCase):
    """`PieceLemmas` reads `DICTIONARY_REFERENCES`, which follows the configured
    database path, so a reset of the references must empty it with the others."""

    def setUp(self):
        patcher = patch.object(PieceLemmas, "_lemmas", lemmas_from_table)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(dictionary_references._reset_reference_state)

    def test_the_reference_reset_empties_every_engine_cache(self):
        forms, verbs = database([TAJ]), database([PRATI_I])
        lemmas = lemma_database([("manaḥ", "manas")])

        def answers():
            with Session(forms) as one, Session(verbs) as two, Session(lemmas) as three:
                return (CompoundForms(one).target("taj"), CompoundVerbIndex(two).headwords(["prati"], "i"),
                        PieceLemmas(three)("manaḥ"))

        self.assertEqual(answers(), ("tad", ["pratī"], {"manas"}))
        for engine, table in ((forms, "mw"), (verbs, "mw"), (lemmas, "lemmas")):
            with engine.begin() as connection:
                connection.execute(text(f"DELETE FROM {table}"))
        self.assertEqual(answers(), ("tad", ["pratī"], {"manas"}))
        dictionary_references._reset_reference_state()
        self.assertEqual(answers(), (None, [], frozenset()))

    def test_the_reference_reset_forgets_the_members(self):
        asked = []
        with patch.object(LemmaLexicon, "_compound_members", lambda lexicon: CountingMembers(asked)):
            engine = create_engine("sqlite://")
            for _ in range(2):
                with Session(engine) as session:
                    LemmaLexicon(session).members("sukha")
                dictionary_references._reset_reference_state()
        self.assertEqual(asked, ["sukha", "sukha"])


if __name__ == "__main__":
    unittest.main()
