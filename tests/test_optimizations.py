"""Regression tests for the low-memory SQLite optimization path."""

import sqlite3
import unittest

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from process_sanskrit.functions.SQLiteFind import (
    SQLite_find_name,
    SQLite_find_verb,
)
from process_sanskrit.functions.dictionaryLookup import multidict
from process_sanskrit.functions.rootAnyWord import _direct_roots
from process_sanskrit.utils import loadResources
from process_sanskrit.utils.databaseSetup import get_engine, get_session
from process_sanskrit.utils.dictionary_references import DICTIONARY_REFERENCES


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = get_engine()
        cls.session = get_session()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_read_engine_is_small_and_read_only(self):
        self.assertEqual(self.engine.pool.size(), 2)
        with self.engine.connect() as connection:
            self.assertEqual(connection.exec_driver_sql("PRAGMA query_only").scalar(), 1)
            self.assertEqual(
                connection.exec_driver_sql("PRAGMA cache_size").scalar(),
                -8192,
            )
            with self.assertRaises(OperationalError):
                connection.exec_driver_sql("CREATE TEMP TABLE must_fail (id)")

    def test_dictionary_reference_compatibility_overlay(self):
        self.assertEqual(len(DICTIONARY_REFERENCES), 246955)
        self.assertEqual(DICTIONARY_REFERENCES["*ai"], ["ddsa"])
        self.assertEqual(
            DICTIONARY_REFERENCES["a"],
            ["ap90", "bhs", "cae", "ddsa", "gra", "mw"],
        )
        self.assertEqual(
            DICTIONARY_REFERENCES["yoga"],
            ["bhs", "cae", "cped", "ddsa", "gra", "mw"],
        )

    def test_mw_key_list_stays_lazy(self):
        self.assertNotIn("mwdictionaryKeys", loadResources.__dict__)

    def test_exact_dictionary_lookup(self):
        component, dictionaries = multidict("yoga", "MW", session=self.session)
        self.assertEqual(component, "yoga")
        self.assertIn("yoga", dictionaries["MW"])

    def test_morphology_output_and_immutable_memo(self):
        noun = SQLite_find_name("pratiprasave", session=self.session)
        verb = SQLite_find_verb("gacchathaḥ", session=self.session)
        self.assertEqual(noun[0][0:3], ["pratiprasava", "m_a", [("Loc", "Sg")]])
        self.assertEqual(verb[0][0:3], ["gam", "pre-1a", [("Second", "Du")]])

        memo = {}
        first = _direct_roots("pratiprasave", self.session, memo)
        first[0][0] = "mutated"
        second = _direct_roots("pratiprasave", self.session, memo)
        self.assertEqual(second[0][0], "pratiprasava")

    def test_exact_and_morphology_plans_use_indexes(self):
        plans = []
        statements = (
            (
                "SELECT keys_iast, components, cleaned_body "
                "FROM mw WHERE keys_iast = :word",
                {"word": "yoga"},
            ),
            (
                "SELECT t2.key, t2.model, t2.stem, t1.refs, t1.data "
                "FROM lgtab2 AS t2 JOIN lgtab1 AS t1 "
                "ON t1.stem=t2.stem AND t1.model=t2.model "
                "WHERE t2.key=:word AND t1.rowid=("
                "SELECT MIN(candidate.rowid) FROM lgtab1 AS candidate "
                "WHERE candidate.stem=t2.stem AND candidate.model=t2.model)",
                {"word": "viṃśatyai"},
            ),
            (
                "SELECT t2.key, t2.model, t2.stem, t1.refs, t1.data "
                "FROM vlgtab2 AS t2 JOIN vlgtab1 AS t1 "
                "ON t1.stem=t2.stem AND t1.model=t2.model "
                "WHERE t2.rowid=(SELECT candidate.rowid FROM vlgtab2 AS candidate "
                "WHERE candidate.key=:word ORDER BY candidate.rowid DESC LIMIT 1) "
                "ORDER BY t1.rowid LIMIT 1",
                {"word": "gacchathaḥ"},
            ),
        )
        with self.engine.connect() as connection:
            for statement, parameters in statements:
                rows = connection.execute(
                    text("EXPLAIN QUERY PLAN " + statement), parameters
                )
                plan = [row[3] for row in rows]
                plans.append(plan)
                self.assertTrue(
                    any("USING" in detail and "INDEX" in detail for detail in plan),
                    plan,
                )
                self.assertFalse(
                    any(detail.startswith("SCAN ") for detail in plan),
                    plan,
                )


if __name__ == "__main__":
    unittest.main()
