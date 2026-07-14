"""Regression tests for dictionary rows whose ``components`` column is NULL."""

import unittest

from process_sanskrit import process
from process_sanskrit.functions.cleanResults import roots_splitted
from process_sanskrit.functions.dictionaryLookup import multidict
from process_sanskrit.utils.databaseSetup import get_database_path
from tests.datasets.yogaSutra import ys


class _StaticRows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _StaticSession:
    """Small SQLAlchemy-session stand-in for a deterministic dictionary row."""

    def __init__(self, rows):
        self._rows = rows

    def execute(self, _statement, _parameters):
        return _StaticRows(self._rows)


class NullDictionaryComponentsTests(unittest.TestCase):
    def test_multidict_uses_headword_when_components_is_null(self):
        session = _StaticSession(
            [("samādhibhāvanā", None, "cultivation of concentration")]
        )

        component, dictionaries = multidict(
            "samādhibhāvanā", "CPED", session=session
        )

        self.assertEqual(component, "samādhibhāvanā")
        self.assertEqual(
            dictionaries,
            {"CPED": {"samādhibhāvanā": ["cultivation of concentration"]}},
        )

    def test_parts_formatting_falls_back_to_headword_for_null_components(self):
        entries = [
            [
                "samādhibhāvanā",
                None,
                {"CPED": {"samādhibhāvanā": ["cultivation of concentration"]}},
            ]
        ]

        self.assertEqual(
            roots_splitted(entries),
            {"samādhibhāvanā": ["samādhibhāvanā"]},
        )

    def test_seven_field_parts_formatting_falls_back_for_null_components(self):
        entries = [
            [
                "samādhibhāvanā",
                "f",
                [("Nom", "Sg")],
                ["samādhibhāvanā"],
                "samādhibhāvanā",
                None,
                {
                    "CPED": {
                        "samādhibhāvanā": ["cultivation of concentration"]
                    }
                },
            ]
        ]

        self.assertEqual(
            roots_splitted(entries),
            {"samādhibhāvanā": ["samādhibhāvanā"]},
        )


@unittest.skipUnless(
    get_database_path().exists(), "packaged lexicon database is not installed"
)
class YogaSutraNullComponentsTests(unittest.TestCase):
    def test_sutra_53_full_line_parts_mode_succeeds(self):
        result = process(ys[52], "MW", mode="parts", cached=False)

        self.assertIsInstance(result, dict)
        self.assertTrue(result)
        self.assertTrue(
            all(
                isinstance(part, str)
                for components in result.values()
                for part in components
            )
        )


if __name__ == "__main__":
    unittest.main()
