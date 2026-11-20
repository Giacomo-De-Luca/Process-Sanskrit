"""Small independent checks for the read-only Huet coverage assessment."""

import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.assess_huet_merge import AssessmentConfiguration, Coverage, HuetMergeAssessment


class HuetAssessmentTests(unittest.TestCase):
    def test_coverage_counts_unique_forms_and_preserves_existing_inventory(self):
        result = Coverage.compare({"a", "b", "c"}, {"b", "d"})
        self.assertEqual(result, {"current": 3, "huet": 2, "overlap": 1,
                                 "new": 1, "union": 4, "gain_percent": 33.3333})

    def test_normalization_only_folds_unicode_and_final_anusvara(self):
        self.assertEqual(Coverage.surface("ra\u0304mam"), Coverage.surface("rāmaṃ"))
        self.assertNotEqual(Coverage.surface("samaya"), Coverage.surface("saṃaya"))
        self.assertNotEqual(Coverage.surface("rāmas"), Coverage.surface("rāmaḥ"))
        self.assertEqual(Coverage.lemma("a-kāma"), "akāma")

    def test_source_connections_reject_writes_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.db"
            sqlite3.connect(path).close()
            with HuetMergeAssessment.connect(path) as database:
                with self.assertRaises(sqlite3.OperationalError):
                    database.execute("CREATE TABLE forbidden (value)")
            with self.assertRaises(sqlite3.OperationalError):
                HuetMergeAssessment.connect(Path(directory) / "missing.db")

    def test_current_pairs_use_lemma_reference_and_ignore_orphan_keys(self):
        with sqlite3.connect(":memory:") as database:
            database.executescript("""
                CREATE TABLE lgtab1 (model, stem, refs, data);
                CREATE TABLE lgtab2 (key, model, stem);
                INSERT INTO lgtab1 VALUES ('m_a', 'a-kāma', '1,akāma', 'akāmaḥ');
                INSERT INTO lgtab2 VALUES ('akāmam', 'm_a', 'a-kāma');
                INSERT INTO lgtab2 VALUES ('?', 'm_a', 'a-kāma');
                INSERT INTO lgtab2 VALUES ('orphan', 'm_a', 'missing');
            """)
            stats, forms, lemmas, missing = HuetMergeAssessment.current_inventory(
                database, "lgtab", {("akāmaṃ", "akāma"), ("new", "new")})
        self.assertEqual(forms, {"akāmam"})
        self.assertEqual(lemmas, {"akāma"})
        self.assertEqual(missing, {("new", "new")})
        self.assertEqual(stats["discarded_index_rows"], {"?": 1, "orphan_index_rows": 1})

    def test_duplicate_changed_spellings_and_corrupt_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            assessment = HuetMergeAssessment(AssessmentConfiguration(root=Path(directory)))
            assessment.huet_directory.mkdir(parents=True)
            columns = ["form_iast", "stem_iast", "stem_index", "case_value", "number_value", "gender"]
            canonical = ("rāmaḥ", "rāma", "", "nom", "sg", "mas")
            original = ("rāmas",) + canonical[1:]
            with (assessment.huet_directory / "nominal_forms.tsv").open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(columns)
                writer.writerows([canonical, canonical])
            with sqlite3.connect(":memory:") as database:
                database.execute("CREATE TABLE nominal_forms (id INTEGER PRIMARY KEY, " + ",".join(columns) + ")")
                database.executemany("INSERT INTO nominal_forms VALUES (NULL,?,?,?,?,?,?)", [original, original, canonical, canonical])
                stats, forms, pairs, stems, rows = assessment.huet_inventory(database, "nominal_forms")
                self.assertEqual(stats["tsv_rows"], 2)
                self.assertEqual(stats["unique_tsv_records"], 1)
                self.assertEqual(stats["sqlite_identical_duplicate_rows"], 2)
                self.assertEqual(forms, {"rāmaḥ"})
                database.execute("UPDATE nominal_forms SET gender='neu' WHERE id=4")
                with self.assertRaisesRegex(ValueError, "metadata differs"):
                    assessment.huet_inventory(database, "nominal_forms")


if __name__ == "__main__":
    unittest.main()
