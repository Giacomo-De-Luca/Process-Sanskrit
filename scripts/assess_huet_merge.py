"""Read-only inventory comparison; settings live in AssessmentConfiguration.

This measures database coverage, not end-to-end annotation accuracy. It does
not import the processing library or modify either source database.
"""

import csv
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path


@dataclass(frozen=True)
class AssessmentConfiguration:
    root: Path = Path(__file__).resolve().parents[1]
    output: str = "build/huet-merge-assessment.json"


class Coverage:
    """Compare inventories without counting repeated spellings as extra forms."""

    @staticmethod
    def text(value):
        return unicodedata.normalize("NFC", value.strip())

    @classmethod
    def surface(cls, value):
        value = cls.text(value)
        return value[:-1] + "ṃ" if value.endswith("m") else value

    @classmethod
    def lemma(cls, value):
        return cls.text(value).replace("-", "")

    @staticmethod
    def conventional_iast(value):
        return re.fullmatch("[aāiīuūṛṝḷḹeokgṅcjñṭḍṇtdnpbmyrlvśṣshṃṁḥ]+", value) is not None

    @staticmethod
    def compare(current, added):
        overlap = len(current & added)
        novel = len(added - current)
        return {
            "current": len(current), "huet": len(added), "overlap": overlap,
            "new": novel, "union": len(current | added),
            "gain_percent": round(100 * novel / len(current), 4) if current else None,
        }


class HuetMergeAssessment:
    def __init__(self, config=AssessmentConfiguration()):
        self.config = config
        self.resources = config.root / "process_sanskrit/resources"
        self.huet_directory = self.resources / "HuetInflectionTables"

    @staticmethod
    def connect(path):
        return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)

    @staticmethod
    def fingerprint(path):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}

    def huet_inventory(self, database, table):
        """Verify both DB halves against every TSV row, then use canonical TSV."""
        count = database.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        columns = [row[1] for row in database.execute(f"PRAGMA table_info({table})")][1:]
        half = count // 2
        first = database.execute(f"SELECT {','.join(columns)} FROM {table} ORDER BY id LIMIT ?", (half,))
        second = database.execute(f"SELECT {','.join(columns)} FROM {table} ORDER BY id LIMIT -1 OFFSET ?", (half,))
        raw_forms, pairs, stems, records = set(), set(), set(), set()
        cells, features, differences = {}, {key: Counter() for key in columns[2:]}, Counter()
        normalized_rows = []
        with (self.huet_directory / (table + ".tsv")).open(newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if reader.fieldnames != columns:
                raise ValueError(f"Unexpected columns in {table}: {reader.fieldnames}")
            n = 0
            for n, triplet in enumerate(zip_longest(reader, first, second), 1):
                row, original, canonical = triplet
                if row is None or original is None or canonical is None:
                    raise ValueError(f"TSV/DB row counts differ in {table}")
                record = tuple(row.values())
                canonical = tuple("" if value is None else str(value) for value in canonical)
                original = tuple("" if value is None else str(value) for value in original)
                if record != canonical or original[1:] != record[1:]:
                    raise ValueError(f"TSV/DB metadata differs at {table} row {n}")
                if original[0] != record[0]:
                    differences[original[0][-1] + "→" + record[0][-1]] += 1
                    if original[0][:-1] != record[0][:-1] or original[0][-1] not in "sr" or record[0][-1] != "ḥ":
                        raise ValueError(f"Unexpected spelling difference at {table} row {n}")
                records.add(record)
                raw_forms.add(Coverage.text(row["form_iast"]))
                form, stem = Coverage.surface(row["form_iast"]), Coverage.lemma(row["stem_iast"])
                pairs.add((form, stem))
                stems.add(stem)
                for key in features:
                    features[key][row[key]] += 1
                if table == "nominal_forms":
                    group = (stem, row["stem_index"], row["gender"])
                    cell = (row["case_value"], row["number_value"])
                else:
                    group = tuple(row[key] for key in columns[1:-2])
                    cell = (row["person"], row["number_value"])
                cells.setdefault(group, {}).setdefault(cell, set()).add(form)
                normalized_rows.append((form, stem, row.get("conjugation_type", row.get("gender"))))
        multiform = sum(len(forms) > 1 for group in cells.values() for forms in group.values())
        distinct_database_rows = database.execute(
            f"SELECT COUNT(*) FROM (SELECT DISTINCT {','.join(columns)} FROM {table})"
        ).fetchone()[0]
        stats = {
            "sqlite_rows": count, "tsv_rows": n, "unique_tsv_records": len(records),
            "verified_db_halves": True, "raw_to_tsv_changes": dict(differences),
            "sqlite_identical_duplicate_rows": count - distinct_database_rows,
            "features": {key: dict(value) for key, value in features.items()},
            "paradigm_groups": len(cells), "cells_with_multiple_forms": multiform,
            "groups_with_multiple_forms": sum(any(len(forms) > 1 for forms in group.values()) for group in cells.values()),
            "group_cell_counts": dict(sorted(Counter(len(group) for group in cells.values()).items())),
        }
        return stats, raw_forms, pairs, stems, normalized_rows

    @staticmethod
    def current_inventory(database, prefix, candidate_pairs):
        aliases, lemmas, models, lengths = {}, set(), Counter(), Counter()
        missing = set(candidate_pairs)
        for model, stem, refs, data in database.execute(f"SELECT model,stem,refs,data FROM {prefix}1 ORDER BY rowid"):
            reference = re.search(r",([^\W\d_]+)", refs or "")
            lemma = Coverage.lemma(reference.group(1) if reference else stem)
            aliases.setdefault((stem, model), lemma)
            lemmas.add(lemma)
            models[model] += 1
            lengths[len((data or "").split(":"))] += 1
        forms, discarded = set(), Counter()
        rows = 0
        for key, model, stem in database.execute(f"SELECT key,model,stem FROM {prefix}2"):
            rows += 1
            if not key or key == "?":
                discarded[str(key)] += 1
                continue
            if (stem, model) not in aliases:
                discarded["orphan_index_rows"] += 1
                continue
            form = Coverage.text(key)
            forms.add(form)
            missing.discard((Coverage.surface(form), aliases[(stem, model)]))
        return {
            "paradigm_rows": sum(models.values()), "index_rows": rows,
            "models": dict(models), "paradigm_lengths": dict(lengths),
            "discarded_index_rows": dict(discarded),
        }, forms, lemmas, missing

    def run(self):
        source_paths = [self.resources / "SQliteDB.sqlite", self.huet_directory / "sanskrit_forms.db",
                        self.huet_directory / "nominal_forms.tsv", self.huet_directory / "verbal_forms.tsv"]
        fingerprints = {str(path.relative_to(self.config.root)): self.fingerprint(path) for path in source_paths}
        report, current_all, huet_all = {"sources": fingerprints}, set(), set()
        with self.connect(source_paths[0]) as current, self.connect(source_paths[1]) as huet:
            dictionary = {Coverage.lemma(row[0]) for row in current.execute("SELECT keys_iast FROM word_list")}
            for kind, table, prefix in [("nominal", "nominal_forms", "lgtab"), ("verbal", "verbal_forms", "vlgtab")]:
                print(f"Comparing {kind} inventories", flush=True)
                added_stats, added_forms, pairs, added_stems, added_rows = self.huet_inventory(huet, table)
                base_stats, base_forms, base_stems, missing_pairs = self.current_inventory(current, prefix, pairs)
                base_normalized = {Coverage.surface(form) for form in base_forms}
                added_normalized = {Coverage.surface(form) for form in added_forms}
                novel = added_normalized - base_normalized
                new_stems = added_stems - base_stems
                by_feature = {}
                for form, stem, feature in added_rows:
                    by_feature.setdefault(feature, set()).add(form)
                section = {
                    "current_inventory": base_stats, "huet_inventory": added_stats,
                    "exact_surface_coverage": Coverage.compare(base_forms, added_forms),
                    "normalized_surface_coverage": Coverage.compare(base_normalized, added_normalized),
                    "lemma_coverage": Coverage.compare(base_stems, added_stems),
                    "huet_lemmas_in_dictionary": len(added_stems & dictionary),
                    "new_lemmas_in_dictionary": len(new_stems & dictionary),
                    "form_lemma_pairs": {"huet": len(pairs), "new": len(missing_pairs),
                        "new_on_existing_surface": sum(form in base_normalized for form, stem in missing_pairs)},
                    "new_forms_by_feature_nonadditive": {key: len(value - base_normalized) for key, value in by_feature.items()},
                    "new_forms_with_current_morphology_lemma": len({form for form, stem in pairs if form in novel and stem in base_stems}),
                    "new_forms_with_dictionary_lemma": len({form for form, stem in pairs if form in novel and stem in dictionary}),
                    "nonconventional_huet_surfaces": len({form for form in added_normalized if not Coverage.conventional_iast(form)}),
                    "new_nonconventional_surfaces": sorted(form for form in novel if not Coverage.conventional_iast(form)),
                    "new_examples": sorted((form, stem) for form, stem in pairs if form in novel)[:30],
                    "new_lemma_examples": sorted(new_stems)[:40],
                }
                report[kind] = section
                current_all.update(base_normalized)
                huet_all.update(added_normalized)
                # Keep only compact lookup data needed for the combined result.
                del added_rows, pairs, missing_pairs, base_forms, added_forms, base_normalized, added_normalized
            report["combined_normalized_surface_coverage"] = Coverage.compare(current_all, huet_all)
            report["combined_conventional_iast_surface_coverage"] = Coverage.compare(
                {form for form in current_all if Coverage.conventional_iast(form)},
                {form for form in huet_all if Coverage.conventional_iast(form)},
            )
            report["new_morphology_surfaces_also_dictionary_headwords"] = len((huet_all - current_all) & dictionary)
            # Read the fixture as data rather than importing the package.
            import ast
            fixture = ast.parse((self.config.root / "tests/datasets/yogaSutra.py").read_text())
            sutras = next(ast.literal_eval(node.value) for node in fixture.body
                          if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ys" for t in node.targets))
            tokens = {Coverage.surface(token.strip("'’.,;!?|।॥")) for line in sutras for token in line.split()}
            hits = sorted((tokens & huet_all) - current_all)
            report["yoga_sutra_direct_inventory_probe"] = {"unique_tokens": len(tokens),
                "current_hits": len(tokens & current_all), "union_hits": len(tokens & (current_all | huet_all)),
                "added_hits": hits}
        after = {str(path.relative_to(self.config.root)): self.fingerprint(path) for path in source_paths}
        if after != fingerprints:
            raise RuntimeError("A source changed during the assessment")
        report["source_hashes_unchanged"] = True
        output = self.config.root / self.config.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("combined_normalized_surface_coverage", "yoga_sutra_direct_inventory_probe")}, ensure_ascii=False, indent=2))
        print(f"Full report: {output}")
        return report


if __name__ == "__main__":
    HuetMergeAssessment().run()
