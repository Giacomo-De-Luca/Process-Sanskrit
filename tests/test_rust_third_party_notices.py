"""Contracts for the Cargo-derived third-party notice bundle."""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path
from typing import Optional

from tools.generate_rust_third_party_notices import DependencyGraph, cargo_lock_digest


ROOT = Path(__file__).resolve().parents[1]
NOTICE_PATH = ROOT / "THIRD_PARTY_NOTICES.md"


class DependencyGraphTests(unittest.TestCase):
    def test_selects_extension_production_graph_and_classifies_build_only(self):
        def package(
            name: str, source: Optional[str] = "registry"
        ) -> dict[str, object]:
            return {
                "id": name,
                "name": name,
                "version": "1.0.0",
                "source": source,
                "license": "MIT",
                "manifest_path": f"/{name}/Cargo.toml",
            }

        def edge(package_id: str, kind: Optional[str]) -> dict[str, object]:
            return {
                "pkg": package_id,
                "dep_kinds": [{"kind": kind, "target": None}],
            }

        packages = [
            package("process-sanskrit-python", None),
            package("splitter-core", None),
            package("runtime"),
            package("runtime-transitive"),
            package("build-tool"),
            package("build-helper"),
            package("shared"),
            package("dev-only"),
            package("unrelated-workspace", None),
            package("unrelated-dependency"),
        ]
        nodes = [
            {
                "id": "process-sanskrit-python",
                "deps": [
                    edge("splitter-core", None),
                    edge("runtime", None),
                    edge("build-tool", "build"),
                    edge("dev-only", "dev"),
                ],
            },
            {
                "id": "splitter-core",
                "deps": [edge("shared", None)],
            },
            {
                "id": "runtime",
                "deps": [
                    edge("runtime-transitive", None),
                    edge("shared", None),
                ],
            },
            {"id": "runtime-transitive", "deps": []},
            {
                "id": "build-tool",
                "deps": [
                    edge("build-helper", None),
                    edge("shared", None),
                ],
            },
            {"id": "build-helper", "deps": []},
            {"id": "shared", "deps": []},
            {"id": "dev-only", "deps": []},
            {
                "id": "unrelated-workspace",
                "deps": [edge("unrelated-dependency", None)],
            },
            {"id": "unrelated-dependency", "deps": []},
        ]
        metadata = {
            "packages": packages,
            "resolve": {"nodes": nodes},
        }

        dependencies = DependencyGraph(metadata).production_dependencies()

        self.assertEqual(
            [(dependency.name, dependency.role) for dependency in dependencies],
            [
                ("build-helper", "build-only"),
                ("build-tool", "build-only"),
                ("runtime", "runtime"),
                ("runtime-transitive", "runtime"),
                ("shared", "runtime"),
            ],
        )


class CheckedNoticeTests(unittest.TestCase):
    def test_notice_tracks_lockfile_and_expected_extension_dependencies(self):
        notice = NOTICE_PATH.read_text(encoding="utf-8")
        lock_digest = cargo_lock_digest(ROOT)

        self.assertIn(f"Cargo.lock SHA-256: `{lock_digest}`", notice)
        for dependency in ("fst", "pyo3", "serde", "sha2", "cc"):
            self.assertRegex(notice, rf"(?m)^\| `{re.escape(dependency)}` \|")
        for non_extension_dependency in ("ndarray", "tempfile"):
            self.assertNotRegex(
                notice,
                rf"(?m)^\| `{re.escape(non_extension_dependency)}` \|",
            )
        self.assertIn("Apache License", notice)
        self.assertIn("MIT License", notice)

    def test_distribution_metadata_declares_notice(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertIn('"THIRD_PARTY_NOTICES.md"', pyproject)
        self.assertRegex(manifest, r"(?m)^include THIRD_PARTY_NOTICES\.md$")
        self.assertRegex(
            manifest,
            r"(?m)^include tools/generate_rust_third_party_notices\.py$",
        )


if __name__ == "__main__":
    unittest.main()
