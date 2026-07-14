"""Static release-workflow contracts for native distributions."""

from __future__ import annotations

import runpy
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
WHEEL_WORKFLOW = ROOT / ".github" / "workflows" / "wheels.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
MANIFEST = ROOT / "MANIFEST.in"
VERSION_CHECK = ROOT / ".github" / "scripts" / "check_pypi_version.py"


class NativePublishWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.publish = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
        cls.wheels = WHEEL_WORKFLOW.read_text(encoding="utf-8")
        cls.ci = CI_WORKFLOW.read_text(encoding="utf-8")
        cls.manifest = MANIFEST.read_text(encoding="utf-8")

    def test_reuses_validated_four_platform_wheel_workflow(self):
        self.assertRegex(self.wheels, r"(?m)^  workflow_call:$")
        self.assertIn("uses: ./.github/workflows/wheels.yml", self.publish)
        for artifact in (
            "linux-x86-64",
            "macos-x86-64",
            "macos-arm64",
            "windows-x86-64",
        ):
            self.assertIn(f"artifact: {artifact}", self.wheels)

    def test_native_wheels_have_abi_license_and_backend_smoke_gates(self):
        self.assertIn("*-cp39-abi3-*", self.wheels)
        self.assertIn("License-Expression", self.wheels)
        self.assertIn("Run installed-wheel backend contracts", self.wheels)
        self.assertIn("NativeBackendTests.test_all_complex_compounds_match", self.wheels)
        self.assertIn("site-packages", self.wheels)
        self.assertIn("ps.transliterate", self.wheels)

    def test_wheel_smoke_python_is_shell_literal(self):
        """Markdown backticks in Python must never reach Bash interpolation."""
        self.assertIn("python - <<'PY'", self.wheels)
        self.assertNotRegex(
            self.wheels,
            r'(?s)Import native extension and split canonical input.*?python -c\s+"',
        )

    def test_wheel_inputs_and_static_contracts_trigger_validation(self):
        for path in (
            "tests/test_splitter_backends.py",
            "tests/datasets/sanskrit_compounds_benchmark.json",
            "tests/datasets/sanskrit_compounds_benchmark2.json",
        ):
            self.assertGreaterEqual(self.wheels.count(path), 2)
        self.assertIn("tests.test_publish_workflow", self.ci)

    def test_publish_collects_one_sdist_and_all_wheels_after_validation(self):
        self.assertRegex(
            self.publish,
            r"(?s)build-sdist:.*?python -m build --sdist.*?name: sdist",
        )
        self.assertIn(
            "python -m pip wheel --no-deps --wheel-dir sdist-smoke-wheel",
            self.publish,
        )
        self.assertIn("Smoke-test wheel rebuilt from sdist", self.publish)
        self.assertIn("_native.BUILD_PROFILE == 'release'", self.publish)
        self.assertIn("ps.transliterate", self.publish)
        self.assertRegex(
            self.publish,
            r"(?s)collect-distributions:.*?needs: \[check-version, build-wheels, build-sdist\]",
        )
        self.assertIn("pattern: wheel-*", self.publish)
        self.assertIn("name: release-dist", self.publish)
        self.assertGreaterEqual(self.publish.count("overwrite: true"), 2)
        self.assertIn("overwrite: true", self.wheels)
        self.assertRegex(
            self.publish,
            r"(?s)publish:.*?needs: \[check-version, collect-distributions\]",
        )

    def test_sdist_retains_benchmark_documentation_and_rebuild_tools(self):
        for directive in (
            "recursive-include benchmarks *.json",
            "recursive-include documentation *.md",
            "recursive-include scripts *.md *.py",
            "recursive-include tools *.py",
        ):
            self.assertIn(directive, self.manifest)

        self.assertRegex(
            self.manifest,
            r"recursive-include process_sanskrit/splitter/data .*\*\.npy",
        )

        for required_suffix in (
            "benchmarks/splitter-benchmark.json",
            "documentation/rust-splitter.md",
            "process_sanskrit/splitter/data/log-table.npy",
            "scripts/benchmark_splitter.py",
            "tools/build_splitter_data.py",
        ):
            self.assertIn(required_suffix, self.publish)

    def test_sdist_retains_files_read_by_its_packaging_tests(self):
        for path in (
            ".github/workflows/publish.yml",
            ".github/workflows/wheels.yml",
            ".github/workflows/ci.yml",
            ".github/scripts/check_pypi_version.py",
        ):
            self.assertIn(f"include {path}", self.manifest)

    def test_version_gate_trusted_publishing_and_remote_tag_guard_remain(self):
        self.assertGreaterEqual(
            self.publish.count(
                "if: needs.check-version.outputs.should-publish == 'true'"
            ),
            3,
        )
        self.assertIn("id-token: write", self.publish)
        self.assertIn("pypa/gh-action-pypi-publish@release/v1", self.publish)
        self.assertIn("git ls-remote --exit-code --tags origin", self.publish)
        self.assertIn('"refs/tags/v${VERSION}"', self.publish)
        self.assertIn("EXPECTED_COMMIT: ${{ github.sha }}", self.publish)
        self.assertIn('"refs/tags/v${VERSION}^{}"', self.publish)
        self.assertRegex(
            self.publish,
            r'if \[\[ "\$\{remote_commit\}" == "\$\{EXPECTED_COMMIT\}" \]\]; then',
        )
        self.assertIn(
            'Tag v${VERSION} points to ${remote_commit}, expected ${EXPECTED_COMMIT}.',
            self.publish,
        )

    def test_version_gate_rejects_python_rust_and_lockfile_drift(self):
        read_release_project = runpy.run_path(str(VERSION_CHECK))[
            "read_release_project"
        ]
        project_name, project_version = read_release_project(
            ROOT / "pyproject.toml",
            ROOT / "Cargo.toml",
            ROOT / "Cargo.lock",
        )
        self.assertEqual(project_name, "process-sanskrit")
        self.assertRegex(project_version, r"^\d+\.\d+\.\d+")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pyproject = root / "pyproject.toml"
            cargo = root / "Cargo.toml"
            lock = root / "Cargo.lock"
            pyproject.write_text(
                '[project]\nname = "process-sanskrit"\nversion = "2.0.0"\n',
                encoding="utf-8",
            )
            cargo.write_text(
                '[workspace.package]\nversion = "2.0.1"\n',
                encoding="utf-8",
            )
            lock.write_text(
                'version = 4\n[[package]]\nname = "process-sanskrit-python"\nversion = "2.0.0"\n'
                '[[package]]\nname = "process-sanskrit-resource-builder"\nversion = "2.0.0"\n'
                '[[package]]\nname = "process-sanskrit-splitter-core"\nversion = "2.0.0"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                read_release_project(pyproject, cargo, lock)

            cargo.write_text(
                '[workspace.package]\nversion = "2.0.0"\n',
                encoding="utf-8",
            )
            lock.write_text(lock.read_text(encoding="utf-8").replace(
                'name = "process-sanskrit-splitter-core"\nversion = "2.0.0"',
                'name = "process-sanskrit-splitter-core"\nversion = "1.9.9"',
            ), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Cargo.lock"):
                read_release_project(pyproject, cargo, lock)


if __name__ == "__main__":
    unittest.main()
