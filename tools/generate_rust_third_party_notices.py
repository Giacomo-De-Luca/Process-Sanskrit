#!/usr/bin/env python3
"""Generate the deterministic Cargo third-party notice bundle.

The selected graph is the union of production dependencies for the native
Python extension across Cargo's resolved target-specific edges. Development
dependencies and unrelated workspace members are deliberately excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Set, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md"
ROOT_PACKAGE = "process-sanskrit-python"
NOTICE_FILE_PATTERN = re.compile(
    r"^(?:LICEN[CS]E|COPYING|COPYRIGHT|NOTICE|UNLICENSE)(?:[-_.].*)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DependencyRecord:
    """A third-party Cargo package selected for distribution notices."""

    package_id: str
    name: str
    version: str
    role: str
    license_expression: str
    manifest_path: Path


@dataclass(frozen=True)
class LicenseMaterial:
    """One license/notice source file associated with a Cargo package."""

    dependency: DependencyRecord
    relative_path: str
    text: str
    digest: str


class DependencyGraph:
    """Select the native extension's non-development Cargo graph."""

    def __init__(
        self,
        metadata: Mapping[str, Any],
        root_package: str = ROOT_PACKAGE,
    ) -> None:
        self._packages = {
            package["id"]: package for package in metadata["packages"]
        }
        self._nodes = {
            node["id"]: node for node in metadata["resolve"]["nodes"]
        }
        matching_roots = [
            package["id"]
            for package in metadata["packages"]
            if package["name"] == root_package
        ]
        if len(matching_roots) != 1:
            raise ValueError(
                f"expected exactly one Cargo package named {root_package!r}, "
                f"found {len(matching_roots)}"
            )
        self._root_id = matching_roots[0]

    def production_dependencies(self) -> List[DependencyRecord]:
        """Return registry dependencies, with runtime taking role precedence."""

        modes_by_package: DefaultDict[str, Set[str]] = defaultdict(set)
        queue: deque[Tuple[str, str]] = deque([(self._root_id, "runtime")])

        while queue:
            package_id, incoming_mode = queue.popleft()
            if incoming_mode in modes_by_package[package_id]:
                continue
            modes_by_package[package_id].add(incoming_mode)

            for dependency in self._nodes[package_id].get("deps", []):
                edge_kinds = {
                    dependency_kind.get("kind") or "normal"
                    for dependency_kind in dependency.get("dep_kinds", [])
                }
                edge_kinds.discard("dev")
                for edge_kind in edge_kinds:
                    next_mode = (
                        "runtime"
                        if incoming_mode == "runtime" and edge_kind == "normal"
                        else "build-only"
                    )
                    queue.append((dependency["pkg"], next_mode))

        dependencies: List[DependencyRecord] = []
        for package_id, modes in modes_by_package.items():
            package = self._packages[package_id]
            if package_id == self._root_id or package.get("source") is None:
                continue
            dependencies.append(
                DependencyRecord(
                    package_id=package_id,
                    name=package["name"],
                    version=package["version"],
                    role="runtime" if "runtime" in modes else "build-only",
                    license_expression=package.get("license") or "NOT DECLARED",
                    manifest_path=Path(package["manifest_path"]),
                )
            )

        return sorted(
            dependencies,
            key=lambda dependency: (dependency.name, dependency.version),
        )


class LicenseCollector:
    """Collect the license and notice files shipped in downloaded crates."""

    def __init__(self, packages_by_id: Mapping[str, Mapping[str, Any]]) -> None:
        self._packages_by_id = packages_by_id

    def collect(self, dependencies: Iterable[DependencyRecord]) -> List[LicenseMaterial]:
        materials: List[LicenseMaterial] = []
        for dependency in dependencies:
            package = self._packages_by_id[dependency.package_id]
            crate_root = dependency.manifest_path.parent
            paths = self._notice_paths(crate_root, package.get("license_file"))
            if not paths:
                raise RuntimeError(
                    f"no license or notice files found for "
                    f"{dependency.name} {dependency.version} in {crate_root}"
                )
            for path in paths:
                text = self._read_normalized_text(path)
                materials.append(
                    LicenseMaterial(
                        dependency=dependency,
                        relative_path=path.relative_to(crate_root).as_posix(),
                        text=text,
                        digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    )
                )
        return materials

    @staticmethod
    def _notice_paths(crate_root: Path, license_file: Optional[str]) -> List[Path]:
        paths: Set[Path] = set()
        if license_file:
            paths.add((crate_root / license_file).resolve())

        for path in crate_root.iterdir():
            if path.is_file() and NOTICE_FILE_PATTERN.match(path.name):
                paths.add(path.resolve())

        licenses_directory = crate_root / "LICENSES"
        if licenses_directory.is_dir():
            paths.update(
                path.resolve()
                for path in licenses_directory.rglob("*")
                if path.is_file()
            )

        missing = [path for path in paths if not path.is_file()]
        if missing:
            formatted = ", ".join(str(path) for path in sorted(missing))
            raise RuntimeError(f"declared license files do not exist: {formatted}")
        return sorted(paths, key=lambda path: path.relative_to(crate_root).as_posix())

    @staticmethod
    def _read_normalized_text(path: Path) -> str:
        text = path.read_text(encoding="utf-8-sig")
        return text.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"


class NoticeRenderer:
    """Render dependency inventory and deduplicated full license texts."""

    def __init__(self, cargo_lock_digest: str) -> None:
        self._cargo_lock_digest = cargo_lock_digest

    def render(
        self,
        dependencies: List[DependencyRecord],
        materials: List[LicenseMaterial],
    ) -> str:
        materials_by_package: DefaultDict[str, List[LicenseMaterial]] = defaultdict(list)
        uses_by_digest: DefaultDict[str, List[LicenseMaterial]] = defaultdict(list)
        text_by_digest: Dict[str, str] = {}
        for material in materials:
            materials_by_package[material.dependency.package_id].append(material)
            uses_by_digest[material.digest].append(material)
            previous_text = text_by_digest.setdefault(material.digest, material.text)
            if previous_text != material.text:
                raise RuntimeError(f"SHA-256 collision for license text {material.digest}")

        runtime_count = sum(
            dependency.role == "runtime" for dependency in dependencies
        )
        build_count = len(dependencies) - runtime_count
        lines = [
            "<!-- Generated by tools/generate_rust_third_party_notices.py; do not edit. -->",
            "",
            "# Rust third-party notices",
            "",
            "This bundle covers the third-party Cargo packages used to compile the",
            "Process-Sanskrit native Python extension. It contains the declared",
            "license expression and complete license/notice source files shipped by",
            "each selected crate.",
            "",
            f"Cargo.lock SHA-256: `{self._cargo_lock_digest}`",
            "",
            "## Scope and regeneration",
            "",
            f"The inventory contains **{len(dependencies)} packages**: "
            f"{runtime_count} runtime or code-generation dependencies and "
            f"{build_count} build-only dependencies. It is the union of all "
            "target-specific normal and build edges reachable from "
            f"`{ROOT_PACKAGE}` in Cargo metadata. Development dependencies, "
            "local Process-Sanskrit workspace crates, and the unrelated resource-builder "
            "graph are excluded. When one package is reachable in both roles, it is "
            "reported as runtime.",
            "",
            "Regenerate after every `Cargo.lock` dependency change, once the locked "
            "crate sources are present in Cargo's local cache:",
            "",
            "```console",
            "cargo fetch --locked",
            "uv run --no-project python tools/generate_rust_third_party_notices.py",
            "uv run --no-project python tools/generate_rust_third_party_notices.py --check",
            "```",
            "",
            "Generation itself invokes `cargo metadata --locked --offline`; it does "
            "not contact the network. The checked-in lockfile digest provides a cheap "
            "test for stale output, while `--check` re-resolves the complete graph and "
            "license texts from the local Cargo cache.",
            "",
            "The separately vendored SentencePiece C++ components and the Python "
            "reference splitter are covered by "
            "`process_sanskrit/splitter/NOTICE.md` and the adjacent `LICENSE.*` files; "
            "they are intentionally not duplicated here.",
            "",
            "## Dependency inventory",
            "",
            "| Crate | Version | Build role | Cargo license expression | Included texts |",
            "|---|---:|---|---|---|",
        ]

        for dependency in dependencies:
            references = ", ".join(
                f"`L-{material.digest[:12]}` (`{material.relative_path}`)"
                for material in materials_by_package[dependency.package_id]
            )
            expression = dependency.license_expression.replace("|", "\\|")
            lines.append(
                f"| `{dependency.name}` | `{dependency.version}` | "
                f"{dependency.role} | `{expression}` | {references} |"
            )

        lines.extend(["", "## License and notice texts", ""])
        for digest in sorted(text_by_digest):
            uses = sorted(
                {
                    (
                        material.dependency.name,
                        material.dependency.version,
                        material.relative_path,
                    )
                    for material in uses_by_digest[digest]
                }
            )
            use_list = ", ".join(
                f"`{name} {version}` (`{relative_path}`)"
                for name, version, relative_path in uses
            )
            text = text_by_digest[digest]
            fence = self._markdown_fence(text)
            lines.extend(
                [
                    f"### `L-{digest[:12]}`",
                    "",
                    f"- SHA-256: `{digest}`",
                    f"- Used by: {use_list}",
                    "",
                    f"{fence}text",
                    text.rstrip("\n"),
                    fence,
                    "",
                ]
            )

        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _markdown_fence(text: str) -> str:
        longest_run = max((len(run) for run in re.findall(r"`+", text)), default=0)
        return "`" * max(3, longest_run + 1)


class NoticeApplication:
    """Coordinate metadata loading, generation, and stale-file checks."""

    def __init__(self, repository_root: Path = REPOSITORY_ROOT) -> None:
        self._repository_root = repository_root

    def generate(self) -> str:
        metadata = self._cargo_metadata()
        dependencies = DependencyGraph(metadata).production_dependencies()
        packages_by_id = {
            package["id"]: package for package in metadata["packages"]
        }
        materials = LicenseCollector(packages_by_id).collect(dependencies)
        lock_digest = hashlib.sha256(
            (self._repository_root / "Cargo.lock").read_bytes()
        ).hexdigest()
        return NoticeRenderer(lock_digest).render(dependencies, materials)

    def _cargo_metadata(self) -> Mapping[str, Any]:
        command = [
            "cargo",
            "metadata",
            "--locked",
            "--offline",
            "--format-version",
            "1",
        ]
        completed = subprocess.run(
            command,
            cwd=self._repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked notice differs from freshly generated output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output path (default: {DEFAULT_OUTPUT.relative_to(REPOSITORY_ROOT)})",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    generated = NoticeApplication().generate()
    output = arguments.output.resolve()

    if arguments.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != generated:
            print(
                f"{output} is stale; regenerate it with "
                "tools/generate_rust_third_party_notices.py",
                file=sys.stderr,
            )
            return 1
        print(f"{output} is current")
        return 0

    with output.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(generated)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
