"""Decide whether the version in pyproject.toml still needs publishing to PyPI.

PyPI refuses to overwrite an existing version, so the publish workflow asks this
script first and skips the build entirely when the version is already released.

Writes `version` and `should_publish` to $GITHUB_OUTPUT for downstream jobs.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
REQUEST_TIMEOUT_SECONDS = 30

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_project(pyproject: Path) -> tuple[str, str]:
    """Return (name, version) declared in pyproject.toml."""
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    return project["name"], project["version"]


def published_state(name: str) -> tuple[set[str], str]:
    """Return (every released version, PyPI's own "latest") — empty if never published.

    `latest` comes from PyPI rather than max() because version strings do not sort
    lexicographically: "1.0.9" would outrank "1.0.27".
    """
    request = urllib.request.Request(
        PYPI_JSON_URL.format(name=name),
        headers={"User-Agent": "process-sanskrit-release-check"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return set(), "none"
        raise
    return set(payload.get("releases", {})), payload["info"]["version"]


def emit_outputs(**outputs: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output is None:
        return
    with open(github_output, "a", encoding="utf-8") as handle:
        for key, value in outputs.items():
            handle.write(f"{key}={value}\n")


def main() -> int:
    name, version = read_project(REPO_ROOT / "pyproject.toml")
    released, latest = published_state(name)
    should_publish = version not in released

    if should_publish:
        print(f"{name} {version} is not on PyPI (latest there: {latest}) -> publishing.")
    else:
        print(f"{name} {version} is already on PyPI -> nothing to do.")
        print("Bump the version in pyproject.toml to cut a new release.")

    emit_outputs(version=version, should_publish=str(should_publish).lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
