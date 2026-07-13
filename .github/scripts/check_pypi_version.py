"""Decide whether the version in pyproject.toml still needs publishing to PyPI.

PyPI refuses to overwrite an existing version, so the publish workflow asks this
script first and skips the build entirely when the version is already released.

Writes `version` and `should_publish` to $GITHUB_OUTPUT for downstream jobs.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
REQUEST_TIMEOUT_SECONDS = 30
RETRY_DELAYS_SECONDS = (2, 8)  # a PyPI blip must not redden an unrelated push

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_project(pyproject: Path) -> tuple[str, str]:
    """Return (name, version) declared in pyproject.toml."""
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    return project["name"], project["version"]


def fetch_pypi(name: str) -> dict | None:
    """Fetch PyPI's JSON for `name`, or None if the project was never published.

    Retries transient failures (5xx, DNS/TLS blips) so that a PyPI hiccup does not
    fail an ordinary push. A genuine 404 is an answer, not a failure.
    """
    request = urllib.request.Request(
        PYPI_JSON_URL.format(name=name),
        headers={"User-Agent": "process-sanskrit-release-check"},
    )
    for attempt, delay in enumerate((*RETRY_DELAYS_SECONDS, None), start=1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.code < 500 or delay is None:
                raise
            reason = f"HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError) as error:
            if delay is None:
                raise
            reason = str(error)
        print(f"PyPI request failed ({reason}); retrying in {delay}s [attempt {attempt}]")
        time.sleep(delay)
    raise AssertionError("unreachable")


def published_state(name: str) -> tuple[set[str], str]:
    """Return (every released version, PyPI's own "latest") — empty if never published.

    `latest` comes from PyPI rather than max() because version strings do not sort
    lexicographically: "1.0.9" would outrank "1.0.27".

    Caveat: a *deleted* release disappears from `releases` but PyPI still reserves
    its filenames forever, so re-publishing a deleted version is rejected at upload.
    That case fails loudly at the twine step rather than silently shipping nothing.
    """
    payload = fetch_pypi(name)
    if payload is None:
        return set(), "none"
    return set(payload.get("releases", {})), payload["info"]["version"]


def emit_outputs(**outputs: str) -> None:
    """Hand the decision to the workflow. Absent $GITHUB_OUTPUT is fine locally, but
    under CI it would skip every downstream job and report success while publishing
    nothing — the worst failure mode a release pipeline has. So: fail loudly."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output is None:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            raise RuntimeError("$GITHUB_OUTPUT is unset; refusing to silently skip the release")
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
