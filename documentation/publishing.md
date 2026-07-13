# Publishing to PyPI

Releases are automated by [`.github/workflows/publish.yml`](../.github/workflows/publish.yml).
Bump the version in `pyproject.toml`, push to `main`, and the new version appears on PyPI.

## How a release is decided

PyPI **refuses to overwrite an existing version** — the same version number can never be
uploaded twice, even if the release is later deleted. So the workflow cannot simply publish on
every push; it has to ask whether there is anything new to publish.

That is what [`.github/scripts/check_pypi_version.py`](../.github/scripts/check_pypi_version.py)
does. It reads the version from `pyproject.toml`, asks PyPI which versions already exist, and
emits `should_publish` accordingly. The rest of the workflow is gated on it:

| Push to `main`                            | Result                                     |
| ----------------------------------------- | ------------------------------------------ |
| version in `pyproject.toml` is new to PyPI | build → smoke-test → publish → tag         |
| version already on PyPI                    | workflow ends green, nothing is uploaded   |

So a docs fix or a bugfix push is a no-op, and the release step is the version bump itself.

## The four jobs

1. **check-version** — compares `pyproject.toml` against PyPI; everything below is skipped if
   the version is already released.
2. **build** — `python -m build`, then `twine check`, then installs the wheel into a clean venv
   and asserts `transliterate('rāmaḥ', 'devanagari') == 'रामः'`. That smoke test runs with **no
   database present**, which is the point: `transliterate` is the only database-free entry point,
   so it verifies the wheel imports and its packaged resources resolve on a bare machine. A
   broken wheel fails here rather than on a user's `pip install`.
3. **publish** — uploads via Trusted Publishing (see below).
4. **tag** — pushes an annotated `v<version>` tag once the upload succeeds. Tags therefore
   follow the release rather than triggering it; they cannot drift from what is on PyPI.

## One-time setup

Neither of these exists yet — the workflow will fail at the publish step until both are done.

**1. Register the Trusted Publisher on PyPI.** Go to the project's
[publishing settings](https://pypi.org/manage/project/process-sanskrit/settings/publishing/)
and add a GitHub publisher with exactly these values:

| Field           | Value                |
| --------------- | -------------------- |
| Owner           | `Giacomo-De-Luca`    |
| Repository name | `Process-Sanskrit`   |
| Workflow name   | `publish.yml`        |
| Environment     | `pypi`               |

**2. Create the `pypi` environment on GitHub.** Repository *Settings → Environments → New
environment*, named `pypi`. It needs no secrets. Adding yourself as a *required reviewer* is
worth considering: the run then pauses before the upload and waits for your click, which gives
you a chance to stop an accidental bump.

## Why Trusted Publishing and not an API token

Trusted Publishing uses OIDC: GitHub proves to PyPI that this specific workflow, in this
specific repository, is running, and PyPI mints a token valid for that single upload. There is
no long-lived credential in the repository, nothing to rotate, and nothing that can leak from a
secrets store. It is the method PyPI recommends for GitHub Actions.

The alternative — a `PYPI_API_TOKEN` in GitHub Secrets — works, but it is a standing credential
that can publish to your project from anywhere, forever, until you notice and revoke it.

## What changes when the Rust extension lands

Today the package is pure Python, so a single `python -m build` on Linux produces one universal
`py3-none-any` wheel that installs everywhere. A compiled extension breaks that assumption: a
wheel is then specific to a platform and interpreter, and a Linux-only wheel would leave macOS
and Windows users building from source (or failing outright).

At that point the **build** job — and only the build job — has to become a
[cibuildwheel](https://cibuildwheel.pypa.io/) matrix producing one wheel per platform, plus a
separate job for the sdist, with all artifacts collected into `dist/` before the publish job
runs. The `build-backend` in `pyproject.toml` changes too (`maturin` or `setuptools-rust`).
The **publish** job needs no changes at all: it just uploads whatever lands in `dist/`. That
separation is deliberate — the auth half of the pipeline should not have to be re-touched.

## Cutting a release by hand

The workflow does nothing you cannot do locally:

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

One gotcha: `python -m build` fails in a working tree that has a `build/` directory, because it
shadows the `build` module. CI never hits this (fresh checkout), but locally you may need to
`rm -rf build/` first.
