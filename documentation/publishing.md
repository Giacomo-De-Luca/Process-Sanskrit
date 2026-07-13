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

## The jobs

1. **check-version** — compares `pyproject.toml` against PyPI; everything below is skipped if
   the version is already released.
2. **build** — `python -m build`, then `twine check`.
3. **smoke-test** — installs the built wheel on Python 3.9 (the floor declared in
   `pyproject.toml`) and 3.12, and asserts `transliterate('rāmaḥ', 'devanagari') == 'रामः'`.
   This is the only gate standing in for the test suite, so two details are load-bearing:

   - It runs with **no database present**. `transliterate` is the only database-free entry
     point, so this proves the wheel imports and its packaged resources (`forms.trie`, the
     resource JSON) resolve on a bare machine — a broken `package-data` glob fails here rather
     than on a user's `pip install`.
   - The job **deliberately does not check the repo out**, and asserts `site-packages` is in
     `process_sanskrit.__file__`. `python -c` puts the working directory on `sys.path`, so with
     a checkout present `import process_sanskrit` silently resolves to the *source tree* and the
     test passes even if the wheel is empty or not installed at all. No checkout, no shadowing.

4. **publish** — uploads via Trusted Publishing (see below).
5. **tag** — pushes an annotated `v<version>` tag once the upload succeeds. Tags therefore
   follow the release rather than triggering it; they cannot drift from what is on PyPI. The
   step is a no-op if the tag already exists, so a re-run cannot redden an already-good release.

## One-time setup

**Both of these are already configured** — recorded here because the publish job fails without
them, so they must be recreated if the project is ever moved, renamed, or forked.

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
environment*, named `pypi`. It needs no secrets, but set one protection rule:

- **Deployment branches: selected branches → `main`.** PyPI's Trusted Publisher binds
  owner/repo/workflow/environment — *not* the branch. Without this rule, anyone who can run a
  `workflow_dispatch` could publish from any branch, and PyPI would happily mint a token for it.
  The environment is the real enforcement point, because the publish job is what it gates.

  Use *selected branches* naming `main` explicitly, not the *protected branches* option: the
  latter matches only branches that carry a branch-protection rule, so if `main` has none it
  matches nothing and blocks every deploy. Equivalently, from the CLI:

  ```bash
  gh api --method PUT repos/OWNER/REPO/environments/pypi \
    -F 'deployment_branch_policy[protected_branches]=false' \
    -F 'deployment_branch_policy[custom_branch_policies]=true'
  gh api --method POST repos/OWNER/REPO/environments/pypi/deployment-branch-policies \
    -f name=main -f type=branch
  ```
- **Required reviewer: `Giacomo-De-Luca`.** The publish job pauses and waits for an approval
  click before uploading. A version number spent on PyPI is spent forever — it cannot be
  re-uploaded even after deleting the release — so the gate exists to catch an accidental bump
  before it becomes permanent.

## Approving a release

Because of that reviewer rule, a push to `main` with a bumped version does not publish on its
own. It builds, smoke-tests, and then **waits**: GitHub notifies you and the run sits pending
until you open it and click *Review deployments → Approve and deploy*. Check the version it is
about to ship, then approve. Rejecting it leaves nothing on PyPI.

## When something goes wrong

When retrying, prefer **Re-run failed jobs** over **Re-run all jobs**. Both work, but they differ
in one case that matters: if the upload succeeded and only the `tag` job failed (a tag ruleset
blocking `github-actions[bot]`, say), *Re-run failed jobs* re-runs `tag` with `check-version`'s
original outputs intact. Any other route — a fresh push, a `workflow_dispatch` — is a dead end:
`check-version` now sees the version on PyPI, `should-publish` flips to `false`, and the tag job
is skipped forever. The version would ship untagged with no way for the workflow to fix it. (If
that happens, just `git tag -a v<version> && git push origin v<version>` by hand.)

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
