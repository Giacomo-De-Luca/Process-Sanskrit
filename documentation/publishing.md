# Publishing to PyPI

Releases are automated by [`.github/workflows/publish.yml`](../.github/workflows/publish.yml).
Bump the same version in `pyproject.toml` and `[workspace.package]` in `Cargo.toml`, refresh
`Cargo.lock` and the generated Rust notices, then push to `main`.

## How a release is decided

PyPI **refuses to overwrite an existing version** — the same version number can never be
uploaded twice, even if the release is later deleted. So the workflow cannot simply publish on
every push; it has to ask whether there is anything new to publish.

That is what [`.github/scripts/check_pypi_version.py`](../.github/scripts/check_pypi_version.py)
does. It reads the version from `pyproject.toml`, asks PyPI which versions already exist, and
emits `should_publish` accordingly. Before contacting PyPI it also requires the Python version,
the Cargo workspace version, and all three local package versions in `Cargo.lock` to match. The
rest of the workflow is gated on it:

| Push to `main`                            | Result                                     |
| ----------------------------------------- | ------------------------------------------ |
| version in `pyproject.toml` is new to PyPI | build → smoke-test → collect → publish → tag |
| version already on PyPI                    | workflow ends green, nothing is uploaded   |

So a docs fix or a bugfix push is a no-op, and the release step is the version bump itself.

## Release jobs and gates

1. **check-version** compares `pyproject.toml` against PyPI; everything below is skipped if the
   version is already released.
2. **build-wheels** calls the same reusable [wheel workflow](../.github/workflows/wheels.yml)
   used for pull-request validation. `cibuildwheel` produces exactly four `cp39-abi3` wheels:
   manylinux x86-64, macOS x86-64, macOS arm64, and Windows x86-64. Calling the same workflow
   keeps validation and publication on one platform matrix; the underlying `cibuildwheel`
   configuration lives in `pyproject.toml`.
3. **build-sdist** produces one source archive, runs `twine check`, verifies that the Cargo
   workspace, lockfile, native binding manifest, third-party notices, and vendored
   SentencePiece license are present, then builds and smoke-tests a `cp39-abi3` native wheel
   from that archive outside the source tree.
4. **collect-distributions** runs only after the complete wheel smoke matrix and the sdist job
   have succeeded. It requires exactly four wheels and one sdist, checks all five with `twine`,
   and creates the sole `release-dist` artifact consumed by publishing.
5. **publish** uploads that single checked artifact set via Trusted Publishing (see below).
6. **tag** pushes an annotated `v<version>` tag once the upload succeeds. Tags therefore follow
   the release rather than triggering it; they cannot drift from what is on PyPI. A retry is a
   no-op only when the existing lightweight or annotated tag resolves to the release commit; an
   existing tag on any other commit fails the job instead of concealing drift.

Each wheel is installed outside the checkout and tested on Python 3.9, 3.12, and 3.14 on its
native operating system. The smoke gate checks the `cp39-abi3` extension, release build profile,
native resource schema and SentencePiece version, PEP 639 license metadata and packaged notices,
a canonical sandhi split, the public database-free `transliterate` API, an assertion that imports
resolve from `site-packages`, and the installed-wheel backend contract suite. Linux/Python 3.12
also runs the complete complex-compound Python/Rust differential. None of these checks needs the
583 MB lexical SQLite database.

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

## Native distribution coverage

The Rust extension uses Python's stable ABI from Python 3.9 onward. One wheel per operating
system and CPU architecture therefore covers every supported CPython version; separate wheels
for Python 3.10, 3.11, and so on are neither needed nor published. musllinux, Linux arm64, Windows
arm64, macOS universal2, PyPy, and 32-bit platforms are not currently built. Users on an
unsupported platform receive the sdist and need a compatible Rust/C++ build toolchain.

The standalone wheel workflow remains useful as a pre-release gate on pushes and pull requests.
When called by the publish workflow, its short-lived `wheel-*` artifacts remain internal build
inputs. Only the collected `release-dist` artifact is passed to the Trusted Publishing job.

## Cutting a release by hand

Prepare one synchronized version bump before building:

```bash
# Edit project.version in pyproject.toml and workspace.package.version in Cargo.toml.
cargo check --workspace
uv run python tools/generate_rust_third_party_notices.py
cargo check --workspace --locked
uv run python tools/generate_rust_third_party_notices.py --check
```

The first Cargo check refreshes the three local package versions in `Cargo.lock`; regenerating
the notices records that lockfile's new SHA-256. The locked check and notice check are the
reproducibility gates. The publish workflow refuses to start if the declared or locked versions
diverge.

The source archive can be reproduced and checked locally:

```bash
uv build --sdist --out-dir dist
uvx twine check dist/*
```

Build a native wheel with `uvx cibuildwheel --output-dir wheelhouse`; it produces only the wheel
for the host platform unless the required cross-build environment is configured. A real release
must contain the four CI-built wheels above, so do not replace the workflow's collected artifacts
with a wheel from one developer machine. If a manual upload is ever required, download the
workflow's `release-dist` artifact, run `twine check` over all five files, and upload that directory
as one set.
