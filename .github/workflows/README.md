# Native build workflows

- `ci.yml` runs the Rust workspace on both the declared Rust 1.87 MSRV and the
  current stable toolchain. Stable also owns formatting and Clippy checks. The
  retained Python splitter is tested explicitly on Python 3.9, 3.12, and 3.14.
- `wheels.yml` builds one `cp39-abi3` wheel for each of Linux x86-64, macOS
  x86-64, macOS arm64, and Windows x86-64. Each artifact is installed and smoke
  tested on Python 3.9, 3.12, and 3.14, including the installed-wheel backend
  contract suite. Linux/Python 3.12 also runs the complete complex-compound
  Python/Rust differential. The workflow enforces the 50 MB compressed-wheel
  release ceiling.

The wheel workflow uploads short-lived artifacts and never publishes by itself.
`publish.yml` calls it as a reusable workflow after the PyPI version gate, adds
one checked sdist, and collects the four wheels plus sdist into one release
artifact only after every native smoke job succeeds.
