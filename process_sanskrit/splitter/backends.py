"""Private execution backends for the public split-only :class:`Parser`.

Python continues to own transliteration and public result objects. Backends
accept and return canonical SLP1 strings only, which keeps the native boundary
coarse and makes differential testing straightforward.
"""

from __future__ import annotations

import importlib.resources
import os
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

BACKEND_ENVIRONMENT_VARIABLE = "PROCESS_SANSKRIT_SPLITTER_BACKEND"
_ALLOWED_BACKENDS = frozenset(("rust", "python"))
_selected_backend = None
_selection_lock = threading.Lock()
_native_splitter = None
_native_lock = threading.Lock()


class SplitBackend(ABC):
    """Canonical-SLP1 backend contract."""

    name: str

    @abstractmethod
    def split_slp1(
        self, text: str, *, limit: int, scored: bool
    ) -> Optional[List[List[str]]]:
        """Return candidate token paths, or ``None`` when no root exists."""

    @abstractmethod
    def valid_slp1(self, word: str) -> bool:
        """Return whether *word* is in the exhaustive form set."""

    @abstractmethod
    def score_slp1(self, sequences: Sequence[Sequence[str]]) -> List[float]:
        """Score complete token sequences using the DCS model."""


class PythonBackend(SplitBackend):
    """Vendored Python implementation retained as the parity oracle."""

    name = "python"

    def __init__(self):
        from .sandhi_analyzer import LexicalSandhiAnalyzer

        self.analyzer = LexicalSandhiAnalyzer()
        # The analyzer stores its graph and memo table on the instance.
        self._request_lock = threading.RLock()

    def split_slp1(self, text, *, limit, scored):
        from indic_transliteration import sanscript

        from .sanskrit_base import SanskritNormalizedString

        with self._request_lock:
            value = SanskritNormalizedString(
                text, encoding=sanscript.SLP1, strict_io=True
            )
            graph = self.analyzer.getSandhiSplits(value, pre_segmented=False)
            if graph is None:
                return None
            paths = graph.find_all_paths(
                max_paths=limit, sort=True, score=scored
            )
            return [[str(token) for token in path] for path in paths]

    def valid_slp1(self, word):
        return self.analyzer.forms.valid(word)

    def score_slp1(self, sequences):
        from .scorer import Scorer

        sentences = [" ".join(sequence) for sequence in sequences]
        return Scorer.shared().score_strings(sentences)


class RustBackend(SplitBackend):
    """Thread-safe native implementation backed by immutable resources."""

    name = "rust"

    def __init__(self):
        self._native = _get_native_splitter()

    def split_slp1(self, text, *, limit, scored):
        return self._native.split_slp1(text, limit, scored)

    def valid_slp1(self, word):
        return self._native.valid_slp1(word)

    def score_slp1(self, sequences):
        return self._native.score_slp1([list(sequence) for sequence in sequences])


def create_backend() -> SplitBackend:
    """Construct a backend using the process-wide, first-use selection."""
    name = selected_backend_name()
    if name == "python":
        return PythonBackend()
    return RustBackend()


def selected_backend_name() -> str:
    global _selected_backend
    if _selected_backend is None:
        with _selection_lock:
            if _selected_backend is None:
                value = os.environ.get(BACKEND_ENVIRONMENT_VARIABLE, "rust")
                if value not in _ALLOWED_BACKENDS:
                    allowed = ", ".join(sorted(_ALLOWED_BACKENDS))
                    raise ValueError(
                        f"{BACKEND_ENVIRONMENT_VARIABLE} must be one of "
                        f"{allowed}; got {value!r}"
                    )
                _selected_backend = value
    return _selected_backend


def _get_native_splitter():
    global _native_splitter
    if _native_splitter is None:
        with _native_lock:
            if _native_splitter is None:
                try:
                    from ._native import NativeSplitter
                except (ImportError, OSError) as error:
                    raise RuntimeError(
                        "The Rust sandhi splitter extension could not be imported. "
                        "Install a supported process-sanskrit wheel or rebuild the "
                        "package with a Rust and C++ toolchain. To run the retained "
                        "reference implementation explicitly for testing, set "
                        f"{BACKEND_ENVIRONMENT_VARIABLE}=python before importing "
                        "process_sanskrit."
                    ) from error

                data_dir = importlib.resources.files(__package__).joinpath(
                    "data", "native"
                )
                try:
                    _native_splitter = NativeSplitter(str(data_dir))
                except Exception as error:
                    raise RuntimeError(
                        "The Rust sandhi splitter failed to initialize its verified "
                        f"assets from {data_dir}: {error}"
                    ) from error
    return _native_splitter


def _reset_backend_state_for_tests():
    """Reset first-use state; private because production choice is immutable."""
    global _selected_backend, _native_splitter
    with _selection_lock:
        _selected_backend = None
    with _native_lock:
        _native_splitter = None
