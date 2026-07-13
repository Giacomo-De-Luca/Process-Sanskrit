"""Validity oracle: is a given string a real Sanskrit word form?

Sandhi splitting only ever asks *is this a word* -- it never asks *what are its
tags*.  Upstream answers the first question by way of the second: it keeps two
sqlite databases and a 33 MB pickle of stems and tags, reaches them through
sqlalchemy and the sanskrit_util ORM, and runs a generative stem analysis on
every miss.  All of that machinery exists to serve ``get_tags()``, which the
splitter calls only when tagging is requested.  Process-Sanskrit never requests
it -- morphology is handled by ``process_sanskrit.functions.inflect``.

So we precompute the accept set once (tools/build_splitter_data.py) and answer
from a marisa-trie: 78 MB and three packages become 13 MB and one.

The trie is exhaustive, not approximate.  tests/test_splitter_parity.py asserts
it agrees with upstream's CombinedWrapper.valid() on every query a real corpus
generates.
"""

import threading

import marisa_trie

from .data_manager import data_file_path

_trie = None
_trie_lock = threading.Lock()


def _forms():
    global _trie
    if _trie is None:
        with _trie_lock:
            if _trie is None:
                trie = marisa_trie.Trie()
                trie.load(data_file_path("forms.trie"))
                _trie = trie
    return _trie


class TrieLookup:
    """Drop-in replacement for upstream's CombinedWrapper (valid() only)."""

    def valid(self, word: str) -> bool:
        return word in _forms()
