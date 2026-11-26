"""Data read from a database once per SQLAlchemy engine, dropped with the engine.

Lemma mode keeps four things that depend only on the database:
`CompoundVerbIndex` and `CompoundForms` read one table each, and `PieceLemmas`
and `LemmaLexicon` (its members) fill a dict word by word.  Each holds one
`EngineCache` as a class attribute.

The key is the engine itself, held weakly: every in-memory database has the URL
`sqlite://`, so a URL key would let two test databases share one table, and a
dropped engine must not keep its data alive.  `EngineCache.clear_all()` empties
every instance; `dictionary_references._reset_reference_state` calls it when the
configured database changes.
"""

import threading
import weakref
from typing import Callable, Dict, TypeVar

Value = TypeVar("Value")

## marks an engine with no value yet, so that a `read()` may return None
_MISSING = object()


class EngineCache:
    """One value per database engine, made at most once under a lock."""

    _instances: "weakref.WeakSet[EngineCache]" = weakref.WeakSet()
    _instances_lock = threading.Lock()

    def __init__(self):
        self._entries: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()
        with self._instances_lock:
            self._instances.add(self)

    def table(self, session, read: Callable[[], Value]) -> Value:
        """The value for `session`'s engine, made by `read()` the first time it is asked for."""
        engine = session.get_bind()
        value = self._entries.get(engine, _MISSING)
        if value is _MISSING:
            with self._lock:
                value = self._entries.get(engine, _MISSING)
                if value is _MISSING:
                    value = self._entries[engine] = read()
        return value

    def memo(self, session) -> Dict:
        """A dict for `session`'s engine that the caller fills itself."""
        return self.table(session, dict)

    def clear(self) -> None:
        """Forget every engine's value; the next request makes it again."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """How many engines have a value."""
        return len(self._entries)

    @classmethod
    def clear_all(cls) -> None:
        """Clear every `EngineCache` there is."""
        with cls._instances_lock:
            caches = list(cls._instances)
        for cache in caches:
            cache.clear()


__all__ = ["EngineCache"]
