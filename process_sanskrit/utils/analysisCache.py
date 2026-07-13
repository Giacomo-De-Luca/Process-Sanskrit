"""Persistent cache for expensive sandhi and morphology analyses.

The cache deliberately uses its own writable SQLite database, SQLAlchemy Core
metadata, and engine.  It must never share tables, transactions, or lifecycle
state with the downloaded read-only lexicon.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    cast,
    create_engine,
    delete,
    event,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import DisconnectionError, SQLAlchemyError
from sqlalchemy.pool import QueuePool

from process_sanskrit.utils.resourcePaths import resolve_configured_path


log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
# Bump this whenever a change can alter a split or morphology result. The
# signature is part of the cache key, and superseded rows are evicted on open.
ANALYSIS_ALGORITHM_VERSION = "hybrid-morphology-v3"
# v3 ranks genuine compound headwords ahead of bare variant-reading pointers;
# the short-lived v2 implementation used an additive score penalty instead.
# v1 could persist an unsplit fallback for direct `attempts=1` calls because
# the wrapper mishandled Parser's list return. Keep that stale result contract
# isolated from the corrected statistical splitter without invalidating hybrid
# morphology records that never used the broken branch.
STATISTICAL_ANALYSIS_ALGORITHM_VERSION = "statistical-splitter-v2"
ACTIVE_ANALYSIS_ALGORITHM_SIGNATURES = (
    ANALYSIS_ALGORITHM_VERSION,
    STATISTICAL_ANALYSIS_ALGORITHM_VERSION,
)
ANALYSIS_SIGNATURE_BY_KIND = {
    "hybrid": ANALYSIS_ALGORITHM_VERSION,
    "hybrid_morphology": ANALYSIS_ALGORITHM_VERSION,
    "statistical": STATISTICAL_ANALYSIS_ALGORITHM_VERSION,
}
_TYPE_TAG = "__process_sanskrit_type__"
_KEY_COLUMNS = (
    "normalized_input",
    "analysis_kind",
    "algorithm_signature",
    "lexicon_fingerprint",
    "settings_json",
)


class CacheConfigurationError(ValueError):
    """Raised when persistent-cache configuration is invalid."""


class TaggedJSON:
    """Strict JSON codec that preserves Python list and tuple distinctions."""

    @classmethod
    def dumps(cls, value: Any) -> str:
        return json.dumps(
            cls._encode(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    @classmethod
    def loads(cls, payload: str) -> Any:
        return cls._decode(json.loads(payload))

    @classmethod
    def _encode(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("cache payload floats must be finite")
            return value
        if isinstance(value, list):
            return {_TYPE_TAG: "list", "items": [cls._encode(item) for item in value]}
        if isinstance(value, tuple):
            return {_TYPE_TAG: "tuple", "items": [cls._encode(item) for item in value]}
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("cache payload dictionaries require string keys")
            return {
                _TYPE_TAG: "dict",
                "items": [
                    [key, cls._encode(value[key])] for key in sorted(value)
                ],
            }
        raise TypeError(f"unsupported cache payload type: {type(value).__name__}")

    @classmethod
    def _decode(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("cache payload floats must be finite")
            return value
        if isinstance(value, list):
            # Untagged arrays are not emitted by this codec.  Rejecting them
            # makes malformed or hand-edited cache rows fail closed.
            raise ValueError("untagged array in cache payload")
        if not isinstance(value, dict):
            raise ValueError("invalid cache payload")
        if set(value) != {_TYPE_TAG, "items"}:
            raise ValueError("invalid tagged cache object")
        tag = value[_TYPE_TAG]
        items = value["items"]
        if not isinstance(items, list):
            raise ValueError("tagged cache items must be a list")
        if tag == "list":
            return [cls._decode(item) for item in items]
        if tag == "tuple":
            return tuple(cls._decode(item) for item in items)
        if tag == "dict":
            decoded: Dict[str, Any] = {}
            for item in items:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or not isinstance(item[0], str)
                    or item[0] in decoded
                ):
                    raise ValueError("invalid tagged dictionary item")
                decoded[item[0]] = cls._decode(item[1])
            return decoded
        raise ValueError(f"unknown cache payload tag: {tag!r}")


def _default_cache_path() -> Path:
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        root = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "process-sanskrit" / "analysis-cache.sqlite3"


def _parse_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise CacheConfigurationError(f"{name} must be true or false")


def _parse_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise CacheConfigurationError(f"{name} must be an integer") from error
    if value <= 0:
        raise CacheConfigurationError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool
    retention: str
    max_age_days: int
    path: Path
    busy_timeout_ms: int = 100
    pool_timeout_seconds: float = 0.1
    prune_interval_seconds: int = 86_400
    touch_interval_seconds: int = 86_400
    prune_batch_size: int = 1_000
    cache_kib: int = 2_048
    wal_autocheckpoint_pages: int = 256

    def __post_init__(self) -> None:
        if self.retention not in {"prune", "keep_all"}:
            raise CacheConfigurationError(
                "PROCESS_SANSKRIT_CACHE_RETENTION must be prune or keep_all"
            )
        if self.max_age_days <= 0:
            raise CacheConfigurationError("cache max_age_days must be positive")
        if self.busy_timeout_ms <= 0 or self.pool_timeout_seconds <= 0:
            raise CacheConfigurationError("cache timeouts must be positive")
        if (
            self.prune_interval_seconds <= 0
            or self.touch_interval_seconds <= 0
            or self.prune_batch_size <= 0
            or self.cache_kib <= 0
            or self.wal_autocheckpoint_pages <= 0
        ):
            raise CacheConfigurationError("cache maintenance settings must be positive")
        object.__setattr__(self, "path", Path(self.path).expanduser().absolute())

    @classmethod
    def from_environment(cls) -> "CacheConfig":
        retention = os.getenv("PROCESS_SANSKRIT_CACHE_RETENTION", "prune")
        configured_path = os.getenv("PROCESS_SANSKRIT_CACHE_PATH")
        return cls(
            enabled=_parse_bool("PROCESS_SANSKRIT_CACHE_ENABLED", True),
            retention=retention.strip().lower(),
            max_age_days=_parse_positive_int(
                "PROCESS_SANSKRIT_CACHE_MAX_AGE_DAYS", 90
            ),
            path=Path(configured_path) if configured_path else _default_cache_path(),
        )


@dataclass(frozen=True)
class CacheKey:
    normalized_input: str
    analysis_kind: str
    algorithm_signature: str
    lexicon_fingerprint: str
    settings_json: str

    @classmethod
    def from_settings(
        cls,
        *,
        normalized_input: str,
        analysis_kind: str,
        algorithm_signature: str,
        lexicon_fingerprint: str,
        settings: Mapping[str, Any],
    ) -> "CacheKey":
        return cls(
            normalized_input=normalized_input,
            analysis_kind=analysis_kind,
            algorithm_signature=algorithm_signature,
            lexicon_fingerprint=lexicon_fingerprint,
            settings_json=TaggedJSON.dumps(dict(settings)),
        )

    def values(self) -> Dict[str, str]:
        return {column: getattr(self, column) for column in _KEY_COLUMNS}


@dataclass
class CacheRecord:
    key: CacheKey
    raw_input: str
    split: Any
    grammar: Any = None
    score: Optional[float] = None
    subscores: Any = None
    result_source: str = "unknown"
    status: str = "success"
    compute_ms: Optional[float] = None
    created_at: Optional[int] = None
    last_accessed_at: Optional[int] = None


cache_metadata = MetaData()

analysis_cache_table = Table(
    "analysis_cache",
    cache_metadata,
    Column("id", Integer, primary_key=True),
    Column("normalized_input", Text, nullable=False),
    Column("analysis_kind", String(32), nullable=False),
    Column("algorithm_signature", String(128), nullable=False),
    Column("lexicon_fingerprint", String(128), nullable=False),
    Column("settings_json", Text, nullable=False),
    Column("raw_input", Text, nullable=False),
    Column("split_payload", Text, nullable=False),
    Column("grammar_payload", Text),
    Column("score", Float),
    Column("subscores_payload", Text),
    Column("result_source", String(32), nullable=False),
    Column("status", String(32), nullable=False),
    Column("compute_ms", Float),
    Column("created_at", Integer, nullable=False),
    Column("last_accessed_at", Integer, nullable=False),
    UniqueConstraint(*_KEY_COLUMNS, name="uq_analysis_cache_key"),
)
Index(
    "ix_analysis_cache_last_accessed_id",
    analysis_cache_table.c.last_accessed_at,
    analysis_cache_table.c.id,
)

cache_metadata_table = Table(
    "cache_metadata",
    cache_metadata,
    Column("key", String(64), primary_key=True),
    Column("value", String(255), nullable=False),
)


class AnalysisCache:
    """Lazy SQLAlchemy Core service for persistent analysis records."""

    def __init__(
        self,
        config: CacheConfig,
        *,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        self.config = config
        self._clock = clock
        self._engine: Optional[Engine] = None
        self._engine_lock = threading.RLock()
        self._key_locks_guard = threading.Lock()
        self._key_locks: Dict[CacheKey, list] = {}
        self._maintenance_lock = threading.Lock()
        self._next_prune_check = 0
        self._logged_failures: set[str] = set()

    @property
    def engine(self) -> Engine:
        engine = self._ensure_engine()
        if engine is None:
            raise RuntimeError("analysis cache is unavailable")
        return engine

    def _log_once(self, operation: str, error: BaseException) -> None:
        with self._engine_lock:
            if operation in self._logged_failures:
                return
            self._logged_failures.add(operation)
        log.warning("Analysis cache %s failed; continuing without it: %s", operation, error)

    def _prepare_path(self) -> None:
        parent = self.config.path.parent
        parent_existed = parent.exists()
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt" and not parent_existed:
            try:
                parent.chmod(0o700)
            except OSError:
                pass
        try:
            descriptor = os.open(
                self.config.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError:
            return
        else:
            os.close(descriptor)

    def _create_engine(self) -> Engine:
        self._prepare_path()
        engine = create_engine(
            URL.create("sqlite+pysqlite", database=str(self.config.path)),
            future=True,
            poolclass=QueuePool,
            pool_size=1,
            max_overflow=1,
            pool_timeout=self.config.pool_timeout_seconds,
            connect_args={
                "check_same_thread": False,
                "timeout": self.config.busy_timeout_ms / 1000,
            },
        )
        expected_pid = os.getpid

        @event.listens_for(engine, "connect")
        def configure_connection(dbapi_connection, connection_record):
            connection_record.info["pid"] = expected_pid()
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute(f"PRAGMA busy_timeout={self.config.busy_timeout_ms}")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute(f"PRAGMA cache_size=-{self.config.cache_kib}")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute(
                    "PRAGMA wal_autocheckpoint="
                    f"{self.config.wal_autocheckpoint_pages}"
                )
            finally:
                cursor.close()

        @event.listens_for(engine, "checkout")
        def guard_fork(dbapi_connection, connection_record, connection_proxy):
            if connection_record.info.get("pid") == expected_pid():
                return
            connection_record.dbapi_connection = None
            connection_proxy.dbapi_connection = None
            raise DisconnectionError("SQLite connection belongs to another process")

        return engine

    def _bootstrap(self, engine: Engine) -> None:
        with engine.connect() as connection:
            # These file-level pragmas must not run inside application DML.
            user_version = connection.exec_driver_sql("PRAGMA user_version").scalar()
            table_names = {
                row[0]
                for row in connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            expected_tables = {
                analysis_cache_table.name,
                cache_metadata_table.name,
            }
            unexpected_tables = table_names - expected_tables
            if unexpected_tables:
                names = ", ".join(sorted(unexpected_tables))
                raise CacheConfigurationError(
                    "analysis cache path contains unrelated tables: " + names
                )
            if user_version == 0 and not table_names:
                connection.exec_driver_sql("PRAGMA auto_vacuum=INCREMENTAL")
            # WAL itself allocates database pages, so negotiate it only after
            # auto_vacuum has been selected for a brand-new file.
            connection.exec_driver_sql("PRAGMA journal_mode=WAL")
            connection.commit()

            # BEGIN IMMEDIATE serializes schema bootstrap across web workers.
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                current = connection.exec_driver_sql("PRAGMA user_version").scalar()
                if current not in (0, SCHEMA_VERSION):
                    raise CacheConfigurationError(
                        f"unsupported analysis cache schema version: {current}"
                    )
                cache_metadata.create_all(connection)
                if current == 0:
                    connection.exec_driver_sql(f"PRAGMA user_version={SCHEMA_VERSION}")

                # Keep each active analysis family while removing superseded
                # signatures and known kinds paired with another family's key.
                signature_column = analysis_cache_table.c.algorithm_signature
                kind_column = analysis_cache_table.c.analysis_kind
                superseded_condition = signature_column.not_in(
                    ACTIVE_ANALYSIS_ALGORITHM_SIGNATURES
                )
                for analysis_kind, active_signature in ANALYSIS_SIGNATURE_BY_KIND.items():
                    superseded_condition |= (kind_column == analysis_kind) & (
                        signature_column != active_signature
                    )
                superseded = connection.execute(
                    analysis_cache_table.delete().where(superseded_condition)
                )
                if superseded.rowcount > 0:
                    log.info(
                        "analysis cache: dropped %d record(s) from superseded "
                        "algorithm versions",
                        superseded.rowcount,
                    )

                now = int(self._clock())
                connection.execute(
                    sqlite_insert(cache_metadata_table)
                    .values(key="last_pruned_at", value=str(now))
                    .on_conflict_do_nothing(
                        index_elements=[cache_metadata_table.c.key]
                    )
                )
                last_pruned = connection.scalar(
                    select(cache_metadata_table.c.value).where(
                        cache_metadata_table.c.key == "last_pruned_at"
                    )
                )
                try:
                    last_pruned_at = int(last_pruned)
                except (TypeError, ValueError) as error:
                    raise CacheConfigurationError(
                        "invalid last_pruned_at cache metadata"
                    ) from error
                connection.commit()
                self._next_prune_check = (
                    last_pruned_at + self.config.prune_interval_seconds
                )
            except BaseException:
                connection.rollback()
                raise

    def _ensure_engine(self) -> Optional[Engine]:
        if not self.config.enabled:
            return None
        if self._engine is not None:
            return self._engine
        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            candidate: Optional[Engine] = None
            try:
                candidate = self._create_engine()
                self._bootstrap(candidate)
            except (OSError, SQLAlchemyError, CacheConfigurationError) as error:
                if candidate is not None:
                    candidate.dispose()
                self._log_once("initialization", error)
                return None
            self._engine = candidate
            return candidate

    @contextmanager
    def compute_lock(self, key: CacheKey) -> Iterator[bool]:
        """Serialize an in-process miss without retaining an unbounded lock map."""
        with self._key_locks_guard:
            entry = self._key_locks.get(key)
            if entry is None:
                entry = [threading.RLock(), 0]
                self._key_locks[key] = entry
            entry[1] += 1
        lock = entry[0]
        acquired = lock.acquire(timeout=self.config.pool_timeout_seconds)
        try:
            yield acquired
        finally:
            if acquired:
                lock.release()
            with self._key_locks_guard:
                entry[1] -= 1
                if entry[1] == 0:
                    self._key_locks.pop(key, None)

    @staticmethod
    def _key_predicate(key: CacheKey):
        predicate = analysis_cache_table.c.normalized_input == key.normalized_input
        for column in _KEY_COLUMNS[1:]:
            predicate = predicate & (
                getattr(analysis_cache_table.c, column) == getattr(key, column)
            )
        return predicate

    @staticmethod
    def _decode_row(row: Mapping[str, Any]) -> CacheRecord:
        for column in (
            "normalized_input",
            "analysis_kind",
            "algorithm_signature",
            "lexicon_fingerprint",
            "settings_json",
            "raw_input",
            "split_payload",
            "result_source",
            "status",
        ):
            if not isinstance(row[column], str):
                raise ValueError(f"invalid scalar cache column: {column}")
        for column in ("created_at", "last_accessed_at"):
            if isinstance(row[column], bool) or not isinstance(row[column], int):
                raise ValueError(f"invalid scalar cache column: {column}")
        for column in ("score", "compute_ms"):
            value = row[column]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"invalid scalar cache column: {column}")
        for column in ("grammar_payload", "subscores_payload"):
            if row[column] is not None and not isinstance(row[column], str):
                raise ValueError(f"invalid scalar cache column: {column}")
        split = TaggedJSON.loads(row["split_payload"])
        if not isinstance(split, list) or not all(
            isinstance(part, str) for part in split
        ):
            raise ValueError("cached split must be a list of strings")
        key = CacheKey(
            normalized_input=row["normalized_input"],
            analysis_kind=row["analysis_kind"],
            algorithm_signature=row["algorithm_signature"],
            lexicon_fingerprint=row["lexicon_fingerprint"],
            settings_json=row["settings_json"],
        )
        return CacheRecord(
            key=key,
            raw_input=row["raw_input"],
            split=split,
            grammar=(
                None
                if row["grammar_payload"] is None
                else TaggedJSON.loads(row["grammar_payload"])
            ),
            score=row["score"],
            subscores=(
                None
                if row["subscores_payload"] is None
                else TaggedJSON.loads(row["subscores_payload"])
            ),
            result_source=row["result_source"],
            status=row["status"],
            compute_ms=row["compute_ms"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
        )

    def _delete_corrupt_row(self, engine: Engine, row_id: int) -> None:
        try:
            with engine.begin() as connection:
                connection.execute(
                    delete(analysis_cache_table).where(
                        analysis_cache_table.c.id == row_id
                    )
                )
        except SQLAlchemyError as error:
            self._log_once("corrupt-row cleanup", error)

    def get(self, key: CacheKey) -> Optional[CacheRecord]:
        engine = self._ensure_engine()
        if engine is None:
            return None
        now = int(self._clock())
        try:
            with engine.connect() as connection:
                row = (
                    connection.execute(
                        select(analysis_cache_table).where(self._key_predicate(key))
                    )
                    .mappings()
                    .first()
                )
            if row is None:
                self._maybe_prune(now)
                return None
            try:
                record = self._decode_row(row)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                self._log_once("payload decoding", error)
                self._delete_corrupt_row(engine, row["id"])
                return None

            if now - row["last_accessed_at"] >= self.config.touch_interval_seconds:
                try:
                    with engine.begin() as connection:
                        connection.execute(
                            update(analysis_cache_table)
                            .where(analysis_cache_table.c.id == row["id"])
                            .where(
                                analysis_cache_table.c.last_accessed_at
                                <= now - self.config.touch_interval_seconds
                            )
                            .values(last_accessed_at=now)
                        )
                    record.last_accessed_at = now
                except SQLAlchemyError as error:
                    self._log_once("access touch", error)
            self._maybe_prune(now)
            return record
        except SQLAlchemyError as error:
            self._log_once("lookup", error)
            return None

    def store(self, record: CacheRecord) -> CacheRecord:
        engine = self._ensure_engine()
        if engine is None:
            return record
        if not isinstance(record.split, list) or not all(
            isinstance(part, str) for part in record.split
        ):
            raise ValueError("cached split must be a list of strings")
        now = int(self._clock())
        created_at = now if record.created_at is None else int(record.created_at)
        values = {
            **record.key.values(),
            "raw_input": record.raw_input,
            "split_payload": TaggedJSON.dumps(record.split),
            "grammar_payload": (
                None if record.grammar is None else TaggedJSON.dumps(record.grammar)
            ),
            "score": record.score,
            "subscores_payload": (
                None
                if record.subscores is None
                else TaggedJSON.dumps(record.subscores)
            ),
            "result_source": record.result_source,
            "status": record.status,
            "compute_ms": record.compute_ms,
            "created_at": created_at,
            "last_accessed_at": now,
        }
        try:
            with self.compute_lock(record.key) as acquired:
                if not acquired:
                    return record
                statement = sqlite_insert(analysis_cache_table).values(**values)
                statement = statement.on_conflict_do_nothing(
                    index_elements=[getattr(analysis_cache_table.c, name) for name in _KEY_COLUMNS]
                )
                with engine.begin() as connection:
                    result = connection.execute(statement)
                if result.rowcount == 1:
                    self._maybe_prune(now)
                    return replace(
                        record,
                        created_at=created_at,
                        last_accessed_at=now,
                    )
                canonical = self.get(record.key)
                return (
                    canonical
                    if canonical is not None
                    else replace(
                        record,
                        created_at=created_at,
                        last_accessed_at=now,
                    )
                )
        except (TypeError, ValueError):
            raise
        except SQLAlchemyError as error:
            self._log_once("write", error)
            return record

    def _maybe_prune(self, now: int) -> None:
        if self.config.retention != "prune":
            return
        with self._maintenance_lock:
            if now < self._next_prune_check:
                return
            # Claim locally before touching SQLite. The persisted metadata in
            # _prune coordinates the same work across other processes.
            self._next_prune_check = now + self.config.prune_interval_seconds
        self._prune(now=now, force=False)

    def prune(self, *, force: bool = False) -> int:
        if self.config.retention != "prune":
            return 0
        return self._prune(now=int(self._clock()), force=force)

    def _prune(self, *, now: int, force: bool) -> int:
        engine = self._ensure_engine()
        if engine is None:
            return 0
        deleted_count = 0
        try:
            with engine.begin() as connection:
                connection.execute(
                    sqlite_insert(cache_metadata_table)
                    .values(key="last_pruned_at", value="0")
                    .on_conflict_do_nothing(index_elements=[cache_metadata_table.c.key])
                )
                if not force:
                    claim = connection.execute(
                        update(cache_metadata_table)
                        .where(cache_metadata_table.c.key == "last_pruned_at")
                        .where(
                            cast(cache_metadata_table.c.value, Integer)
                            <= now - self.config.prune_interval_seconds
                        )
                        .values(value=str(now))
                    )
                    if claim.rowcount == 0:
                        return 0
                cutoff = now - self.config.max_age_days * 86_400
                expired_ids = select(analysis_cache_table.c.id).where(
                    analysis_cache_table.c.last_accessed_at < cutoff
                ).order_by(
                    analysis_cache_table.c.last_accessed_at,
                    analysis_cache_table.c.id,
                ).limit(self.config.prune_batch_size)
                result = connection.execute(
                    delete(analysis_cache_table).where(
                        analysis_cache_table.c.id.in_(expired_ids)
                    )
                )
                deleted_count = max(result.rowcount or 0, 0)
                next_value = (
                    now
                    if deleted_count < self.config.prune_batch_size
                    else now - self.config.prune_interval_seconds + 60
                )
                connection.execute(
                    update(cache_metadata_table)
                    .where(cache_metadata_table.c.key == "last_pruned_at")
                    .values(value=str(next_value))
                )
            with self._maintenance_lock:
                self._next_prune_check = (
                    next_value + self.config.prune_interval_seconds
                )
            if deleted_count:
                self._bounded_maintenance(engine)
            return deleted_count
        except SQLAlchemyError as error:
            self._log_once("pruning", error)
            return 0

    def _bounded_maintenance(self, engine: Engine) -> None:
        try:
            with engine.connect() as connection:
                connection.exec_driver_sql("PRAGMA wal_checkpoint(PASSIVE)")
                connection.exec_driver_sql("PRAGMA incremental_vacuum(32)")
                connection.commit()
        except SQLAlchemyError as error:
            self._log_once("maintenance", error)

    def close(self) -> None:
        with self._engine_lock:
            engine, self._engine = self._engine, None
        if engine is not None:
            engine.dispose()

    def reset_after_fork(self) -> None:
        """Drop inherited pool state in a child process."""
        engine, self._engine = self._engine, None
        self._engine_lock = threading.RLock()
        self._key_locks_guard = threading.Lock()
        self._maintenance_lock = threading.Lock()
        self._key_locks.clear()
        if engine is not None:
            engine.dispose(close=False)


def resolve_cache_enabled(
    requested: Optional[bool], *, configured_default: Optional[bool] = None
) -> bool:
    if requested is not None:
        return requested
    if configured_default is not None:
        return configured_default
    return CacheConfig.from_environment().enabled


@lru_cache(maxsize=4)
def lexicon_fingerprint(db_path: Optional[str] = None) -> str:
    """Build a cheap process-level identity without hashing the 569 MiB DB."""
    if db_path is None:
        from process_sanskrit.utils.databaseSetup import get_db_path

        db_path = get_db_path()
    path = resolve_configured_path(db_path)
    stat = path.stat()
    try:
        from process_sanskrit.setup.updateDB import RELEASE_TAG
    except (ImportError, AttributeError):
        release_tag = "unknown"
    else:
        release_tag = RELEASE_TAG
    identity = f"{release_tag}|{path}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


_cache_singleton: Optional[AnalysisCache] = None
_cache_singleton_lock = threading.RLock()


def get_analysis_cache(
    config: Optional[CacheConfig] = None,
    *,
    force_enabled: bool = False,
) -> AnalysisCache:
    global _cache_singleton
    selected = CacheConfig.from_environment() if config is None else config
    if force_enabled and not selected.enabled:
        selected = replace(selected, enabled=True)
    with _cache_singleton_lock:
        if _cache_singleton is not None:
            if _cache_singleton.config.path != selected.path:
                raise CacheConfigurationError(
                    "analysis cache is already initialized for a different path"
                )
            if force_enabled and not _cache_singleton.config.enabled:
                _cache_singleton = AnalysisCache(selected)
        else:
            _cache_singleton = AnalysisCache(selected)
        return _cache_singleton


def reset_analysis_cache() -> None:
    global _cache_singleton
    with _cache_singleton_lock:
        cache, _cache_singleton = _cache_singleton, None
    if cache is not None:
        cache.close()


def _reset_analysis_cache_after_fork() -> None:
    global _cache_singleton_lock
    cache = _cache_singleton
    _cache_singleton_lock = threading.RLock()
    if cache is not None:
        cache.reset_after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_analysis_cache_after_fork)
