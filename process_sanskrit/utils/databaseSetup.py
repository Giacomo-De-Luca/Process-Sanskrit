"""
Database setup and connection management for Sanskrit processing.

This module provides optimized database access patterns for:
1. Local text processing with many queries per word
2. Web application usage with concurrent requests

Features:
- Lazy loading to avoid unnecessary database connections
- Connection pooling for efficient query execution
- Session management utilities for reusing connections
- Proper path resolution for database file
- Robust error handling for missing database scenarios
"""

import os
import logging
import threading
from pathlib import Path
from urllib.parse import quote
from functools import wraps
from contextlib import contextmanager

from typing import Optional, Callable, Any, Generator, TypeVar, cast

from sqlalchemy import create_engine, event
from sqlalchemy.exc import DisconnectionError, OperationalError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool

from process_sanskrit.utils.resourcePaths import (
    get_database_path,
    reset_database_path_cache,
    resolve_configured_path,
)

# Configure logging
log = logging.getLogger(__name__)

# Type variable for decorator return type preservation
F = TypeVar('F', bound=Callable[..., Any])

# --- Custom Exceptions ---
class DatabaseNotFoundError(Exception):
    """Raised when the required database file is not found."""
    pass

# --- Global Variables for Lazy Loading ---
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None
_scoped_session: Optional[scoped_session] = None
_engine_path: Optional[Path] = None
_engine_lock = threading.RLock()
_session_lock = threading.RLock()
_database_fork_locks_held = False


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer database setting from the environment."""
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _nonnegative_int_env(name: str, default: int) -> int:
    """Read a non-negative integer database setting from the environment."""
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < 0:
        raise ValueError(f"{name} must be zero or greater")
    return value

# --- Database Path Resolution ---
def get_db_path() -> str:
    """
    Get the path to the SQLite database file.

    Returns:
        str: The resolved path to the database file

    Note:
        A configured but missing path fails explicitly; there is no fallback to
        the packaged database.
    """
    return str(get_database_path())

def database_exists(db_path: Optional[str] = None) -> bool:
    """
    Check if the database file exists at the expected path.
    
    Args:
        db_path: Optional explicit path to check
        
    Returns:
        bool: True if database exists, False otherwise
    """
    if db_path is None:
        db_path = get_db_path()
    return os.path.exists(db_path)

def _install_pid_guards(engine: Engine) -> None:
    """Invalidate pooled DBAPI connections inherited across ``fork()``."""

    @event.listens_for(engine, "connect")
    def remember_pid(dbapi_connection, connection_record):
        connection_record.info["pid"] = os.getpid()

    @event.listens_for(engine, "checkout")
    def reject_parent_connection(
        dbapi_connection, connection_record, connection_proxy
    ):
        if connection_record.info.get("pid") == os.getpid():
            return
        connection_record.dbapi_connection = None
        connection_proxy.dbapi_connection = None
        raise DisconnectionError("SQLite connection belongs to another process")


def _create_read_only_engine(
    db_path: str,
    *,
    pool_size: int,
    max_overflow: int,
    cache_kib: int,
    mmap_size: int,
    pool_timeout: float = 30,
) -> Engine:
    """Construct, but do not globally publish, a read-only lexicon engine."""
    resolved_path = resolve_configured_path(db_path)
    quoted_path = quote(resolved_path.as_posix(), safe="/:")
    database_url = (
        f"sqlite+pysqlite:///file:{quoted_path}"
        "?immutable=1&mode=ro&uri=true"
    )
    engine = create_engine(
        database_url,
        future=True,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        connect_args={"check_same_thread": False},
    )
    _install_pid_guards(engine)

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA query_only=ON;")
            cursor.execute(f"PRAGMA cache_size=-{cache_kib};")
            if mmap_size:
                cursor.execute(f"PRAGMA mmap_size={mmap_size};")
        except Exception as error:
            log.warning("Could not set all lexicon PRAGMAs: %s", error)
        finally:
            cursor.close()
    return engine


# --- Engine and Session Management ---
def get_engine(db_path: Optional[str] = None) -> Engine:
    """
    Lazily create and return the SQLAlchemy engine with optimized pooling.
    
    Args:
        db_path: Optional explicit database path
        
    Returns:
        Engine: Configured SQLAlchemy engine
        
    Raises:
        DatabaseNotFoundError: If database file doesn't exist
    """
    global _engine, _engine_path

    selected_path = (
        resolve_configured_path(db_path)
        if db_path is not None
        else None
    )
    if _engine is not None:
        if selected_path is not None and selected_path != _engine_path:
            raise ValueError(
                f"lexicon engine already initialized for {_engine_path}, "
                f"not {selected_path}"
            )
        return _engine

    with _engine_lock:
        if _engine is not None:
            if selected_path is not None and selected_path != _engine_path:
                raise ValueError(
                    f"lexicon engine already initialized for {_engine_path}, "
                    f"not {selected_path}"
                )
            return _engine
        if selected_path is None:
            selected_path = get_database_path()
        if not database_exists(str(selected_path)):
            error_msg = (
                f"Database file not found at: {selected_path}\n"
                "Please run 'update-ps-database' to download and setup the database."
            )
            log.error(error_msg)
            raise DatabaseNotFoundError(error_msg)

        candidate: Optional[Engine] = None
        try:
            candidate = _create_read_only_engine(
                str(selected_path),
                pool_size=_positive_int_env("PROCESS_SANSKRIT_DB_POOL_SIZE", 2),
                max_overflow=_nonnegative_int_env(
                    "PROCESS_SANSKRIT_DB_MAX_OVERFLOW", 2
                ),
                cache_kib=_positive_int_env(
                    "PROCESS_SANSKRIT_DB_CACHE_KIB", 8192
                ),
                mmap_size=_nonnegative_int_env(
                    "PROCESS_SANSKRIT_DB_MMAP_SIZE", 0
                ),
            )
            with candidate.connect():
                pass
        except OperationalError as error:
            if candidate is not None:
                candidate.dispose()
            raise DatabaseNotFoundError(
                f"Failed to connect to database: {error}"
            ) from error
        except Exception:
            if candidate is not None:
                candidate.dispose()
            raise

        _engine = candidate
        _engine_path = selected_path
        log.info("Successfully connected to database at %s", selected_path)
        return _engine


def get_session_factory() -> sessionmaker:
    """
    Lazily create and return the SQLAlchemy session factory.
    
    Returns:
        sessionmaker: Configured session factory bound to the engine
    """
    global _session_factory
    
    if _session_factory is None:
        engine = get_engine()
        with _session_lock:
            if _session_factory is None:
                _session_factory = sessionmaker(bind=engine)
                log.debug("Created new session factory")
        
    return _session_factory

def get_scoped_session() -> scoped_session:
    """
    Get a thread-local scoped session for multi-threaded web applications.
    
    Returns:
        scoped_session: Thread-local session factory
        
    Note:
        Use this in web applications for thread safety
    """
    global _scoped_session
    
    if _scoped_session is None:
        session_factory = get_session_factory()
        with _session_lock:
            if _scoped_session is None:
                _scoped_session = scoped_session(
                    session_factory,
                    scopefunc=threading.get_ident,
                )
                log.debug("Created new scoped session")
        
    return _scoped_session

def get_session() -> Session:
    """
    Get a new SQLAlchemy session.
    
    Returns:
        Session: A new database session
        
    Note:
        For most operations, consider using session_scope() instead
    """
    session_factory = get_session_factory()
    return session_factory()

# --- Session Management Utilities ---
@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager that handles the lifecycle of a read-only session.
    
    Yields:
        Session: A session that is closed automatically. Failed work is rolled
                back, while successful read-only work needs no commit.
                
    Usage:
        with session_scope() as session:
            results = session.query(Model).all()
    """
    session = get_session()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def with_session(func: F) -> F:
    """
    Decorator that provides a session to the decorated function.
    
    Args:
        func: Function that should receive a session parameter
        
    Returns:
        Wrapped function that automatically gets a session
        
    Usage:
        @with_session
        def process_word(word, session=None):
            # Use session for database operations
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if session already provided in kwargs
        if 'session' in kwargs and kwargs['session'] is not None:
            # Use existing session
            return func(*args, **kwargs)
        else:
            # Create new session
            with session_scope() as session:
                kwargs['session'] = session
                return func(*args, **kwargs)
    return cast(F, wrapper)

def requires_database(func: F) -> F:
    """
    Decorator that ensures database exists before calling function.
    
    Args:
        func: Function requiring database access
        
    Returns:
        Wrapped function that checks database existence
        
    Raises:
        DatabaseNotFoundError: If database file doesn't exist
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not database_exists():
            raise DatabaseNotFoundError(
                f"Function '{func.__name__}' requires database access, but database file not found. "
                "Please run 'update-ps-database' command to download the database."
            )
        return func(*args, **kwargs)
    return cast(F, wrapper)

def _detach_database_state() -> tuple[Optional[scoped_session], Optional[Engine]]:
    """Atomically detach module-owned sessions and the lexicon engine."""
    global _engine, _session_factory, _scoped_session, _engine_path

    with _engine_lock, _session_lock:
        scoped, _scoped_session = _scoped_session, None
        engine, _engine = _engine, None
        _session_factory = None
        _engine_path = None
    return scoped, engine


def _dispose_database_state(
    scoped: Optional[scoped_session], engine: Optional[Engine]
) -> None:
    """Close detached state while it still belongs to the current process."""
    if scoped is not None:
        scoped.remove()
    if engine is not None:
        engine.dispose()


def _reset_database_state() -> None:
    """Dispose global database state; used by tests and controlled shutdown."""
    _dispose_database_state(*_detach_database_state())
    reset_database_path_cache()


def _prepare_database_state_before_fork() -> None:
    """Close SQLite state in the parent before creating a worker process."""
    global _database_fork_locks_held

    _engine_lock.acquire()
    try:
        _session_lock.acquire()
    except BaseException:
        _engine_lock.release()
        raise
    try:
        _dispose_database_state(*_detach_database_state())
    except BaseException:
        _session_lock.release()
        _engine_lock.release()
        raise
    _database_fork_locks_held = True


def _restore_database_state_after_fork_in_parent() -> None:
    """Release lifecycle locks retained by the parent-side fork hook."""
    global _database_fork_locks_held

    if not _database_fork_locks_held:
        return
    _database_fork_locks_held = False
    _session_lock.release()
    _engine_lock.release()


def _reset_database_state_after_fork() -> None:
    """Reset locks and process-local globals in a newly forked child.

    The memoized path cache is deliberately *retained* here, unlike in
    ``_reset_database_state``: it is keyed on the environment value and holds no
    file descriptors or other process-affine state, so the child inherits a
    still-valid entry.  Clearing it would only make every forked worker re-pay
    the cold resolve.
    """
    global _engine, _session_factory, _scoped_session
    global _engine_path, _engine_lock, _session_lock
    global _database_fork_locks_held

    _scoped_session = None
    _engine = None
    _session_factory = None
    _engine_path = None
    _engine_lock = threading.RLock()
    _session_lock = threading.RLock()
    _database_fork_locks_held = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_prepare_database_state_before_fork,
        after_in_parent=_restore_database_state_after_fork_in_parent,
        after_in_child=_reset_database_state_after_fork,
    )
