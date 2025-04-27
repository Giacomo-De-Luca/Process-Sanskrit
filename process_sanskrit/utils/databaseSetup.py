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
import importlib.resources
import logging
import threading
from functools import wraps
from contextlib import contextmanager

from typing import Optional, Callable, Any, Generator, TypeVar, cast

from sqlalchemy import create_engine, Column, String, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base, Session

# Configure logging
log = logging.getLogger(__name__)

# Type variable for decorator return type preservation
F = TypeVar('F', bound=Callable[..., Any])

# --- Custom Exceptions ---
class DatabaseNotFoundError(Exception):
    """Raised when the required database file is not found."""
    pass

# --- Base Model Definition ---
Base = declarative_base()

# --- Global Variables for Lazy Loading ---
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None
_scoped_session: Optional[scoped_session] = None
_db_path_checked: bool = False

# --- Database Path Resolution ---
def get_db_path() -> str:
    """
    Get the path to the SQLite database file using importlib.resources.
    
    Returns:
        str: The resolved path to the database file
        
    Note:
        Uses a fallback path if importlib.resources resolution fails
    """
    try:
        # Modern approach (Python 3.9+)
        resources_dir = importlib.resources.files('process_sanskrit').joinpath('resources')
        db_file = resources_dir.joinpath('SQliteDB.sqlite')
        
        # Convert to string path
        with importlib.resources.as_file(db_file) as path:
            return str(path)
    except (ImportError, ModuleNotFoundError, AttributeError, FileNotFoundError) as e:
        # Fallback for older Python versions or unexpected package structure
        log.warning(f"Could not resolve database path with importlib: {e}")
        
        # Try relative resolution from this file
        module_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(module_dir)  # Up one level from utils
        fallback_path = os.path.join(parent_dir, 'resources', 'SQliteDB.sqlite')
        
        log.info(f"Using fallback database path: {fallback_path}")
        return fallback_path

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
    global _engine, _db_path_checked
    
    if _engine is None:
        if db_path is None:
            db_path = get_db_path()
        
        # Check database existence
        if not _db_path_checked and not database_exists(db_path):
            _db_path_checked = True
            error_msg = (
                f"Database file not found at: {db_path}\n"
                "Please run 'update-ps-database' to download and setup the database."
            )
            log.error(error_msg)
            raise DatabaseNotFoundError(error_msg)
        
        _db_path_checked = True
        DATABASE_URL = f"sqlite:///{db_path}"
        
        try:
            # Create engine with optimized pooling configuration for Sanskrit processing
            _engine = create_engine(
                DATABASE_URL,
                pool_size=5,          # Keep 5 connections ready
                max_overflow=10,      # Allow up to 10 more during peak loads
                pool_timeout=30,      # Wait up to 30 seconds for a connection
                pool_recycle=1800,    # Recycle connections after 30 minutes
                connect_args={"check_same_thread": False}  # Allow threads to share connections
            )
            
            # Configure SQLite for better performance and concurrency
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute("PRAGMA journal_mode=WAL;")  # Write-Ahead Logging for better concurrency
                    cursor.execute("PRAGMA busy_timeout=5000;")  # Wait up to 5s if db is locked
                    cursor.execute("PRAGMA synchronous=NORMAL;")  # Faster with slight durability tradeoff
                    cursor.execute("PRAGMA foreign_keys=ON;")    # Enforce foreign key constraints
                    cursor.execute("PRAGMA cache_size=-64000;")  # Use ~64MB memory cache (negative = kilobytes)
                except Exception as e:
                    log.warning(f"Could not set all PRAGMA settings: {e}")
                finally:
                    cursor.close()
                    
            # Test connection
            with _engine.connect() as conn:
                log.info(f"Successfully connected to database at {db_path}")
                
        except OperationalError as e:
            _engine = None
            raise DatabaseNotFoundError(f"Failed to connect to database: {e}") from e
        except Exception as e:
            _engine = None
            log.error(f"Unexpected error creating engine: {e}")
            raise
            
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
        _scoped_session = scoped_session(
            session_factory,
            scopefunc=threading.current_thread
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
    Context manager that handles session lifecycle and transactions.
    
    Yields:
        Session: A session that will be automatically committed or
                rolled back based on whether an exception occurs
                
    Usage:
        with session_scope() as session:
            results = session.query(Model).all()
    """
    session = get_session()
    try:
        yield session
        session.commit()
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

# --- Model Definitions ---
class SplitCache(Base):
    """Table for caching sandhi split results to avoid redundant processing."""
    __tablename__ = 'split_cache'
    input = Column(String, primary_key=True)
    splitted_text = Column(String)

# --- Database Initialization ---
def initialize_database():
    """
    Create all defined tables in the database.
    
    Raises:
        DatabaseNotFoundError: If database file doesn't exist
    """
    try:
        engine = get_engine()
        log.info(f"Creating tables in database: {engine.url}")
        Base.metadata.create_all(engine)
        log.info("Tables created successfully")
    except DatabaseNotFoundError:
        log.error("Cannot initialize tables: database file not found")
        raise
    except Exception as e:
        log.error(f"Error during table creation: {e}")
        raise
