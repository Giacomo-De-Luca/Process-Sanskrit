import os
import sys
import importlib.resources
from sqlalchemy import create_engine, Column, String, exc as sqlalchemy_exc
from sqlalchemy.orm import sessionmaker, declarative_base

# --- Base and Models ---
Base = declarative_base()

class SplitCache(Base):
    __tablename__ = 'split_cache'
    input = Column(String, primary_key=True)
    splitted_text = Column(String)

# --- Globals for Lazy Loading ---
_engine = None
_SessionFactory = None # Renamed to avoid clash with Session property if used
_db_path = None
_db_path_checked = False

# --- Helper Functions (Adapted from previous robust example) ---

def _get_expected_db_path():
    """Internal function to determine the expected DB path using importlib.resources."""
    global _db_path
    if _db_path is None:
        try:
            resource_ref = importlib.resources.files('process_sanskrit').joinpath('resources', 'SQliteDB.sqlite')
            _db_path = str(resource_ref)
        except Exception as e:
            print(f"Error finding DB path: {e}", file=sys.stderr)
            _db_path = None
    return _db_path

def _check_db_exists_and_initialize(engine_instance, db_path_str):
     """Checks DB file, creates tables if needed."""
     try:
         Base.metadata.create_all(engine_instance)
         return True
     except Exception as e:
         print(f"Warning: Error during table creation/check for {db_path_str}: {e}", file=sys.stderr)
         return False


# --- Lazy Loader Functions ---

def _lazy_get_engine():
    """Lazily creates and returns the SQLAlchemy engine. Includes checks."""
    global _engine, _db_path_checked
    if _engine is None:
        db_path = _get_expected_db_path()
        if not db_path:
            _db_path_checked = True
            raise RuntimeError("Database path could not be determined.")

        if not os.path.exists(db_path):
            _db_path_checked = True
            raise FileNotFoundError(
                f"Database file 'SQliteDB.sqlite' not found at expected location: {db_path}. "
                "Please run 'update-ps-database'."
            )

        try:
            DATABASE_URL = f"sqlite:///{db_path}"
            created_engine = create_engine(DATABASE_URL, echo=False)

            if not _check_db_exists_and_initialize(created_engine, db_path):
                 raise sqlalchemy_exc.OperationalError(f"Failed to initialize tables for database at {db_path}")

            _engine = created_engine # Assign only on success
            _db_path_checked = True

        except Exception as e: # Catch engine creation or init errors
            _engine = None
            _db_path_checked = True
            print(f"Error: Failed to create/initialize engine for {db_path}: {e}", file=sys.stderr)
            # Re-raise a more specific error or a generic one
            raise ConnectionError(f"Failed to establish database connection to {db_path}") from e

    return _engine

def _lazy_get_session_factory():
    """Lazily creates and returns the SQLAlchemy session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        engine_instance = _lazy_get_engine() # Get engine (handles creation/errors)
        _SessionFactory = sessionmaker(bind=engine_instance)
    return _SessionFactory

# --- Public Access via Properties (Your Idea) ---
# Provides the original names engine, Session (for factory)
engine = property(lambda: _lazy_get_engine())
Session = property(lambda: _lazy_get_session_factory())

# --- Strongly Recommended Usage (No global 'session' instance) ---
#
# def some_function():
#     try:
#         # Access the factory via the property
#         SessionFactory = Session
#         with SessionFactory() as session_instance: # Create instance per unit of work
#             # Use session_instance
#             result = session_instance.query(SplitCache).first()
#             # ... etc ...
#             session_instance.commit()
#     except (FileNotFoundError, ConnectionError, sqlalchemy_exc.OperationalError) as db_err:
#         print(f"Database not ready or error occurred: {db_err}")
#     except Exception as e:
#         print(f"An unexpected error: {e}")
#