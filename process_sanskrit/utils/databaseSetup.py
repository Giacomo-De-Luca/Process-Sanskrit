from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
import os
import importlib.resources

def get_db_path():
    """Get the actual filesystem path to the database file"""
    try:
        # Use importlib.resources to find the correct path
        resources = importlib.resources.files('process_sanskrit.resources')
        db_path = resources.joinpath('SQliteDB.sqlite')
        return str(db_path)
    except Exception as e:
        print(f"Error finding database path: {e}", file=sys.stderr)
        return None
db_path = get_db_path()

if not db_path:
    raise RuntimeError("Database path could not be determined.")
else:
    # Use importlib.resources to get the correct path to the database
    DATABASE_URL = f"sqlite:///{db_path}"
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    Base = declarative_base()
    # Define the split_cache model
    class SplitCache(Base):
        __tablename__ = 'split_cache'    
        input = Column(String, primary_key=True)
        splitted_text = Column(String)

    # Create the table if it doesn't exist
    Base.metadata.create_all(engine)


