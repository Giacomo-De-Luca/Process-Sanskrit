from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite://process_sanskrit/resources/SQliteDB.sqlite"
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

