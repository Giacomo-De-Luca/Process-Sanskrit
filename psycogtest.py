import os
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

try:
    # Get the database URL from environment variable
    DATABASE_URL = os.environ['DATABASE_URL']
    
    # Establish the connection
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    
    # Create a cursor object to interact with the database
    cursor = conn.cursor()
    
    # Execute a simple query to check connection (e.g., version of PostgreSQL)
    cursor.execute('SELECT version();')
    
    # Fetch and print the result
    db_version = cursor.fetchone()
    print(f"Connected to the database. PostgreSQL version: {db_version[0]}")
    
except Exception as e:
    # Print the error if connection fails
    print(f"Error connecting to the database: {e}")
    
finally:
    # Close the connection
    if 'conn' in locals():
        cursor.close()
        conn.close()
        print("Database connection closed.")



try:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL)
    connection = engine.connect()
    print("Connection successful! wo oh oh ! ")
    connection.close()
except Exception as e:
    print(f"Error: {e}")