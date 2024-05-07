# connects to SQL database from data/raw and provides functions to query data
from pathlib import Path
from sqlalchemy import create_engine
import pandas as pd
import pkgutil
import os
import dotenv

def get_db_locations(max_depth=10):
    """Gets path for db file

    Args:
        max_depth (int, optional): max times to move back a directory. Defaults to 10.

    Raises:
        FileNotFoundError: uses README.md to locate root of project directory, so if not found an error is thrown

    Returns:
        strings: paths to database and table_schema json
    """

    DB_PATH = '/home/reece/sports/mlb_simulator/data/databases/mlb.db'
    TABLE_SCHEMA_PATH = '/home/reece/sports/mlb_simulator/data/databases/table_schema.json'
    
    #project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    #dotenv_path = os.path.join(project_dir, '.env')
    #dotenv.load_dotenv(dotenv_path)

    return DB_PATH, TABLE_SCHEMA_PATH #os.environ.get("DB_PATH"), os.environ.get("TABLE_SCHEMA_PATH")

def query_mlb_db(query_str):
    try:
        db_path, _ = get_db_locations()
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        df = pd.read_sql(query_str, engine)
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return None


def get_mlb_db_engine():
    try:
        db_path, _ = get_db_locations()
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        return engine
    except Exception as e:
        print(f'Error creating engine: {e}')
        return None