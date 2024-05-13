# connects to SQL database from data/raw and provides functions to query data
from pathlib import Path
from sqlalchemy import create_engine
import pandas as pd
import os
import pathlib

def get_db_locations():
    """gets paths for database and table schemas

    Returns:
        str, str: db, table schema paths
    """
    home_dir = pathlib.Path.home()

    DB_PATH = os.path.join(home_dir, 'sports', 'mlb_simulator', 'data', 'databases', 'mlb.db')
    TABLE_SCHEMA_PATH = os.path.join(home_dir, 'sports', 'mlb_simulator', 'data', 'databases', 'table_schema.json')

    return DB_PATH, TABLE_SCHEMA_PATH 

def get_mlb_db_engine():
    """Get an engine for the database. Can be used to run an upload script, etc.

    Returns:
        engine: sql engine
    """
    try:
        db_path, _ = get_db_locations()
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        return engine
    except Exception as e:
        print(f'Error creating engine: {e}')
        return None

def query_mlb_db(query_str) -> pd.DataFrame:
    """Function to query the mlb database

    Args:
        query_str (str): the query to send to the mlb database

    Returns:
        pd.DataFrame: dataframe of query, or None if error
    """
    try:
        engine = get_mlb_db_engine()
        df = pd.read_sql(query_str, engine)
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return None


