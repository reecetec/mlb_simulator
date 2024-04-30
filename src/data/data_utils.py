# connects to SQL database from data/raw and provides functions to query data
from pathlib import Path
from sqlalchemy import create_engine
import pandas as pd

def get_db_locations(max_depth=10):
    """Gets path for db file

    Args:
        max_depth (int, optional): max times to move back a directory. Defaults to 10.

    Raises:
        FileNotFoundError: uses README.md to locate root of project directory, so if not found an error is thrown

    Returns:
        strings: paths to database and table_schema json
    """
    
    current_dir = Path(__file__).resolve().parent

    for _ in range(max_depth):
        if (current_dir / 'README.md').exists():
            break
        current_dir = current_dir.parent
    else:
        raise FileNotFoundError("README.md not found within the specified depth.")

    # Append path to db to parent dir
    db_path = current_dir / 'data' / 'databases' / 'mlb.db'
    table_schema_path = current_dir / 'data' / 'databases' / 'table_schema.json'

    return db_path, table_schema_path

def query_mlb_db(query_str):
    try:
        db_path, _ = get_db_locations()
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        df = pd.read_sql(query_str, engine)
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return None