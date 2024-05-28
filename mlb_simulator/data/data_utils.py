# connects to SQL database from data/raw and provides functions to query data
import pathlib
from sqlalchemy import create_engine, engine 
import pandas as pd
import numpy as np
import os
import pathlib
import subprocess

def get_db_location():
    """gets path for database 

    Returns:
        str, str: db, table schema paths
    """
    home_dir = pathlib.Path.home()
    DB_PATH = os.path.join(home_dir, 'sports', 'mlb_simulator', 'data',
                           'databases', 'mlb.db')
    return DB_PATH 

def get_mlb_db_engine() -> engine.Engine | None:
    """Get an engine for the database. Can be used to run an upload script, etc.

    Returns:
        engine: sql engine
    """
    try:
        db_path = get_db_location()
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        return engine
    except Exception as e:
        print(f'Error creating engine: {e}')
        return None

def query_mlb_db(query_str) -> pd.DataFrame | None:
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

def git_pull(repo_path, logger):
    try:
        subprocess.run(['git', 'pull'], cwd=repo_path, check=True)
        logger.info("Repository successfully pulled")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error pulling repository: {e}")

def git_clone(repo_url, save_path, logger):
    try:
        subprocess.run(['git', 'clone', repo_url, save_path], check=True)
        logger.info("Repository successfully cloned")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error cloning repository: {e}")



