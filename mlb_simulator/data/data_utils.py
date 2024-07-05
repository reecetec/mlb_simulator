# connects to SQL database from data/raw and provides functions to query data
import pathlib
from sqlalchemy import create_engine, engine
import pandas as pd
import os
import subprocess


def get_db_location() -> str:
    """Gets path for mlb.db database

    Default location is set to ~/sports/mlb_simulator/data/databases/mlb.db

    Returns:
        str: db path
    """

    home_dir = pathlib.Path.home()
    db_path = os.path.join(home_dir, 'sports', 'mlb_simulator', 'data',
                           'databases', 'mlb.db')
    return db_path


def get_models_location() -> str:
    """Gets path for models folder

    Returns:
        str: path to models folder
    """

    home_dir = pathlib.Path.home()
    path = os.path.join(home_dir, 'sports', 'mlb_simulator', 'models')

    return path


def get_mlb_db_engine() -> engine.Engine:
    """Get a sqlalchemy engine for the mlb.db

    Returns:
        engine: sqlalchemy engine connected to the master db
    """

    try:
        db_path = get_db_location()
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
    except Exception as e:
        print(f'Error creating engine: {e}')
        raise
    return engine


def query_mlb_db(query_str) -> pd.DataFrame:
    """Function to query the mlb database

    Using get_mlb_db_engine(), obtains an engine

    Args:
        query_str (str): the query to send to the mlb database

    Returns:
        pd.DataFrame: dataframe of query
    """

    try:
        engine = get_mlb_db_engine()
        df = pd.read_sql(query_str, engine)
    except Exception as e:
        print(f"Error executing query: {e}")
        raise
    return df


def git_pull(repo_path) -> None:
    """Function to pull a git repo

    Args:
        repo_path (str): The path to the local git repo you wish to pull
    """

    try:
        subprocess.run(['git', 'pull'], cwd=repo_path, check=True)
        print(f"{repo_path} sucessfully pulled")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling repository: {e}")
        raise


def git_clone(repo_url, save_path):
    """Function to clone a git repo

    Args:
        repo_url (str): The url to the git repo you wish to pull
        save_path (str): The path you wish to save the pulled repo to
    """

    try:
        subprocess.run(['git', 'clone', repo_url, save_path], check=True)
        print(f"{repo_url} successfully cloned to {save_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        raise
