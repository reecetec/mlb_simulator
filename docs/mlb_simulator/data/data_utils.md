Module mlb_simulator.data.data_utils
====================================

Functions
---------

    
`get_db_location() ‑> str`
:   Gets path for mlb.db database 
    
    Default location is set to ~/sports/mlb_simulator/data/databases/mlb.db
    
    Returns:
        str: db path

    
`get_mlb_db_engine() ‑> sqlalchemy.engine.base.Engine`
:   Get a sqlalchemy engine for the mlb.db 
    
    Returns:
        engine: sqlalchemy engine connected to the master db

    
`get_models_location() ‑> str`
:   Gets path for models folder
    
    Returns:
        str: path to models folder

    
`git_clone(repo_url, save_path)`
:   Function to clone a git repo
    
    Args:
        repo_url (str): The url to the git repo you wish to pull
        save_path (str): The path you wish to save the pulled repo to

    
`git_pull(repo_path) ‑> None`
:   Function to pull a git repo
    
    Args:
        repo_path (str): The path to the local git repo you wish to pull

    
`query_mlb_db(query_str) ‑> pandas.core.frame.DataFrame`
:   Function to query the mlb database
    
    Using get_mlb_db_engine(), obtains an engine 
    
    Args:
        query_str (str): the query to send to the mlb database
    
    Returns:
        pd.DataFrame: dataframe of query