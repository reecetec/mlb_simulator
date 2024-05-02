import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import sqlite3
import json
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
from pybaseball import statcast, cache
from data_utils import get_db_locations

#DB_LOCATION = '../../data/databases/mlb.db'
#TABLE_SCHEMAS_PATH = '../../data/databases/table_schema.json'
DB_LOCATION, TABLE_SCHEMAS_PATH = get_db_locations()

logger = logging.getLogger(__name__)

def validate_schema_path():
    """Validates table schema file exists, otherwise script will end
    """
    if not Path(TABLE_SCHEMAS_PATH).exists():
        logger.critical(f'table_schema.json not found at {TABLE_SCHEMAS_PATH}')
        exit()

def get_table_schema(table_name):
    """Check if table is defined in table_schema json, and if so, return it, otherwise, throw error

    Args:
        table_name (Str): table name of schema to look up

    Returns:
        Dict: table schema 
    """
    with open(TABLE_SCHEMAS_PATH, 'r') as file:
        schema = json.load(file)
    
    # validate table schema exists
    if table_name in schema:
        table_schema = schema[table_name]
        return table_schema
    else:
        logger.critical(f'no {table_name} schema found in table_schema.json')
        exit()


def create_table(table_name = 'Statcast'):
    """Creates tables in data/databases/mlb.db for table schemas found in the table_schema.json file

    Args:
        table_name (str, optional): name of table to initialize from table_schema.json file. Defaults to 'Statcast'.
    """
    validate_schema_path()
    table_schema = get_table_schema(table_name)

    # initialize db
    conn = sqlite3.connect(DB_LOCATION)
    c = conn.cursor()
    # generate create table string from schema
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ("
    for column in table_schema['columns']:
        create_table_sql += f"{column['name']} {column['type']}, "
    # add the primary key constraint
    if 'primaryKey' in table_schema:
        create_table_sql += f"PRIMARY KEY ({', '.join(table_schema['primaryKey'])})"
    create_table_sql += ")"

    # execute the SQL statement
    c.execute(create_table_sql)
    conn.commit()
    conn.close()
    logger.info(f'Creation successful for {table_name}')

def update_statcast_table():

    if not Path(DB_LOCATION).exists():
        logger.critical('mlb.db doesnt exist, something went wrong in the creation')
        exit()

    validate_schema_path()

    engine = create_engine(f'sqlite:///{DB_LOCATION}', echo=False)
    max_date = pd.read_sql('select max(game_date) from Statcast',engine)['max(game_date)'][0]
    
    cache.enable()
    today = datetime.today().strftime('%Y-%m-%d')
    # if no data uploaded, get all data from 2017 to today.
    if max_date==None:
        df = statcast('2017-01-01', '2020-01-01', verbose=False)#today)
        if not df.empty:

            # drop garbage
            df = df.drop(columns=['pitcher.1','fielder_2.1'], errors='ignore')

            # ensure no duplicates
            cur_keys = pd.read_sql('select game_date, game_pk, at_bat_number, pitch_number from Statcast', engine)
            key_match_df = pd.merge(df, cur_keys, how='inner', on=['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
            df.drop(key_match_df.index, inplace=True)

            df.to_sql('Statcast', engine, if_exists='append', index=False,
                      chunksize=1000)
        logger.info('Successfully uploaded statcast data from 2017 to 2019')

    # otherwise, get all data from 
    else:
        max_date = datetime.strptime(max_date, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y-%m-%d')
        key_date = (datetime.strptime(max_date, '%Y-%m-%d') - timedelta(days=2)).strftime('%Y-%m-%d')
        df = statcast(max_date, today, verbose=False)
        if not df.empty:
            # drop garbage
            df.drop(columns=['pitcher.1','fielder_2.1'], errors='ignore', inplace=True)

            # ensure no duplicates
            cur_keys = pd.read_sql(f'select game_date, game_pk, at_bat_number, pitch_number from Statcast where game_date > {key_date}', engine)
            cur_keys['game_date'] = pd.to_datetime(cur_keys['game_date'])
            merged = pd.merge(df,cur_keys,how='outer',indicator=True)
            upload_df = merged.loc[merged["_merge"] == "left_only"].drop("_merge", axis=1)

            upload_df.to_sql('Statcast', engine, if_exists='append', index=False,
                      chunksize=1000)
            
        logger.info(f'Successfully uploaded {len(upload_df)} rows to table Statcast')


def main():
    logger.info('Starting SQLite db creation/updates')

    #For each table in the list, make sure it is created with desired schema
    required_tables = ['Statcast']
    for table in required_tables:
        create_table(table)

    #run update function on each table
    logger.info(f'Updating Statcast')
    update_statcast_table()
    
    logger.info('DB creation/updates complete')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
