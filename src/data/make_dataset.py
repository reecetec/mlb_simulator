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
from pybaseball import statcast
from scipy.stats import zscore
from data_utils import get_mlb_db_engine, query_mlb_db, get_db_locations

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
    
    #cache.enable()
    today = datetime.today().strftime('%Y-%m-%d')
    # if no data uploaded, get all data from 2017 to today in yr chunks (avoid memory overload)
    if max_date==None:
        for year in range(2017, datetime.now().year + 1):
            df = statcast(f'{year}-01-01', f'{year+1}-01-01', verbose=False)
            if not df.empty:
                # drop garbage
                df = df.drop(columns=['pitcher.1','fielder_2.1'], errors='ignore')
                # upload
                df.to_sql('Statcast', engine, if_exists='append', index=False,
                        chunksize=1000)
            #cache.purge()
            logger.info(f'Successfully uploaded statcast data from {year} to {year + 1}')

    # otherwise, get all data from max date in table
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


def update_woba_strike_tables(min_pitch_count=50, min_hit_count=15, backtest_yr=None):

    if backtest_yr:
        backtest_yr = f'and game_year < {backtest_yr}'
    else:
        backtest_yr = ''

    batter_df = query_mlb_db(f"""
        select batter, pitch_type, type, woba_value
        from Statcast
        where pitch_type <> 'PO' 
            and pitch_type is not null
            and type is not null
            {backtest_yr}
    """)


    """
    get each batters strike percentage for each pitch in pitch arsenal given they have
    at least min_pitch_count of the pitch type thrown to them
    standardize across all batters
    """
    filtered_data = batter_df.groupby(['batter', 'pitch_type']).filter(lambda x: len(x) >= min_pitch_count)
    strike_percentage = filtered_data.groupby(['batter', 'pitch_type'])['type'].apply(lambda x: (x == 'S').mean() * 100)
    strike_percentage_standardized = strike_percentage.groupby('pitch_type').transform(zscore).reset_index()    
    strike_df = strike_percentage_standardized.pivot(index='batter', columns='pitch_type', values='type').fillna(0)

    """
    do the same as above but for avg woba when they hit the ball
    """
    filtered_data = batter_df[batter_df['type'] == 'X'].groupby(['batter', 'pitch_type']).filter(lambda x: (x['type'] == 'X').sum() >= min_hit_count)
    average_woba = filtered_data.groupby(['batter', 'pitch_type'])['woba_value'].mean()
    average_woba_standardized = average_woba.groupby('pitch_type').transform(zscore).reset_index()
    woba_df = average_woba_standardized.pivot(index='batter', columns='pitch_type', values='woba_value').fillna(0)

    #upload df's
    engine = get_mlb_db_engine()

    strike_df.to_sql('BatterStrikePctByPitchType', engine, if_exists='replace')
    woba_df.to_sql('BatterAvgWobaByPitchType', engine, if_exists='replace')


def main():
    logger.info('Starting SQLite db creation/updates')

    #For each table in the list, make sure it is created with desired schema
    required_tables = ['Statcast']
    for table in required_tables:
        create_table(table)

    # update tables
    logger.info(f'Updating Statcast')
    update_statcast_table()

    logger.info(f'Updating BatterBatterStrikePctByPitchType and BatterAvgWobaByPitchType')
    update_woba_strike_tables()
    
    logger.info('DB creation/updates complete')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
