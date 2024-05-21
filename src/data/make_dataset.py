import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
from tqdm import tqdm
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import requests
import pathlib
import os
import sqlite3
import json
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
import re
import pathlib
import requests
from pybaseball import statcast
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from data_utils import get_mlb_db_engine, query_mlb_db, get_db_locations, git_clone, git_pull
import statsapi
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

def update_chadwick_repo():
    home_dir = pathlib.Path.home()
    repo_save_path = os.path.join(home_dir, 'sports', 'mlb_simulator', 'data', 'raw', 'chadwick')
    
    if os.path.exists(repo_save_path):
        git_pull(repo_save_path, logger)
    else:
        repo_url = 'https://github.com/chadwickbureau/register.git'
        git_clone(repo_url, repo_save_path, logger)

def update_player_name_map():
    home_dir = pathlib.Path.home()
    save_path = os.path.join(home_dir, 'sports', 'mlb_simulator', 'data', 'raw', 'name_map.csv')

    url = 'https://www.smartfantasybaseball.com/PLAYERIDMAPCSV'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    res = requests.get(url, headers=headers)

    if res.status_code == 200:
        # Delete the existing file if it exists
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"Existing file '{save_path}' deleted.")

        # Save the content of the response (CSV file) to the specified path
        open(save_path, 'wb').write(res.content)
        print(f"New file downloaded and saved to '{save_path}'")
    else:
        print(f"Failed to download CSV file. Status code: {res.status_code}")

def update_similar_sz_table():

    player_sz_data = query_mlb_db('''select batter, avg(sz_top) as sz_top, avg(sz_bot) as sz_bot
                                    from Statcast
                                    where sz_top & sz_bot is not null
                                    group by batter''')
    
    X = player_sz_data[['sz_top', 'sz_bot']].values
    
    # normalization or standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # clustering
    kmeans = KMeans(n_clusters=10)  
    kmeans.fit(X_scaled)
    cluster_labels = kmeans.labels_
    
    # add cluster labels 
    player_sz_data['cluster'] = cluster_labels

    engine = get_mlb_db_engine()

    try:
        if engine:
            upload_df = player_sz_data[['batter','cluster']]
            upload_df.to_sql('BatterStrikezoneCluster', engine, if_exists='replace', index=False)
        else:
            raise NameError('Could not create db engine')
    except NameError as e:
        print(e.args)


def update_sz_lookup():

    player_sz_data = query_mlb_db('''select batter, round(avg(sz_top),3) as sz_top, round(avg(sz_bot),3) as sz_bot
                                    from Statcast
                                    where sz_top & sz_bot is not null
                                    group by batter''')

    engine = get_mlb_db_engine()

    try:
        if engine:
            upload_df = player_sz_data
            upload_df.to_sql('BatterStrikezoneLookup', engine, if_exists='replace', index=False)
        else:
            raise NameError('Could not create db engine')
    except NameError as e:
        print(e.args)



def update_stadium_table_old():
    date_range_df = query_mlb_db('select min(game_date) as min_date, max(game_date) as max_date from Statcast')
    min_dt = datetime.strptime(date_range_df['min_date'][0], '%Y-%m-%d %H:%M:%S.%f')
    min_date = min_dt.strftime('%m/%d/%Y')
    max_dt = datetime.strptime(date_range_df['max_date'][0], '%Y-%m-%d %H:%M:%S.%f')
    max_date = max_dt.strftime('%m/%d/%Y')
    games = statsapi.schedule(start_date=min_date,end_date=max_date)
    games_df = pd.DataFrame(games)[['game_id', 'game_datetime', 'game_type', 'venue_id', 'venue_name']]

    engine = get_mlb_db_engine()
    games_df.to_sql('GamePkParkMapping', engine, if_exists='replace', index=False)

def update_park_factors():
    url = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=distance-all&year=2023&batSide=&stat=index_wOBA&condition=All&rolling="
    response = requests.get(url)
    data = re.search(r"data = (.*);", response.text).group(1)
    data = json.loads(data)
    df = pd.DataFrame(data)
    engine = get_mlb_db_engine()
    df.to_sql('ParkFactors', engine, if_exists='replace', index=False)

def update_venue_game_pk_mapping():
    game_pks = query_mlb_db('select distinct game_pk from Statcast')
    venue_game_pks = query_mlb_db('select distinct game_pk from VenueGamePkMapping')
    missing_game_pks = game_pks[~game_pks['game_pk'].isin(venue_game_pks['game_pk'])]['game_pk'].to_list()

    if missing_game_pks:
        sleep_time = 1/10
        venue_data = []
        for game_pk in tqdm(missing_game_pks, total=len(missing_game_pks)):
            res = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?gamePk={game_pk}')
            if res.status_code == 200:
                game_json = res.json()
                venue = game_json['dates'][0]['games'][0]['venue']
                venue['game_pk'] = game_pk
                venue_data.append(venue)
                #dont get rate limited... apparently max is 25/sec, 10/sec to be on safe side
                time.sleep(sleep_time)
        df = pd.DataFrame(venue_data)
        upload_df = df.rename(columns={"name": "venue_name", "id": "venue_id"})[['game_pk','venue_name','venue_id']]
        engine = get_mlb_db_engine()
        upload_df.to_sql('VenueGamePkMapping', engine, if_exists='append', index=False)

        logger.info(f'Successfully uploaded {len(upload_df)} rows to table VenueGamePkMapping')
    else:
        logger.info(f'Successfully uploaded {0} rows to table VenueGamePkMapping')

def update_player_speed():
    url = f"https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=distance-all&year={datetime.now().year}&batSide=&stat=index_wOBA&condition=All&rolling="
    response = requests.get(url)
    data = re.search(r"data = (.*);", response.text).group(1)
    data = json.loads(data)
    df = pd.DataFrame(data)
    engine = get_mlb_db_engine()
    df.to_sql('PlayerSpeed', engine, if_exists='replace', index=False)

def update_oaa():
    url = f"https://baseballsavant.mlb.com/leaderboard/outs_above_average?type=Fielding_Team&startYear=2021&endYear={datetime.now().year}&split=yes&team=&range=year&min=q&pos=of&roles=&viz=hide&sort=5&sortDir=desc"

    response = requests.get(url)
    data = re.search(r"data = (.*);", response.text).group(1)
    data = json.loads(data)
    df = pd.DataFrame(data)

    df = df[['year', 'entity_id', 'entity_name', 'outs_above_average_rhh', 'outs_above_average_lhh']].sort_values(by=['entity_id', 'year'])

    means = df.groupby('year')[['outs_above_average_rhh', 'outs_above_average_lhh']].mean()
    stds = df.groupby('year')[['outs_above_average_rhh', 'outs_above_average_lhh']].std()

    for col in ('outs_above_average_rhh', 'outs_above_average_lhh'):
        df[f'{col}_standardized'] = df.apply(lambda row: (row[col] - means.loc[row['year']][col]) / stds.loc[row['year']][col], axis=1)
    
    column_renames = {
        'outs_above_average_rhh': 'oaa_rhh',
        'outs_above_average_rhh_standardized': 'oaa_rhh_standardized',
        'outs_above_average_lhh': 'oaa_lhh',
        'outs_above_average_lhh_standardized': 'oaa_lhh_standardized'
    }

    df.rename(columns=column_renames, inplace=True)

    engine = get_mlb_db_engine()
    df.to_sql('TeamOAA', engine, if_exists='replace', index=False)

def main():
    logger.info('Starting SQLite db creation/updates')

    # only update this once a month.
    if datetime.now().day == 1:
        logger.info('Updating player name mapping')
        update_player_name_map()
        logger.info('Updating chadwick repo')
        update_chadwick_repo()


    #For each table in the list, make sure it is created with desired schema
    required_tables = ['Statcast']
    for table in required_tables:
        create_table(table)

    # update tables
    logger.info(f'Updating Statcast')
    update_statcast_table()

    logger.info(f'Updating BatterBatterStrikePctByPitchType and BatterAvgWobaByPitchType')
    update_woba_strike_tables()

    logger.info(f'Updating BatterStrikezoneCluster')
    update_similar_sz_table()

    logger.info(f'Updating BatterStrikezoneLookup')
    update_sz_lookup()

    logger.info(f'Updating GamePkParkMapping')
    update_park_factors()

    logger.info(f'Updating VenueGamePkMapping')
    update_venue_game_pk_mapping()

    logger.info(f'Updating TeamOAA')
    update_oaa()
        
    logger.info('DB creation/updates complete')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
